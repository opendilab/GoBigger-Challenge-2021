import os
import numpy as np
import copy
from tensorboardX import SummaryWriter
import sys
sys.path.append('..')

from ding.config import compile_config
from ding.worker import BaseLearner, BattleSampleSerialCollector, BattleInteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import SyncSubprocessEnvManager
from ding.policy import DQNPolicy
from ding.utils import set_pkg_seed, DistributedWriter
from ding.framework import Task, Context, Parallel
from ding.rl_utils import get_epsilon_greedy_fn
from gobigger.agents import BotAgent

from envs import GoBiggerEnv
from model import GoBiggerStructedNetwork
from config.gobigger_no_spatial_config import main_config


class RandomPolicy:
    def __init__(self, action_type_shape: int, player_num: int):
        self.action_type_shape = action_type_shape
        self.player_num = player_num

    def forward(self, data: dict) -> dict:
        return {
            env_id: {
                'action':
                np.random.randint(0,
                                  self.action_type_shape,
                                  size=(self.player_num))
            }
            for env_id in data.keys()
        }

    def reset(self, data_id: list = []) -> None:
        pass


class RulePolicy:
    def __init__(self, team_id: int, player_num_per_team: int):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id +
                                                     1) * player_num_per_team
        self.bot = {str(i): BotAgent(str(i)) for i in range(start, end)}

    def forward(self, data: dict, **kwargs) -> dict:
        ret = {}
        for env_id in data.keys():
            action = []
            for o in data[env_id]:  # len(data[env_id]) = player_num_per_team
                raw_obs = o['collate_ignore_raw_obs']
                raw_obs['overlap']['clone'] = [[
                    x[0], x[1], x[2], int(x[3]),
                    int(x[4])
                ] for x in raw_obs['overlap']['clone']]
                key = str(int(o['player_name']))
                bot = self.bot[key]
                action.append(bot.step(raw_obs))
            ret[env_id] = {'action': np.array(action)}
        return ret

    def reset(self, data_id: list = []) -> None:
        pass


def collecting(task: Task, cfg, tb_logger, policy, seed=0):
    collector = None
    epsilon_greedy = None
    train_iter = None

    def save_learn_session(learn_session):
        nonlocal train_iter
        train_iter = learn_session["train_iter"]

    def _collecting(ctx: Context):
        nonlocal collector, epsilon_greedy
        if collector is None:
            task.on("learn_session", save_learn_session)
            collector_env_num = cfg.env.collector_env_num
            collector_env_cfg = copy.deepcopy(cfg.env)
            collector_env_cfg.train = True
            collector_env = SyncSubprocessEnvManager(env_fn=[
                lambda: GoBiggerEnv(collector_env_cfg)
                for _ in range(collector_env_num)
            ],
                                                     cfg=cfg.env.manager)
            collector_env.seed(seed)
            team_num = cfg.env.team_num
            rule_collect_policy = [
                RulePolicy(team_id, cfg.env.player_num_per_team)
                for team_id in range(1, team_num)
            ]
            collector = BattleSampleSerialCollector(
                cfg.policy.collect.collector,
                collector_env, [policy.collect_mode] + rule_collect_policy,
                tb_logger,
                exp_name=cfg.exp_name)
            eps_cfg = cfg.policy.other.eps
            epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end,
                                                   eps_cfg.decay, eps_cfg.type)

        print("............ Collecting")
        eps = epsilon_greedy(collector.envstep)
        # Sampling data from environments
        new_data, _ = collector.collect(train_iter=train_iter or 0,
                                        policy_kwargs={'eps': eps})

        collect_session = {"new_data": new_data, "env_step": collector.envstep}
        task.emit("collect_session", collect_session)

    return _collecting


def learning(task: Task, cfg, tb_logger, learner, policy, seed=0):
    replay_buffer = None
    env_step = None
    last_eval_iter = 0

    def save_collect_session(collect_session):
        nonlocal replay_buffer, env_step
        if replay_buffer:
            new_data, env_step = collect_session["new_data"], collect_session[
                "env_step"]
            replay_buffer.push(new_data[0], cur_collector_envstep=env_step)
            replay_buffer.push(new_data[1], cur_collector_envstep=env_step)

    def save_eval_session(eval_session):
        nonlocal last_eval_iter
        last_eval_iter = eval_session["last_eval_iter"]

    def _learning(ctx):
        nonlocal learner, replay_buffer, env_step, last_eval_iter
        if replay_buffer is None:
            task.on("collect_session", save_collect_session)
            task.on("eval_session", save_eval_session)
            replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer,
                                              tb_logger=tb_logger,
                                              exp_name=cfg.exp_name)

        print("............ Learning")
        for _ in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(
                learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                return
            learner.train(train_data, env_step)

        learn_session = {
            "train_iter": learner.train_iter,
            "env_step": env_step,
            "state_dict": policy._model.state_dict()
        }
        task.emit("learn_session", learn_session)
        if learner.train_iter - last_eval_iter > cfg.policy.eval.evaluator.eval_freq:
            task.emit_remote("learn_session", learn_session)

    return _learning


def random_evaluating(task: Task,
                      cfg,
                      tb_logger,
                      policy,
                      save_ckpt_fn,
                      seed=0):
    evaluator = None

    def _random_evaluating(ctx: Context):
        nonlocal evaluator
        if evaluator is None:  # Lazy initialize
            evaluator_env_num = cfg.env.evaluator_env_num
            team_num = cfg.env.team_num
            evaluator_env_cfg = copy.deepcopy(cfg.env)
            evaluator_env_cfg.train = False
            evaluator_env = SyncSubprocessEnvManager(env_fn=[
                lambda: GoBiggerEnv(evaluator_env_cfg)
                for _ in range(evaluator_env_num)
            ],
                                                     cfg=cfg.env.manager)
            evaluator_env.seed(seed, dynamic_seed=False)
            eval_policy = RandomPolicy(cfg.policy.model.action_type_shape,
                                       cfg.env.player_num_per_team)
            evaluator = BattleInteractionSerialEvaluator(
                cfg.policy.eval.evaluator,
                evaluator_env, [policy.eval_mode] +
                [eval_policy for _ in range(team_num - 1)],
                tb_logger,
                exp_name=cfg.exp_name,
                instance_name='random_evaluator')

        if ctx.total_step == 0:
            train_iter = 0
            env_step = 0
        else:
            learn_session = task.wait_for("learn_session")[0][0]
            train_iter, env_step, state_dict = learn_session[
                "train_iter"], learn_session["env_step"], learn_session[
                    "state_dict"]
            policy._model.load_state_dict(state_dict)

        print("............ Random Evaluating")
        stop_flag, _, _ = evaluator.eval(save_ckpt_fn, train_iter, env_step)

        eval_session = {"last_eval_iter": train_iter}
        task.emit("eval_session", eval_session)
        task.emit_remote("eval_session", eval_session)

        if stop_flag:
            task.emit("random_stop_flag", True)
            task.emit_remote("random_stop_flag", True)

    return _random_evaluating


def rule_evaluating(task: Task, cfg, tb_logger, policy, save_ckpt_fn, seed=0):
    evaluator = None

    def _rule_evaluating(ctx: Context):
        nonlocal evaluator
        if evaluator is None:  # Lazy initialize
            evaluator_env_num = cfg.env.evaluator_env_num
            team_num = cfg.env.team_num
            evaluator_env_cfg = copy.deepcopy(cfg.env)
            evaluator_env_cfg.train = False
            evaluator_env = SyncSubprocessEnvManager(env_fn=[
                lambda: GoBiggerEnv(evaluator_env_cfg)
                for _ in range(evaluator_env_num)
            ],
                                                     cfg=cfg.env.manager)
            evaluator_env.seed(seed, dynamic_seed=False)
            eval_policy = [
                RulePolicy(team_id, cfg.env.player_num_per_team)
                for team_id in range(1, team_num)
            ]
            evaluator = BattleInteractionSerialEvaluator(
                cfg.policy.eval.evaluator,
                evaluator_env, [policy.eval_mode] + eval_policy,
                tb_logger,
                exp_name=cfg.exp_name,
                instance_name='rule_evaluator')

        if ctx.total_step == 0:
            train_iter = 0
            env_step = 0
        else:
            learn_session = task.wait_for("learn_session")[0][0]
            train_iter, env_step, state_dict = learn_session[
                "train_iter"], learn_session["env_step"], learn_session[
                    "state_dict"]
            policy._model.load_state_dict(state_dict)

        print("............ Rule Evaluating")
        stop_flag, _, _ = evaluator.eval(save_ckpt_fn, train_iter, env_step)

        eval_session = {"last_eval_iter": train_iter}
        task.emit("eval_session", eval_session)
        task.emit_remote("eval_session", eval_session)

        if stop_flag:
            task.emit("rule_stop_flag", True)
            task.emit_remote("rule_stop_flag", True)

    return _rule_evaluating


def main(seed=0, max_iterations=int(1e10)):
    with Task(async_mode=True, n_async_workers=6, auto_sync_ctx=False) as task:
        cfg = compile_config(main_config,
                             SyncSubprocessEnvManager,
                             DQNPolicy,
                             BaseLearner,
                             BattleSampleSerialCollector,
                             BattleInteractionSerialEvaluator,
                             NaiveReplayBuffer,
                             save_cfg=True)
        LOGGER_PATH = os.environ.get("LOGGER_PATH") or os.path.join(
            './{}/log/'.format(cfg.exp_name), 'serial')
        tb_logger = DistributedWriter(LOGGER_PATH).plugin(
            task, is_writer=("node.0" in task.labels))
        set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
        model = GoBiggerStructedNetwork(**cfg.policy.model)
        policy = DQNPolicy(cfg.policy, model=model)
        learner = BaseLearner(cfg.policy.learn.learner,
                              policy.learn_mode,
                              tb_logger,
                              exp_name=cfg.exp_name,
                              instance_name='learner')
        # Pipeline start
        task.use(random_evaluating(task,
                                   cfg=cfg,
                                   tb_logger=tb_logger,
                                   policy=policy,
                                   save_ckpt_fn=learner.save_checkpoint,
                                   seed=seed),
                 filter_labels=["standalone", "node.1"])
        task.use(rule_evaluating(task,
                                 cfg=cfg,
                                 tb_logger=tb_logger,
                                 policy=policy,
                                 save_ckpt_fn=learner.save_checkpoint,
                                 seed=seed),
                 filter_labels=["standalone", "node.2"])
        task.use(collecting(task,
                            cfg=cfg,
                            tb_logger=tb_logger,
                            policy=policy,
                            seed=0),
                 filter_labels=["standalone", "node.0"])
        task.use(learning(task,
                          cfg=cfg,
                          tb_logger=tb_logger,
                          learner=learner,
                          policy=policy,
                          seed=0),
                 filter_labels=["standalone", "node.0"])
        task.run(max_step=max_iterations)


if __name__ == "__main__":
    # main()
    Parallel.runner(n_parallel_workers=3, topology="star")(main)
