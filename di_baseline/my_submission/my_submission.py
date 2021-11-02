import random
import os
import gym
import numpy as np
import copy
import torch
import time

from ding.config import compile_config
from ding.policy import DQNPolicy
from .envs import GoBiggerEnv
from .model import GoBiggerStructedNetwork
from .config.gobigger_no_spatial_config import main_config


class BaseSubmission:

    def __init__(self, team_name, player_names):
        self.team_name = team_name
        self.player_names = player_names

    def get_actions(self, obs):
        '''
        Overview:
            You must implement this function.
        '''
        raise NotImplementedError


class MySubmission(BaseSubmission):

    def __init__(self, team_name, player_names):
        super(MySubmission, self).__init__(team_name, player_names)
        self.cfg = copy.deepcopy(main_config)
        self.cfg = compile_config(
            self.cfg,
            policy=DQNPolicy,
            save_cfg=False,
        )
        print(self.cfg)
        self.root_path = os.path.abspath(os.path.dirname(__file__))
        self.model = GoBiggerStructedNetwork(**self.cfg.policy.model)
        self.model.load_state_dict(torch.load(os.path.join(self.root_path, 'supplements', 'ckpt_best.pth.tar'), map_location='cpu')['model'])
        self.policy = DQNPolicy(self.cfg.policy, model=self.model).eval_mode
        self.env = GoBiggerEnv(self.cfg.env)

    def get_actions(self, obs):
        obs_transform = self.env._obs_transform(obs)[0]
        obs_transform = {0: obs_transform}
        raw_actions = self.policy.forward(obs_transform)[0]['action']
        raw_actions = raw_actions.tolist()
        actions = {n: GoBiggerEnv._to_raw_action(a) for n, a in zip(obs[1].keys(), raw_actions)}
        return actions
