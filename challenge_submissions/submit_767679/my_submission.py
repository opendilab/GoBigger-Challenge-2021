import collections
import copy
import functools
import math
import os
import queue
from collections import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def single_as_batch(func):
    def _recursive_processing(x, squeeze=False):
        if isinstance(x, Sequence):
            return (_recursive_processing(_, squeeze) for _ in x)
        elif isinstance(x, dict):
            return {k: _recursive_processing(v, squeeze) for k, v in x.items()}
        else:
            return x.squeeze(0) if squeeze else x.unsqueeze(0)

    @functools.wraps(func)
    def wrap(self, *tensors):
        tensors = _recursive_processing(tensors)
        result = func(self, *tensors)
        return _recursive_processing(result, squeeze=True)

    return wrap


class GoBiggerModel(nn.Module):
    def __init__(self, n_ball, n_pi_output, n_v_output):
        super().__init__()
        self.n_ball = n_ball

        n_basic_input = 4344
        n_self_input = 79
        n_ally_input = 79
        n_oppo_input = 369

        self.n_self_fc_hidden = 200
        self.n_ally_fc_hidden = 200
        self.n_oppo_fc_hidden = 300

        self.basic_fc = nn.Sequential(
            nn.Linear(n_basic_input, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.self_fc = nn.Sequential(
            nn.Linear(n_self_input, self.n_self_fc_hidden),
            nn.ReLU(),
            nn.Linear(self.n_self_fc_hidden, self.n_self_fc_hidden),
            nn.ReLU(),
        )
        self.ally_fc = nn.Sequential(
            nn.Linear(n_ally_input, self.n_ally_fc_hidden),
            nn.ReLU(),
            nn.Linear(self.n_ally_fc_hidden, self.n_ally_fc_hidden),
            nn.ReLU(),
        )
        self.oppo_fc = nn.Sequential(
            nn.Linear(n_oppo_input, self.n_oppo_fc_hidden),
            nn.ReLU(),
            nn.Linear(self.n_oppo_fc_hidden, self.n_oppo_fc_hidden),
            nn.ReLU(),
        )

        hidden_in = 1024 + 200 + 200 + 300
        n_hidden = 1024

        self.policy = nn.Sequential(
            nn.Linear(hidden_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, n_pi_output),
        )
        self.values = nn.Sequential(
            nn.Linear(hidden_in * n_ball, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_v_output),
        )

    @single_as_batch
    def infer(self, state):
        return self.forward(state)

    def forward(self, state):
        batch_size = state['global_vec'].shape[0]
        device = state['global_vec'].device

        basic_states = []
        self_states = []
        ally_states = []
        oppo_states = []
        clone_masks = []
        for i in range(self.n_ball):
            basic_state = torch.cat([
                state['global_vec'],
                state[f'player{i}_vec']['basic'],
                state[f'player{i}_vec']['fst_rings'].reshape((batch_size, -1)),
                state[f'player{i}_vec']['ally_rings'].reshape((batch_size, -1)),
                state[f'player{i}_vec']['oppo_rings'].reshape((batch_size, -1)),
                state[f'player{i}_vec']['food'].reshape((batch_size, -1)),
                state[f'player{i}_vec']['spore'].reshape((batch_size, -1)),
                state[f'player{i}_vec']['thorn'].reshape((batch_size, -1)),
            ], dim=-1)
            common_part = torch.cat([
                state[f'player{i}_vec']['per_self_clone'],
                torch.eye(16, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
            ], dim=-1)
            clone_masks.append(state[f'player{i}_vec']['clone_mask'])

            self_state = torch.cat([state[f'player{i}_vec']['self'], common_part], dim=-1)
            ally_state = torch.cat([state[f'player{i}_vec']['ally'], common_part], dim=-1)
            oppo_state = torch.cat([state[f'player{i}_vec']['oppo'], common_part], dim=-1)
            basic_states.append(basic_state)
            self_states.append(self_state)
            ally_states.append(ally_state)
            oppo_states.append(oppo_state)

        x_basic = torch.stack(basic_states, dim=1)  # [batch_size, n_ball, n_feature]
        x_self = torch.stack(self_states, dim=1)  # [batch_size, n_ball, n_self_clone, n_feature]
        x_ally = torch.stack(ally_states, dim=1)
        x_oppo = torch.stack(oppo_states, dim=1)
        clone_masks = torch.stack(clone_masks, dim=1).unsqueeze(-1)

        basic_hidden = self.basic_fc(x_basic)
        self_hidden = (self.self_fc(x_self) * clone_masks.repeat(1, 1, 1, self.n_self_fc_hidden)).max(2)[0]
        ally_hidden = (self.ally_fc(x_ally) * clone_masks.repeat(1, 1, 1, self.n_ally_fc_hidden)).max(2)[0]
        oppo_hidden = (self.oppo_fc(x_oppo) * clone_masks.repeat(1, 1, 1, self.n_oppo_fc_hidden)).max(2)[0]
        rel_hidden = torch.cat([basic_hidden, self_hidden, ally_hidden, oppo_hidden], dim=-1)

        logits = self.policy(rel_hidden)
        value_in = torch.cat([rel_hidden.reshape(rel_hidden.shape[0], -1)], dim=-1)
        values = self.values(value_in)
        return logits, values


def tensorize_state(func):
    def _recursive_processing(state, device):
        if not isinstance(state, torch.Tensor):
            if isinstance(state, dict):
                for k, v in state.items():
                    state[k] = _recursive_processing(state[k], device)
            else:
                state = torch.FloatTensor(state).to(device)
        return state

    @functools.wraps(func)
    def wrap(self, state, *arg, **kwargs):
        state = copy.deepcopy(state)
        state = _recursive_processing(state, self.device)
        return func(self, state, *arg, **kwargs)

    return wrap


def legal_mask(logit, legal):
    mask = torch.ones_like(legal) * -math.inf
    logit = torch.where(legal == 1., logit, mask)
    return logit


class Agent:
    def __init__(self, use_gpu: bool, *args, **kwargs):
        self.use_gpu = use_gpu

        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
        self.state_handler_dict = {}

        torch.set_num_threads(1)
        self.training_iter = 0

    def register_model(self, name, model):
        assert isinstance(model, nn.Module)
        if name in self.state_handler_dict:
            raise KeyError(f"model named with {name} reassigned.")
        self.state_handler_dict[name] = model

    def loads(self, agent_dict):
        self.training_iter = agent_dict['training_iter']

        for name, np_dict in agent_dict['model_dict'].items():
            model = self.state_handler_dict[name]  # alias
            state_dict = {
                k: torch.as_tensor(v.copy(), device=self.device)
                for k, v in zip(model.state_dict().keys(), np_dict.values())
            }
            model.load_state_dict(state_dict)


class GoBiggerAgent(Agent):
    def __init__(self, use_gpu, *, net_conf):
        super().__init__(use_gpu)
        self.net = GoBiggerModel(**net_conf).to(self.device)
        self.register_model('net', self.net)

        self._n_ball = net_conf['n_ball']

    @tensorize_state
    def infer(self, state):
        with torch.no_grad():
            logits, value = self.net.infer(state)
            legals = torch.stack([state[f'player{i}_legal'] for i in range(self._n_ball)])
            logits = legal_mask(logits, legals)

            # prob = F.softmax(logits, dim=-1).squeeze()
            # action = prob.argmax(dim=-1).numpy()

            prob = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=prob)
            action = dist.sample()
            action = action.numpy()
        return action


def one_hot_generator(n_feature, index):
    one_hot = np.zeros(n_feature,)
    one_hot[index] = 1
    return one_hot


def multi_hot_generator(n_feature, index):
    one_hot = np.zeros(n_feature,)
    one_hot[:index] = 1
    return one_hot


def history_queues(n_queue, n_item, item_dim):
    history = [
        collections.deque([np.zeros(item_dim,)] * n_item, maxlen=n_item)
        for _ in range(n_queue)
    ]
    return history


ACTION_TYPE_MAPPING = {
    -1: 'move',
    0: 'eject',
    1: 'split',
    2: 'stop',
    3: 'eject',
    4: 'split',
}
ACTION_TYPES = sorted(list(set(ACTION_TYPE_MAPPING.values())))

N_MOVE_DIR = 8
N_EJECT_DIR = 8
N_SPLIT_DIR = 32


class AtomicActions:
    def __init__(self):
        self.atoms = []
        for action_type in [-1, 0, 1]:  # TODO: enable type 3/4
            self.atoms.append([None, None, action_type])
            n_dir = {-1: N_MOVE_DIR, 0: N_EJECT_DIR, 1: N_SPLIT_DIR}[action_type]
            for angle in [2 * np.pi / n_dir * i for i in range(1, n_dir + 1)]:
                x = self._round(np.cos(angle))
                y = self._round(np.sin(angle))
                self.atoms.append([x, y, action_type])
        self.atoms.append([None, None, 2])
        self.queuing_atoms = queue.Queue()

    @staticmethod
    def _round(z):
        if abs(z) < 1e-6:
            z = 0.
        if abs(z - 1) < 1e-6:
            z = 1.
        return z

    def get_atom(self, idx):
        if idx < len(self.atoms):
            atom = self.atoms[idx]
        else:
            # call for macro
            if self.queuing_atoms.qsize() == 0:
                self.queuing_atoms.put([None, None, 2])
                self.queuing_atoms.put([None, None, -1])
                self.queuing_atoms.put([None, None, -1])
                self.queuing_atoms.put([None, None, -1])
                self.queuing_atoms.put([None, None, -1])
                self.queuing_atoms.put([None, None, -1])
                self.queuing_atoms.put([None, None, -1])
                self.queuing_atoms.put([None, None, 0])
                self.queuing_atoms.put([None, None, 0])
                self.queuing_atoms.put([None, None, 0])
                self.queuing_atoms.put([None, None, 0])
                self.queuing_atoms.put([None, None, 0])
                self.queuing_atoms.put([None, None, 0])
                self.queuing_atoms.put([None, None, 0])
                self.queuing_atoms.put([None, None, 0])
            atom = self.queuing_atoms.get()
        return atom

    def __len__(self):
        return len(self.atoms) + 1  # 1 macro for middle gathering

    def get_legal(self, ejectable, splittable, midgatherable):
        mask = np.ones(len(self))
        if self.queuing_atoms.qsize() > 0:
            mask[:-1] = 0
        else:
            if not ejectable:
                mask[N_MOVE_DIR + 1: N_MOVE_DIR + N_EJECT_DIR + 2] = 0
            if not splittable:
                mask[N_MOVE_DIR + N_EJECT_DIR + 2: -1] = 0
            if not midgatherable:
                mask[-1] = 0
        return mask


class Translator:
    MAX_CLONE_BALLS = 16
    MIN_SKILL_RADIUS = 10

    def __init__(self, len_center_hist=20, len_speed_hist=5, len_action_hist=10):
        self.len_center_hist = len_center_hist
        self.len_speed_hist = len_speed_hist
        self.len_action_hist = len_action_hist

        self.atom_actions = None
        self.raw_obs = None
        self.n_team = None
        self.n_player_per_team = None
        self.n_player = None
        self.team_name = None
        self.controlled_player_names = None
        self.all_player_names = None
        self.centroid_history = None  # record motion trail
        self.avg_pos_history = None  # record motion trail
        self.centroid_speed_history = None  # can infer acceleration
        self.avg_pos_speed_history = None  # can infer acceleration
        self.action_history = None
        self.n_frame_since_last_action = None
        self.n_frame_an_action_last = None

    def reset(self, init_obs):
        global_obs, players_obs = init_obs

        self.n_team = len(global_obs['leaderboard'])
        self.n_player_per_team = len(players_obs)
        self.n_player = self.n_team * self.n_player_per_team
        self.team_name = next(iter(players_obs.values()))['team_name']
        self.controlled_player_names = self._get_player_names_by_team(self.team_name)
        self.all_player_names = list(map(str, range(self.n_player)))

        self.centroid_history = history_queues(
            self.n_player_per_team, self.len_center_hist, item_dim=2)
        self.avg_pos_history = history_queues(
            self.n_player_per_team, self.len_center_hist, item_dim=2)
        self.centroid_speed_history = history_queues(
            self.n_player_per_team, self.len_speed_hist, item_dim=2)
        self.avg_pos_speed_history = history_queues(
            self.n_player_per_team, self.len_speed_hist, item_dim=2)
        self.action_history = history_queues(
            self.n_player_per_team, self.len_action_hist, item_dim=2 + len(ACTION_TYPE_MAPPING))

        self.n_frame_since_last_action = [
            {k: 0 for k in ACTION_TYPES}
            for _ in range(self.n_player_per_team)
        ]
        self.n_frame_an_action_last = [
            {k: 0 for k in ACTION_TYPES}
            for _ in range(self.n_player_per_team)
        ]

        self.atom_actions = [AtomicActions() for _ in range(self.n_player_per_team)]

    def handle_obs(self, raw_obs):
        self.raw_obs = raw_obs

        global_obs, players_obs = raw_obs
        last_time = global_obs['last_time']

        global_vec = self.extract_global_vec(global_obs)
        teamview_clones = sum([player_obs['overlap']['clone'] for _, player_obs in players_obs.items()], [])
        teamview_clones = set(tuple(c) for c in teamview_clones)  # remove duplicates

        player_vecs = []
        player_legals = []
        assert list(players_obs.keys()) == sorted(players_obs, key=int)
        for player_idx, (_, player_data) in enumerate(players_obs.items()):
            player_feature, legal_mask = self.extract_player_vec(
                player_idx, player_data, teamview_clones, last_time)
            player_vecs.append(player_feature)
            player_legals.append(legal_mask)

        features = {
            'global_vec': global_vec,
            **{f'player{i}_vec': vec for i, vec in enumerate(player_vecs)},
            **{f'player{i}_legal': vec for i, vec in enumerate(player_legals)},
        }
        return features

    def handle_action(self, action):
        atomized = {}

        _, players_obs = self.raw_obs
        for i, (player_name, player_action) in enumerate(zip(self.controlled_player_names, action)):
            atomized[player_name] = self.atom_actions[i].get_atom(player_action)
            action_type_idx = int(atomized[player_name][-1])
            action_dir = atomized[player_name][:2]

            action_feature = np.concatenate([
                action_dir if action_dir != [None, None] else [0, 0],
                one_hot_generator(len(ACTION_TYPE_MAPPING), action_type_idx),
            ])
            self.action_history[i].append(action_feature)

            action_type_name: str = ACTION_TYPE_MAPPING[action_type_idx]
            for k in ACTION_TYPES:
                if k != action_type_name:
                    self.n_frame_since_last_action[i][k] = min(self.n_frame_since_last_action[i][k] + 1, 100)
                    self.n_frame_an_action_last[i][k] = 0
                else:
                    self.n_frame_since_last_action[i][k] = 0
                    self.n_frame_an_action_last[i][k] = min(self.n_frame_an_action_last[i][k] + 1, 20)
        return atomized

    def _get_player_names_by_team(self, team_name):
        player_names = [
            str(i + int(team_name) * self.n_player_per_team)
            for i in range(self.n_player_per_team)
        ]
        return player_names

    def extract_global_vec(self, global_obs):
        game_progress = global_obs['last_time'] / global_obs['total_time']
        progress_multihot = multi_hot_generator(n_feature=10, index=int(game_progress * 10) + 1)

        team_idx = int(self.team_name)
        team_scores = [s / 3000. for s in global_obs['leaderboard'].values()]
        my_score = team_scores[team_idx]
        del team_scores[team_idx]
        other_scores = np.asarray(sorted(team_scores))
        other_advantages = other_scores - my_score

        features = np.concatenate([
            [game_progress],
            progress_multihot,
            [my_score],
            other_scores,
            other_advantages,
        ])
        return features

    def extract_player_vec(self, idx_in_team, player_obs, teamview_clones, last_time, map_size=1000):
        n_player = len(self.all_player_names)
        n_player_per_team = len(self.controlled_player_names)
        n_team = n_player // n_player_per_team

        my_name = self.controlled_player_names[idx_in_team]
        my_team_name = player_obs['team_name']
        ally_names = [name for name in self.controlled_player_names if name != my_name]
        oppo_names = [name for name in self.all_player_names if name not in self.controlled_player_names]
        my_centroid_history = self.centroid_history[idx_in_team]
        my_avg_pos_history = self.avg_pos_history[idx_in_team]
        my_centroid_speed_history = self.centroid_speed_history[idx_in_team]
        my_avg_pos_speed_history = self.avg_pos_speed_history[idx_in_team]

        rectangle = player_obs['rectangle']
        food = player_obs['overlap']['food']
        thorns = player_obs['overlap']['thorns']
        spore = player_obs['overlap']['spore']

        my_clones = [c for c in teamview_clones if c[-2] == my_name]
        ally_clones = [c for c in teamview_clones if c[-2] in ally_names]
        ally1_clones = [c for c in teamview_clones if c[-2] in ally_names[0]]
        ally2_clones = [c for c in teamview_clones if c[-2] in ally_names[1]]
        oppo_clones = [c for c in teamview_clones if c[-2] in oppo_names]

        my_centroid = np.mean([(c[0] * c[2] ** 2, c[1] * c[2] ** 2) for c in my_clones], axis=0) / np.sum([c[2] ** 2 for c in my_clones])
        ally1_centroid = np.mean([(c[0] * c[2] ** 2, c[1] * c[2] ** 2) for c in ally1_clones], axis=0) / np.sum([c[2] ** 2 for c in ally1_clones])
        ally2_centroid = np.mean([(c[0] * c[2] ** 2, c[1] * c[2] ** 2) for c in ally2_clones], axis=0) / np.sum([c[2] ** 2 for c in ally2_clones])
        ally1_cent_dir = ally1_centroid - my_centroid
        ally2_cent_dir = ally2_centroid - my_centroid
        ally1_cent_dist = np.linalg.norm(ally1_cent_dir)
        ally2_cent_dist = np.linalg.norm(ally2_cent_dir)

        my_avg_pos = np.mean([(c[0], c[1]) for c in my_clones], axis=0)
        ally1_avg_pos = np.mean([(c[0], c[1]) for c in ally1_clones], axis=0)
        ally2_avg_pos = np.mean([(c[0], c[1]) for c in ally2_clones], axis=0)
        ally1_avg_pos_dir = ally1_avg_pos - my_avg_pos
        ally2_avg_pos_dir = ally2_avg_pos - my_avg_pos
        ally1_avg_pos_dist = np.linalg.norm(ally1_avg_pos_dir)
        ally2_avg_pos_dist = np.linalg.norm(ally2_avg_pos_dir)

        close_to_margins = np.array([
            any([c[0] - c[2] < 5 for c in my_clones]),
            any([c[1] - c[2] < 5 for c in my_clones]),
            any([map_size - (c[0] + c[2]) < 5 for c in my_clones]),
            any([map_size - (c[1] + c[2]) < 5 for c in my_clones]),
        ])
        very_close_to_margins = np.array([
            any([c[0] - c[2] < 1 for c in my_clones]),
            any([c[1] - c[2] < 1 for c in my_clones]),
            any([map_size - (c[0] + c[2]) < 1 for c in my_clones]),
            any([map_size - (c[1] + c[2]) < 1 for c in my_clones]),
        ])

        # sort by radius, so that the features of the largest will always be placed at the beginning, then the 2nd, 3rd ...
        my_clones = sorted(my_clones, key=lambda c: c[2])
        n_my_clone = len(my_clones)

        my_clones_radius = np.pad([c[2] for c in my_clones], (0, self.MAX_CLONE_BALLS - n_my_clone))
        n_clone_multihot = multi_hot_generator(n_feature=self.MAX_CLONE_BALLS, index=n_my_clone)
        my_clone_pos = np.pad(np.array([c[:2] for c in my_clones]), ((0, self.MAX_CLONE_BALLS - n_my_clone), (0, 0)))
        my_clone_margins = np.pad(1000. - np.array([c[:2] for c in my_clones]), ((0, self.MAX_CLONE_BALLS - n_my_clone), (0, 0)))

        per_self_clone_arr = np.concatenate([
            np.sqrt(my_clones_radius).reshape(-1, 1) / 10.,
            my_clone_pos / 1000.,
            my_clone_margins / 1000.,
        ], axis=-1)

        left_x, top_y, right_x, bottom_y = rectangle
        right_margin, bottom_margin = map_size - right_x, map_size - bottom_y
        width, height = right_x - left_x, bottom_y - top_y
        # center = np.array([(left_x + right_x) / 2, (top_y + bottom_y) / 2])
        my_centroid_history.append(my_centroid)
        my_avg_pos_history.append(my_avg_pos)
        if last_time > 0:
            centroid_speed = my_centroid_history[-1] - my_centroid_history[-2]
            my_centroid_speed_history.append(centroid_speed)
            avg_pos_speed = my_avg_pos_history[-1] - my_avg_pos_history[-2]
            my_avg_pos_speed_history.append(avg_pos_speed)

        # legal action related
        ejectable = any(clone[2] >= self.MIN_SKILL_RADIUS for clone in my_clones)
        splittable = ejectable and n_my_clone < self.MAX_CLONE_BALLS
        midgatherable = n_my_clone >= 9 and my_clones[4][2] > 14  # at least 9 clone balls, and the radius of the 5th largest clone is greater than 14
        legal_mask = self.atom_actions[idx_in_team].get_legal(ejectable, splittable, midgatherable)

        basic_arr = np.concatenate([
            my_centroid / 1000.,
            ally1_centroid / 1000.,
            ally2_centroid / 1000.,
            ally1_cent_dir / 1000.,
            ally2_cent_dir / 1000.,
            [ally1_cent_dist / 1000., ally2_cent_dist / 1000.],
            my_avg_pos / 1000.,
            ally1_avg_pos / 1000.,
            ally2_avg_pos / 1000.,
            ally1_avg_pos_dir / 1000.,
            ally2_avg_pos_dir / 1000.,
            [ally1_avg_pos_dist / 1000., ally2_avg_pos_dist / 1000.],
            close_to_margins,
            very_close_to_margins,
            np.asarray(rectangle) / 1000.,
            np.asarray([right_margin, bottom_margin, width, height]) / 1000.,
            [ejectable, splittable],
            n_clone_multihot,
            my_clones_radius / 10.,
            np.concatenate(my_centroid_history) / 1000.,
            np.concatenate(my_centroid_speed_history) / 10.,
            np.concatenate(my_avg_pos_history) / 1000.,
            np.concatenate(my_avg_pos_speed_history) / 10.,
        ])

        def extract_rings(triplets, n_dir=8, scaling=100.):
            dist_slots = np.array([3, 6, 12, 24, 48, 96])

            arrs = []
            for my_cl in my_clones:
                arr = np.zeros((len(dist_slots), n_dir))
                if len(triplets) > 0:
                    cl_x, cl_y, cl_r = my_cl[:3]
                    cl_center = np.asarray([cl_x, cl_y])
                    points = np.asarray(triplets)
                    points[:, :2] = points[:, :2] - cl_center
                    points = points[np.linalg.norm(points, axis=1) < dist_slots[-1]]

                    for point in points:
                        norm_idx = sum(dist_slots < np.linalg.norm(point[:2]))
                        angle_idx = min(n_dir - 1, int(n_dir / 2 * (np.arctan2(*point[:2]) / np.pi + 1)))
                        arr[norm_idx, angle_idx] += point[-1] ** 2
                arr = arr.flatten() / scaling
                arrs.append(arr)

            arrs = np.stack(arrs)
            arrs = np.pad(arrs, ((0, self.MAX_CLONE_BALLS - arrs.shape[0]), (0, 0)))
            # arrs = arrs.flatten()
            return arrs

        def team_name_reorder(team_name: float) -> float:
            if team_name == 0.:
                return my_team_name
            elif team_name == float(my_team_name):
                return 0
            else:
                return team_name

        def player_name_reorder(player_name: float) -> float:
            controlled = list(map(float, self.controlled_player_names))
            if player_name in map(float, controlled):
                return controlled.index(player_name)
            elif player_name in range(n_player_per_team):
                return player_name + controlled[0]
            else:
                return player_name

        def extract_rel(targets, limit, source, target_type, rel):
            dim_final = dict(food=5, spore=7, thorn=10, clone=29)[target_type]
            if len(targets) == 0:
                return np.zeros(limit * dim_final)

            targets = np.asarray(targets, dtype='float')
            targets[:, :2] = targets[:, :2] - source[:2]  # use relative position
            if rel == 'nearest':
                # keep the nearest #limit target balls
                sort_idx = np.argsort(np.linalg.norm(targets[:, :2], axis=1))
                targets = targets[sort_idx]  # sort by distance
            elif rel == 'biggest_eatable':
                # keep the biggest #limit target balls that can be eaten by splitting
                sort_idx = np.argsort(targets[:, 2])
                targets = targets[sort_idx]  # sort by radius
                targets = targets[np.linalg.norm(targets[:, :2], axis=1) < source[2] * 2 + 20]  # 20 further, aware to move towards it
                targets = targets[targets[:, 2] < np.sqrt(0.5) * source[2]]
                if len(targets) == 0:
                    return np.zeros(limit * dim_final)
            elif rel == 'biggest_can_eat_me':
                # keep the biggest #limit target balls that can eat source by splitting
                sort_idx = np.argsort(targets[:, 2])
                targets = targets[sort_idx]  # sort by radius
                targets = targets[targets[:, 2] * 2 + 20 > np.linalg.norm(targets[:, :2], axis=1)]
                targets = targets[targets[:, 2] * np.sqrt(0.5) > source[2]]
                if len(targets) == 0:
                    return np.zeros(limit * dim_final)
            else:
                raise ValueError('Undefined ball relation', rel)
            targets = targets[:limit]

            # common features
            offset = targets[:, :2]
            source_radius = source[None, 2:3].repeat(len(offset), axis=0)
            dist = np.linalg.norm(offset, axis=1, keepdims=True)
            cos = offset[:, 0:1] / (dist + 1e-6)
            sin = offset[:, 1:2] / (dist + 1e-6)
            arr = np.concatenate((
                offset / 100.,
                dist / 100.,
                cos,
                sin,
            ), axis=-1)

            if target_type in ['spore', 'thorn', 'clone']:
                target_radius = targets[:, 2:3]
                source_can_eat_target = source_radius - target_radius
                source_split_can_eat_target = np.sqrt(0.5) * source_radius - target_radius
                arr = np.concatenate((
                    arr,
                    source_can_eat_target / 100.,
                    source_split_can_eat_target / 100.,
                ), axis=-1)

                if target_type in ['thorn', 'clone']:
                    collide_from_source = source_radius - dist
                    collide_from_source_split = (2 + np.sqrt(0.5)) * source_radius - dist
                    arr = np.concatenate((
                        arr,
                        target_radius / 30.,
                        collide_from_source / 100.,
                        collide_from_source_split / 100.,
                    ), axis=-1)

                    if target_type == 'clone':
                        collide_from_target = target_radius - dist
                        collide_from_target_split = (2 + np.sqrt(0.5)) * target_radius - dist
                        target_split_can_eat_source = np.sqrt(0.5) * target_radius - source_radius
                        team_onehot = np.stack([one_hot_generator(n_team, t) for t in [int(team_name_reorder(t)) for t in targets[:, -1]]])  # -1: team_name
                        player_onehot = np.stack([one_hot_generator(n_player, t) for t in [int(player_name_reorder(t)) for t in targets[:, -2]]])  # -2: player_name
                        arr = np.concatenate((
                            arr,
                            collide_from_target / 100.,
                            collide_from_target_split / 100.,
                            target_split_can_eat_source / 100.,
                            team_onehot,
                            player_onehot,
                        ), axis=-1)

            arr = np.pad(arr, ((0, limit - len(arr)), (0, 0)))
            arr = arr.flatten()
            return arr

        def extract_nearests(triplets, limit, target_type, relations=None):
            if relations is None:
                relations = ['nearest']

            arrs = []
            for i, my_cl in enumerate(my_clones):
                cl_arr = np.asarray(my_cl[:-2])
                arr = []
                for rel in relations:
                    per_rel_arr = extract_rel(triplets, limit, cl_arr, target_type, rel)
                    arr.append(per_rel_arr)
                arr = np.concatenate(arr)
                arrs.append(arr)

            arrs = np.stack(arrs)
            arrs = np.pad(arrs, ((0, self.MAX_CLONE_BALLS - arrs.shape[0]), (0, 0)))
            return arrs

        fst_rings = extract_rings(food) + extract_rings(spore) + extract_rings(thorns)
        ally_rings = extract_rings(np.array([c[:3] for c in ally_clones]), scaling=500.)
        oppo_rings = extract_rings(np.array([c[:3] for c in oppo_clones]), scaling=500.)
        food_nearests = extract_nearests(food, limit=5, target_type='food')
        spore_nearests = extract_nearests(spore, limit=5, target_type='spore')
        thorns_nearests = extract_nearests(thorns, limit=5, target_type='thorn')
        self_nearests = extract_nearests(my_clones, limit=2, target_type='clone')
        ally_nearests = extract_nearests(ally_clones, limit=2, target_type='clone')
        oppo_nearests = extract_nearests(oppo_clones, limit=4, target_type='clone',
                                         relations=['nearest', 'biggest_eatable', 'biggest_can_eat_me'])

        features = {
            'basic': basic_arr,
            'fst_rings': fst_rings,
            'ally_rings': ally_rings,
            'oppo_rings': oppo_rings,
            'food': food_nearests,
            'spore': spore_nearests,
            'thorn': thorns_nearests,
            'self': self_nearests,
            'ally': ally_nearests,
            'oppo': oppo_nearests,
            'per_self_clone': per_self_clone_arr,
            'clone_mask': n_clone_multihot,
        }

        # encode action related
        if self.len_action_hist > 0:
            features['basic'] = np.concatenate([features['basic'], np.concatenate(self.action_history[idx_in_team])])
        frame_since = [self.n_frame_since_last_action[idx_in_team][k] / 100. for k in ACTION_TYPES]
        features['basic'] = np.concatenate([features['basic'], frame_since])
        action_last = [self.n_frame_an_action_last[idx_in_team][k] / 20. for k in ACTION_TYPES]
        features['basic'] = np.concatenate([features['basic'], action_last])

        return features, legal_mask


class BaseSubmission:
    def __init__(self, team_name, player_names):
        self.team_name = team_name
        self.player_names = player_names

    def get_actions(self, obs):
        raise NotImplementedError


class MySubmission(BaseSubmission):
    def __init__(self, team_name, player_names):
        super().__init__(team_name, player_names)
        self.team_name = team_name
        self.player_names = player_names

        net_conf = dict(n_ball=3, n_pi_output=53, n_v_output=3)
        self.agent = GoBiggerAgent(False, net_conf=net_conf)
        model_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'supplements/model.pth')
        agent_dict = torch.load(model_path)
        self.agent.loads(agent_dict)

        self.translator = Translator()

    def get_actions(self, obs):
        if obs[0]['last_time'] == 0:
            self.translator.reset(init_obs=obs)
        state = self.translator.handle_obs(obs)
        action = self.agent.infer(state)
        action = self.translator.handle_action(action)
        return action
