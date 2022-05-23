import copy
import math
import os
import random
import time
from functools import wraps

import numpy as np
import torch
from gobigger.agents import BotAgent
from pygame import Vector2


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.__name__, str(t1 - t0))
              )
        return result

    return function_timer


class RulePolicy:
    def __init__(self, team_id: int, player_num_per_team: int):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.bot = [BotAgent(str(i)) for i in range(start, end)]

    def forward(self, data: dict, **kwargs) -> dict:
        ret = {}
        for env_id in data.keys():
            action = []
            for bot, raw_obs in zip(self.bot, data[env_id]['collate_ignore_raw_obs']):
                raw_obs['overlap']['clone'] = [[x[0], x[1], x[2], int(x[3]), int(x[4])] for x in
                                               raw_obs['overlap']['clone']]
                action.append(bot.step(raw_obs))
            ret[env_id] = {'action': np.array(action)}
        return ret

    def reset(self, data_id: list = []) -> None:
        pass


class BotPolicy:
    def __init__(self, team_id: int, player_num_per_team: int, botagent):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.bot = [botagent(name=str(i), team=str(team_id)) for i in range(start, end)]

    def forward(self, data, **kwargs) -> dict:
        obs = TeamBotPolicy.obs_transform(data)
        ret = {}
        for env_id in data.keys():
            action = []
            for bot, raw_obs in zip(self.bot, obs[env_id]):
                action.append(bot.step(raw_obs))
            ret[env_id] = {'action': np.array(action)}
        return ret

    def reset(self, data_id: list = []) -> None:
        pass


class TeamBotPolicyOld:
    def __init__(self, team_id: int, player_num_per_team: int, botagent):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.bot = [botagent(name=str(i), team=str(team_id)) for i in range(start, end)]
        self.team_danger_balls = []

    def forward(self, data, **kwargs) -> dict:
        obs = self.obs_transform(data)
        ret = {}
        for env_id in obs.keys():
            action = []
            team_danger_balls = []

            for bot, raw_obs in zip(self.bot, obs[env_id]):
                raw_obs['overlap']['clone'] = [[x[0], x[1], x[2], int(x[3]), int(x[4])] for x in
                                               raw_obs['overlap']['clone']]

                act, help_ball = bot.step(raw_obs, self.team_danger_balls)
                if help_ball is not None:
                    team_danger_balls.append(help_ball)
                action.append(act)
                self.team_danger_balls = team_danger_balls
            ret[int(env_id)] = {'action': np.array(action)}
        return ret

    @staticmethod
    def obs_transform(data):
        # bot_env
        rec = {}

        if isinstance(data, dict):
            for env_id in data.keys():
                tmp = []
                for obs in data[env_id]:
                    tmp.append(obs[1])
                rec[env_id] = tmp

        # submission
        elif isinstance(data, list):
            # 删除feature_layers
            for k in data[1].keys():
                if 'feature_layers' in data[1][k]:
                    del (data[1][k]['feature_layers'])

            player_info = data[1]
            for player_id in player_info.keys():
                rec.append(player_info[player_id])
        return rec

    def reset(self, data_id: list = []) -> None:
        pass


class OpenBallPolicy:
    def __init__(self, team_id: int, player_num_per_team: int, botagent):
        self.collect_data = False  # necessary
        self.team_id = int(team_id)
        self.player_num = player_num_per_team
        self.start, self.end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.solo_agent = botagent
        self.master_id = -1
        self.server_id = -1
        self.action_rec = dict()
        self.thorns_history = []

    def find_master_and_servant(self, clone, thorns):
        if 1:
            return True
        else:
            return False

    def team_play(self, clone, thorns):
        master_id, servant_id = self.find_master_and_servant(clone, thorns)
        # no master/servant
        if master_id is None or servant_id is None:
            return dict()

    def forward(self, data, **kwargs) -> dict:
        obs = self.obs_transform(data)
        ret = {}
        for env_id in obs.keys():
            action = []
            o = obs[env_id]

            # share thorns & clone
            share_history_thorns = self.share_history_thorns(o)
            share_clone = self.share_clone(o)
            self.action_rec = self.team_play(share_clone, share_history_thorns)

            for player_id, raw_obs in zip(range(self.start, self.end), o):
                raw_obs['overlap']['clone'] = share_clone
                raw_obs['overlap']['thorns'] = share_history_thorns
                # team mode
                if player_id in self.action_rec.keys():
                    act = self.action_rec.get(player_id)
                # solo mode
                else:
                    act = self.solo_agent(name=player_id, team=self.team_id).step(raw_obs)
                action.append(act)
            ret[int(env_id)] = {'action': np.array(action)}
        return ret

    @staticmethod
    def obs_transform(data):
        # bot_env
        rec = {}

        if isinstance(data, dict):
            for env_id in data.keys():
                tmp = []
                for obs in data[env_id]:
                    player_info = obs[1]
                    global_info = obs[0]
                    player_info.update(global_info)
                    tmp.append(player_info)
                rec[env_id] = tmp

        # submission
        elif isinstance(data, list):
            # 删除feature_layers
            for k in data[1].keys():
                if 'feature_layers' in data[1][k]:
                    del (data[1][k]['feature_layers'])

            player_info = data[1]
            for player_id in player_info.keys():
                rec[0].append(player_info[player_id])
        return rec

    def share_history_thorns(self, obs):
        thorns_history = self.thorns_history

        # share the thorns inside rect
        thorns_rec = []
        for o in obs:
            thorns = o['overlap']['thorns']
            if thorns:
                thorns = [np.array(th).reshape(1, 3) for th in thorns]
                thorns = np.concatenate(thorns)
                thorns_rec.append(thorns)

        # check_history
        for th_history in thorns_history:
            inview = False
            x, y = th_history[0], th_history[1]
            for o in obs:
                left_top_x, left_top_y, right_bottom_x, right_bottom_y = o['rectangle']
                if left_top_x <= x <= right_bottom_x and left_top_y <= y <= right_bottom_y:
                    inview = True

            # append if not in team view
            if not inview:
                thorns_rec.append(th_history.reshape(1, 3))

        if thorns_rec:
            thorns_rec = np.concatenate(thorns_rec)
            thorns_rec = np.unique(thorns_rec, axis=0)
        self.thorns_history = thorns_rec
        return thorns_rec

    def share_clone(self, obs):
        # share the thorns inside rect
        clone_rec = []
        for o in obs:
            clone = o['overlap']['clone']
            if clone:
                clone = [np.array(cl).reshape(-1, 5) for cl in clone]
                clone = np.concatenate(clone)
                clone_rec.append(clone)

        if clone_rec:
            clone_rec = np.concatenate(clone_rec, axis=0)
            clone_rec = np.unique(clone_rec, axis=0)
        return clone_rec

    def reset(self, data_id: list = []) -> None:
        self.thorns_history = []


class TeamBotPolicy:
    def __init__(self, team_id: int, player_num_per_team: int, botagent, greedy_thorn=False, player_names=None):
        self.collect_data = False  # necessary
        self.greedy_thorn = greedy_thorn
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        if player_names:
            self.bot = [botagent(name=str(n), team=str(team_id)) for n in player_names]
        else:
            self.bot = [botagent(name=str(i), team=str(team_id)) for i in range(start, end)]
        self.team_danger_balls = []
        self.thorns_history = []

    def forward(self, data, **kwargs) -> dict:
        obs = self.obs_transform(data)
        ret = {}
        for env_id in obs.keys():
            action = []
            team_danger_balls = []
            # share thorns
            share_history_thorns = self.share_history_thorns(obs[env_id])
            # share clone
            team_id = int(self.team_id)
            share_clone = self.share_clone(obs[env_id])
            team_clone = share_clone[share_clone[:, -1] == team_id]

            if not self.greedy_thorn:
                thorns_split = self.avoid_eat_same_thorn(team_clone, share_history_thorns)
            else:
                thorns_split = self.avoid_eat_same_thorn_greedy(team_clone, share_history_thorns)

            for bot, raw_obs, th in zip(self.bot, obs[env_id], thorns_split):
                raw_obs['overlap']['clone'] = share_clone
                raw_obs['overlap']['thorns'] = th

                act, help_ball = bot.step(raw_obs, self.team_danger_balls)
                if help_ball is not None:
                    team_danger_balls.append(help_ball)
                action.append(act)
                self.team_danger_balls = team_danger_balls
            ret[int(env_id)] = {'action': np.array(action)}
        return ret

    @staticmethod
    def obs_transform(data):
        # bot_env
        rec = {}

        if isinstance(data, dict):
            for env_id in data.keys():
                tmp = []
                for obs in data[env_id]:
                    player_info = obs[1]
                    global_info = obs[0]
                    player_info.update(global_info)
                    tmp.append(player_info)
                rec[env_id] = tmp

        # submission
        elif isinstance(data, list):
            rec[0] = []
            # 删除feature_layers
            for k in data[1].keys():
                if 'feature_layers' in data[1][k]:
                    del (data[1][k]['feature_layers'])

            global_info = data[0]
            player_info = data[1]
            for player_id in player_info.keys():
                player_info[player_id]['overlap']['clone'] = [[x[0], x[1], x[2], int(x[3]), int(x[4])] for x in
                                                              player_info[player_id]['overlap']['clone']]
                player_info[player_id].update(global_info)
                rec[0].append(player_info[player_id])
        return rec

    def share_history_thorns(self, obs):
        thorns_history = self.thorns_history

        # share the thorns inside rect
        thorns_rec = []
        for o in obs:
            thorns = o['overlap']['thorns']
            if thorns:
                thorns = [np.array(th).reshape(1, 3) for th in thorns]
                thorns = np.concatenate(thorns)
                thorns_rec.append(thorns)

        # check_history
        for th_history in thorns_history:
            inview = False
            x, y = th_history[0], th_history[1]
            for o in obs:
                left_top_x, left_top_y, right_bottom_x, right_bottom_y = o['rectangle']
                if left_top_x <= x <= right_bottom_x and left_top_y <= y <= right_bottom_y:
                    inview = True

            # append if not in team view
            if not inview:
                thorns_rec.append(th_history.reshape(1, 3))

        if thorns_rec:
            thorns_rec = np.concatenate(thorns_rec)
            thorns_rec = np.unique(thorns_rec, axis=0)
        self.thorns_history = thorns_rec
        return thorns_rec

    def avoid_eat_same_thorn(self, team_clone, thorns):
        bot = self.bot[0]

        fake_overlap = dict()
        fake_overlap['clone'] = team_clone
        fake_overlap['thorns'] = thorns
        fake_overlap = bot.preprocess(fake_overlap)
        score_rec = bot.process_thorns_balls(fake_overlap['thorns'], fake_overlap['clone'], mode=1)

        player_match_thorn = {}
        locked_thorn_id = []
        if score_rec:
            score_rec.sort(key=lambda a: a[2], reverse=True)
            score_arr = np.array(score_rec)

            for line in score_arr:
                player_id, th_id, _ = line

                # player has not match thorn
                if player_id not in player_match_thorn.keys():
                    # this thorn has been locked
                    if th_id in locked_thorn_id:
                        continue
                    player_match_thorn[player_id] = th_id
                    locked_thorn_id.append(th_id)

        # no locked thorn
        if not locked_thorn_id:
            return [thorns] * 3

        player_thorn_rec = []
        for b in self.bot:
            thorn_rec = []

            for i, th in enumerate(thorns):
                if i not in locked_thorn_id:
                    thorn_rec.append(th)

            player_id = int(b.name)
            if player_id in player_match_thorn.keys():
                thorn_id = int(player_match_thorn[player_id])
                thorn_rec.append(thorns[thorn_id])

            if thorn_rec:
                thorn_rec = [th.reshape(-1, 3) for th in thorn_rec]
                thorn_rec = np.concatenate(thorn_rec)
            player_thorn_rec.append(thorn_rec)
        return player_thorn_rec

    def avoid_eat_same_thorn_greedy(self, team_clone, thorns):
        bot = self.bot[0]

        fake_overlap = dict()
        fake_overlap['clone'] = team_clone
        fake_overlap['thorns'] = thorns
        fake_overlap = bot.preprocess(fake_overlap)
        score_rec = bot.process_thorns_balls(fake_overlap['thorns'], fake_overlap['clone'], mode=1)

        locked_thorn = {}
        if score_rec:
            score_rec.sort(key=lambda a: a[2], reverse=True)
            score_arr = np.array(score_rec)

            for line in score_arr:
                player_id, th_id, _ = line

                # this thorn has not been locked
                if th_id not in locked_thorn.keys():
                    locked_thorn[th_id] = player_id

        # no locked thorn
        if not locked_thorn:
            return [thorns] * 3

        player_thorn_rec = []
        for b in self.bot:
            thorn_rec = []

            for i, th in enumerate(thorns):
                if i not in locked_thorn.keys():
                    thorn_rec.append(th)

            player_id = int(b.name)
            for th_id, p_id in locked_thorn.items():
                if player_id == p_id:
                    thorn_rec.append(thorns[int(th_id)])

            if thorn_rec:
                thorn_rec = [th.reshape(-1, 3) for th in thorn_rec]
                thorn_rec = np.concatenate(thorn_rec)
            player_thorn_rec.append(thorn_rec)
        return player_thorn_rec

    def share_clone(self, obs):
        # share the thorns inside rect
        clone_rec = []
        for o in obs:
            clone = o['overlap']['clone']
            if clone:
                clone = [np.array(cl).reshape(-1, 5) for cl in clone]
                clone = np.concatenate(clone)
                clone_rec.append(clone)

        if clone_rec:
            clone_rec = np.concatenate(clone_rec, axis=0)
            clone_rec = np.unique(clone_rec, axis=0)
        return clone_rec

    def reset(self, data_id: list = []) -> None:
        self.thorns_history = []


class TeamBotPolicyV2:
    def __init__(self, team_id: int, player_num_per_team: int, botagent, greedy_thorn=False, player_names=None):
        self.collect_data = False  # necessary
        self.greedy_thorn = greedy_thorn
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        if player_names:
            self.bot = [botagent(name=str(n), team=str(team_id)) for n in player_names]
        else:
            self.bot = [botagent(name=str(i), team=str(team_id)) for i in range(start, end)]
        self.team_danger_balls = []
        self.thorns_history = []

        self.last_obs = dict()
        self.player_merge_cd = [0.0] * 12

    def forward(self, data, **kwargs) -> dict:
        obs = self.obs_transform(data)
        ret = {}
        team_id = int(self.team_id)
        for env_id in obs.keys():
            cur_env_obs = obs[env_id]
            # share thorns
            share_history_thorns = self.share_history_thorns(cur_env_obs)
            # share clone
            share_clone = self.share_clone(cur_env_obs)
            team_clone = share_clone[share_clone[:, -1] == team_id]

            if not self.greedy_thorn:
                thorns_split = self.avoid_eat_same_thorn(team_clone, share_history_thorns)
            else:
                thorns_split = self.avoid_eat_same_thorn_greedy(team_clone, share_history_thorns)

            # process clone and th
            for raw_obs, th in zip(cur_env_obs, thorns_split):
                raw_obs['overlap']['clone'] = share_clone
                raw_obs['overlap']['thorns'] = th

            # fresh merge cd
            last_env_obs = self.last_obs.get(int(env_id), None)
            if last_env_obs:
                self.fresh_merge_cd(last_env_obs, cur_env_obs)

            for raw_obs in cur_env_obs:
                raw_obs['player_merge_cd'] = self.player_merge_cd

            # fake_th = self.fake_thorn_ball(share_clone, share_history_thorns)
            # if fake_th:
            #     self.team_danger_balls += fake_th

            action = []
            team_danger_balls = []
            for bot, raw_obs in zip(self.bot, cur_env_obs):
                act, help_ball = bot.step(raw_obs, self.team_danger_balls)
                if help_ball is not None:
                    team_danger_balls.append(help_ball)
                action.append(act)

            self.team_danger_balls = team_danger_balls
            self.last_obs[int(env_id)] = cur_env_obs
            ret[int(env_id)] = {'action': np.array(action)}
        return ret

    def fresh_merge_cd(self, last_obs, cur_obs):
        last_clones = last_obs[0]['overlap']['clone']
        cur_clones = cur_obs[0]['overlap']['clone']
        for player_id in range(12):
            split = False
            last_player_clone = last_clones[last_clones[:, -2] == player_id].reshape(-1, 5)
            cur_player_clone = cur_clones[cur_clones[:, -2] == player_id].reshape(-1, 5)
            last_player_n = last_player_clone.shape[0]
            cur_player_n = cur_player_clone.shape[0]

            # missing
            if last_player_n == 0 or cur_player_n == 0:
                continue

            last_biggest_r = np.max(last_player_clone[:, 2])
            cur_biggest_r = np.max(cur_player_clone[:, 2])

            if (cur_player_n > last_player_n) and (cur_biggest_r < last_biggest_r * 0.99):
                split = True

            # 最大球裂开
            if split:
                # cur_time = cur_obs[0]['last_time']
                # print("%.1f %i号 玩家裂开" % (cur_time, player_id))
                self.player_merge_cd[player_id] = 20.

        # fresh merge cd
        for player_id in range(12):
            if self.player_merge_cd[player_id] > 0.0:
                self.player_merge_cd[player_id] -= 0.2
            if self.player_merge_cd[player_id] < 0.2:
                self.player_merge_cd[player_id] = 0.0

    def fake_thorn_ball(self, clones, thorns):
        """
        在两个人无法吃荆棘球时，制造fake ball 进行对冲操作
        :return:
        """
        team = int(self.team_id)

        team_clones = clones[clones[:, -1] == team].reshape(-1, 5)
        enemy_clones = clones[clones[:, -1] != team].reshape(-1, 5)

        i, j, score = self.get_help_id(team_clones, enemy_clones, thorns)
        if score == 0.:
            return None

        i_clones = team_clones[team_clones[:, -2] == i].reshape(-1, 5)
        i_biggest_cl = i_clones[np.argmax(i_clones[:, 2])].flatten()
        i_smallest_cl = i_clones[np.argmin(i_clones[:, 2])].flatten()
        j_clones = team_clones[team_clones[:, -2] == j].reshape(-1, 5)
        j_biggest_cl = j_clones[np.argmax(j_clones[:, 2])].flatten()
        j_smallest_cl = j_clones[np.argmin(j_clones[:, 2])].flatten()

        i_biggest_cl_pos = Vector2(i_biggest_cl[0], i_biggest_cl[1])
        j_biggest_cl_pos = Vector2(j_biggest_cl[0], j_biggest_cl[1])

        i_smallest_cl_pos = Vector2(i_smallest_cl[0], i_smallest_cl[1])
        j_smallest_cl_pos = Vector2(j_smallest_cl[0], j_smallest_cl[1])
        fake_th_rec = []
        fake_th_i = dict()
        fake_th_i['position'] = j_biggest_cl_pos
        fake_th_i['radius'] = 12
        fake_th_i['team'] = team
        fake_th_i['player'] = -i

        fake_th_j = dict()
        fake_th_j['position'] = i_biggest_cl_pos
        fake_th_j['radius'] = 12
        fake_th_j['team'] = team
        fake_th_j['player'] = -j
        fake_th_rec.append(fake_th_i)
        fake_th_rec.append(fake_th_j)
        return fake_th_rec

    def get_help_id(self, team_clones, enemy_clones, thorns):
        best_rec = [-1, -1, 0.0]
        for i in range(int(self.team_id) * 3, int(self.team_id) * 3 + 3):
            i_clones = team_clones[team_clones[:, -2] == i].reshape(-1, 5)
            i_clones_n = i_clones.shape[0]
            i_avg_r = np.mean(i_clones[:, 2])
            i_avg_spd = get_spd(i_avg_r)
            i_center_pos = get_center_np(i_clones)
            i_max_r = np.max(i_clones[:, 2])
            i_min_r = np.min(i_clones[:, 2])
            for j in range(i + 1, int(self.team_id) * 3 + 3):
                j_clones = team_clones[team_clones[:, -2] == j].reshape(-1, 5)
                j_clones_n = j_clones.shape[0]
                j_avg_r = np.mean(j_clones[:, 2])
                j_avg_spd = get_spd(j_avg_r)
                j_center_pos = get_center_np(j_clones)
                j_max_r = np.max(j_clones[:, 2])
                j_min_r = np.min(j_clones[:, 2])

                i_can_eat = 20 > i_max_r > j_min_r and j_clones_n >= 2
                j_can_eat = 20 > j_max_r > i_min_r and i_clones_n >= 2
                if not (i_can_eat and j_can_eat):
                    continue

                mid_center_pos = (i_center_pos + j_center_pos) / 2

                dis_i_to_mid = (i_center_pos - mid_center_pos).length()
                dis_j_to_mid = (j_center_pos - mid_center_pos).length()
                dis_i_to_j = (i_center_pos - j_center_pos).length()
                farther_dis = max(dis_i_to_mid, dis_j_to_mid)
                if dis_i_to_j > 250.:
                    continue

                enemy_clones_danger = enemy_clones[enemy_clones[:, 2] >= min(i_min_r, j_min_r)]
                if enemy_clones_danger.size > 0:
                    enemy_to_mid_dis = get_dis(enemy_clones[:, 0], enemy_clones[:, 1], mid_center_pos.x,
                                               mid_center_pos.y)
                    enemy_to_mid_dis_min = np.min(enemy_to_mid_dis)
                    # enemy nearby danger
                    if enemy_to_mid_dis_min < farther_dis:
                        continue

                th_dis = get_dis(thorns[:, 0], thorns[:, 1], mid_center_pos.x, mid_center_pos.y)
                th_dis_min_arg = np.argmin(th_dis)
                t = thorns[th_dis_min_arg].flatten()
                t_v = t[2] * t[2]

                deserve_eat = np.sum(th_dis < 400.) >= 4
                if deserve_eat:
                    spend_time = (j_center_pos - i_center_pos).length() / (j_avg_spd + i_avg_spd)
                    score = t_v / spend_time
                    if score > best_rec[-1]:
                        best_rec = [i, j, score]
        return best_rec

    @staticmethod
    def obs_transform(data):
        # bot_env
        rec = {}

        if isinstance(data, dict):
            for env_id in data.keys():
                tmp = []
                for obs in data[env_id]:
                    player_info = obs[1]
                    global_info = obs[0]
                    player_info['overlap']['clone'] = [[x[0], x[1], x[2], int(x[3]), int(x[4])] for x in
                                                       player_info['overlap']['clone']]
                    player_info.update(global_info)
                    tmp.append(player_info)
                rec[env_id] = tmp

        # submission
        elif isinstance(data, list):
            rec[0] = []
            # 删除feature_layers
            for k in data[1].keys():
                if 'feature_layers' in data[1][k]:
                    del (data[1][k]['feature_layers'])

            global_info = data[0]
            player_info = data[1]
            for player_id in player_info.keys():
                player_info[player_id]['overlap']['clone'] = [[x[0], x[1], x[2], int(x[3]), int(x[4])] for x in
                                                              player_info[player_id]['overlap']['clone']]
                player_info[player_id].update(global_info)
                rec[0].append(player_info[player_id])
        return rec

    def share_history_thorns(self, obs):
        thorns_history = self.thorns_history

        # share the thorns inside rect
        thorns_rec = []
        for o in obs:
            thorns = o['overlap']['thorns']
            if thorns:
                thorns = [np.array(th).reshape(1, 3) for th in thorns]
                thorns = np.concatenate(thorns)
                thorns_rec.append(thorns)

        # check_history
        for th_history in thorns_history:
            inview = False
            x, y = th_history[0], th_history[1]
            for o in obs:
                left_top_x, left_top_y, right_bottom_x, right_bottom_y = o['rectangle']
                if left_top_x <= x <= right_bottom_x and left_top_y <= y <= right_bottom_y:
                    inview = True

            # append if not in team view
            if not inview:
                thorns_rec.append(th_history.reshape(1, 3))

        if thorns_rec:
            thorns_rec = np.concatenate(thorns_rec)
            thorns_rec = np.unique(thorns_rec, axis=0)
        self.thorns_history = thorns_rec
        return thorns_rec

    def avoid_eat_same_thorn(self, team_clone, thorns):
        bot = self.bot[0]

        fake_overlap = dict()
        fake_overlap['clone'] = team_clone
        fake_overlap['thorns'] = thorns
        fake_overlap = bot.preprocess(fake_overlap)
        score_rec = bot.process_thorns_balls(fake_overlap['thorns'], fake_overlap['clone'], mode=1)

        player_match_thorn = {}
        locked_thorn_id = []
        if score_rec:
            score_rec.sort(key=lambda a: a[2], reverse=True)
            score_arr = np.array(score_rec)

            for line in score_arr:
                player_id, th_id, _ = line

                # player has not match thorn
                if player_id not in player_match_thorn.keys():
                    # this thorn has been locked
                    if th_id in locked_thorn_id:
                        continue
                    player_match_thorn[player_id] = th_id
                    locked_thorn_id.append(th_id)

        # no locked thorn
        if not locked_thorn_id:
            return [thorns] * 3

        player_thorn_rec = []
        for b in self.bot:
            thorn_rec = []

            for i, th in enumerate(thorns):
                if i not in locked_thorn_id:
                    thorn_rec.append(th)

            player_id = int(b.name)
            if player_id in player_match_thorn.keys():
                thorn_id = int(player_match_thorn[player_id])
                thorn_rec.append(thorns[thorn_id])

            if thorn_rec:
                thorn_rec = [th.reshape(-1, 3) for th in thorn_rec]
                thorn_rec = np.concatenate(thorn_rec)
            player_thorn_rec.append(thorn_rec)
        return player_thorn_rec

    def avoid_eat_same_thorn_greedy(self, team_clone, thorns):
        bot = self.bot[0]

        fake_overlap = dict()
        fake_overlap['clone'] = team_clone
        fake_overlap['thorns'] = thorns
        fake_overlap = bot.preprocess(fake_overlap)
        score_rec = bot.process_thorns_balls(fake_overlap['thorns'], fake_overlap['clone'], mode=1)

        locked_thorn = {}
        if score_rec:
            score_rec.sort(key=lambda a: a[2], reverse=True)
            score_arr = np.array(score_rec)

            for line in score_arr:
                player_id, th_id, _ = line

                # this thorn has not been locked
                if th_id not in locked_thorn.keys():
                    locked_thorn[th_id] = player_id

        # no locked thorn
        if not locked_thorn:
            return [thorns] * 3

        player_thorn_rec = []
        for b in self.bot:
            thorn_rec = []

            for i, th in enumerate(thorns):
                if i not in locked_thorn.keys():
                    thorn_rec.append(th)

            player_id = int(b.name)
            for th_id, p_id in locked_thorn.items():
                if player_id == p_id:
                    thorn_rec.append(thorns[int(th_id)])

            if thorn_rec:
                thorn_rec = [th.reshape(-1, 3) for th in thorn_rec]
                thorn_rec = np.concatenate(thorn_rec)
            player_thorn_rec.append(thorn_rec)
        return player_thorn_rec

    def share_clone(self, obs):
        # share the thorns inside rect
        clone_rec = []
        for o in obs:
            clone = o['overlap']['clone']
            if clone:
                clone = [np.array(cl).reshape(-1, 5) for cl in clone]
                clone = np.concatenate(clone)
                clone_rec.append(clone)

        if clone_rec:
            clone_rec = np.concatenate(clone_rec, axis=0)
            clone_rec = np.unique(clone_rec, axis=0)
        return clone_rec

    def reset(self, data_id: list = []) -> None:
        self.thorns_history = []


class TeamBotPolicyWT:
    def __init__(self, team_id: int, player_num_per_team: int, botagent):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.bot = [botagent(name=str(i), team=str(team_id)) for i in range(start, end)]
        self.thorns_history = []

    def forward(self, data, **kwargs) -> dict:
        obs = self.obs_transform(data)
        ret = {}
        for env_id in obs.keys():
            action = []
            team_danger_balls = []
            # share thorns
            share_history_thorns = self.share_history_thorns(obs[env_id])
            # share clone
            team_id = int(self.bot[0].team)
            share_clone = self.share_clone(obs[env_id])
            team_clone = share_clone[share_clone[:, -1] == team_id]

            thorns_split = self.avoid_eat_same_thorn(team_clone, share_history_thorns)
            for bot, raw_obs, th in zip(self.bot, obs[env_id], thorns_split):
                raw_obs['overlap']['clone'] = share_clone
                raw_obs['overlap']['thorns'] = th
                act = bot.step(raw_obs)
                action.append(act)
            ret[int(env_id)] = {'action': np.array(action)}
        return ret

    @staticmethod
    def obs_transform(data):
        # bot_env
        rec = {}

        if isinstance(data, dict):
            for env_id in data.keys():
                tmp = []
                for obs in data[env_id]:
                    player_info = obs[1]
                    global_info = obs[0]
                    player_info.update(global_info)
                    tmp.append(player_info)
                rec[env_id] = tmp

        # submission
        elif isinstance(data, list):
            # 删除feature_layers
            for k in data[1].keys():
                if 'feature_layers' in data[1][k]:
                    del (data[1][k]['feature_layers'])

            player_info = data[1]
            for player_id in player_info.keys():
                rec.append(player_info[player_id])
        return rec

    def share_history_thorns(self, obs):
        thorns_history = self.thorns_history

        # share the thorns inside rect
        thorns_rec = []
        for o in obs:
            thorns = o['overlap']['thorns']
            if thorns:
                thorns = [np.array(th).reshape(1, 3) for th in thorns]
                thorns = np.concatenate(thorns)
                thorns_rec.append(thorns)

        # check_history
        for th_history in thorns_history:
            inview = False
            x, y = th_history[0], th_history[1]
            for o in obs:
                left_top_x, left_top_y, right_bottom_x, right_bottom_y = o['rectangle']
                if left_top_x <= x <= right_bottom_x and left_top_y <= y <= right_bottom_y:
                    inview = True

            # append if not in team view
            if not inview:
                thorns_rec.append(th_history.reshape(1, 3))

        if thorns_rec:
            thorns_rec = np.concatenate(thorns_rec)
            thorns_rec = np.unique(thorns_rec, axis=0)
        self.thorns_history = thorns_rec
        return thorns_rec

    def avoid_eat_same_thorn(self, team_clone, thorns):
        bot = self.bot[0]

        fake_overlap = dict()
        fake_overlap['clone'] = team_clone
        fake_overlap['thorns'] = thorns
        fake_overlap = bot.preprocess(fake_overlap)
        score_rec = bot.process_thorns_balls(fake_overlap['thorns'], fake_overlap['clone'], mode=1)

        player_match_thorn = {}
        locked_thorn_id = []
        if score_rec:
            score_rec.sort(key=lambda a: a[2], reverse=True)
            score_arr = np.array(score_rec)

            for line in score_arr:
                player_id, th_id, _ = line

                # player has not match thorn
                if player_id not in player_match_thorn.keys():
                    # this thorn has been locked
                    if th_id in locked_thorn_id:
                        continue
                    player_match_thorn[player_id] = th_id
                    locked_thorn_id.append(th_id)

        # no locked thorn
        if not locked_thorn_id:
            return [thorns] * 3

        player_thorn_rec = []
        for b in self.bot:
            thorn_rec = []

            for i, th in enumerate(thorns):
                if i not in locked_thorn_id:
                    thorn_rec.append(th)

            player_id = int(b.name)
            if player_id in player_match_thorn.keys():
                thorn_id = int(player_match_thorn[player_id])
                thorn_rec.append(thorns[thorn_id])

            if thorn_rec:
                thorn_rec = [th.reshape(-1, 3) for th in thorn_rec]
                thorn_rec = np.concatenate(thorn_rec)
            player_thorn_rec.append(thorn_rec)
        return player_thorn_rec

    def share_clone(self, obs):
        # share the thorns inside rect
        clone_rec = []
        for o in obs:
            clone = o['overlap']['clone']
            if clone:
                clone = [np.array(cl).reshape(-1, 5) for cl in clone]
                clone = np.concatenate(clone)
                clone_rec.append(clone)

        if clone_rec:
            clone_rec = np.concatenate(clone_rec, axis=0)
            clone_rec = np.unique(clone_rec, axis=0)
        return clone_rec

    def reset(self, data_id: list = []) -> None:
        self.thorns_history = []


class FoodScore:
    def __init__(self, food_queue, score, keep_one=True):
        self.food_queue = food_queue
        self.score = score
        if keep_one:
            self.keep_one_food()

    def keep_one_food(self):
        if self.food_queue:
            self.food_queue = [self.food_queue[0]]

    def get_food_tar(self):
        if self.food_queue:
            return self.food_queue[0][1]
        return None

    def remove_head(self):
        if self.food_queue:
            self.food_queue = self.food_queue[1:]


def item_to_str(cl):
    x = str(round(cl['position'].x))
    y = str(round(cl['position'].y))
    s = '-'.join([x, y])
    return s


def xy_to_str(x, y):
    x = str(round(x))
    y = str(round(y))
    s = '-'.join([x, y])
    return s


class ChaseInfo:
    def __init__(self, src, tar, merge_cl, spend_time, move_dis, action_type=-1):
        self.src = src
        self.tar = tar
        self.merge_cl = merge_cl
        self.spend_time = spend_time
        self.move_dis = move_dis
        self.action_type = action_type


class ChaseAnalyzer:
    def __init__(self):
        self.map_width = 1000
        self.map_height = 1000

    def init_map_info(self, obs):
        if self.map_width > 0 and self.map_height > 0:
            return
        else:
            map_width, map_height = obs['border'].tolist()
            self.map_width = map_width
            self.map_height = map_height

    def chase_analyze(self, clone, tar, chase_factor=2):
        assert isinstance(clone, np.ndarray) and isinstance(tar, np.ndarray)
        assert clone.size > 0 and tar.size > 0
        assert chase_factor > 0.

        clone = clone.flatten()
        tar = tar.flatten()

        clone_x, clone_y, clone_r = clone[0:3]
        tar_x, tar_y, tar_r = tar[0:3]
        tar_can_move = False if tar.shape[-1] == 3 else True
        same_player = True if (tar_can_move and tar[-2] == clone[-2]) else False
        same_team = True if (tar_can_move and tar[-1] == clone[-1]) else False

        dt_x = tar_x - clone_x
        dt_y = tar_y - clone_y

        # avoid / zero
        if dt_x == 0:
            dt_x += 1e-4
        if dt_y == 0:
            dt_y += 1e-4

        dis_center = np.sqrt(np.power(dt_x, 2) + np.power(dt_y, 2))
        dis_cur_eat = dis_center - max(clone_r, tar_r)
        merge_r = np.sqrt(np.power(clone_r, 2) + np.power(tar_r, 2))

        if not tar_can_move:
            merge_player = clone[-2]
            merge_team = clone[-1]
        else:
            merge_player = clone[-2] if clone_r > tar_r else tar[-2]
            merge_team = clone[-1] if clone_r > tar_r else tar[-1]

        if dis_cur_eat <= 0:
            spend_time = 0.1
            if clone_r > tar_r:
                final_x = clone_x
                final_y = clone_y
            else:
                final_x = tar_x
                final_y = tar_y
        else:
            spd_clone = get_spd(clone_r)
            if tar_can_move:
                spd_tar = get_spd(tar_r)
                to_board_dis, to_board_time = self.chase_to_border(clone, tar)
                if same_player:
                    if spd_clone <= spd_tar:
                        chase_dis = 1000
                        chase_time = 100.
                    else:
                        chase_time = dis_cur_eat / (spd_clone - spd_tar)
                        chase_dis = chase_time * spd_clone
                else:
                    chase_dis = dis_cur_eat
                    chase_time = chase_dis / spd_clone
                    chase_time = max(chase_time, np.power(chase_time, chase_factor))

                # if to_board_time < chase_time:
                #     dis = to_board_dis
                #     spend_time = to_board_time
                # else:
                dis = chase_dis
                spend_time = chase_time
            else:
                dis = dis_cur_eat
                spend_time = dis / spd_clone

            final_x = clone_x + (dt_x / dis_center) * dis
            final_y = clone_y + (dt_y / dis_center) * dis

        merge_cl = np.array([final_x, final_y, merge_r, merge_player, merge_team])
        move_dis = get_dis(clone_x, clone_y, final_x, final_y)
        chase_info = ChaseInfo(clone, tar, merge_cl, spend_time, move_dis)
        return chase_info

    def split_analyze(self, clone, tar):
        assert isinstance(clone, np.ndarray) and isinstance(tar, np.ndarray)
        assert clone.size > 0 and tar.size > 0

        clone = clone.flatten()
        tar = tar.flatten()
        dis = get_dis(clone[0], clone[1], tar[0], tar[1])

        clone_r = clone[2]

        # can not split
        if clone_r < 10 or dis > 2.12 * clone_r:
            return None

        tar_can_move = False if tar.shape[-1] == 3 else True
        same_player = True if (tar_can_move and tar[-2] == clone[-2]) else False
        same_team = True if (tar_can_move and tar[-1] == clone[-1]) else False
        tar_r = tar[2]

        # feed enemy
        if not same_team and clone_r / np.sqrt(2) < tar_r:
            return None

        clone_x, clone_y, clone_r = clone[0:3]
        tar_x, tar_y, tar_r = tar[0:3]

        dt_x = tar_x - clone_x
        dt_y = tar_y - clone_y

        if clone_r > tar_r:
            final_x = clone_x + (dt_x / dis) * 2.12 * clone_r
            final_y = clone_y + (dt_y / dis) * 2.12 * clone_r
        else:
            final_x = tar_x
            final_y = tar_y

        # avoid / zero
        if dt_x == 0:
            dt_x += 1e-4
        if dt_y == 0:
            dt_y += 1e-4

        merge_r = np.sqrt(np.power(clone_r, 2) + np.power(tar_r, 2))

        if not tar_can_move:
            merge_player = clone[-2]
            merge_team = clone[-1]
        else:
            merge_player = clone[-2] if clone_r > tar_r else tar[-2]
            merge_team = clone[-1] if clone_r > tar_r else tar[-1]

        spend_time = 0.1
        merge_cl = np.array([final_x, final_y, merge_r, merge_player, merge_team])
        move_dis = get_dis(clone_x, clone_y, final_x, final_y)
        split_info = ChaseInfo(clone, tar, merge_cl, spend_time, move_dis, action_type=4)
        return split_info

    def chase_to_border(self, src_cl, tar_cl):
        src_x, src_y, src_r = src_cl[0:3]
        src_spd = get_spd(src_r)

        border_x, border_y = self.chase_to_border_pos(src_cl, tar_cl)
        dt_x = border_x - src_x
        dt_y = border_y - src_y

        to_border_dis = np.sqrt(np.power(dt_x, 2) + np.power(dt_y, 2))
        to_border_time = to_border_dis / src_spd
        return to_border_dis, to_border_time

    def chase_to_border_pos(self, src_cl, tar_cl):
        map_width = self.map_width
        map_height = self.map_height
        assert map_width > 0 and map_height > 0, "map width/height not initialized"

        src_x, src_y, src_r = src_cl[0:3]
        tar_x, tar_y, tar_r = tar_cl[0:3]

        dt_x = tar_x - src_x
        dt_y = tar_y - src_y
        # avoid / zero
        if dt_x == 0:
            dt_x += 1e-4
        if dt_y == 0:
            dt_y += 1e-4

        k = dt_y / dt_x
        b = src_y - k * src_x

        border_top_y = tar_r
        border_top_x = y_to_x(k, b, border_top_y)

        border_bottom_y = map_height - tar_r
        border_bottom_x = y_to_x(k, b, border_bottom_y)

        border_left_x = tar_r
        border_left_y = x_to_y(k, b, border_left_x)

        border_right_x = map_width - tar_r
        border_right_y = x_to_y(k, b, border_right_x)

        group = [[border_top_x, border_top_y], [border_bottom_x, border_bottom_y], [border_left_x, border_left_y],
                 [border_right_x, border_right_y]]

        for x, y in group:
            if not (0 <= x <= map_width and 0 <= y <= map_height):
                continue
            if x == src_x and y == src_y:
                continue

            border_dt_x = x - src_x
            border_dt_y = y - src_y

            if border_dt_x == 0:
                border_dt_x += 1e-4
            if border_dt_y == 0:
                border_dt_y += 1e-4
            k_border = border_dt_y / border_dt_x
            if k_border * k > 0:
                return x, y

        assert False, "Border ERROR"

    def filt_beheind_clone(self, obs_cl, clone, thorns=None):
        """

        :param obs_cl:
        :param clone: 所有clone 根据半径排序
        :return:
        """
        obs_cl_x, obs_cl_y, obs_cl_r = obs_cl[0:3]
        visible_angle_rec = np.zeros(360)

        keep_clone = []
        keep_thorns = []

        if clone is not None:
            clone = sort_clone_by_r(clone)
        if thorns is not None:
            thorns = sort_clone_by_r(thorns)

        # filt clone
        for tar_cl in clone:
            x, y, r = tar_cl[0:3]
            dt_x = x - obs_cl_x
            dt_y = y - obs_cl_y
            dis = math.sqrt(dt_x * dt_x + dt_y * dt_y)
            dis = max(dis, r + 1)
            ang = round(vector2angle(dt_x, dt_y))
            ang = min(ang, 359)

            if visible_angle_rec[ang] > r:
                continue

            if obs_cl_r < r:
                ang_dt = np.arctan(r / dis)
                ang_range_start = math.floor(ang - ang_dt)
                ang_range_end = math.ceil(ang + ang_dt)

                if 0 <= ang_range_start <= ang_range_end < 360:
                    visible_angle_rec[ang_range_start:ang_range_end + 1] = r
                elif ang_range_end >= 360:
                    visible_angle_rec[ang_range_start:360] = r
                    visible_angle_rec[0:ang_range_end - 360] = r
                elif ang_range_start < 0:
                    visible_angle_rec[ang_range_start + 360:360] = r
                    visible_angle_rec[0:ang_range_end] = r
                else:
                    assert False, "filt_behind_clone_error "

            keep_clone.append(tar_cl)

        # filt thorns
        if thorns is not None:
            for th in thorns:
                x, y, r = th[0:3]
                dt_x = x - obs_cl_x
                dt_y = y - obs_cl_y
                ang = round(vector2angle(dt_x, dt_y))
                ang = min(ang, 359)
                if visible_angle_rec[ang] > r:
                    continue

                keep_thorns.append(th)

        keep_clone = np.concatenate(keep_clone)
        keep_clone = keep_clone.reshape(-1, 5)
        if keep_thorns:
            keep_thorns = np.concatenate(keep_thorns)
            keep_thorns = keep_thorns.reshape(-1, 3)
        else:
            keep_thorns = None
        return keep_clone, keep_thorns


class Bot_wt_env_Policy:
    def __init__(self, team_id: int, player_num_per_team: int, botagent):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.bot = [botagent(name=str(i), team=str(team_id)) for i in range(start, end)]

    def forward(self, data: dict, **kwargs) -> dict:
        ret = {}
        for env_id in data.keys():
            action = []
            for bot, raw_obs in zip(self.bot, data[env_id]):
                action.append(bot.step(data[env_id]))
            ret[env_id] = {'action': np.array(action)}
        return ret

    def reset(self, data_id: list = []) -> None:
        pass


def ang_to_rad(ang):
    return ang / 180. * np.pi


def rad_to_ang(rad):
    return rad * 180. / np.pi


def sort_clone_by_r(clone, r_dim=2):
    return clone[np.argsort(-clone[:, r_dim])]


def vector_normalize(x, y):
    dis = np.sqrt(np.power(x, 2) + np.power(y, 2))
    assert (dis != 0).all()
    return x / dis, y / dis


def overlap_process_to_np(overlap):
    overlap_np = {}
    for k, v in overlap.items():
        overlap_np[k] = group_process_to_np(overlap[k])
    return overlap_np


def group_process_to_np(group):
    if not group:
        return None
    group_list = [item_process_to_np(item) for item in group]
    group_np = np.concatenate(group_list)
    return group_np


def item_process_to_np(item):
    if len(item) == 2:
        item = np.array([float(item['position'][0]), float(item['position'][1]), float(item['radius'])]).reshape(-1, 3)
    elif len(item) == 4:
        item = np.array(
            [float(item['position'][0]), float(item['position'][1]), float(item['radius']), int(item['player']),
             int(item['team'])]).reshape(-1, 5)
    return item


def group_process_np_to_dict(arr):
    if arr.size == 0:
        return []
    group = []
    arr = arr.reshape(-1, arr.shape[-1])
    for i in range(arr.shape[0]):
        item = item_process_np_to_dict(arr[i])
        group.append(item)
    group.sort(key=lambda a: a['radius'], reverse=True)
    return group


def item_process_np_to_dict(arr):
    item = dict()
    arr = arr.flatten()
    item['position'] = Vector2(float(arr[0]), float(arr[1]))
    item['radius'] = float(arr[2])
    if arr.size == 5:
        item['player'] = int(arr[3])
        item['team'] = int(arr[4])
    return item


def calculate_spd(last_clones, cur_clones):
    # 按r排序
    last_clones = sort_clone_by_r(last_clones, 2)
    cur_clones = sort_clone_by_r(cur_clones, 2)
    # 计算重心
    last_center_x, last_center_y = get_zhongxin(last_clones)
    cur_center_x, cur_center_y = get_zhongxin(cur_clones)
    avg_move_x = cur_center_x - last_center_x
    avg_move_y = cur_center_y - last_center_y
    return avg_move_x, avg_move_y


def spore_init_dis():
    spore_init_spd = 250
    spore_vel_to_zero_time = 0.3
    spore_init_avg_spd = (spore_init_spd + spore_init_spd * 1 / 3) / 2
    spore_init_move_dis = 0.2 * spore_init_avg_spd + 3
    return spore_init_move_dis


def spore_end_dis():
    spore_init_spd = 250
    spore_vel_to_zero_time = 0.3
    spore_end_move_dis = 0.3 * spore_init_spd / 2 + 3
    return spore_end_move_dis


def match_clones(last_clones, cur_clones, avg_move_x, avg_move_y):
    """
    匹配同一玩家的所有clones
    :param avg_move_y:
    :param avg_move_x:
    :param last_clones:
    :param cur_clones:
    :return:
    """
    last_clones = last_clones.reshape((-1, last_clones.shape[-1]))
    cur_clones = cur_clones.reshape((-1, cur_clones.shape[-1]))
    assert last_clones.shape[0] <= 16 and cur_clones.shape[0] <= 16
    # 按r排序
    last_clones = sort_clone_by_r(last_clones, 2)
    cur_clones = sort_clone_by_r(cur_clones, 2)

    n_last = last_clones.shape[0]
    n_cur = cur_clones.shape[0]
    # x, y, r, team, player, last_idx, cur_idx
    last_clones_plus = np.c_[last_clones, np.zeros(n_last), np.zeros(n_last)]
    cur_clones_plus = np.c_[cur_clones, np.zeros(n_cur), np.zeros(n_cur)]

    last_clones_plus[:, -2] = np.arange(0, n_last)
    cur_clones_plus[:, -1] = np.arange(0, n_cur)
    last_clones_plus[:, -1] = -1
    cur_clones_plus[:, -2] = -1

    adj_cur_clones = cur_clones.copy()
    adj_cur_clones[:, 0] -= avg_move_x
    adj_cur_clones[:, 1] -= avg_move_y

    if n_cur == n_last:
        if n_cur == 1:
            last_clones_plus[0, -1] = 0
            cur_clones_plus[0, -2] = 0
        else:
            for cur_idx in range(n_cur):
                cl = adj_cur_clones[cur_idx]

                dis = get_dis(cl[0], cl[1], last_clones_plus[:, 0], last_clones_plus[:, 1])
                # sus_dis = dis[dis < (500 / 13) * 0.2 * 2]
                # if sus_dis.shape[0] == 0:
                #     print("Error:n_last == n_cur Match No suspect cl")
                #     continue
                # 按距离排序
                idx = np.argmin(dis)
                last_idx = int(last_clones_plus[idx, -2])
                last_clones_plus[last_idx, -1] = cur_idx
                cur_clones_plus[cur_idx, -2] = last_idx

    elif n_cur > n_last:
        return last_clones_plus, cur_clones_plus

    else:  # n_cur < n_last
        for cur_idx in range(n_cur):
            cl = adj_cur_clones[cur_idx]
            dis = get_dis(cl[0], cl[1], last_clones_plus[:, 0], last_clones_plus[:, 1])
            if dis.shape[0] == 0:
                print("Error:n_cur < n_last Match No suspect cl")
                continue
            # 按距离排序
            idx = np.argmin(dis)
            last_idx = int(last_clones_plus[idx, -2])
            last_clones_plus[last_idx, -1] = cur_idx
            cur_clones_plus[cur_idx, -2] = last_idx
            pass

    return last_clones_plus, cur_clones_plus


def dis_point_2_linear(x0, y0, A, B, C):
    """
    (x0,y0) Ax + By + C = 0
    :param x0:
    :param y0:
    :param A:
    :param B:
    :param C:
    :return:
    """
    up = np.abs(A * x0 + B * y0 + C)
    down = np.sqrt(A * A + B * B) + 1e-5
    return up / down


def get_zhongxin(clone):
    x_arr = []
    y_arr = []
    r_arr = []
    for p in clone:
        x_arr.append(p[0])
        y_arr.append(p[1])
        r_arr.append(p[2])

    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    r_arr = np.array(r_arr)

    w_arr = np.power(r_arr, 2)
    x_center = np.sum(x_arr * w_arr) / np.sum(w_arr)
    y_center = np.sum(y_arr * w_arr) / np.sum(w_arr)

    return x_center, y_center


def vector2angle(x, y):
    theta = np.arctan2(y, x)
    angle = theta * 180 / np.pi
    if isinstance(angle, np.float64) or isinstance(angle, float):
        if abs(angle) < 1e-4:
            angle = 0
        elif angle < 0:
            angle += 360
        return angle
    angle[np.abs(angle) < 1e-4] = 0
    angle[angle < 0] = angle[angle < 0] + 360.
    return angle


def get_dis(x1, y1, x2, y2):
    dis = np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))
    return dis


def collect_nearest_food_balls(food_balls, my_max_clone_ball):
    min_distance = 10000
    min_food_ball = None
    for food_ball in food_balls:
        distance = (food_ball['position'] - my_max_clone_ball['position']).length()
        if distance < min_distance:
            min_distance = distance
            min_food_ball = copy.deepcopy(food_ball)
    return min_distance, min_food_ball


def get_move_vector(last_clone, cur_clone):
    last_zhongxin = get_zhongxin(last_clone)
    cur_zhongxin = get_zhongxin(cur_clone)
    dt_x = cur_zhongxin[0] - last_zhongxin[0]
    dt_y = cur_zhongxin[1] - last_zhongxin[1]
    return dt_x, dt_y


def insec(x, y, R, a, b, S):
    d = np.sqrt(np.abs(np.power(a - x, 2) + np.power(b - y, 2)))

    A = (np.power(R, 2) - np.power(S, 2) + np.power(d, 2)) / (2 * d)
    h = np.sqrt(np.power(R, 2) - np.power(A, 2))
    x2 = x + A * (a - x) / d
    y2 = y + A * (b - y) / d
    x3 = x2 - h * (b - y) / d
    y3 = y2 + h * (a - x) / d
    x4 = x2 + h * (b - y) / d
    y4 = y2 - h * (a - x) / d
    print(x3, y3)
    print(x4, y4)
    c1 = np.array([x3, y3])
    c2 = np.array([x4, y4])
    return c1, c2


def food_filter(food, clone):
    if food is None or food.size == 0:
        return None
    filt_food = food.copy()

    n_clone = clone.shape[0]
    min_r = np.min(clone[:, 2])
    min_clone = clone[clone[:, 2] == min_r]
    min_clone = min_clone.flatten()

    # filt border food
    fake_r = np.sqrt(np.sum(np.power(clone[:, 2], 2)))
    border_length = min(50, fake_r)
    food_temp = filt_food.copy()
    food_temp = food_temp[food_temp[:, 0] > border_length]
    food_temp = food_temp[food_temp[:, 0] < 1000 - border_length]
    food_temp = food_temp[food_temp[:, 1] > border_length]
    food_temp = food_temp[food_temp[:, 1] < 1000 - border_length]
    # replace_food
    if food_temp.size > 0:
        filt_food = food_temp

    # 单球筛选
    if n_clone == 1:
        least_count = 50
    else:
        least_count = 100

    dt_x = filt_food[:, 0] - min_clone[0]
    dt_y = filt_food[:, 1] - min_clone[1]
    dis = np.sqrt(np.power(dt_x, 2) + np.power(dt_y, 2))

    for r in range(100, 1000, 100):
        inside = (dis <= r)
        if inside.size > least_count:
            filt_food = filt_food[inside].reshape(-1, 3)
            break

    return filt_food


def food_filter_dict(food_balls, my_clone_ball):
    food_filter_rec = []
    my_clone_r = my_clone_ball[0]["radius"]
    for fb in food_balls:
        x = fb['position'].x
        y = fb['position'].y

        if my_clone_r < x < 1000 - my_clone_r and my_clone_r < y < 1000 - my_clone_r:
            food_filter_rec.append(fb)
    return food_filter_rec


def food_revalue(food, clone):
    clone = clone.reshape(-1, 5)
    score_rec = [0.] * food.shape[0]
    r_factor = 1.

    for i, f in enumerate(food):
        if clone.shape[0] == 1:
            clone_r = clone[0, 2]
            dis = get_dis(f[0], f[1], food[:, 0], food[:, 1])
            score = 4. * np.sum((dis < r_factor * clone_r))
        else:
            score = 4.
        score_rec[i] = score
    return np.array(score_rec)


class EventCollector:
    def __init__(self, player_num=12):
        self.player_num = player_num
        self.leader_board_record = [27. for _ in range(4)]
        self.last_obs = None
        self.cur_obs = None
        self.last_clone = None
        self.cur_clone = None
        self.last_spd = [[0., 0.] for _ in range(player_num)]

    def update(self, last_obs, cur_obs):
        self.last_obs = last_obs
        self.cur_obs = cur_obs
        self.last_clone = [np.array(self.last_obs[1][str(i)]['overlap']['clone']) for i in range(self.player_num)]
        self.cur_clone = [np.array(self.cur_obs[1][str(i)]['overlap']['clone']) for i in range(self.player_num)]

    def reset(self):
        self.leader_board_record = [27. for _ in range(4)]
        self.last_spd = [[0., 0.] for _ in range(12)]

    def collect_event(self):
        """
        evt_rec:记录字典
            score_change:根据体积变化计算分数变化
            eat_clone_event:记录clone互吃事件（包含队友吃，对手吃，自己吃）
            farm_score:score_change抵消eat_clone_event的分数（单纯发育/衰减得分）
        :return: evt_rec
        """
        evt_rec = dict()

        score_change = self.score_change()
        player_dead_event = self.player_dead_event()
        gg_event = self.gg_event()
        clone_lost = self.clone_lost_event(player_dead_event)

        eat_clone_event = self.eat_clone_event(clone_lost)

        evt_rec['score_change'] = score_change
        evt_rec['eat_clone_event'] = eat_clone_event
        evt_rec['farm_score'] = self.farm_score(score_change, eat_clone_event)
        evt_rec['player_dead_event'] = player_dead_event
        evt_rec['gg_event'] = gg_event
        return evt_rec

    def collect_reward(self, evt_rec):
        # farm_score
        farm_score = evt_rec['farm_score']
        # move_score
        move_score = self.collect_score_move()

        # fight_score
        eat_clone_event = evt_rec['eat_clone_event']
        cur_clone = self.cur_clone
        cur_obs = self.cur_obs
        fight_score = self.collect_score_fight(eat_clone_event, cur_obs, cur_clone)

        # player_dead_score
        player_dead_event = evt_rec['player_dead_event']
        player_dead_score = self.collect_score_playerdead(player_dead_event)

        farm_score = np.array(farm_score)
        move_score = np.array(move_score)
        fight_score = np.array(fight_score)

        reward = farm_score + move_score + fight_score
        reward /= 1000

        player_dead_score = np.array(player_dead_score)
        reward += player_dead_score
        rec = [reward[i * 3: (i + 1) * 3].flatten() for i in range(4)]
        return rec

    def collect_score_fight(self, eat_clone_event, cur_obs, cur_clone, team_sprit=0.):
        fight_score = [0. for _ in range(self.player_num)]

        for eat_evt in eat_clone_event:
            eat_player_id = int(eat_evt['eat_player_id'])
            feed_player_id = int(eat_evt['feed_player_id'])
            eat_player_team = eat_player_id // 3
            feed_player_team = feed_player_id // 3

            eat_score = eat_evt['eat_score']
            eat_x = eat_evt['eat_x']
            eat_y = eat_evt['eat_y']

            # 自己吃自己
            if eat_player_id == feed_player_id:
                continue

            # 队友互吃
            elif eat_player_team == feed_player_team:
                pass
                # # 补偿被吃的损失
                # fight_score[feed_player_id] += 1.0 * eat_score
                # # 抵消吃队友的奖励
                # fight_score[eat_player_id] -= 1.0 * eat_score

            # 敌对关系
            else:
                # 当事人
                fight_score[feed_player_id] -= 2 * eat_score
                fight_score[eat_player_id] += 2 * eat_score

                if team_sprit == 0.0:
                    continue

                # 队友被吃
                for i in range(3 * feed_player_team, 3 * (feed_player_team + 1)):
                    # 跳过本人
                    if i == feed_player_id:
                        continue

                    # 在视野内
                    l_t_x, l_t_y, r_b_x, r_b_y = cur_obs[1][str(i)]['rectangle']
                    if not (l_t_x < eat_x < r_b_x and l_t_y < eat_y < r_b_y):
                        continue

                    clone = cur_clone[i]
                    my_clone = clone[clone[:, 3].astype(int) == i]
                    dis_mean = np.mean(get_dis(my_clone[:, 0], my_clone[:, 1], eat_x, eat_y))
                    if dis_mean == 0:
                        dis_mean = 1
                    ratio = 100. / dis_mean
                    ratio = np.clip(ratio, 0.2, 1)
                    # f = ratio
                    fight_score[i] -= team_sprit * eat_score

                # 队友吃人
                for i in range(3 * eat_player_team, 3 * (eat_player_team + 1)):
                    # 跳过本人
                    if i == eat_player_id:
                        continue

                    # 在视野内
                    l_t_x, l_t_y, r_b_x, r_b_y = cur_obs[1][str(i)]['rectangle']
                    if not (l_t_x < eat_x < r_b_x and l_t_y < eat_y < r_b_y):
                        continue
                    clone = cur_clone[i]
                    my_clone = clone[clone[:, 3].astype(int) == i]
                    dis_mean = np.mean(get_dis(my_clone[:, 0], my_clone[:, 1], eat_x, eat_y))
                    if dis_mean == 0:
                        dis_mean = 1
                    ratio = 100 / dis_mean
                    ratio = np.clip(ratio, 0.2, 1)
                    fight_score[i] += team_sprit * eat_score
        return fight_score

    def collect_score_playerdead(self, gameover_event):
        n_player = self.player_num
        score = [0. for _ in range(n_player)]
        for evt in gameover_event:
            player_id = evt['player_id']
            score[player_id] = -2.
        return score

    def collect_score_move(self):
        n_player = self.player_num
        score_move = [0. for _ in range(n_player)]
        for i in range(n_player):
            last_clone = self.last_clone[i]
            cur_clone = self.cur_clone[i]
            last_player_clone = last_clone[last_clone[:, -2] == i]
            cur_player_clone = cur_clone[cur_clone[:, -2] == i]
            last_center_x, last_center_y = get_zhongxin(last_player_clone)
            cur_center_x, cur_center_y = get_zhongxin(cur_player_clone)

            # 荆棘球奖励
            score_thorn = 0.
            cur_thorns = np.array(self.cur_obs[1][str(i)]['overlap']['thorns'])
            if cur_thorns.shape[0] > 0:
                thorn = cur_thorns[0]
                min_dis = 1000
                # 找到最近荆棘球
                for t in cur_thorns:
                    t_x, t_y, t_r = t
                    # 荆棘
                    dis = get_dis(t_x, t_y, cur_center_x, cur_center_y)
                    if dis < min_dis:
                        min_dis = dis
                        thorn = t

                min_dis = 1000
                target_clone = None
                for c in cur_player_clone:
                    c_x, c_y, c_r = c[:3]
                    t_x, t_y, t_r = thorn
                    dis = get_dis(t_x, t_y, c_x, c_y)
                    if c_r > t_r:
                        if dis < min_dis:
                            min_dis = dis
                            target_clone = c

                if target_clone is not None:
                    x_thorn, y_thorn = thorn[:2]
                    x_clone, y_clone = target_clone[:2]
                    # v1: clone -> thorn向量
                    v1 = np.array([x_thorn - x_clone, y_thorn - y_clone])
                    l1 = get_dis(x_thorn, y_thorn, x_clone, y_clone)

                    x_move = cur_center_x - last_center_x
                    y_move = cur_center_y - last_center_y
                    # v2: speed向量
                    v2 = np.array([x_move, y_move])
                    vdot = np.vdot(v1, v2)
                    score_thorn = vdot / (l1 * l1)
                    score_thorn = np.clip(score_thorn, 0, 10)

            # 角落惩罚
            score_corner = 0.
            for clone in cur_player_clone:
                x = clone[0]
                y = clone[1]
                r = clone[2]
                if not ((3 + r <= x <= 1000 - 3 - r) and (3 + r <= y <= 1000 - 3 - r)):
                    score_corner -= 1

            score_move[i] = score_thorn + score_corner
        return score_move

    def score_change(self):
        rec = []

        player_num = self.player_num
        for i in range(player_num):
            last_clone = self.last_clone[i]
            cur_clone = self.cur_clone[i]
            last_player_clone = last_clone[last_clone[:, -2] == i]
            cur_player_clone = cur_clone[cur_clone[:, -2] == i]
            last_score = np.sum(np.power(last_player_clone[:, 2], 2))
            cur_score = np.sum(np.power(cur_player_clone[:, 2], 2))
            rec.append(cur_score - last_score)
        return rec

    def eat_food_event(self):
        rec = []
        return rec

    def eat_thorn_event(self):
        """
        player_id|thorn_r
        :return:
        """
        rec = []
        return rec

    def farm_score(self, score_change, eat_clone_event):
        farm_score = score_change.copy()
        if not eat_clone_event:
            return farm_score
        for eat_evt in eat_clone_event:
            eat_player_id = int(eat_evt['eat_player_id'])
            feed_player_id = int(eat_evt['feed_player_id'])
            eat_score = eat_evt['eat_score']
            farm_score[eat_player_id] -= eat_score
            farm_score[feed_player_id] += eat_score
        return farm_score

    def eat_spore_event(self):
        rec = []
        return rec

    def player_dead_event(self):
        event = []
        n_player = self.player_num
        for i in range(n_player):
            last_clone = self.last_clone[i]
            cur_clone = self.cur_clone[i]
            last_my_clone = last_clone[last_clone[:, -2] == i]
            cur_my_clone = cur_clone[cur_clone[:, -2] == i]

            n_cur_clone = cur_my_clone.shape[0]
            if np.sum(last_my_clone[-1, 2]) >= 4.3 and n_cur_clone == 1 and cur_my_clone[0, 2] <= 4.2:
                info = dict()
                info['player_id'] = i
                event.append(info)
        return event

    def gg_event(self):
        gg_event_rec = []
        leader_board = self.cur_obs[0]['leaderboard']
        for team, score in leader_board.items():
            team_id = int(team)
            self.leader_board_record[team_id] = max(self.leader_board_record[team_id], score)
            if score < self.leader_board_record[team_id] / 3:
                gg_event = dict()
                gg_event['team_id'] = team_id
                gg_event_rec.append(gg_event)
        return gg_event_rec

    def clone_lost_event(self, gg_event):
        rec = []
        player_num = self.player_num

        for i in range(player_num):
            last_clone = self.last_clone[i]
            cur_clone = self.cur_clone[i]
            last_player_clone = last_clone[last_clone[:, -2] == i]
            cur_player_clone = cur_clone[cur_clone[:, -2] == i]

            n_last_clones = last_player_clone.shape[0]
            n_cur_clones = cur_player_clone.shape[0]

            gg = False
            for evt in gg_event:
                player_id = [evt['player_id']]
                if i == player_id:
                    gg = True

            if gg:
                lost_clone = last_player_clone
                self.last_spd[i] = [0.0, 0.0]
            else:
                if n_last_clones == n_cur_clones:
                    avg_move_x, avg_move_y = calculate_spd(last_player_clone, cur_player_clone)
                    self.last_spd[i] = [avg_move_x, avg_move_y]
                else:
                    # 读取上一个数据
                    avg_move_x, avg_move_y = self.last_spd[i]

                last_clones_plus, cur_clones_plus = match_clones(last_player_clone, cur_player_clone, avg_move_x,
                                                                 avg_move_y)

                # 只考虑数量减小
                if n_cur_clones > n_last_clones:
                    continue

                lost_clone = last_clones_plus[last_clones_plus[:, -1] == -1]

            if lost_clone.shape[0] > 0:
                rec.append(lost_clone.reshape((-1, lost_clone.shape[-1])))

        return rec

    def eat_clone_event(self, clone_lost_rec):
        rec = []
        player_num = self.player_num
        clones = []
        if not clone_lost_rec:
            return rec

        for i in range(player_num):
            cur_clone = self.cur_clone[i]
            cur_player_clone = cur_clone[cur_clone[:, -2] == i]
            clones.append(cur_player_clone)

        clones = np.concatenate(clones)
        clone_lost = np.concatenate(clone_lost_rec)
        for cl in clone_lost:
            eat_r = cl[2]
            sus_clones = clones[clones[:, 2] > eat_r]
            dis = get_dis(sus_clones[:, 0], sus_clones[:, 1], cl[0], cl[1])
            if dis.size == 0:
                # 吃荆棘球爆炸
                print("Error 找不到吃人的球")
                continue
            eater = sus_clones[np.argmin(dis)]
            eat_score = eat_r * eat_r
            eat_player_id = eater[3]
            feed_player_id = cl[3]
            eat_info = dict()
            eat_info['eat_player_id'] = eat_player_id
            eat_info['feed_player_id'] = feed_player_id
            eat_info['eat_score'] = eat_score
            eat_info['eat_x'] = cl[0]
            eat_info['eat_y'] = cl[1]
            rec.append(eat_info)
        return rec


class ActionKit:
    def __init__(self):
        pass

    @staticmethod
    def split_obs(obs):
        if obs[0] is None:
            return obs
        else:
            obs_split = []
            for o in obs:
                n = len(o['scalar'])
                obs_dicts = []
                for i in range(n):
                    obs_dicts.append({})
                for t, v in o.items():
                    for j in range(len(o[t])):
                        obs_dicts[j][t] = v[j]
                obs_split += obs_dicts
            return obs_split

    def eat_food(self, obs):
        overlap = obs['collate_ignore_raw_obs']['overlap']
        food = np.array(overlap['food'])
        clone = np.array(overlap['clone'])
        clone_n = clone.shape[0]
        food_n = food.shape[0]
        min_r = np.argmin(clone[:, 2])
        min_clone = clone[min_r]

        if food_n == 0:
            return 500 - min_clone[0], 500 - min_clone[1]

        v_center_x = obs['scalar'][4] * 40
        v_center_y = obs['scalar'][5] * 40

        food = self.filter_food(food, min_clone, v_center_x, v_center_y)

        select_food, select_clone = self.find_food(food, clone, v_center_x, v_center_y,
                                                   consider_vel_direction=True)

        if select_food is not None and select_clone is not None:
            dt_x = select_food[0] - select_clone[0]
            dt_y = select_food[1] - select_clone[1]
            force_x, force_y = self.give_force_to_pos(dt_x, dt_y, v_center_x, v_center_y)
            return force_x, force_y
        else:
            return None, None

    def filter_food(self, food, clone, vel_x, vel_y):
        dis_to_center = np.sqrt(np.power(food[:, 0] - 500, 2) + np.power(food[:, 1] - 500, 2))
        food = food[dis_to_center < 500]

        last_vel_angle = vector2angle(vel_x, vel_y)
        if vel_x == 0 and vel_y == 0:
            return food
        max_speed = 500 / (10 + clone[2])
        cur_spd = np.sqrt(np.power(vel_x, 2) + np.power(vel_y, 2))
        angle_range = 1 - 0.5 * (cur_spd / max_speed)

        dt_x = food[:, 0] - clone[0]
        dt_y = food[:, 1] - clone[1]
        dt_dis = np.sqrt(np.power(dt_x, 2) + np.power(dt_y, 2))
        # for r in [200, 300, 400]:
        #     # 筛选部分范围内的食物球
        #     if np.sum(dt_dis < r) > 0:
        #         food = food[dt_dis < r]
        #         break
        relative_angle = vector2angle(dt_x, dt_y)
        r1 = np.abs(relative_angle - last_vel_angle) < 180 * angle_range
        r2 = np.abs(relative_angle - last_vel_angle + 360) < 180 * angle_range
        r3 = np.abs(relative_angle - last_vel_angle - 360) < 180 * angle_range

        filtered_food = food[r1 | r2 | r3]
        # if angle_low >= 0 and angle_high <= 360:
        #     filtered_food = food[(relative_angle > angle_low) & (relative_angle < angle_high)]
        # elif angle_low < 0:
        #     filtered_food = food[(relative_angle > angle_low + 360) | (relative_angle < angle_high)]
        # else:
        #     # angle_high> 360:
        #     filtered_food = food[(relative_angle > angle_high) | (relative_angle < angle_high - 360)]

        if filtered_food.shape[0] > 0:
            return filtered_food
        else:
            return food

    def find_food(self, food, my_clone, v_center_x, v_center_y, consider_vel_direction=True, consider_food_density=True,
                  filt_food=True):
        n_my_clone = my_clone.shape[0]
        if filt_food:
            food = food_filter(food, my_clone)

        if food is None or food.size == 0 or n_my_clone == 0:
            return None, None, 0.0

        min_clone = my_clone[np.argmin(my_clone[:, 2])].flatten()
        if consider_vel_direction:
            if n_my_clone == 1:
                cl = min_clone
                spend_time_list = [self.move_analyzer(cl[0], cl[1], cl[2], v_center_x, v_center_y, f[0], f[1], f[2])[-1]
                                   for f in food]
                spend_time = np.array(spend_time_list)
                spend_time[spend_time < 0.1] = 0.1
                value = food_revalue(food, my_clone)
                score = value / spend_time

                food_score = np.max(score)
                select_food = food[np.argmax(score)].tolist()
                select_clone = cl
            else:
                my_clone = my_clone.reshape(-1, 5)
                K = 5

                first_score = []
                # first round score
                for my_cl in my_clone:
                    spend_time_list = [
                        self.move_analyzer(my_cl[0], my_cl[1], my_cl[2], v_center_x, v_center_y, f[0], f[1], f[2])[-1]
                        for f in food]
                    spend_time = np.array(spend_time_list)
                    spend_time[spend_time < 0.1] = 0.1
                    value = food_revalue(food, my_clone)
                    score = value / spend_time

                    # more than 5 food
                    if food.shape[0] > 5:
                        best_score_idx = np.argpartition(-score, K)[:K]
                        select_food = food[best_score_idx]
                        select_score = score[best_score_idx]
                    # less than 5 food
                    else:
                        select_food = food
                        select_score = score

                    for i in range(select_food.shape[0]):
                        rec = [select_food[i][0], select_food[i][1], my_cl[0], my_cl[1], select_score[i]]
                        first_score.append(rec)

                # second round score
                best_score = 0.
                best_score_idx = 0

                first_score_n = len(first_score)
                for i in range(first_score_n):
                    score_dict = dict()
                    rec_base = first_score[i]
                    f_x_base, f_y_base = rec_base[0], rec_base[1]
                    cl_x_base, cl_y_base = rec_base[2], rec_base[3]
                    dt_x_base = f_x_base - cl_x_base
                    dt_y_base = f_y_base - cl_y_base
                    dt_x_base, dt_y_base = vector_normalize(dt_x_base, dt_y_base)
                    score_base = rec_base[-1]
                    k_base = xy_to_str(f_x_base, f_y_base)
                    score_dict[k_base] = score_base

                    for j in range(first_score_n):
                        if i == j:
                            continue
                        rec_new = first_score[j]
                        f_x_new, f_y_new = rec_new[0], rec_new[1]
                        cl_x_new, cl_y_new = rec_new[2], rec_new[3]
                        dt_x_new = f_x_new - cl_x_new
                        dt_y_new = f_y_new - cl_y_new
                        dt_x_new, dt_y_new = vector_normalize(dt_x_new, dt_y_new)
                        k_new = xy_to_str(f_x_new, f_y_new)
                        cur_score = score_dict.get(k_new, 0.0)
                        cor = dt_x_base * dt_x_new + dt_y_base * dt_y_base
                        score_new = rec_new[-1] * cor
                        if score_new > cur_score:
                            score_dict[k_new] = score_new

                    score_sum = sum(score_dict.values())
                    if score_sum > best_score:
                        best_score_idx = i

                best_rec = first_score[best_score_idx]
                food_return = [best_rec[0], best_rec[1]]
                cl_return = [best_rec[2], best_rec[3]]
                return food_return, cl_return, best_score
        else:
            dt_x = food[:, 0] - min_clone[0]
            dt_y = food[:, 1] - min_clone[1]
            dis = np.sqrt(np.power(dt_x, 2) + np.power(dt_y, 2)) - min_clone[2] - food[:, 2]
            select_food = food[np.argmin(dis)].tolist()
            cl = my_clone[-1].flatten()
            spend_time = self.move_analyzer(cl[0], cl[1], cl[2], v_center_x, v_center_y, select_food[0], select_food[1],
                                            select_food[2])[-1]
            spend_time = np.array(spend_time)
            spend_time[spend_time < 0.1] = 0.1
            value = food_revalue(food, my_clone)
            score = value / spend_time
            food_score = np.max(score)
            select_clone = cl

        return select_food, select_clone, float(food_score)

    def find_food_plus(self, food, clone, spd_x, spd_y, travel_time):
        if food.size == 0 or clone.size == 0:
            return None, None
        min_clone = clone[np.argmin(clone[:, 2])].flatten()

        # filt border food
        border_length = min(50, min_clone[2])
        food = food[food[:, 0] > border_length]
        food = food[food[:, 0] < 1000 - border_length]
        food = food[food[:, 1] > border_length]
        food = food[food[:, 1] < 1000 - border_length]
        find = False

        dt_x = food[:, 0] - min_clone[0]
        dt_y = food[:, 1] - min_clone[1]
        dis = np.sqrt(np.power(dt_x, 2) + np.power(dt_y, 2))

        for r in range(100, 600, 100):
            inside = food[dis <= r]
            if inside.size > 0:
                food = inside.reshape(-1, 3)
                find = True
                break

        if not find:
            return None

        n_food = food.shape[0]
        food_flag = [-1] * n_food

        cl_x = float(min_clone[0])
        cl_y = float(min_clone[1])
        cl_r = float(min_clone[2])

        score_max, best_food_flag = self.dfs_search_food(cl_x, cl_y, cl_r, spd_x, spd_y, food, food_flag, 0.0,
                                                         travel_time, 0., 0)
        food_queue = []
        for i, flag in enumerate(best_food_flag):
            if flag >= 0:
                food_queue.append([flag, food[i]])
        food_queue.sort(key=lambda f: f[0], reverse=False)
        score = score_max / travel_time
        fs = FoodScore(food_queue, score)
        return fs

    def dfs_search_food(self, clone_x, clone_y, clone_r, v_x, v_y, food, food_flag, used_time, rest_time, score, idx):
        # no available food
        if -1 not in food_flag:
            return score / (used_time + 1e-3), copy.deepcopy(food_flag)

        score_max = score
        best_food_flag = copy.deepcopy(food_flag)
        for i in range(len(food_flag)):
            # food has been used
            if food_flag[i] != -1:
                continue

            food_x, food_y, food_r = food[i]
            clone_x_last, clone_y_last, clone_r_last, v_x_last, v_y_last, spend_time = self.move_analyzer(clone_x,
                                                                                                          clone_y,
                                                                                                          clone_r, v_x,
                                                                                                          v_y, food_x,
                                                                                                          food_y,
                                                                                                          food_r)

            if rest_time >= spend_time:
                food_flag[i] = idx
                dt_score = float(pow(clone_r_last, 2) - pow(clone_r, 2))
                t = float(rest_time - spend_time)
                s = float(score + dt_score)
                u = float(used_time + spend_time)
                new_score, new_food_flag = self.dfs_search_food(clone_x_last, clone_y_last, clone_r_last, v_x_last,
                                                                v_y_last, food, food_flag, u, t, s, idx + 1)

                # TODO:最大分数改成平均耗时
                if new_score > score_max:
                    score_max = new_score
                    best_food_flag = new_food_flag
                food_flag[i] = -1
            else:
                food_flag[i] = -2
        return score_max, best_food_flag

    def move_analyzer(self, clone_x, clone_y, clone_r, v_x, v_y, tar_x, tar_y, tar_r):
        ang_v = vector2angle(v_x, v_y)
        spd_value = np.sqrt(np.power(v_x, 2) + np.power(v_y, 2))
        dt_x = tar_x - clone_x
        dt_y = tar_y - clone_y

        ang_to_tar = vector2angle(dt_x, dt_y)
        dt_ang = float(np.abs(ang_to_tar - ang_v))
        if dt_ang > 180.:
            dt_ang = 360. - dt_ang
        dt_rad = ang_to_rad(dt_ang)

        dis_center = get_dis(clone_x, clone_y, tar_x, tar_y)
        dis_move = dis_center - max(clone_r, tar_r)
        clone_r_last = np.sqrt(np.power(clone_r, 2) + np.power(tar_r, 2))

        if dis_move <= 0:
            return float(clone_x), float(clone_y), float(clone_r_last), float(v_x), float(v_y), 0.0

        v_0_length = float(np.cos(dt_rad) * spd_value)
        acc = 100
        # V末 * V末 - v0 * v0 = 2 * a * s
        # 假设acc恒为100 per second
        v_last_length = float(np.sqrt(2 * acc * dis_move + np.power(v_0_length, 2)))
        v_max_length = 500. / (clone_r + 10)
        if v_last_length > v_max_length:
            v_last_length = v_max_length
            acc_time = (v_last_length - v_0_length) / acc
            has_move = (v_last_length + v_0_length) / 2 * acc_time
            rest_move = max(0, dis_move - has_move)
            rest_time = rest_move / v_last_length
            spend_time = acc_time + rest_time

        else:
            spend_time = (v_last_length - v_0_length) / acc

        clone_x_last = clone_x + dt_x * dis_move / dis_center
        clone_y_last = clone_y + dt_y * dis_move / dis_center

        v_x_last = v_last_length * dt_x / dis_center
        v_y_last = v_last_length * dt_y / dis_center

        return float(clone_x_last), float(clone_y_last), float(clone_r_last), float(v_x_last), float(v_y_last), float(
            spend_time)

    def give_force_to_pos(self, dt_x, dt_y, v_x, v_y):
        '''
        考虑速度的给力方向
        :param dt_x:
        :param dt_y:
        :param v_x:
        :param v_y:
        :return: x方向的力，y方向的力
        '''
        if v_x == 0.0 and v_y == 0.0:
            return dt_x, dt_y

        ang_v = vector2angle(v_x, v_y)
        spd_value = np.sqrt(np.power(v_x, 2) + np.power(v_y, 2))

        ang_to_target = vector2angle(dt_x, dt_y)
        dt_ang = np.abs(ang_v - ang_to_target)
        if dt_ang >= 180.:
            dt_ang = 360. - dt_ang

        dt_ang_rad = ang_to_rad(dt_ang)
        spd_to_target = spd_value * np.cos(dt_ang_rad)
        spd_qie_target = spd_value * np.sin(dt_ang_rad)

        during = 0.1
        force_max = 100.
        force_qie_target = np.clip(spd_qie_target / during, -force_max, force_max)
        force_to_target = np.sqrt(np.power(force_max, 2) - np.power(force_qie_target, 2))

        if ang_v > ang_to_target:
            if ang_v - ang_to_target <= 180.:
                offset = -90.
            else:
                offset = 90.
        else:
            if ang_to_target - ang_v <= 180.:
                offset = 90.
            else:
                offset = -90.

        ang_offset_qie = ang_to_target + offset
        rad_offset_qie = ang_to_rad(ang_offset_qie)
        rad_to_target = ang_to_rad(ang_to_target)
        force_x = np.cos(rad_to_target) * force_to_target + np.cos(rad_offset_qie) * force_qie_target
        force_y = np.sin(rad_to_target) * force_to_target + np.sin(rad_offset_qie) * force_qie_target
        return force_x, force_y

    @staticmethod
    def find_thorn(obs):
        # t = dis / v
        # v = 500 / (10 + r)
        # t = dis / 500 * (10+r) = dis * (10 + r)
        overlap = obs['collate_ignore_raw_obs']['overlap']
        select_thorn = None
        select_clone = None
        min_t = 100000000
        clone = overlap['clone']
        thorn = overlap['thorns']
        if not (clone and thorn):
            return None, None
        for cl in clone:
            for th in thorn:
                if cl[2] > th[2]:
                    dis = pow(cl[0] - th[0], 2) + pow(cl[1] - th[1], 2)
                    t = dis * (10 + cl[2])
                    if t < min_t:
                        min_t = t
                        select_thorn = th
                        select_clone = cl
        if select_thorn is None or select_clone is None:
            return None, None

        select_thorn = np.array(select_thorn).flatten()
        select_clone = np.array(select_clone).flatten()
        return select_thorn, select_clone

    @staticmethod
    def find_move_attack_target_by_clone(my_cl, other_cl):
        other_cl = other_cl.reshape((-1, other_cl.shape[-1]))
        if other_cl.shape[0] == 0:
            return None
        smaller_cl = other_cl[other_cl[:, 2] < my_cl[2]]
        if smaller_cl.shape[0] == 0:
            return None
        move_to_eat_dis = get_dis(my_cl[0], my_cl[1], smaller_cl[:, 0], smaller_cl[:, 1]) - my_cl[2]
        move_to_eat_dis = np.clip(move_to_eat_dis, 0.1, 100000)
        add_v = np.power(smaller_cl[:, 2], 2)
        target_arg = np.argmax(add_v / move_to_eat_dis)
        return smaller_cl[target_arg]

    @staticmethod
    def find_split_attack_target_by_clone(my_cl, other_cl):
        if other_cl.shape[0] == 0:
            return None
        split_eat_cl = other_cl[other_cl[:, 2] < my_cl[2] * np.sqrt(0.5)]
        if split_eat_cl.shape[0] == 0:
            return None
        dis = np.clip(get_dis(my_cl[0], my_cl[1], split_eat_cl[:, 0], split_eat_cl[:, 1]), 0.1, 100000)
        split_eat_cl = split_eat_cl[dis <= 2.12 * my_cl[2]]
        if split_eat_cl.shape[0] == 0:
            return None
        add_v = np.power(split_eat_cl[:, 2], 2)
        target_arg = np.argmax(add_v)
        return split_eat_cl[target_arg]

    @staticmethod
    def find_move_feed_target_by_clone(my_cl, other_cl):
        if other_cl.shape[0] == 0:
            return None
        bigger_cl = other_cl[other_cl[:, 2] > my_cl[2]]
        if bigger_cl.shape[0] == 0:
            return None
        move_to_eat_dis = get_dis(my_cl[0], my_cl[1], bigger_cl[:, 0], bigger_cl[:, 1]) - bigger_cl[:, 2]
        move_to_eat_dis = np.clip(move_to_eat_dis, 0.1, 100000)
        vel = 500 / (10 + bigger_cl[:, 2])
        target_arg = np.argmin(move_to_eat_dis / vel)
        return bigger_cl[target_arg]

    @staticmethod
    def find_split_feed_target_by_clone(my_cl, other_cl):
        if other_cl.shape[0] == 0:
            return None
        split_eat_cl = other_cl[other_cl[:, 2] * np.sqrt(0.5) > my_cl[2]]
        if split_eat_cl.shape[0] == 0:
            return None
        dis = np.clip(get_dis(my_cl[0], my_cl[1], other_cl[:, 0], other_cl[:, 1]), 0.1, 100000)
        split_eat_cl = split_eat_cl[dis <= 2.12 * other_cl[:, 2]]
        if split_eat_cl.shape[0] == 0:
            return None
        move_to_eat_dis = get_dis(my_cl[0], my_cl[1], split_eat_cl[:, 0], split_eat_cl[:, 1]) - split_eat_cl[:, 2]
        move_to_eat_dis = np.clip(move_to_eat_dis, 0.1, 100000)
        target_arg = np.argmax(move_to_eat_dis)
        return split_eat_cl[target_arg]

    @staticmethod
    def find_attack_target_by_relation(cl_rela):
        follow = (cl_rela[:, 5] == 1)
        eat = (cl_rela[:, -6] == 1)
        is_enemy = (cl_rela[:, -1] == 1)
        can_attack = follow & eat & is_enemy
        if np.sum(follow & eat & is_enemy) == 0:
            return None
        # 4: r_2 / dis 兼顾被吃物体的大小 和距离
        r2_dis_argsort = np.argsort(cl_rela[:, 4])
        for sort_idx in r2_dis_argsort:
            if can_attack[sort_idx]:
                return sort_idx
        return None

    @staticmethod
    def find_feed_target_by_relation(cl_rela):
        be_follow = cl_rela[:, 6] == 1
        be_eat = cl_rela[:, -5] == 1
        is_enemy = cl_rela[:, -1] == 1
        can_feed = be_follow & be_eat & is_enemy
        if np.sum(can_feed) == 0:
            return None
        r2_dis_argsort = np.argsort(cl_rela[:, 4])
        for sort_idx in r2_dis_argsort:
            if can_feed[sort_idx]:
                return sort_idx
        return None

    @staticmethod
    def aft_eat(atk_cl, feed_cl):
        pass

    @staticmethod
    def dir_src_to_tar(src_clone, tar_clone):
        src_x = src_clone[0]
        src_y = src_clone[1]
        tar_x = tar_clone[0]
        tar_y = tar_clone[1]
        dir_x = tar_x - src_x
        dir_y = tar_y - src_y
        return dir_x, dir_y

    # @staticmethod
    # def corner_adjust(x,y):
    @staticmethod
    def attack_split(src_clone, tar_clone):
        src_x = src_clone[0]
        src_y = src_clone[1]
        tar_x = tar_clone[0]
        tar_y = tar_clone[1]
        dir_x = tar_x - src_x
        dir_y = tar_y - src_y
        action_type = 4
        return [dir_x, dir_y, action_type]

    @staticmethod
    def run_from_enemy(src_cl, tar_cl):
        run_dir_x = src_cl[0] - tar_cl[0]
        run_dir_y = src_cl[1] - tar_cl[1]
        return [run_dir_x, run_dir_y, -1]

    @staticmethod
    def avoid_feed_to_enemy(obs):
        clone = np.array(obs['clone'])
        # 还原

        my_clone = clone[clone[:, -3] == 1].reshape(-1, clone.shape[1])
        enmey_clone = clone[clone[:, -1] == 1].reshape(-1, clone.shape[1])

        my_feed_clone = None
        enmey_big_clone = None
        for m_c in my_clone:
            m_x, m_y, m_r = m_c[:3]
            for e_c in enmey_clone:
                e_x, e_y, e_r = e_c[:3]
                if e_r <= m_r:
                    continue
                dis = np.sqrt(np.power(e_x - m_x, 2) + np.power(e_y - m_y, 2))
                if dis <= 4 * m_c[2]:
                    if (enmey_big_clone is None) or (enmey_big_clone[2] < e_r):
                        my_feed_clone = m_c
                        enmey_big_clone = e_c
        return my_feed_clone, enmey_big_clone

    @staticmethod
    def generate_mask(args, action_type_n, player_num_per_team=1):
        mask_list = []
        for arg in args:
            batch = arg['batch']
            mask = torch.zeros(batch, player_num_per_team, action_type_n)

            thorn_relation = arg['thorn_relation']
            # clone_relation = arg['clone_relation']
            my_clones = arg['my_clones']
            atk_target_clones = arg['atk_target_clones']
            feed_target_clones = arg['feed_target_clones']
            my_clones = [c.cpu().numpy() if isinstance(c, torch.Tensor) else c for c in my_clones]
            atk_target_clones = [a.cpu().numpy() if isinstance(a, torch.Tensor) else a for a in atk_target_clones]
            feed_target_clones = [r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in feed_target_clones]
            if thorn_relation.ndim == 3:
                thorn_relation = thorn_relation.view(1, *thorn_relation.shape)

            # 默认开启移动 + 无操作
            # mask[:, 0, :5] = 1
            # 默认开启吃食物球
            mask[:, 0, 8] = 1
            # 默认无法移动进攻
            atk_move_pos = 10
            run_pos = 26

            for b in range(batch):
                n_my_clone = np.sum(my_clones[b][:, 2] != 0)
                # 开启吐孢子
                # if n_my_clone > 1 and my_clones[b][0, 2] >= 10:
                #     mask[b, 0, 5] = 1
                # 开启分裂
                # if n_my_clone < 16 and my_clones[b][0, 2] >= 10:
                #     mask[b, 0, 6] = 1
                # 开启静止
                # if n_my_clone > 8:
                #     mask[b, 0, 7] = 1
                # 开启吃荆棘球
                cnt_eat_thorn = int(torch.sum(thorn_relation[b, :, :, -2] == 1))
                if cnt_eat_thorn > 0:
                    mask[b, 0, 9] = 1

                # 判断能否atk_move,atk_split,run
                for idx in range(n_my_clone):
                    my_cl = my_clones[b][idx].reshape(1, -1)
                    if my_cl[0, 2] == 0:
                        continue

                    # atk_move
                    can_attack_move = (atk_target_clones[b][idx, 2] != 0)
                    # atk_split
                    atk_tar_cl = atk_target_clones[b][idx].reshape(1, -1)
                    atk_relation = wt_relation_encode(my_cl, atk_tar_cl, clone=True).flatten()
                    # enemy + split_eat + split_collide
                    can_split_by_relation = (atk_relation[-1] == 1) & (atk_relation[-4] == 1) & (atk_relation[-8] == 1)
                    # self
                    can_split_by_self = (idx < 16 - n_my_clone) and (my_cl[idx, 2] > 10)
                    # run
                    run_tar_cl = feed_target_clones[b][idx].reshape(1, -1)
                    can_run = (run_tar_cl[0, 2] != 0)

                    if can_attack_move:
                        mask[b, 0, atk_move_pos + idx] = 1
                    if can_run:
                        mask[b, 0, run_pos + idx] = 1

            mask_list.append(mask)
        return mask_list

    @staticmethod
    def generate_mask_team(args, action_type_n):
        mask_list = []
        for arg in args:
            batch = arg['batch']
            player_num_per_team = arg['player_num_per_team']
            n = int(batch * player_num_per_team)
            mask = torch.zeros(n, action_type_n)

            thorn_relation = arg['thorn_relation']
            # clone_relation = arg['clone_relation']
            my_clones = arg['my_clones']
            atk_enemy_clones = arg['atk_enemy_clones']
            my_clones = [c.cpu().numpy() if isinstance(c, torch.Tensor) else c for c in my_clones]
            atk_enemy_clones = [a.cpu().numpy() if isinstance(a, torch.Tensor) else a for a in atk_enemy_clones]
            if thorn_relation.ndim == 3:
                thorn_relation = thorn_relation.view(1, *thorn_relation.shape)

            # 默认无法移动进攻
            atk_move_pos = 11
            atk_split_pos = 27

            for i in range(n):
                my_clone_n = np.sum(my_clones[i][:, 2] != 0)
                my_clone_can_split_n = np.sum(my_clones[i][:, 2] >= 10)
                my_clone_max_r = my_clones[i][0, 2]

                # 开启吃食物球
                mask[i, 9] = 1
                # 开启移动
                mask[i, 0:8] = 1
                # 开启吃荆棘球
                cnt_eat_thorn = int(torch.sum(thorn_relation[i, :, :, -2] == 1))
                if cnt_eat_thorn > 0:
                    mask[i, 10] = 1

                if my_clone_n + my_clone_can_split_n <= 8 and my_clone_max_r >= 25:
                    # 开启分裂
                    mask[i, 8] = 1

                # 判断能否atk_move,atk_split,run
                for idx in range(my_clone_n):
                    my_cl = my_clones[i][idx].reshape(1, -1)
                    my_cl_r = my_cl[0, 2]
                    if my_cl_r == 0:
                        continue

                    can_attack_move = (atk_enemy_clones[i][idx, 2] != 0)
                    if not can_attack_move:
                        continue

                    # 开启atk_move
                    mask[i, atk_move_pos + idx] = 1

                    # 判断atk_split
                    if idx >= 8:
                        continue

                    atk_tar_cl = atk_enemy_clones[i][idx].reshape(1, -1)
                    atk_relation = wt_relation_encode(my_cl, atk_tar_cl, clone=True).flatten()
                    # enemy + split_eat + split_collide
                    can_split_by_relation = (atk_relation[-1] == 1) & (atk_relation[-4] == 1) & (atk_relation[-8] == 1)
                    # self
                    can_split_by_self = (idx < 16 - my_clone_n) and (my_cl_r >= 10)

                    if can_split_by_relation and can_split_by_self:
                        mask[i, atk_split_pos + idx] = 1

            mask_list.append(mask)
        return mask_list


def fake_eat_by_move(src_cl, tar_cl):
    src_cl_x, src_cl_y, src_cl_r = src_cl[0], src_cl[1], src_cl[2]
    tar_cl_x, tar_cl_y, tar_cl_r = tar_cl[0], tar_cl[1], tar_cl[2]
    if src_cl_r < tar_cl_r:
        return None
    dt_x = tar_cl_x - src_cl_x
    dt_y = tar_cl_y - src_cl_y
    dis = np.sqrt(dt_x * dt_x + dt_y * dt_y)
    move_dis = dis - src_cl_r
    aft_move_x = src_cl_x + move_dis * (dt_x / dis)
    aft_move_y = src_cl_y + move_dis * (dt_y / dis)
    aft_move_r = np.sqrt(src_cl_r * src_cl_r + tar_cl_r * tar_cl_r)
    return np.array([aft_move_x, aft_move_y, aft_move_r, 0, 0])


def fake_eat_by_split(src_cl, tar_cl, split_ratio=2.1213):
    src_cl_x, src_cl_y, src_cl_r = src_cl[0], src_cl[1], src_cl[2]
    tar_cl_x, tar_cl_y, tar_cl_r = tar_cl[0], tar_cl[1], tar_cl[2]
    if src_cl_r * np.sqrt(0.5) < tar_cl_r:
        return None
    dt_x = tar_cl_x - src_cl_x
    dt_y = tar_cl_y - src_cl_y
    dis = np.sqrt(dt_x * dt_x + dt_y * dt_y)
    move_dis = split_ratio * src_cl_r
    if move_dis < dis:
        return None

    aft_split_x = src_cl_x + move_dis * (dt_x / dis)
    aft_split_y = src_cl_y + move_dis * (dt_y / dis)
    aft_split_r = np.sqrt(src_cl_r * src_cl_r / 2 + tar_cl_r * tar_cl_r)
    return np.array([aft_split_x, aft_split_y, aft_split_r, 0, 0])


def find_relation_between_clones(cl1, cl2, top_n=1):
    cl1 = cl1.reshape(1, cl1.shape[-1])
    cl2 = cl2.reshape(-1, cl2.shape[-1])
    relation = wt_relation_encode(cl1, cl2, clone=True)[0]
    dim_n = relation.shape[-1]

    atk_relation_rec = np.zeros((top_n, dim_n))
    feed_relation_rec = np.zeros((top_n, dim_n))
    relation = relation[relation[:, 4] != 0]
    if relation.shape[0] > 0:
        atk_relation = relation[relation[:, -6] == 1]
        feed_relation = relation[relation[:, -5] == 1]
        if atk_relation.shape[0] > 0:
            atk_relation = atk_relation[np.argsort(-atk_relation[:, 4])][:top_n]
            n_atk = atk_relation.shape[0]
            atk_relation_rec[:n_atk] = atk_relation
        if feed_relation.shape[0] > 0:
            feed_relation = feed_relation[np.argsort(-feed_relation[:, 4])][:top_n]
            n_feed = feed_relation.shape[0]
            feed_relation_rec[:n_feed] = feed_relation
    return atk_relation_rec, feed_relation_rec


def wt_relation_encode(point_1, point_2, clone=True, split_ratio=2.12, follow_ratio=4):
    if len(point_1.shape) == 1:
        point_1 = point_1.reshape(-1, point_1.shape[0])
    if len(point_2.shape) == 1:
        point_2 = point_2.reshape(-1, point_2.shape[0])
    n1 = point_1.shape[0]
    n2 = point_2.shape[0]
    if (point_2[:, None, 2:3] == 0).any():
        useless = True
    else:
        useless = False

    r_1 = point_1[:, None, 2:3]
    r_2 = point_2[None, :, 2:3]
    # 相对位置
    dt_xy = (point_2[None, :, :2] - point_1[:, None, :2])  # relative position
    # 距离
    dt_dis = np.linalg.norm(dt_xy, ord=2, axis=2, keepdims=True)  # distance
    dt_dis[dt_dis < 3] = 3
    # 方向角
    ang_cos = dt_xy[:, :, 0].reshape(dt_dis.shape) / dt_dis
    ang_sin = dt_xy[:, :, 1].reshape(dt_dis.shape) / dt_dis

    # (被）碰撞系数
    r1_dis_ratio = r_1 / dt_dis  # whether source's split collides with target
    r2_dis_ratio = r_2 / dt_dis  # whether target's split collides with source
    # （被） follow * r >= dt_dis
    follow = np.zeros_like(r1_dis_ratio)
    be_follow = np.zeros_like(r2_dis_ratio)
    follow[follow_ratio * r1_dis_ratio >= 1] = 1  # follow[follow_ratio * r_1 >= dt_dis] = 1
    be_follow[follow_ratio * r2_dis_ratio >= 1] = 1  # be_follow[follow_ratio * r_2 >= dt_dis] = 1

    # 分裂后是否（被）碰撞
    split_collide = np.zeros_like(r1_dis_ratio)  # whether source's split collides with target
    split_collide[r_1 * split_ratio >= dt_dis] = 1
    be_split_collide = np.zeros_like(r2_dis_ratio)  # whether target's split collides with source
    be_split_collide[r_2 * split_ratio + 15 >= dt_dis] = 1

    # 半径占比
    rds_weight = r_1 / (r_1 + r_2)

    # 是否能吃
    eat = np.zeros(rds_weight.shape)
    eat[rds_weight > 0.5] = 1
    # 是否能被吃
    be_ate = np.zeros(rds_weight.shape)
    be_ate[rds_weight < 0.5] = 1
    # 分裂后是否能吃
    split_eat = np.zeros(rds_weight.shape)
    split_eat[(r_1 > 10) & (np.sqrt(0.5) * r_1 > r_2)] = 1
    # 分裂后是否能被吃
    be_split_ate = np.zeros(rds_weight.shape)
    be_split_ate[(r_2 > 10) & (r_1 < np.sqrt(0.5) * r_2)] = 1

    if clone:
        # 判断归属关系
        t_1 = point_1[:, None, -2]
        p_1 = point_1[:, None, -1]
        t_2 = point_2[None, :, -2]
        p_2 = point_2[None, :, -1]

        same_owner = 1 - (4 * (t_1 - t_2) + (p_2 - p_1))
        same_owner[same_owner != 1] = 0
        same_owner = same_owner.reshape(n1, n2, 1)

        is_enemy = t_1 - t_2
        is_enemy = is_enemy.reshape(n1, n2, 1)
        is_enemy[is_enemy != 0] = 1

        r = [ang_cos, ang_sin, rds_weight, r1_dis_ratio, r2_dis_ratio, follow, be_follow, split_collide,
             be_split_collide, eat, be_ate, split_eat, be_split_ate, same_owner, is_enemy]
    else:
        r = [ang_cos, ang_sin, rds_weight, r1_dis_ratio, follow, split_collide, eat, split_eat]
    relation = np.concatenate(r, axis=2)

    relation[relation[:, :, 0] == 0] = 0
    if useless:
        relation[:, :, :] = 0
    return relation


def get_spd(r):
    return 500. / (10. + r)


def generate_custom_init(border_x=1000, border_y=1000, player_n=12, team_n=4, init_n=6, r_min=3, r_max=15):
    custom_init = {}
    custom_init['food'] = []
    custom_init['thorns'] = []
    custom_init['spore'] = []

    player_each_team = int(player_n / team_n)
    custom_clones = []
    for i in range(player_n):
        center_x = random.randint(int(border_x * 0.1), int(border_x * 0.9))
        center_y = random.randint(int(border_y * 0.1), int(border_y * 0.9))

        for j in range(init_n):
            c = [0] * 23
            offset_x = int((random.random() - 0.5) * border_x * 0.1)
            offset_y = int((random.random() - 0.5) * border_y * 0.1)
            c[0] = center_x + offset_x
            c[1] = center_y + offset_y
            c[2] = random.randint(r_min, r_max)
            c[3] = str(i)
            c[4] = str(i // player_each_team)
            c[17] = 20
            custom_clones.append(c)
    custom_init['clone'] = custom_clones
    return custom_init


def y_to_x(k, b, y):
    x = (y - b) / k
    return x


def x_to_y(k, b, x):
    y = k * x + b
    return y


def rename_videos(save_floder, seed):
    assert os.path.isdir(save_floder), "save_floder %s 不是文件夹" % save_floder
    seed = str(seed)
    video_names = os.listdir(save_floder)
    for name in video_names:
        if not name.endswith('.mp4'):
            continue
        if 'seed' not in name:
            tail = name.split("-")[-1]
            name_head = 'seed' + '-' + seed
            tar_floder = os.path.join(save_floder, name_head)
            os.makedirs(tar_floder, exist_ok=True)

            tar_name = name_head + '-' + tail

            src_path = os.path.join(save_floder, name)
            tar_path = os.path.join(tar_floder, tar_name)

            if os.path.exists(tar_path):
                print('seed:' + seed + "已存在")
                os.remove(src_path)
            else:
                os.rename(src_path, tar_path)


def team_chase_dir(tar_cl, friend_clones):
    tar_cl_r = tar_cl['radius']
    tar_cl_v = tar_cl_r * tar_cl_r
    tar_cl_pos = tar_cl['position']

    chase_cl = None
    t_min = 10000
    for b in friend_clones:
        b_r = b['radius']
        b_v = b_r * b_r
        b_pos = b['position']
        # too far / smaller
        if b_v <= tar_cl_v or (b_pos - tar_cl_pos).length() >= 3. * b_r:
            continue

        if b_r > 10 and b_v / 2 < tar_cl_v:
            dis = (b_pos - tar_cl_pos).length() - b_r
        else:
            dis = (b_pos - tar_cl_pos).length() - 2.12 * b_r
        if dis <= 0.0:
            continue
        b_spd = 500. / (10 + b_r)
        t = dis / b_spd
        if t < t_min:
            t_min = t
            chase_cl = b
    if (chase_cl is None) or (tar_cl_pos == chase_cl['position']):
        return None

    chase_direct = (tar_cl_pos - chase_cl['position']).normalize()
    return chase_direct


def get_split_eat_enemy_volumn(my_cl, enemy_cl, player_clone_np, my_clones, my_merging_clones, friend_clones,
                               enemy_clones, thorns, my_density=0, split_eat_thorns=False):
    n_my_clones = len(my_clones)
    my_cl_pos = my_cl['position']
    my_cl_r = my_cl['radius']
    my_cl_v = my_cl_r * my_cl_r

    n_can_split = 0
    n_can_split_twice = 0
    my_cl_idx = 0
    my_total_v = 0.0
    for idx, b in enumerate(my_clones):
        if b['radius'] > 10.:
            n_can_split += 1
        elif b['radius'] > 20.:
            n_can_split_twice += 1
        if b['position'] == my_cl_pos:
            my_cl_idx = idx
        my_total_v += b['radius'] * b['radius']

    n_aft_split = min(16, n_my_clones + n_can_split)
    n_aft_two_split = min(16, n_my_clones + n_can_split_twice)
    my_cl_can_split_once = my_cl_r > 10. and n_my_clones + my_cl_idx < 16
    my_cl_can_split_twice = my_cl_r > 20. and my_cl_idx == 0 and n_aft_split + my_cl_idx < 16

    if not my_cl_can_split_once:
        return 0.0

    split_score = 0.0
    enemy_cl_pos = enemy_cl['position']
    enemy_cl_r = enemy_cl['radius']
    enemy_cl_v = enemy_cl_r * enemy_cl_r

    enemy_cl_name = int(enemy_cl['player'])
    enemy_player_clone = player_clone_np[enemy_cl_name]
    n_enemy_cl = enemy_player_clone.shape[0]
    enemy_idx = 0

    for j, temp_cl in enumerate(enemy_player_clone):
        if int(enemy_cl_pos[0]) == int(temp_cl[0]) and int(enemy_cl_pos[1]) == int(temp_cl[1]):
            enemy_idx = j
            break

    enemy_can_split_sum = np.sum(enemy_player_clone[:, 2] > 10.)
    enemy_can_split_twice = enemy_idx == 0 and n_enemy_cl + enemy_can_split_sum - 1 < 16 and enemy_cl_r > 20.
    my_to_enemy_dis = (enemy_cl_pos - my_cl_pos).length()
    if my_to_enemy_dis == 0.0:
        direction = Vector2(0.1, 0.1)
    else:
        direction = (enemy_cl_pos - my_cl_pos).normalize()

    # collide bug
    if my_to_enemy_dis < my_cl_r:
        return 0.0

    # can split
    split_danger = False
    split_temp_v = 0.0
    split_thorn_v = 0.0
    # split once and kill target
    if my_to_enemy_dis < 3.2 * my_cl_r:
        # eat other ball by the way
        fake_ball_v = my_cl_v / 2
        fake_ball_r = math.sqrt(fake_ball_v)
        fake_ball_x = my_cl_pos.x + direction.x * 2 * fake_ball_r
        fake_ball_y = my_cl_pos.y + direction.y * 2 * fake_ball_r

        fake_ball_x = np.clip(fake_ball_x, fake_ball_r, 1000. - fake_ball_r)
        fake_ball_y = np.clip(fake_ball_y, fake_ball_r, 1000. - fake_ball_r)

        # danger of collide
        if enemy_clones and (not split_danger):
            for b in enemy_clones:
                fake_dis = math.sqrt(math.pow(fake_ball_x - b['position'].x, 2) + math.pow(
                    fake_ball_y - b['position'].y, 2))
                b_r = b['radius']
                # collide and be eat
                if fake_dis < b_r and fake_ball_r < b_r:
                    # eat by other
                    split_danger = True
                    break

        # safe collide
        if not split_danger:
            # eat friend
            if friend_clones:
                for b in friend_clones:
                    fake_dis = math.sqrt(math.pow(fake_ball_x - b['position'].x, 2) + math.pow(
                        fake_ball_y - b['position'].y, 2))
                    b_r = b['radius']
                    b_v = b_r * b_r
                    # collide and be eat
                    if fake_dis < fake_ball_r and fake_ball_r > b_r:
                        fake_ball_v += b_v

            # eat enemy
            if enemy_clones:
                for b in enemy_clones:
                    fake_dis = math.sqrt(math.pow(fake_ball_x - b['position'].x, 2) + math.pow(
                        fake_ball_y - b['position'].y, 2))
                    b_r = b['radius']
                    b_v = b_r * b_r
                    # collide and eat enemy
                    if fake_ball_r > b_r and fake_dis < fake_ball_r:
                        fake_ball_v += b_v
                        split_temp_v += b_v

            # eat thorns
            if thorns:
                temp_n = copy.deepcopy(n_aft_split)
                for th in thorns:
                    th_r = th['radius']
                    th_v = th_r * th_r
                    if th_v > fake_ball_v:
                        continue

                    to_th_dis = math.sqrt(math.pow(fake_ball_x - th['position'].x, 2) + math.pow(
                        fake_ball_y - th['position'].y, 2))

                    if to_th_dis < math.sqrt(fake_ball_v):
                        split_n = min(16 - temp_n, 10)
                        if split_n > 0:
                            merge_v = fake_ball_v + th_r * th_r
                            split_r = min(math.sqrt(merge_v / (split_n + 1)), 20)
                            split_v = split_r * split_r
                            middle_v = merge_v - split_v * split_n
                            fake_ball_v = middle_v
                            temp_n = min(16, temp_n + split_n)
                        else:
                            fake_ball_v += th_v
                            split_thorn_v += th_v / 2

            # eat my merging cl
            if my_merging_clones and (not split_danger):
                for b in my_merging_clones:
                    b_r = b['radius']
                    b_v = b_r * b_r
                    to_my_merge_dis = math.sqrt(math.pow(fake_ball_x - b['position'].x, 2) + math.pow(
                        fake_ball_y - b['position'].y, 2))
                    # collide and be eat
                    if to_my_merge_dis > fake_ball_r:
                        continue
                    if fake_ball_v / 2 < b_v:
                        fake_ball_x = b['position'].x
                        fake_ball_y = b['position'].y
                    fake_ball_v += b_v

        # aft split be eat
        if enemy_clones and (not split_danger):
            # new ball
            for b in enemy_clones:
                fake_dis = math.sqrt(math.pow(fake_ball_x - b['position'].x, 2) + math.pow(
                    fake_ball_y - b['position'].y, 2))
                b_r = b['radius']
                b_v = b_r * b_r
                # collide
                if fake_ball_v < b_v and fake_dis < b_r + 15:
                    split_danger = True
                    break
                # enemy split once
                elif fake_ball_v < b_v / 2 and fake_dis < 2.2 * b_r + 15 and b_r > 10.:
                    split_danger = True
                    break
                # enemy split twice
                elif fake_ball_v < b_v / 4 and fake_dis < 3.0 * b_r + 18 and b_r > 20. and enemy_can_split_twice:
                    split_danger = True
                    break

            # raw ball
            for b in enemy_clones:
                fake_dis = (my_cl_pos - b['position']).length()
                b_r = b['radius']
                b_v = b_r * b_r
                if my_cl_v / 2 < b_v / 2 and fake_dis < 2.2 * b_r + 15 and b_r > 10.:
                    split_danger = True
                    break

        if split_danger:
            return -my_cl_v

        split_eat_volume = split_temp_v
        my_biggest_r = my_clones[0]['radius']
        my_biggest_v = my_biggest_r * my_biggest_r

        if split_eat_volume > 0.:
            split_eat_volume += split_thorn_v
        if split_eat_volume >= my_cl_v / 12:
            return split_eat_volume

        # split twice
        if split_eat_volume >= 0.0 and my_cl_can_split_twice:
            # eat other ball by the way
            fake_ball_v_1 = fake_ball_v
            fake_ball_r_1 = math.sqrt(fake_ball_v_1)
            fake_ball_x_1 = fake_ball_x
            fake_ball_y_1 = fake_ball_y

            fake_dis = math.sqrt(
                math.pow(fake_ball_x_1 - enemy_cl_pos.x, 2) + math.pow(
                    fake_ball_y_1 - enemy_cl_pos.y, 2))

            second_split_collide = fake_dis < 2.2 * fake_ball_r_1
            if second_split_collide:
                fake_ball_v_2 = fake_ball_v_1 / 2
                fake_ball_r_2 = math.sqrt(fake_ball_v_2)
                fake_ball_x_2 = fake_ball_x_1 + direction.x * 2 * fake_ball_r_2
                fake_ball_y_2 = fake_ball_y_1 + direction.y * 2 * fake_ball_r_2

                fake_ball_x_2 = np.clip(fake_ball_x_2, fake_ball_r_2, 1000. - fake_ball_r_2)
                fake_ball_x_2 = np.clip(fake_ball_x_2, fake_ball_r_2, 1000. - fake_ball_r_2)
                for enemy_b in enemy_clones:
                    fake_dis = math.sqrt(
                        math.pow(fake_ball_x_2 - enemy_b['position'].x, 2) + math.pow(
                            fake_ball_y_2 - enemy_b['position'].y, 2))
                    enemy_b_r = enemy_b['radius']
                    enemy_b_v = enemy_b_r * enemy_b_r
                    # collide bigger
                    if enemy_b_r > fake_ball_r_2 and fake_dis < enemy_b_r:
                        split_danger = True
                        break
                    # collide eat smaller
                    elif (fake_ball_r_2 > fake_dis) and (fake_ball_r_2 > enemy_b_r):
                        fake_ball_v_2 += enemy_b_v
                        split_eat_volume += enemy_b_v

                if not split_danger:
                    for enemy_b in enemy_clones:
                        fake_dis = math.sqrt(
                            math.pow(fake_ball_x_2 - enemy_b['position'].x, 2) + math.pow(
                                fake_ball_y_2 - enemy_b['position'].y, 2))
                        enemy_b_r = enemy_b['radius']
                        enemy_b_v = enemy_b_r * enemy_b_r
                        if enemy_b_v / 2 > fake_ball_v_2 and fake_dis < 2.2 * enemy_b_r + 15:
                            split_danger = True
                            break
                        elif enemy_b_v / 4 > fake_ball_v_2 and fake_dis < 3.0 * enemy_b_r + 18 and enemy_b_r > 20. and enemy_can_split_twice:
                            split_danger = True
                            break

                if not split_danger:
                    if thorns:
                        temp_n = copy.deepcopy(n_aft_two_split)
                        for th in thorns:
                            th_r = th['radius']
                            th_v = th_r * th_r
                            if th_v > fake_ball_v:
                                continue

                            to_th_dis = math.sqrt(math.pow(fake_ball_x - th['position'].x, 2) + math.pow(
                                fake_ball_y - th['position'].y, 2))

                            if to_th_dis < math.sqrt(fake_ball_v):
                                split_n = min(16 - temp_n, 10)
                                if split_n > 0:
                                    merge_v = fake_ball_v + th_r * th_r
                                    split_r = min(math.sqrt(merge_v / (split_n + 1)), 20)
                                    split_v = split_r * split_r
                                    middle_v = merge_v - split_v * split_n
                                    fake_ball_v = middle_v
                                    temp_n = min(16, temp_n + split_n)
                                else:
                                    fake_ball_v += th_v
                                    split_thorn_v += th_v / 4

        if split_danger:
            split_eat_volume = -my_cl_v
            return split_eat_volume

        # ignore small volumn
        if split_eat_volume < my_cl_v / 6:
            split_eat_volume = 0.0
            # split for farm
            if split_eat_thorns:
                if split_thorn_v > 0 and my_cl_idx == 0 and n_my_clones == 15 and my_biggest_v > 10000 and my_density > 0.5:
                    split_eat_volume += split_thorn_v
        return split_eat_volume
    else:
        return 0.0


def check_split_eat_by_enemy(my_cl, enemy_cl, player_clone_np, my_clones, my_merging_clones, friend_clones,
                             enemy_clones, thorns, split_num=1):
    '''

    :param my_cl:
    :param enemy_cl:
    :param player_clone_np:
    :param my_clones:
    :param my_merging_clones:
    :param friend_clones:
    :param enemy_clones:
    :param thorns:
    :return:
    True:对我的cl有威胁
    False：对我的cl没威胁
    '''
    n_my_clones = len(my_clones)
    my_cl_pos = my_cl['position']
    my_cl_r = my_cl['radius']
    my_cl_v = my_cl_r * my_cl_r
    team = int(my_clones[0]['team'])
    my_player = int(my_clones[0]['player'])

    n_can_split = 0
    n_can_split_twice = 0
    my_cl_idx = 0
    my_total_v = 0.0
    for idx, b in enumerate(my_clones):
        if b['radius'] > 10.:
            n_can_split += 1
        elif b['radius'] > 20.:
            n_can_split_twice += 1
        if b['position'] == my_cl_pos:
            my_cl_idx = idx
        my_total_v += b['radius'] * b['radius']

    n_aft_split = min(16, n_my_clones + n_can_split)
    n_aft_two_split = min(16, n_my_clones + n_can_split_twice)
    my_cl_can_split_once = my_cl_r > 10. and n_my_clones + my_cl_idx - 1 < 16
    my_cl_can_split_twice = my_cl_r > 20. and my_cl_idx == 0 and n_aft_split + my_cl_idx - 1 < 16

    enemy_cl_pos = enemy_cl['position']
    enemy_cl_r = enemy_cl['radius']
    enemy_cl_v = enemy_cl_r * enemy_cl_r

    enemy_cl_name = int(enemy_cl['player'])
    enemy_cl_team = int(enemy_cl['team'])
    enemy_player_clone = player_clone_np[enemy_cl_name]
    n_enemy_cl = enemy_player_clone.shape[0]
    enemy_idx = 0

    for j, temp_cl in enumerate(enemy_player_clone):
        if int(enemy_cl_pos[0]) == int(temp_cl[0]) and int(enemy_cl_pos[1]) == int(temp_cl[1]):
            enemy_idx = j
            break

    n_aft_split = min(16, n_my_clones + n_can_split)

    my_to_enemy_dis = (my_cl_pos - enemy_cl_pos).length()
    if my_to_enemy_dis == 0.0:
        direction = Vector2(0.1, 0.1)
    else:
        direction = (my_cl_pos - enemy_cl_pos).normalize()

    # collide bug
    if my_to_enemy_dis < my_cl_r:
        return True

    # can split
    split_danger = False

    # eat other ball by the way
    fake_ball_v_1 = enemy_cl_v / 2
    fake_ball_r_1 = math.sqrt(fake_ball_v_1)
    fake_ball_x_1 = enemy_cl_pos.x + direction.x * 2 * fake_ball_r_1
    fake_ball_y_1 = enemy_cl_pos.y + direction.y * 2 * fake_ball_r_1

    fake_ball_x_1 = np.clip(fake_ball_x_1, fake_ball_r_1, 1000. - fake_ball_r_1)
    fake_ball_y_1 = np.clip(fake_ball_y_1, fake_ball_r_1, 1000. - fake_ball_r_1)

    # danger of collide my cl
    if my_clones:
        for b in my_clones:
            fake_dis = math.sqrt(math.pow(fake_ball_x_1 - b['position'].x, 2) + math.pow(
                fake_ball_y_1 - b['position'].y, 2))
            b_r = b['radius']
            # collide and be eat
            if fake_dis < b_r and fake_ball_r_1 < b_r:
                # eat by other
                split_danger = True
                break

    # danger of collide friend cl
    if friend_clones and (not split_danger):
        for b in friend_clones:
            fake_dis = math.sqrt(math.pow(fake_ball_x_1 - b['position'].x, 2) + math.pow(
                fake_ball_y_1 - b['position'].y, 2))
            b_r = b['radius']
            # collide and be eat
            if fake_dis < b_r and fake_ball_r_1 < b_r:
                # eat by other
                split_danger = True
                break

    if split_danger:
        return False

    # safe collide
    if not split_danger:
        # eat friend
        if friend_clones:
            for b in friend_clones:
                fake_dis = math.sqrt(math.pow(fake_ball_x_1 - b['position'].x, 2) + math.pow(
                    fake_ball_y_1 - b['position'].y, 2))
                b_r = b['radius']
                b_v = b_r * b_r
                # collide and be eat
                fake_ball_r_1 = math.sqrt(fake_ball_v_1)
                if fake_dis < fake_ball_r_1 and fake_ball_r_1 > b_r:
                    fake_ball_v_1 += b_v

        # eat my cl
        if my_clones:
            for b in my_clones:
                fake_dis = math.sqrt(math.pow(fake_ball_x_1 - b['position'].x, 2) + math.pow(
                    fake_ball_y_1 - b['position'].y, 2))
                b_r = b['radius']
                b_v = b_r * b_r
                # collide and eat enemy
                fake_ball_r_1 = math.sqrt(fake_ball_v_1)
                if fake_ball_r_1 > b_r and fake_dis < fake_ball_r_1:
                    fake_ball_v_1 += b_v

        # eat thorns
        if thorns:
            temp_n = copy.deepcopy(n_aft_split)
            for th in thorns:
                th_r = th['radius']
                th_v = th_r * th_r
                if th_v > fake_ball_v_1:
                    continue

                to_th_dis = math.sqrt(math.pow(fake_ball_x_1 - th['position'].x, 2) + math.pow(
                    fake_ball_y_1 - th['position'].y, 2))

                if to_th_dis < math.sqrt(fake_ball_v_1):
                    split_n = min(16 - temp_n, 10)
                    if split_n > 0:
                        merge_v = fake_ball_v_1 + th_r * th_r
                        split_r = min(math.sqrt(merge_v / (split_n + 1)), 20)
                        split_v = split_r * split_r
                        middle_v = merge_v - split_v * split_n
                        fake_ball_v_1 = middle_v
                        temp_n = min(16, temp_n + split_n)
                    else:
                        fake_ball_v_1 += th_v

    if split_num == 1:
        fake_enemy = dict()
        fake_enemy['position'] = Vector2(float(fake_ball_x_1), float(fake_ball_y_1))
        fake_enemy['radius'] = fake_ball_r_1
        fake_enemy['player'] = enemy_cl_name
        fake_enemy['team'] = enemy_cl_team

        # aft split be eat
        team = int(my_clones[0]['team'])
        my_player = int(my_clones[0]['player'])

        enemy_clones_add_fake = copy.deepcopy(enemy_clones)
        enemy_clones_add_fake.append(fake_enemy)
        enemy_clones_add_fake.remove(enemy_cl)
        for player_id in range(3 * team, 3 * team + 3):
            player_clones = group_process_np_to_dict(player_clone_np[player_id])

            if not player_clones:
                continue
            player_n = len(player_clones)

            if player_id != my_player:
                can_split = player_n < 16
            else:
                can_split = True

            b = player_clones[0]
            fake_dis = (fake_enemy['position'] - b['position']).length()
            b['radius'] = b['radius'] / 1.414
            b_r = b['radius']
            b_v = b_r * b_r
            # collide

            if fake_ball_v_1 < b_v and fake_dis < b_r:
                split_danger = True
                break

            # split once
            elif fake_ball_v_1 < b_v / 3 and fake_dis < 2.1 * b_r + 15 and can_split:
                eat_v = get_split_eat_enemy_volumn(b, fake_enemy, player_clone_np, player_clones, None, None,
                                                   enemy_clones_add_fake, thorns)
                if eat_v > 0:
                    split_danger = True
                    break

        if split_danger:
            return False
        else:
            return True

    # split twice
    fake_ball_v_2 = fake_ball_v_1 / 2
    fake_ball_r_2 = math.sqrt(fake_ball_v_2)
    fake_ball_x_2 = fake_ball_x_1 + direction.x * 2 * fake_ball_r_2
    fake_ball_y_2 = fake_ball_y_1 + direction.y * 2 * fake_ball_r_2

    fake_ball_x_2 = np.clip(fake_ball_x_2, fake_ball_r_2, 1000. - fake_ball_r_2)
    fake_ball_y_2 = np.clip(fake_ball_y_2, fake_ball_r_2, 1000. - fake_ball_r_2)

    fake_ball_2 = dict()
    fake_ball_2['position'] = Vector2(float(fake_ball_x_2), float(fake_ball_y_2))
    fake_ball_2['player'] = enemy_cl_name
    fake_ball_2['team'] = enemy_cl_team

    for b in my_clones:
        fake_dis = (b['position'] - fake_ball_2['position']).length()
        b_r = b['radius']
        b_v = b_r * b_r
        # collide bigger
        if b_r > fake_ball_r_2 and fake_dis < b_r:
            split_danger = True
            break
        # collide eat smaller
        elif (fake_ball_r_2 > fake_dis) and (fake_ball_r_2 > b_r):
            fake_ball_v_2 += b_v

    fake_ball_r_2 = math.sqrt(fake_ball_v_2)
    for b in friend_clones:
        fake_dis = (b['position'] - fake_ball_2['position']).length()
        b_r = b['radius']
        b_v = b_r * b_r
        # collide bigger
        if b_r > fake_ball_r_2 and fake_dis < b_r:
            split_danger = True
            break
        # collide eat smaller
        elif (fake_ball_r_2 > fake_dis) and (fake_ball_r_2 > b_r):
            fake_ball_v_2 += b_v
    fake_ball_2['radius'] = fake_ball_r_2
    if split_danger:
        return False

    if split_num == 2:
        enemy_clones_add_fake = copy.deepcopy(enemy_clones)
        enemy_clones_add_fake.append(fake_ball_2)
        enemy_clones_add_fake.remove(enemy_cl)
        for player_id in range(3 * team, 3 * team + 3):
            player_clones = group_process_np_to_dict(player_clone_np[player_id])

            if not player_clones:
                continue
            player_n = len(player_clones)

            if player_id != my_player:
                can_split = player_n < 16
            else:
                can_split = True

            b = player_clones[0]
            fake_dis = (b['position'] - fake_ball_2['position']).length()
            b['radius'] = b['radius'] / 1.414
            b_r = b['radius']
            b_v = b_r * b_r

            # collide
            if fake_ball_v_1 < b_v / 2 and fake_dis < b_r:
                split_danger = True
                break
            # split once
            elif fake_ball_v_1 < b_v / 3 and fake_dis < 2.1 * b_r + 15 and can_split:
                eat_v = get_split_eat_enemy_volumn(b, fake_ball_2, player_clone_np, player_clones, None, None,
                                                   enemy_clones_add_fake,
                                                   thorns, 1, False)
                if eat_v > 0:
                    split_danger = True
                    break

        if split_danger:
            return False
        return True

    # split three
    fake_ball_v_3 = fake_ball_v_2 / 2
    fake_ball_r_3 = math.sqrt(fake_ball_v_3)
    fake_ball_x_3 = fake_ball_x_2 + direction.x * 2 * fake_ball_r_3
    fake_ball_y_3 = fake_ball_y_2 + direction.y * 2 * fake_ball_r_3

    fake_ball_x_3 = np.clip(fake_ball_x_3, fake_ball_r_3, 1000. - fake_ball_r_3)
    fake_ball_y_3 = np.clip(fake_ball_y_3, fake_ball_r_3, 1000. - fake_ball_r_3)

    fake_ball_3 = dict()
    fake_ball_3['position'] = Vector2(float(fake_ball_x_3), float(fake_ball_y_3))
    fake_ball_3['player'] = enemy_cl_name
    fake_ball_3['team'] = enemy_cl_team

    for b in my_clones:
        fake_dis = (b['position'] - fake_ball_3['position']).length()
        b_r = b['radius']
        b_v = b_r * b_r
        # collide bigger
        if b_r > fake_ball_r_3 and fake_dis < b_r:
            split_danger = True
            break
        # collide eat smaller
        elif (fake_ball_r_3 > fake_dis) and (fake_ball_r_3 > b_r):
            fake_ball_v_3 += b_v
    if split_danger:
        return False

    fake_ball_r_3 = math.sqrt(fake_ball_v_3)
    for b in friend_clones:
        fake_dis = (b['position'] - fake_ball_3['position']).length()
        b_r = b['radius']
        b_v = b_r * b_r
        # collide bigger
        if b_r > fake_ball_r_3 and fake_dis < b_r:
            split_danger = True
            break
        # collide eat smaller
        elif (fake_ball_r_3 > fake_dis) and (fake_ball_r_3 > b_r):
            fake_ball_v_3 += b_v
    fake_ball_3['radius'] = fake_ball_r_3

    enemy_clones_add_fake = copy.deepcopy(enemy_clones)
    enemy_clones_add_fake.append(fake_ball_3)
    enemy_clones_add_fake.remove(enemy_cl)
    for player_id in range(3 * team, 3 * team + 3):
        player_clones = group_process_np_to_dict(player_clone_np[player_id])

        if not player_clones:
            continue
        player_n = len(player_clones)

        if player_id != my_player:
            can_split = player_n < 16
        else:
            can_split = True

        b = player_clones[0]
        fake_dis = (b['position'] - fake_ball_3['position']).length()
        b['radius'] = b['radius'] / 1.414
        b_r = b['radius']
        b_v = b_r * b_r

        # collide
        if fake_ball_v_3 < b_v / 2 and fake_dis < b_r:
            split_danger = True
            break
        # split once
        elif fake_ball_v_3 < b_v / 3 and fake_dis < 2.1 * b_r + 15 and can_split:
            eat_v = get_split_eat_enemy_volumn(b, fake_ball_3, player_clone_np, player_clones, None, None,
                                               enemy_clones_add_fake,
                                               thorns, 1, False)
            if eat_v > 0:
                split_danger = True
                break
    if split_danger:
        return False
    return True

def get_center(clones):
    center = Vector2(0., 0.)
    size = 0.0
    for cl in clones:
        pos = cl['position']
        r = cl['radius']
        v = r * r
        size += v
        center += pos * v
    center /= size
    return center


def get_center_np(clones_np):
    center = Vector2(0., 0.)
    clones_v = clones_np[:, 2] * clones_np[:, 2]
    sum_v = np.sum(clones_v)
    sum_x = np.sum(clones_np[:, 0] * clones_v)
    sum_y = np.sum(clones_np[:, 1] * clones_v)
    center.x = float(sum_x / sum_v)
    center.y = float(sum_y / sum_v)
    return center


def get_avg_dis(clones):
    center = get_center(clones)
    for cl in clones:
        pass
