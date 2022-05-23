import math
import os
import random
import logging
import copy
import queue
from pygame.math import Vector2
import time
import numpy as np
import sys

parent = os.path.dirname(os.path.realpath(__file__))
sup = os.path.join(parent, 'supplements')
sys.path.append(sup)

import util

class BaseAgent:
    '''
    Overview:
        The base class of all agents
    '''
    def __init__(self):
        pass

    def step(self, obs):
        raise NotImplementedError

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
        team_id = int(team_name)
        bot_agent = BotAgentV6SPlus2
        self.policy = util.TeamBotPolicyV2(team_id=team_id, player_num_per_team=3, botagent=bot_agent,player_names=player_names)

    def get_actions(self, obs):
        actions = self.policy.forward(obs)[0]['action']
        actions_dict = {n: a.tolist() for n, a in zip(obs[1].keys(), actions)}
        return actions_dict

class BotAgentV6SPlus2(BaseAgent):
    """
    基于6splus改
    （1）防三重分裂
    （2）weak merge
    （3）fix merge_volumn
    """

    def __init__(self, name=None, team=None, level=3):
        self.name = name
        self.team = team
        self.actions_queue = queue.Queue()
        self.level = level

        self.average_delta_volume = 0.0
        self.last_my_volume = 9.0
        self.timer = time.time()
        self.start_time = time.time()
        self.total_action_time = 0.0
        self.second_count = 0

        self.last_direction = None
        self.last_my_clone = None
        self.last_action_type = -1
        self.last_spd = [0., 0.]
        self.merge_cd = 0.0
        self.run_init_num = 1.0
        self.run_cd = self.run_init_num
        self.chase_analyzer = util.ChaseAnalyzer()
        self.player_v = dict()
        self.player_n = dict()
        self.last_player_clone = dict()
        self.player_clone = dict()
        self.player_clone_np = dict()
        self.weak = False
        self.boss = False
        self.cur_time = 0
        self.merge_count = 0
        self.avg_farm_spd = 9.0
        self.player_center = None
        # self.last_target_ball = None
        # self.last_my_ball = None

    def step(self, obs, team_danger_balls):
        act, help_ball = self.bot_brain(obs, team_danger_balls)
        return act, help_ball

    def bot_brain(self, obs, team_danger_balls):
        self.cur_time = int(obs['last_time'])
        # self.test_bug(158,1)

        overlap = obs['overlap']
        overlap = self.preprocess(overlap)
        food_balls = overlap['food'] + overlap['spore']
        thorns_balls = overlap['thorns']
        # spore_balls = overlap['spore']
        clone_balls = overlap['clone']
        rect = obs['rectangle']

        self.player_merge_cd = obs.get('player_merge_cd', None)
        my_clone_balls, others_clone_balls, team_clone_balls = self.process_clone_balls(clone_balls)
        # collect clone ball info
        my_center, my_distribution, my_density, my_size = self.process_statistic(my_clone_balls)
        self.process_clone_statistic(clone_balls)

        self.weak = self.is_weak()
        self.boss = self.is_boss()

        # spd
        self.evaluate_spd(self.last_my_clone, my_clone_balls)
        n_my_clone = len(my_clone_balls) if my_clone_balls is not None else 1
        n_my_clone_last = len(self.last_my_clone) if self.last_my_clone is not None else 1
        my_clones_np = util.group_process_to_np(my_clone_balls)
        enemy_clones_np = util.group_process_to_np(others_clone_balls)
        total_clones_np = util.group_process_to_np(clone_balls)
        thorns_np = util.group_process_to_np(thorns_balls)

        n_my_clone = len(my_clone_balls)
        n_aft_split = n_my_clone
        for b in my_clone_balls:
            if b['radius'] > 10.:
                n_aft_split += 1
        n_aft_split = min(16, n_aft_split)

        # collect my statistic
        lead_my_ball, lead_direction, lead_action, lead_score, help_ball \
            = self. \
            estimate_value_balls_friend_verison(my_clone_balls, others_clone_balls, my_center,
                                                my_distribution, my_density, my_size, team_danger_balls,
                                                team_clone_balls, thorns_balls)

        # give the best direction and action for enemy/teammate, and give my sos info
        # thorns_my_ball, thorns_direction, thorns_action, thorns_score = self.process_thorns_balls(thorns_balls, my_clone_balls)

        thorns_my_ball, thorns_direction, thorns_action, thorns_score = self.process_thorns_balls_pro(total_clones_np,
                                                                                                      thorns_np)
        if thorns_score == 0 or thorns_score is None and thorns_balls and self.cur_time < 90:
            thorns_my_ball, thorns_direction, thorns_action, thorns_score = \
                self.process_gather_thorns(my_clone_balls, team_clone_balls, thorns_balls, food_balls)
        # # evaluate self merge
        # merge_score = self.estimate_time_merge_to_center(my_clones_np, enemy_clones_np)
        # merge_direction = Vector2(0, 0)
        # merge_action = 2

        # give the best direction and action for thorns ball
        if self.last_direction is None:
            self.last_direction = (Vector2(500.0, 500.0) - my_clone_balls[0]['position']).normalize()
            # initiate last direction

        food_target_ball, food_direction, food_action, food_score = None, None, None, 0.0
        if thorns_my_ball is None:

            if n_my_clone == 1:
                good_balls = self.process_good_fb(food_balls, team_clone_balls, others_clone_balls, my_clone_balls[-1],
                                                  rect)
                food_balls_np = util.group_process_to_np(good_balls)
                food_target_ball, food_direction, food_action, food_score = self.process_food_balls_np(food_balls_np,
                                                                                                       my_clones_np)
            else:
                # food_balls = util.food_filter_dict(food_balls, my_clone_balls)
                # food_target_ball, food_direction, food_action, food_score = self.process_food_balls(food_balls,
                #                                                                                     my_clone_balls[-1])
                food_balls = util.food_filter_dict(food_balls, my_clone_balls)
                view_center = Vector2(0.5 * (rect[0] + rect[2]), 0.5 * (rect[1] + rect[3]))
                mb_chaser = self.process_food_eater(view_center, my_clone_balls)
                food_target_ball, food_direction, food_action, food_score = self.process_food_balls_pd(food_balls,
                                                                                                       mb_chaser,
                                                                                                       team_clone_balls,
                                                                                                       my_clone_balls,
                                                                                                       others_clone_balls,
                                                                                                       rect)
        if lead_my_ball is not None:
            if (thorns_my_ball is not None) and lead_score < thorns_score:
                direction, action_type = thorns_direction, thorns_action
            elif (thorns_my_ball is None) and (food_target_ball is not None) and lead_score < food_score:
                direction, action_type = food_direction, food_action
            # elif lead_score < merge_score:
            #     direction, action_type = merge_direction, merge_action
            else:
                direction, action_type = lead_direction, lead_action
        else:
            if thorns_score > 0.0:
                direction, action_type = thorns_direction, thorns_action
            elif food_target_ball is not None:
                direction, action_type = food_direction, food_action
            else:
                direction, action_type = (Vector2(500., 500.) - my_clone_balls[0]['position']).normalize(), -1
        # give the overall direction and action

        # record
        self.last_direction = direction
        self.last_my_clone = my_clone_balls

        # fresh merge cd timer
        if (action_type == 4) or (n_my_clone > n_my_clone_last and (n_my_clone_last < 4 or my_size <= 3000)):
            self.merge_cd = 20.
        elif self.merge_cd > 0:
            self.merge_cd -= 0.2
        else:
            self.merge_cd = 0.

        # merge to eat enemy
        if action_type == -1 and (direction is not None) and n_my_clone > 1:
            biggest_position = my_clone_balls[0]['position']
            # bug
            if biggest_position != my_center:
                center_to_biggest_dir = Vector2(biggest_position.x - my_center.x,
                                                biggest_position.y - my_center.y).normalize()
                go_to_bigger = (action_type == -1) and (
                        center_to_biggest_dir.x * direction.x + center_to_biggest_dir.y * direction.y > math.sqrt(
                    2) / 2)
            else:
                go_to_bigger = True
        else:
            go_to_bigger = False
        # go_to_bigger = True

        # keep merge
        merge_v_ratio = 0.8
        biggest_v = my_clone_balls[0]['radius'] * my_clone_balls[0]['radius']
        rest_too_small = biggest_v > my_size * merge_v_ratio
        fake_merge_v = my_size * merge_v_ratio
        fake_merge_r = math.sqrt(fake_merge_v)
        merge_center = copy.deepcopy(my_center)
        if others_clone_balls:
            biggest_enemy = others_clone_balls[0]
            biggest_enemy_v = biggest_enemy['radius'] * biggest_enemy['radius']
        else:
            biggest_enemy_v = 0.0

        split_eat = False
        nearby_danger = self.nearby_danger(merge_center, fake_merge_r, thorns_balls, others_clone_balls)
        low_density = False and (my_density < 0.125) and help_ball
        keep_atk_merge = (self.last_action_type in [2, 3]) and my_density < 0.75 and (not rest_too_small)
        start_atk_merge = my_size >= 10000. and my_density < 0.5 and (not go_to_bigger) and (not keep_atk_merge)

        if start_atk_merge:
            estimate_time = self.estimate_time_merge_to_center(my_clone_balls, v_ratio=merge_v_ratio,
                                                               start_merge=True)
            merge_save_v = my_size * 0.8
            merge_save = self.check_merge_save(my_clone_balls, others_clone_balls, merge_save_v)
            if not merge_save:
                start_atk_merge = False
            else:
                # pre start merge
                if 0.0 < self.merge_cd <= min(10, estimate_time / 2.):
                    self.merge_cd = 0.0
        else:
            estimate_time = 0.0

        can_merge = (n_my_clone > 1) and not (
                action_type == 4 and lead_score > fake_merge_v / 16) and self.merge_cd == 0.0

        if can_merge and (start_atk_merge or keep_atk_merge):
            for enemy_idx, enemy_cl in enumerate(others_clone_balls):
                enemy_cl_pos = enemy_cl['position']
                enemy_cl_r = enemy_cl['radius']
                enemy_cl_v = enemy_cl_r * enemy_cl_r
                dis = (enemy_cl_pos - merge_center).length()

                if keep_atk_merge:
                    split_once = fake_merge_v / 2 > enemy_cl_v >= fake_merge_v / 8 and dis < fake_merge_r * 2.1 + 15
                    split_twice = fake_merge_v / 4 > enemy_cl_v >= fake_merge_v / 16 and dis < fake_merge_r * 2.9 + 18
                # start_atk_merge
                else:
                    spd = 500. / (10. + enemy_cl_r)
                    move_dis = spd * estimate_time
                    if dis > 0:
                        move_dir = (enemy_cl_pos - merge_center).normalize()
                    else:
                        move_dir = Vector2(0.01, 0.01)
                    move_tar_pos = enemy_cl_pos + move_dir * move_dis
                    move_tar_x = max(0., min(1000., move_tar_pos.x))
                    move_tar_y = max(0., min(1000., move_tar_pos.y))
                    move_tar_pos = Vector2(move_tar_x, move_tar_y)
                    estimate_dis = (move_tar_pos - merge_center).length()
                    split_once = fake_merge_v / 2 > enemy_cl_v >= fake_merge_v / 8 and estimate_dis < fake_merge_r * 2.1 + 15
                    split_twice = fake_merge_v / 4 > enemy_cl_v >= fake_merge_v / 16 and estimate_dis < fake_merge_r * 2.9 + 18

                if not (split_once or split_twice):
                    continue

                split_eat = True
                if split_once:
                    for danger_cl in others_clone_balls:
                        danger_cl_pos = danger_cl['position']
                        danger_cl_r = danger_cl['radius']
                        danger_cl_v = danger_cl_r * danger_cl_r
                        danger_dis = (danger_cl_pos - enemy_cl_pos).length()
                        if danger_cl_v / 2 > (fake_merge_v / 2 + enemy_cl_v) and danger_dis < 2.2 * danger_cl_r:
                            split_eat = False
                            break
                elif split_twice:
                    for danger_cl in others_clone_balls:
                        danger_cl_pos = danger_cl['position']
                        danger_cl_r = danger_cl['radius']
                        danger_cl_v = danger_cl_r * danger_cl_r
                        danger_dis = (danger_cl_pos - enemy_cl_pos).length()
                        if danger_cl_v / 2 > (fake_merge_v / 4 + enemy_cl_v) and danger_dis < 2.2 * danger_cl_r:
                            split_eat = False
                            break
                else:
                    split_eat = False

                # some ball can eat
                if split_eat:
                    break

        # if surround_merge and can_merge and split_eat:
        #     print(self.name, self.cur_time, 'surround')

        if can_merge and split_eat:
            use_spore = True
            action_type = 2
            spore_for_split_twice = n_aft_split >= 16
            spore_save_v = my_size * 0.8
            spore_save = self.check_spore_save(my_clone_balls, others_clone_balls, spore_save_v)
            if use_spore:
                if self.merge_count < 12:
                    # stop and move to center
                    self.merge_count += 1
                elif self.merge_count >= 60:
                    if self.check_merge_stop(my_clone_balls):
                        self.merge_count = 0.0
                        action_type = -1
                    else:
                        self.merge_count -= 5
                elif n_my_clone_last < n_my_clone:
                    self.merge_count = 0
                elif spore_for_split_twice and spore_save and (not nearby_danger):
                    # split to center
                    action_type = 3

        else:
            self.merge_count = 0

        # weak merge
        if my_clone_balls[0]['radius'] < 20 and self.weak and (help_ball is not None) and can_merge:
            action_type = 2

        if action_type == 3:
            self.actions_queue.put([None, None, action_type])
        else:
            self.actions_queue.put([direction.x, direction.y, action_type])
        action_ret = self.actions_queue.get()
        self.last_action_type = action_type
        self.last_player_clone = copy.deepcopy(self.player_clone)
        if self.actions_queue.qsize() > 0:
            return self.actions_queue.get(), None
        return action_ret, help_ball

    def nearby_danger(self, center_pos, r, thorns, enemy_clones):
        """
        判断merge区域中 是否存在荆棘球
        如果存在 会影响合球
        :param center_pos:
        :param r:
        :param thorns:
        :param enemy_clones:
        :return:
        """
        for th in thorns:
            th_pos = th['position']
            dis = (center_pos - th_pos).length()
            if dis <= r:
                return True
        for cl in enemy_clones:
            cl_pos = cl['position']
            dis = (center_pos - cl_pos).length()
            if dis <= r:
                return True
        return False

    def check_merge_stop(self, my_clones):
        my_biggest_cl = my_clones[0]
        stop_merge = True
        for idx, my_cl in enumerate(my_clones):
            if idx == 0:
                continue
            dis = (my_cl - my_biggest_cl).length()
            l_sum = my_cl['radius'] + my_biggest_cl['radius']
            if dis < l_sum * 0.99 or dis > l_sum * 1.01:
                stop_merge = False
                break
        return stop_merge

    def check_spore_save(self, my_clones, enemy_clones, fake_merge_v):
        cum_size = 0.0
        for my_cl in my_clones:
            my_cl_pos = my_cl['position']
            my_cl_r = my_cl['radius']
            if my_cl_r < 10:
                continue
            my_cl_v = my_cl_r * my_cl_r
            for enemy_cl in enemy_clones:
                enemy_cl_pos = enemy_cl['position']
                enemy_cl_r = enemy_cl['radius']
                enemy_cl_v = enemy_cl_r * enemy_cl_r

                # no dangerous
                if enemy_cl_v < my_cl_v:
                    continue

                dis_enemy_to_my_cl = (my_cl_pos - enemy_cl_pos).length()
                in_range = dis_enemy_to_my_cl < 2.2 * enemy_cl_r + 15
                spore_die = enemy_cl_v > 10. and enemy_cl_v / 2 > my_cl_v - 9.
                if in_range and spore_die:
                    return False
            cum_size += my_cl_v
            if cum_size >= fake_merge_v:
                break
        return True

    def test_bug(self, record_second, player_id):
        if record_second == self.cur_time and player_id == int(self.name):
            print('now')

    def evaluate_spd(self, last_my_clone, cur_my_clone):
        cur_my_clones_np = util.group_process_to_np(cur_my_clone)
        if last_my_clone:
            last_my_clones_np = util.group_process_to_np(last_my_clone)
            n_last_my_clones = last_my_clones_np.shape[0]
            n_cur_my_clones = cur_my_clones_np.shape[0]
            if n_last_my_clones == n_cur_my_clones:
                move_x, move_y = util.calculate_spd(last_my_clones_np, cur_my_clones_np)
                self.last_spd[0] = move_x
                self.last_spd[1] = move_y

    def is_weak(self):
        player_id = int(self.name)
        team_id = int(self.team)

        # merge when V very small
        my_sum_v = self.player_v[player_id]
        for p in range(3*team_id, 3*team_id+3):
            if p == player_id:
                continue
            friend_v = self.player_v[p]
            if friend_v > 10000. and my_sum_v < friend_v / 16.:
                return True
        return False

    def is_boss(self):
        player_id = int(self.name)
        team_id = int(self.team)

        # merge when V very small
        my_sum_v = self.player_v[player_id]
        friend_max_v = 0.0
        for k, v in self.player_v.items():
            if int(k // 3) == team_id:
                friend_max_v = max(friend_max_v, v)

        if my_sum_v == friend_max_v:
            return True
        return False

    def evaluate_acc_time(self, cl_r):
        acc = 20. / math.sqrt(cl_r)
        v_max = 500. / (10 + cl_r)
        acc_time = v_max / acc
        return acc_time

    def check_merge_save(self, my_clones, enemy_clones, fake_merge_v):
        cum_size = 0.0
        stop_time = 1.0
        for my_cl in my_clones:
            my_cl_pos = my_cl['position']
            my_cl_r = my_cl['radius']
            my_cl_v = my_cl_r * my_cl_r
            acc_time = self.evaluate_acc_time(my_cl_r)
            for enemy_cl in enemy_clones:
                enemy_cl_pos = enemy_cl['position']
                enemy_cl_r = enemy_cl['radius']
                enemy_cl_v = enemy_cl_r * enemy_cl_r
                enemy_cl_spd = util.get_spd(enemy_cl_r)
                # no dangerous
                if enemy_cl_v <= my_cl_v:
                    continue
                elif enemy_cl_v / 2 < my_cl_v:
                    dead_dis = (enemy_cl_pos - my_cl_pos).length() - enemy_cl_r
                else:
                    dead_dis = (enemy_cl_pos - my_cl_pos).length() - 2.12 * enemy_cl_r
                dis_chase = stop_time * enemy_cl_spd + 0.5 * acc_time * enemy_cl_spd
                if dead_dis < dis_chase:
                    return False
            cum_size += my_cl_v
            if cum_size >= fake_merge_v:
                break
        return True

    def estimate_time_merge_to_center(self, my_clones, v_ratio=0.8, start_merge=True):
        n_my_clones = len(my_clones)
        if n_my_clones <= 1:
            return 0.0

        center_pos = util.get_center(my_clones)
        my_size = sum([cl['radius'] * cl['radius'] for cl in my_clones])

        # raw_t
        t_raw = []
        for cl in my_clones:
            cl_pos = cl['position']
            cl_r = cl['radius']
            dis = (cl_pos - center_pos).length()

            acc = 20. / math.sqrt(cl_r)
            v_max = 500. / (10 + cl_r)
            if start_merge:
                t_acc = v_max / acc
                dis_during_acc = t_acc * v_max / 2
                dis_rest = dis - dis_during_acc
                if dis_rest >= 0:
                    t_rest = dis_rest / v_max
                else:
                    t_rest = 0.0
                    t_acc = math.sqrt(2 * dis / acc)
                t_total = t_acc + t_rest
            else:
                t_total = dis / v_max
            t_raw.append(t_total)

        # new_t
        t_raw = np.array(t_raw)
        argsort = np.argsort(t_raw).tolist()
        idx_sort = []
        for idx in range(n_my_clones):
            idx_sort.append(argsort.index(idx))

        v_cum = 0.0
        t_cum = 0.0
        for idx in idx_sort:
            cl = my_clones[int(idx)]
            cl_pos = cl['position']
            cl_r = cl['radius']
            cl_v = cl_r * cl_r
            r_fake = math.sqrt(v_cum)

            dis = (cl_pos - center_pos).length() - r_fake
            v_max = 500. / (10 + cl_r)
            if start_merge:
                acc = 20. / math.sqrt(cl_r)

                t_acc = v_max / acc
                dis_during_acc = t_acc * v_max / 2

                t_max_vel = t_cum - t_acc if t_cum >= t_acc else 0.0
                dis_max_vel = t_max_vel * v_max
                dis_rest = dis - dis_during_acc - dis_max_vel
                if dis_rest > 0:
                    t_rest = dis_rest / v_max
                else:
                    t_rest = 0.0
                if t_cum == 0.0:
                    t_cum += t_acc
            else:
                t_rest = dis / v_max

            t_cum += t_rest
            # cum v
            v_cum += cl_v
            if v_cum >= v_ratio * my_size:
                break

        if start_merge:
            t_stop = 1.
            return t_stop + t_cum
        else:
            return t_cum

    def process_statistic(self, my_clone_balls):
        my_center = Vector2(0, 0)
        my_distribution = 0.0
        my_density = 0.0
        my_size = 0.0
        size_ratio = np.zeros(len(my_clone_balls))

        for my_ball in my_clone_balls:
            size = my_ball['radius'] * my_ball['radius']
            size_ratio[my_clone_balls.index(my_ball)] = size
            my_size += size
            my_center += my_ball['position'] * size

        delta_time = time.time() - self.timer
        self.timer = time.time()
        self.average_delta_volume = 0.998 * self.average_delta_volume + 0.002 * (
                my_size - self.last_my_volume) / delta_time
        # official match: delta_time = 0.2s, player mode: delta_time = 0.05s *
        self.last_my_volume = my_size

        size_ratio = size_ratio / my_size
        my_center = my_center / my_size

        for idx, my_ball in enumerate(my_clone_balls):
            ratio = (my_ball['position'] - my_center).length() / my_ball['radius']
            my_distribution += ratio * ratio * size_ratio[idx]
            my_density += size_ratio[idx] * size_ratio[idx]

        my_distribution = my_distribution / len(my_clone_balls)

        return my_center, my_distribution, my_density, my_size

    def process_clone_statistic(self, total_clone):
        player_v = dict()
        player_n = dict()
        clone = dict()
        clone_np = dict()
        center = dict()
        for cl in total_clone:
            name = int(cl['player'])
            # volumn
            r = cl['radius']
            cur_v = player_v.get(name, 0.0)
            player_v[name] = cur_v + r * r
            # n
            cur_n = player_n.get(name, 0.0)
            player_n[name] = cur_n + 1
            # cl
            cl_rec = clone.get(name, [])
            cl_rec.append(cl)
            clone[name] = cl_rec

        for k in clone.keys():
            clone[k].sort(key=lambda a: a['radius'], reverse=True)
            clone_np[k] = util.group_process_to_np(clone[k])
            center[k] = util.get_center(clone[k])

        self.player_clone = clone
        self.player_clone_np = clone_np
        self.player_center = center
        self.player_v = player_v
        self.player_n = player_n

    def estimate_value_balls(self, my_clone_balls, others_clone_balls, my_center, my_distribution, my_density,
                             my_size, team_danger_balls, team_clone_balls):
        move_score = []
        split_score = []
        attention_my_balls = []
        attention_other_balls = []
        attention_direction = []

        lead_my_ball = None
        help_ball = None

        final_action = -1
        final_direction = (Vector2(0, 0) - my_clone_balls[0]['position']).normalize()
        final_score = 0.0

        density_ratio = 1 - (my_density - 1 / 16) * 16 / 15
        # small density brings small give-up ratio
        giveup_ratio = (1 - density_ratio) * 0.1

        # consider the enemy to be merged as the same enemy
        new_other_volume = [0.0] * len(others_clone_balls)
        for i in range(0, len(others_clone_balls)):
            new_other_volume[i] = others_clone_balls[i]['radius'] * others_clone_balls[i]['radius']
            for j in range(0, len(others_clone_balls)):
                if j != i:
                    distance = (others_clone_balls[j]['position'] - others_clone_balls[i]['position']).length()
                    merge_distance = max(others_clone_balls[j]['radius'], others_clone_balls[i]['radius'])
                    if (distance - merge_distance) / min(others_clone_balls[j]['radius'],
                                                         others_clone_balls[i]['radius']) < 0.4:
                        # almost merge
                        if others_clone_balls[j]['radius'] < others_clone_balls[i]['radius']:
                            # big balls swallow small balls
                            new_other_volume[i] += others_clone_balls[j]['radius'] * others_clone_balls[j]['radius']
                        else:
                            new_other_volume[i] = 0.0
                            # small balls be swallowed
                            break
        # update new clone ball list
        pt = 0
        while pt < len(others_clone_balls):
            if new_other_volume[pt] == 0.0:
                del others_clone_balls[pt]
                del new_other_volume[pt]
            else:
                others_clone_balls[pt]['radius'] = math.sqrt(new_other_volume[pt])
                pt += 1
        # first round scoring
        for my_ball in my_clone_balls:
            my_ball_volume = my_ball['radius'] * my_ball['radius']
            if my_ball_volume / my_size > giveup_ratio:
                # give up small balls
                for other_ball in others_clone_balls:
                    other_ball_volume = other_ball['radius'] * other_ball['radius']
                    distance = (my_ball['position'] - other_ball['position']).length()
                    dead_distance = max(my_ball['radius'], other_ball['radius'])
                    move_dead_time = max((distance - dead_distance) * (dead_distance + 10.0) / 500.0, 0.1)
                    # speed = 500.0 / (r + 10.0), assume target is static and move towards it with max speed
                    # 1 second = 10 frames so min dead_time = 0.1s
                    split_dead_time = 0.2
                    # assume split needs 0.2s
                    # regular move score
                    if my_ball['radius'] > other_ball['radius']:
                        direction = (other_ball['position'] - my_ball['position']).normalize()  # forward
                        move_dead_time = max(move_dead_time, math.pow(move_dead_time, 2.0))
                        # assume catch dynamic ball needs extra reach_time
                        # assume reach_time > 1s has much less chance than reach_time < 1s
                        move_eat_volume = other_ball_volume

                        if my_clone_balls.index(my_ball) < 16 - len(my_clone_balls):  # can split
                            if my_ball['radius'] * 0.7 > other_ball['radius'] > my_ball[
                                'radius'] * 0.2 and distance < 2.0 * my_ball['radius'] + 15:
                                split_eat_volume = other_ball_volume
                                # split once and kill target
                            elif my_ball['radius'] * 0.7 > other_ball['radius'] or distance > (
                                    1.5 * my_ball['radius'] + other_ball['radius']):
                                split_eat_volume = 0.0
                                # get zero benefit
                            else:
                                split_eat_volume = - 0.7 * my_ball_volume
                                # my split ball may lose
                        else:
                            split_eat_volume = 0.0
                            # can't split, get zero benefit
                    else:
                        direction = (my_ball['position'] - other_ball['position']).normalize()  # back
                        move_eat_volume = my_ball_volume
                        if other_ball['radius'] * 0.72 > my_ball['radius'] and distance < 2.2 * other_ball[
                            'radius'] + 15:
                            move_dead_time = 0.2  # other split attack (real: other may not split)
                        else:
                            move_dead_time = max(move_dead_time, math.pow(move_dead_time, 2.0))
                            # have extra time to escape

                        if my_clone_balls.index(my_ball) < 16 - len(my_clone_balls):  # can split
                            if distance < other_ball['radius'] * 2.2:
                                split_eat_volume = - 0.7 * my_ball_volume
                                # your split ball will be killed
                            else:
                                split_eat_volume = 0.0
                        else:
                            split_eat_volume = 0.0

                    mv_score = 2.0 * move_eat_volume / move_dead_time
                    sp_score = 2.0 * split_eat_volume / split_dead_time
                    # delta_volume / reach_time, delta_volume (2.0) means (my reward volume - target's lost volume)
                    if mv_score < 2.0 and 0.0 <= sp_score < 2.0:
                        continue
                    # filter low score

                    attention_direction.append(direction)
                    move_score.append(mv_score)
                    split_score.append(sp_score)
                    attention_my_balls.append(my_ball)
                    attention_other_balls.append(other_ball)
                    # Don't use deepcopy if not change its elements

                    for i in range(len(move_score) - 1, 0, -1):
                        if move_score[i] > move_score[i - 1]:
                            attention_my_balls[i - 1], attention_my_balls[i] = attention_my_balls[i], \
                                                                               attention_my_balls[i - 1]
                            attention_other_balls[i - 1], attention_other_balls[i] = attention_other_balls[i], \
                                                                                     attention_other_balls[i - 1]
                            move_score[i - 1], move_score[i] = move_score[i], move_score[i - 1]
                            split_score[i - 1], split_score[i] = split_score[i], split_score[i - 1]
                            attention_direction[i - 1], attention_direction[i] = attention_direction[i], \
                                                                                 attention_direction[i - 1]
                        else:
                            break
                    if len(move_score) > 128:
                        move_score.pop()
                        split_score.pop()
                        attention_my_balls.pop()
                        attention_other_balls.pop()
                        attention_direction.pop()
                    # sort value pairs by their move scores
                    # choose top 128 pairs

            # receive teammates' sos info
            for team_ball in team_danger_balls:
                if team_ball['player'] == self.name or my_size < 2000.0 or len(my_clone_balls) == 1:
                    # help teammates only my total volume >= 2000.0
                    # can't help if you have only one ball
                    continue
                direction = (team_ball['position'] - my_ball['position']).normalize()
                team_ball_volume = team_ball['radius'] * team_ball['radius']
                distance = (my_ball['position'] - team_ball['position']).length()
                dead_distance = max(my_ball['radius'], team_ball['radius'])
                move_dead_time = max((distance - dead_distance) * (dead_distance + 10.0) / 500.0, 0.1)
                # move_dead_time = max(move_dead_time, math.pow(move_dead_time, 1.2))
                # help teammates need less time than catch enemies
                move_eat_volume = team_ball_volume
                # may save the team ball's volume
                mv_score = 2.0 * move_eat_volume / move_dead_time
                sp_score = 0.0
                # ignore split action

                team_ball['radius'] = my_ball['radius'] * 0.01
                # consider team ball smaller than me
                if mv_score < 2.0 and 0.0 <= sp_score < 2.0:
                    continue

                attention_direction.append(direction)
                move_score.append(mv_score)
                split_score.append(sp_score)
                attention_my_balls.append(my_ball)
                attention_other_balls.append(team_ball)

                for i in range(len(move_score) - 1, 0, -1):
                    if move_score[i] > move_score[i - 1]:
                        attention_my_balls[i - 1], attention_my_balls[i] = attention_my_balls[i], \
                                                                           attention_my_balls[i - 1]
                        attention_other_balls[i - 1], attention_other_balls[i] = attention_other_balls[i], \
                                                                                 attention_other_balls[i - 1]
                        move_score[i - 1], move_score[i] = move_score[i], move_score[i - 1]
                        split_score[i - 1], split_score[i] = split_score[i], split_score[i - 1]
                        attention_direction[i - 1], attention_direction[i] = attention_direction[i], \
                                                                             attention_direction[i - 1]
                    else:
                        break
                if len(move_score) > 128:
                    move_score.pop()
                    split_score.pop()
                    attention_my_balls.pop()
                    attention_other_balls.pop()
                    attention_direction.pop()

            '''
            for team_ball in team_clone_balls:
                direction = (team_ball['position'] - my_ball['position']).normalize()
                team_ball_radius = team_ball['radius']
                team_ball_volume = team_ball_radius * team_ball_radius
                distance = (my_ball['position'] - team_ball['position']).length()
                dead_distance = max(my_ball['radius'], team_ball_radius)
                move_dead_time = max((distance - dead_distance) * (dead_distance + 10.0) / 500.0, 0.1)
                if my_ball['radius'] < team_ball['radius'] and my_density < 0.2 and my_size > 4000.0:
                    move_dead_time = max(move_dead_time, math.pow(move_dead_time, 1.2))
                    move_eat_volume = my_ball_volume
                    mv_score = move_eat_volume / move_dead_time
                    sp_score = 0.0

                    attention_direction.append(direction)
                    move_score.append(mv_score)
                    split_score.append(sp_score)
                    attention_distance.append(distance)
                    attention_my_balls.append(my_ball)
                    attention_other_balls.append(copy.deepcopy(team_ball))
            '''

        final_move_score = [0.0] * len(move_score)
        final_split_score = [0.0] * len(split_score)
        # second round scoring, each value pair will influence other pairs
        for i in range(0, len(move_score)):
            other_record = []
            my_record = []
            if attention_my_balls[i]['radius'] > attention_other_balls[i]['radius']:
                other_record = [[attention_other_balls[i], move_score[i], split_score[i]]]  # get score
            else:
                my_record = [[attention_my_balls[i], move_score[i], split_score[i]]]  # lose score
            for j in range(0, len(move_score)):
                if i != j:
                    correlation = attention_direction[j].x * attention_direction[i].x + attention_direction[j].y * \
                                  attention_direction[i].y
                    related_move_score = correlation * move_score[j]
                    related_split_score = 0.0
                    if correlation >= 0.9:
                        related_split_score = split_score[j]
                    elif split_score[j] < 0:
                        related_split_score = split_score[j]
                    if correlation < 0.0 and attention_my_balls[j]['radius'] < attention_other_balls[j]['radius'] and \
                            (attention_my_balls[j]['position'] - attention_other_balls[j]['position']).length() < 1.5 * \
                            attention_my_balls[j]['radius'] + 2.1 * attention_other_balls[j]['radius']:
                        related_split_score = - 2.0 * attention_my_balls[j]['radius'] * attention_my_balls[j][
                            'radius'] / 0.2
                    ball_his = False
                    if attention_my_balls[j]['radius'] > attention_other_balls[j]['radius']:
                        for k in range(0, len(other_record)):
                            if attention_other_balls[j] == other_record[k][0]:
                                ball_his = True
                                other_record[k][1] = max(other_record[k][1], related_move_score)
                                if related_split_score < 0.0:
                                    other_record[k][2] = min(other_record[k][2], related_split_score)
                                break
                        if not ball_his:
                            other_record.append([attention_other_balls[j], related_move_score, related_split_score])
                    else:
                        for k in range(0, len(my_record)):
                            if attention_my_balls[j] == my_record[k][0]:
                                ball_his = True
                                my_record[k][1] = max(my_record[k][1], related_move_score)
                                if related_split_score < 0.0:
                                    my_record[k][2] = min(my_record[k][2], related_split_score)
                                break
                        if not ball_his:
                            my_record.append([attention_my_balls[j], related_move_score, related_split_score])

            for k in range(0, len(my_record)):
                final_move_score[i] += my_record[k][1]
                final_split_score[i] += my_record[k][2]
            for k in range(0, len(other_record)):
                final_move_score[i] += other_record[k][1]
                final_split_score[i] += other_record[k][2]
        # final selection
        if len(move_score) > 0:
            best_move_id = final_move_score.index(max(final_move_score))
            mv_my_ball = attention_my_balls[best_move_id]
            mv_target_ball = attention_other_balls[best_move_id]
            mv_score = final_move_score[best_move_id]
            mv_direction = attention_direction[best_move_id]

            best_split_id = final_split_score.index(max(final_split_score))
            sp_my_ball = attention_my_balls[best_split_id]
            sp_target_ball = attention_other_balls[best_split_id]
            sp_score = final_split_score[best_split_id]
            sp_direction = attention_direction[best_split_id]

            if mv_score >= sp_score:
                final_action = -1
                final_direction = mv_direction
                final_score = mv_score
                lead_my_ball = mv_my_ball
                if mv_my_ball['radius'] < mv_target_ball['radius']:
                    help_ball = copy.deepcopy(mv_my_ball)
                    if mv_my_ball['position'].x - mv_my_ball['radius'] < 0.5 and - final_direction.x > 0.0:
                        if final_direction.y == 0.0:
                            final_direction.y = 1.0
                        final_direction = Vector2(0.05, final_direction.y).normalize()
                    elif mv_my_ball['position'].y - mv_my_ball['radius'] < 0.5 and - final_direction.y > 0.0:
                        if final_direction.x == 0.0:
                            final_direction.x = 1.0
                        final_direction = Vector2(final_direction.x, 0.05).normalize()
                    elif mv_my_ball['position'].x + mv_my_ball['radius'] > 999.5 and final_direction.x > 0.0:
                        if final_direction.y == 0.0:
                            final_direction.y = 1.0
                        final_direction = Vector2(-0.05, final_direction.y).normalize()
                    elif mv_my_ball['position'].y + mv_my_ball['radius'] > 999.5 and final_direction.y > 0.0:
                        if final_direction.x == 0.0:
                            final_direction.x = 1.0
                        final_direction = Vector2(final_direction.x, -0.05).normalize()
            else:
                final_action = 4
                final_direction = sp_direction
                final_score = sp_score
                lead_my_ball = sp_my_ball

        return lead_my_ball, final_direction, final_action, final_score, help_ball

    def estimate_value_balls_friend_verison(self, my_clone_balls, enemy_clone_balls, my_center, my_distribution,
                                            my_density,
                                            my_size, team_help_balls, team_clone_balls, thorns):
        move_score = []
        split_score = []
        attention_my_balls = []
        attention_other_balls = []
        attention_direction = []

        lead_my_ball = None
        help_ball = None

        final_action = -1

        center_pos = Vector2(500., 500.)
        if my_clone_balls[0]['position'] != center_pos:
            final_direction = (center_pos - my_clone_balls[0]['position']).normalize()
        else:
            final_direction = Vector2(1, 0)
        final_score = 0.0

        density_ratio = 1 - (my_density - 1 / 16) * 16 / 15
        # small density brings small give-up ratio
        giveup_ratio = (1 - density_ratio) * 0.1
        my_total_v = self.player_v[int(self.name)]
        n_my_cl = len(my_clone_balls)
        split_dead_time = 0.2

        if n_my_cl == 1:
            ignore_dt_v = 40
        else:
            ignore_dt_v = 20

        # split_n
        can_split_n = 0
        for my_ball in my_clone_balls:
            if my_ball['radius'] > 10:
                can_split_n += 1
            else:
                break

        center_pos = my_center
        enemy_clone_balls_ignore_merge = copy.deepcopy(enemy_clone_balls)
        # consider the enemy to be merged as the same enemy
        merge_volumn = [0.0] * len(enemy_clone_balls)

        for i in range(0, len(enemy_clone_balls)):
            if merge_volumn[i] != 0.0:
                continue
            i_team = enemy_clone_balls[i]['team']
            for j in range(0, len(enemy_clone_balls)):
                j_team = enemy_clone_balls[j]['team']
                # not same ball & same team
                if j == i or i_team != j_team:
                    continue
                tmp_dis = (enemy_clone_balls[j]['position'] - enemy_clone_balls[i]['position']).length()
                merge_distance = max(enemy_clone_balls[j]['radius'], enemy_clone_balls[i]['radius'])
                if (tmp_dis - merge_distance) / min(enemy_clone_balls[j]['radius'],
                                                    enemy_clone_balls[i]['radius']) < 0.4:
                    # almost merge
                    if enemy_clone_balls[j]['radius'] < enemy_clone_balls[i]['radius']:
                        # big balls swallow small balls
                        merge_volumn[i] += enemy_clone_balls[j]['radius'] * enemy_clone_balls[j]['radius']
                        merge_volumn[j] -= enemy_clone_balls[j]['radius'] * enemy_clone_balls[j]['radius']
                    else:
                        merge_volumn[i] -= enemy_clone_balls[j]['radius'] * enemy_clone_balls[j]['radius']
                        merge_volumn[j] += enemy_clone_balls[j]['radius'] * enemy_clone_balls[j]['radius']
                        break

        n_aft_split = min(16, can_split_n + n_my_cl)
        # first round score-enemy
        for my_idx, my_ball in enumerate(my_clone_balls):
            my_cl_pos = my_ball['position']
            my_cl_r = my_ball['radius']
            my_cl_v = my_cl_r * my_cl_r
            my_cl_can_split = my_cl_r > 10. and (my_idx + n_my_cl < 16)
            my_cl_can_split_twice = my_cl_r > 20. and (n_aft_split + my_idx < 16)

            # give up small balls
            if my_cl_v / my_size < giveup_ratio:
                continue

            enemy_nearby = []
            friend_nearby = []
            th_nearby = []
            my_going_merge = []
            if my_cl_can_split:
                # enemy nearby
                for b in enemy_clone_balls:
                    b_pos = b['position']
                    cl_to_b_dis = (my_cl_pos - b_pos).length()
                    b_r = b['radius']
                    if max(my_cl_r, b_r) * 2 * 2.2 + 15 > cl_to_b_dis:
                        enemy_nearby.append(b)

                # friend_nearby
                for b in team_clone_balls:
                    b_pos = b['position']
                    cl_to_b_dis = (my_cl_pos - b_pos).length()
                    if my_cl_r * 2.5 > cl_to_b_dis:
                        friend_nearby.append(b)

                # thorn nearby
                for b in thorns:
                    b_pos = b['position']
                    cl_to_b_dis = (my_cl_pos - b_pos).length()
                    if my_cl_r * 2.5 > cl_to_b_dis:
                        th_nearby.append(b)

                if my_idx == 0:
                    for my_merge_idx, b in enumerate(my_clone_balls):
                        if my_merge_idx == my_idx:
                            continue
                        b_pos = b['position']
                        b_r = b['radius']
                        if my_cl_r < b_r:
                            continue
                        cl_to_b_dis = (my_cl_pos - b_pos).length()
                        is_merging = cl_to_b_dis < (my_cl_r + b_r) * 0.99
                        this_cl_can_split = my_merge_idx + n_my_cl < 16 and b_r > 10.
                        if my_cl_can_split and is_merging and (not this_cl_can_split):
                            my_going_merge.append(b)

            for enemy_idx, enemy_cl in enumerate(enemy_clone_balls):
                enemy_cl_pos = enemy_cl['position']
                enemy_cl_r = enemy_cl['radius']
                enemy_cl_v = enemy_cl_r * enemy_cl_r

                enemy_cl_name = int(enemy_cl['player'])
                enemy_player_clone = self.player_clone_np[enemy_cl_name]
                n_enemy_cl = enemy_player_clone.shape[0]
                enemy_idx = 0

                for j, temp_cl in enumerate(enemy_player_clone):
                    if int(enemy_cl_pos[0]) == int(temp_cl[0]) and int(enemy_cl_pos[1]) == int(temp_cl[1]):
                        enemy_idx = j
                        break

                enemy_can_split_sum = np.sum(enemy_player_clone[:, 2] > 10.)
                enemy_can_split_twice = enemy_idx == 0 and enemy_cl_r > 20.
                enemy_can_split_third = enemy_idx == 0 and len(enemy_player_clone) <= 2 and enemy_cl_r > 40.

                my_to_enemy_dis = (my_cl_pos - enemy_cl_pos).length()
                if my_to_enemy_dis > 0:
                    direction = (enemy_cl_pos - my_cl_pos).normalize()
                else:
                    direction = Vector2(0.1, 0.1)

                dead_distance = max(my_cl_r, enemy_cl_r)
                move_spd = 500. / (10. + dead_distance)
                move_dead_time = max((my_to_enemy_dis - dead_distance) / move_spd, 0.2)



                # speed = 500.0 / (r + 10.0), assume target is static and move towards it with max speed
                # 1 second = 10 frames so min dead_time = 0.1s

                # assume split needs 0.2s
                # regular move score

                merge_adjust_v = enemy_cl_v + merge_volumn[enemy_idx]
                merge_adjust_r = math.sqrt(merge_adjust_v) if merge_adjust_v > 0 else 0
                if my_cl_v > max(enemy_cl_v, merge_adjust_v):
                    # ignore near V
                    if my_cl_v - ignore_dt_v < enemy_cl_v:
                        move_eat_volume = 0.
                    else:
                        move_eat_volume = merge_adjust_v

                    # ignore hide in corder
                    if math.sqrt(2) * enemy_cl_r < my_cl_r * (math.sqrt(2) - 1):
                        for corner_pos in [Vector2(0., 0.), Vector2(0., 1000.), Vector2(1000., 0.),
                                           Vector2(1000., 1000.)]:
                            dis_to_corner = (enemy_cl_pos - corner_pos).length()
                            if dis_to_corner < 2. * my_cl_r * (math.sqrt(2) - 1):
                                move_eat_volume = 0.
                                break

                    if my_cl_can_split:
                        split_eat_enemy_volume = util.get_split_eat_enemy_volumn(my_ball, enemy_cl,
                                                                                 self.player_clone_np, my_clone_balls,
                                                                                 my_going_merge, friend_nearby,
                                                                                 enemy_nearby, th_nearby)
                    else:
                        split_eat_enemy_volume = 0.0
                else:
                    direction = (my_cl_pos - enemy_cl_pos).normalize()  # back
                    move_eat_volume = my_cl_v
                    split_eat_enemy_volume = 0.0

                    # enemy split once
                    if merge_adjust_v / 2 > my_cl_v and my_to_enemy_dis < 2.2 * merge_adjust_r + 15:
                        in_danger = util.check_split_eat_by_enemy(my_ball, enemy_cl, self.player_clone_np,
                                                                  my_clone_balls, None, team_clone_balls,
                                                                  enemy_clone_balls, team_clone_balls, split_num=1)
                        if in_danger:
                            move_dead_time = 0.3  # other split attack (real: other may not split)
                    # enemy split twice
                    elif merge_adjust_v / 4 > my_cl_v and my_to_enemy_dis < 3. * merge_adjust_r + 18 and enemy_can_split_twice:
                        in_danger = util.check_split_eat_by_enemy(my_ball, enemy_cl, self.player_clone_np,
                                                                  my_clone_balls, None, team_clone_balls,
                                                                  enemy_clone_balls, team_clone_balls, split_num=2)
                        if in_danger:
                            move_dead_time = 0.4  # other split attack (real: other may not split)
                    # enemy split third
                    elif merge_adjust_v / 8 > my_cl_v and my_to_enemy_dis < 3.5 * merge_adjust_r + 18 and enemy_can_split_third:
                        in_danger = util.check_split_eat_by_enemy(my_ball, enemy_cl, self.player_clone_np,
                                                                  my_clone_balls, None, team_clone_balls,
                                                                  enemy_clone_balls, team_clone_balls, split_num=3)
                        if in_danger:
                            move_dead_time = 0.6

                    if my_cl_can_split:  # can split
                        if enemy_cl_v > my_cl_v and my_to_enemy_dis < enemy_cl_r * 2.2 + 15 and enemy_cl_r > 10.:
                            split_eat_enemy_volume = -my_cl_v
                        elif enemy_cl_v / 2 > my_cl_v and my_to_enemy_dis < enemy_cl_r * 3.0 + 18 and enemy_can_split_twice:
                            split_eat_enemy_volume = -my_cl_v
                            # your split ball will be killed

                # delta_volume / reach_time, delta_volume (2.0) means (my reward volume - target's lost volume)
                # define chase mode
                enemy_to_center = (enemy_cl_pos - Vector2(500., 500.)).length()
                mb_to_center = (my_cl_pos - Vector2(500., 500.)).length()
                enemy_out = enemy_to_center > mb_to_center
                new_pos = enemy_cl_pos + 60 * (enemy_cl_pos - my_cl_pos).normalize()
                enemy_trapped = new_pos.x > 1000 or new_pos.x < 0 or new_pos.y > 1000 or new_pos.y < 0
                enemy_cornered = enemy_to_center > 500.

                my_cl_np = util.item_process_to_np(my_ball).flatten()
                tar_cl_np = util.item_process_to_np(enemy_cl).flatten()
                to_board_dis, to_board_time = self.chase_analyzer.chase_to_border(my_cl_np, tar_cl_np)

                team_chase_dir = util.team_chase_dir(enemy_cl, team_clone_balls)
                chase_index = 2.0
                if team_chase_dir is not None:
                    team_chase_aug = team_chase_dir.x * direction.x + team_chase_dir.y * direction.y
                    if team_chase_aug <= -math.sqrt(2) / 2:
                        chase_index = 1.2
                    elif team_chase_aug <= 0:
                        chase_index = 1.5
                else:
                    if self.cur_time < 90:
                        if enemy_trapped:
                            chase_index = 1.6
                        elif enemy_out:
                            chase_index = 1.8
                        else:
                            chase_index = 2.5
                if n_my_cl == 1:
                    if my_cl_r > enemy_cl_r:
                        if enemy_out and (enemy_cornered or enemy_trapped):
                            # can catch
                            if n_enemy_cl == 1:
                                chase_index = 1.5
                            else:
                                player_merge_cd = self.player_merge_cd[enemy_cl_name]
                                if to_board_time < player_merge_cd:
                                    move_eat_volume = self.player_v[enemy_cl_name]
                                    chase_index = 1.2
                                    # if enemy_idx == 0 and n_enemy_cl >= 5.:
                                    #     can_split_eat = n_my_cl == 1 and my_cl_r > 10. and my_cl_v / 2 > enemy_cl_v + 20.
                                    #
                                    #     if can_split_eat:
                                    #         th_danger = False
                                    #         for th in thorns:
                                    #             th_pos = th['position']
                                    #             th_r = th['radius']
                                    #             th_v = th_r * th_r
                                    #             th_to_enemy_dis = (th_pos - enemy_cl_pos).length()
                                    #             if th_to_enemy_dis < my_to_enemy_dis and my_cl_v / 2 > th_v:
                                    #                 th_danger = True
                                    #                 break
                                    #
                                    #         if not th_danger:
                                    #             split_eat_enemy_volume = enemy_cl_v
                                    #             print('strike %.1f %s' % (self.cur_time, self.name))
                        elif n_enemy_cl == 1:
                            chase_index = 2.5

                move_dead_time = max(move_dead_time, math.pow(move_dead_time, chase_index))
                mv_score = 2.0 * move_eat_volume / move_dead_time
                sp_score = 2.0 * split_eat_enemy_volume / split_dead_time
                if mv_score < 5.0 and 0.0 <= sp_score < 2.0:
                    continue
                # filter low score

                attention_direction.append(direction)
                move_score.append(mv_score)
                split_score.append(sp_score)
                attention_my_balls.append(my_ball)
                attention_other_balls.append(enemy_cl)
                # Don't use deepcopy if not change its elements

                for i in range(len(move_score) - 1, 0, -1):
                    if abs(move_score[i]) > abs(move_score[i - 1]) or (abs(split_score[i]) > abs(split_score[i - 1])):
                        attention_my_balls[i - 1], attention_my_balls[i] = attention_my_balls[i], \
                                                                           attention_my_balls[i - 1]
                        attention_other_balls[i - 1], attention_other_balls[i] = attention_other_balls[i], \
                                                                                 attention_other_balls[i - 1]
                        move_score[i - 1], move_score[i] = move_score[i], move_score[i - 1]
                        split_score[i - 1], split_score[i] = split_score[i], split_score[i - 1]
                        attention_direction[i - 1], attention_direction[i] = attention_direction[i], \
                                                                             attention_direction[i - 1]
                    else:
                        break
                if len(move_score) > 128:
                    move_score.pop()
                    split_score.pop()
                    attention_my_balls.pop()
                    attention_other_balls.pop()
                    attention_direction.pop()
                # sort value pairs by their move scores
                # choose top 128 pairs

            # receive teammates' sos info
            for team_ball in team_help_balls:
                name = abs(int(team_ball['player']))
                team = int(team_ball['team'])
                team_ball_volume = team_ball['radius'] * team_ball['radius']
                distance = (my_ball['position'] - team_ball['position']).length()
                if distance == 0.:
                    continue

                # fake_thorn name:
                fake_thorn = isinstance(team_ball['player'], int)
                if not fake_thorn:
                    continue
                    if name == int(self.name) or team != int(self.team) or len(
                            self.player_clone[name]) == 1 or team_ball_volume > my_cl_v:
                        # help teammates only my total volume >= 2000.0
                        # can't help if you have only one ball
                        continue
                else:
                    if self.cur_time >= 60. or name != int(self.name) or team != int(self.team):
                        continue
                    # print("merge help %.1f %s" % (self.cur_time, name))

                direction = (team_ball['position'] - my_ball['position']).normalize()

                distance = (my_ball['position'] - team_ball['position']).length()
                dead_distance = max(my_ball['radius'], team_ball['radius'])
                move_dead_time = max((distance - dead_distance) * (dead_distance + 10.0) / 500.0, 0.1)
                chase_index = 2
                move_dead_time = max(move_dead_time, math.pow(move_dead_time, chase_index))
                # move_dead_time = max(move_dead_time, math.pow(move_dead_time, 1.2))
                # help teammates need less time than catch enemies
                move_eat_volume = team_ball_volume
                # may save the team ball's volume
                mv_score = 2.0 * move_eat_volume / move_dead_time
                sp_score = 0.0
                # ignore split action

                team_ball_copy = copy.deepcopy(team_ball)
                team_ball_copy['radius'] = my_ball['radius'] * 0.01
                # consider team ball smaller than me
                if mv_score < 2.0 and 0.0 <= sp_score < 2.0:
                    continue

                attention_direction.append(direction)
                move_score.append(mv_score)
                split_score.append(sp_score)
                attention_my_balls.append(my_ball)
                attention_other_balls.append(team_ball_copy)

                for i in range(len(move_score) - 1, 0, -1):
                    if move_score[i] > move_score[i - 1]:
                        attention_my_balls[i - 1], attention_my_balls[i] = attention_my_balls[i], \
                                                                           attention_my_balls[i - 1]
                        attention_other_balls[i - 1], attention_other_balls[i] = attention_other_balls[i], \
                                                                                 attention_other_balls[i - 1]
                        move_score[i - 1], move_score[i] = move_score[i], move_score[i - 1]
                        split_score[i - 1], split_score[i] = split_score[i], split_score[i - 1]
                        attention_direction[i - 1], attention_direction[i] = attention_direction[i], \
                                                                             attention_direction[i - 1]
                    else:
                        break
                if len(move_score) > 128:
                    move_score.pop()
                    split_score.pop()
                    attention_my_balls.pop()
                    attention_other_balls.pop()
                    attention_direction.pop()

            # first score-friend
            # for friend_cl in []:
            for friend_cl in team_clone_balls:
                friend_player = int(friend_cl['player'])
                friend_player_idx = self.player_clone[friend_player].index(friend_cl)
                assert friend_player_idx >= 0
                n_friend_cl = self.player_n[friend_player]

                # # eat friend small
                # if my_total_v >= friend_total_v:
                #     continue

                friend_cl_pos = friend_cl['position']
                friend_cl_r = friend_cl['radius']
                friend_cl_v = friend_cl_r * friend_cl_r
                my_to_friend_dis = (my_cl_pos - friend_cl_pos).length()
                dead_distance = max(my_cl_r, friend_cl_r)
                move_spd = 500. / (10 + my_cl_r)
                move_dead_time = max((my_to_friend_dis - dead_distance) / move_spd, 0.1)
                move_dead_time = max(move_dead_time, math.pow(move_dead_time, 2.0))
                if friend_cl_pos == my_cl_pos:
                    continue
                direction = (friend_cl_pos - my_cl_pos).normalize()

                mv_score = 0.0
                sp_score = 0.0
                # ==== move relation with friend ====
                # eat small friend
                eat_small_friend = True
                move_merge_friend = False
                split_merge_friend = True

                if eat_small_friend:
                    if my_cl_r > friend_cl_r:
                        if self.weak and friend_player_idx >= 14:
                            move_eat_volume = friend_cl_v
                            mv_score = move_eat_volume / move_dead_time
                        # # far friend
                        # else:
                        #     temp_friend_clones = self.player_clone[friend_player]
                        #     friend_center = self.player_center[friend_player]
                        #     if friend_center is not None:
                        #         dis_to_center = [(b['position'] - friend_center).length() for b in temp_friend_clones]
                        #         dis_to_center_median = float(np.median(dis_to_center))
                        #         dis_friend_cl_to_center = (friend_cl['position'] - friend_center).length()
                        #         if dis_friend_cl_to_center > 2 * dis_to_center_median:
                        #             move_eat_volume = friend_cl_v
                        #             mv_score = move_eat_volume / move_dead_time

                if move_merge_friend:
                    # move feed friend-> split eat
                    if my_cl_r < friend_cl_r:
                        merge_cl_idx = friend_player_idx
                        n_merge = n_friend_cl
                        merge_cl_x = my_cl_pos.x + direction.x * (my_to_friend_dis - dead_distance)
                        merge_cl_y = my_cl_pos.y + direction.y * (my_to_friend_dis - dead_distance)

                        merge_cl_v = my_cl_v + friend_cl_v
                        merge_cl_r = math.sqrt(merge_cl_v)
                        merge_can_split = (merge_cl_idx + n_merge < 16) and merge_cl_r > 10
                        weight = my_cl_v / merge_cl_v

                        if merge_can_split:
                            for enemy_cl in enemy_clone_balls:
                                enemy_cl_pos = enemy_cl['position']
                                enemy_cl_r = enemy_cl['radius']
                                enemy_cl_v = enemy_cl_r * enemy_cl_r
                                merge_to_enemy_dis = math.sqrt(
                                    math.pow(merge_cl_x - enemy_cl_pos.x, 2) + math.pow(merge_cl_y - enemy_cl_pos.y, 2))

                                # only after merge can split eat
                                if (merge_cl_v / 2 > enemy_cl_v) and (merge_to_enemy_dis < 2.0 * merge_cl_r):
                                    mv_eat_score = min(enemy_cl_v * weight, my_cl_v) / (
                                                move_dead_time + split_dead_time)
                                    mv_score = max(mv_score, mv_eat_score)

                if split_merge_friend:
                    # split merge friend-> split eat
                    if my_cl_can_split and (my_to_friend_dis < my_cl_r * 2.12 + 15) and my_idx == 0:
                        if (my_cl_v / 2 > friend_cl_v and n_friend_cl > 1) or my_cl_v / 2 < friend_cl_v:
                            # eat friend
                            if my_cl_v / 2 > friend_cl_v and n_friend_cl > 1:
                                merge_cl_x = my_cl_pos.x + direction.x * 1.414 * my_cl_r
                                merge_cl_y = my_cl_pos.y + direction.y * 1.414 * my_cl_r
                            # feed friend
                            else:
                                merge_cl_x = friend_cl_pos.x
                                merge_cl_y = friend_cl_pos.y

                            merge_cl_v = my_cl_v / 2 + friend_cl_v
                            merge_cl_r = math.sqrt(merge_cl_v)
                            merge_cl_x = np.clip(merge_cl_x, merge_cl_r, 1000. - merge_cl_r)
                            merge_cl_y = np.clip(merge_cl_y, merge_cl_r, 1000. - merge_cl_r)

                            split_eat_enemy_volume = 0.0
                            # first split
                            split_danger = False

                            for enemy_b in enemy_nearby:
                                fake_dis = math.sqrt(
                                    math.pow(merge_cl_x - enemy_b['position'].x, 2) + math.pow(
                                        merge_cl_y - enemy_b['position'].y, 2))
                                enemy_b_r = enemy_b['radius']
                                # collide bigger
                                if enemy_b_r > merge_cl_r and fake_dis < enemy_b_r:
                                    split_danger = True
                                    break

                            if (not split_danger) and my_cl_v / 2 > friend_cl_v:
                                for friend_b in team_clone_balls:
                                    fake_dis = math.sqrt(
                                        math.pow(merge_cl_x - friend_b['position'].x, 2) + math.pow(
                                            merge_cl_y - friend_b['position'].y, 2))
                                    friend_b_r = friend_b['radius']
                                    friend_b_v = friend_b_r * friend_b_r
                                    # collide bigger
                                    if (merge_cl_r > fake_dis) and (merge_cl_r > friend_b_r):
                                        merge_cl_v += friend_b_v
                                        # if n_friend_cl == 16:
                                        #     split_eat_volume += friend_b_v

                            for enemy_b in enemy_clone_balls:
                                if split_danger:
                                    break
                                fake_dis = math.sqrt(
                                    math.pow(merge_cl_x - enemy_b['position'].x, 2) + math.pow(
                                        merge_cl_y - enemy_b['position'].y, 2))
                                enemy_b_r = enemy_b['radius']
                                enemy_b_v = enemy_b_r * enemy_b_r
                                if enemy_b_v / 2 > merge_cl_v and fake_dis < 2.2 * enemy_b_r + 15:
                                    split_danger = True
                                    break
                                elif (merge_cl_r > fake_dis) and (merge_cl_r > enemy_b_r):
                                    merge_cl_v += enemy_b_v
                                    split_eat_enemy_volume = max(split_eat_enemy_volume, enemy_b_v)
                                elif my_cl_v / 2 > friend_cl_v and (merge_cl_r * 2.1 > fake_dis) and (
                                        merge_cl_v / 2 > enemy_b_v):
                                    merge_cl_v += enemy_b_v
                                    split_eat_enemy_volume = max(split_eat_enemy_volume, enemy_b_v)

                            if not split_danger:
                                if my_cl_v / 4 > split_eat_enemy_volume:
                                    sp_score = 0
                                else:
                                    if my_cl_v / 2 > friend_cl_v:
                                        weight = 1.
                                    else:
                                        weight = 0.5 * my_cl_v / merge_cl_v
                                    sp_score = (split_eat_enemy_volume * weight) / split_dead_time

                if mv_score == 0 and sp_score == 0:
                    continue

                attention_direction.append(direction)
                move_score.append(mv_score)
                split_score.append(sp_score)
                attention_my_balls.append(my_ball)
                attention_other_balls.append(friend_cl)

                for i in range(len(move_score) - 1, 0, -1):
                    if (move_score[i] > move_score[i - 1]) or (abs(split_score[i]) > abs(split_score[i - 1])):
                        attention_my_balls[i - 1], attention_my_balls[i] = attention_my_balls[i], \
                                                                           attention_my_balls[i - 1]
                        attention_other_balls[i - 1], attention_other_balls[i] = attention_other_balls[i], \
                                                                                 attention_other_balls[i - 1]
                        move_score[i - 1], move_score[i] = move_score[i], move_score[i - 1]
                        split_score[i - 1], split_score[i] = split_score[i], split_score[i - 1]
                        attention_direction[i - 1], attention_direction[i] = attention_direction[i], \
                                                                             attention_direction[i - 1]
                    else:
                        break
                if len(move_score) > 128:
                    move_score.pop()
                    split_score.pop()
                    attention_my_balls.pop()
                    attention_other_balls.pop()
                    attention_direction.pop()

        final_move_score = [0.0] * len(move_score)
        final_split_score = [0.0] * len(split_score)

        # second round scoring, each value pair will influence other pairs
        for i in range(0, len(move_score)):
            other_record = dict()
            my_record = dict()

            if attention_my_balls[i]['radius'] > attention_other_balls[i]['radius']:
                key = util.item_to_str(attention_other_balls[i])
                other_record[key] = [move_score[i], split_score[i]]  # get score
            else:
                key = util.item_to_str(attention_my_balls[i])
                my_record[key] = [move_score[i], split_score[i]]  # lose score
            for j in range(0, len(move_score)):
                if i != j:
                    correlation = attention_direction[j].x * attention_direction[i].x + attention_direction[j].y * \
                                  attention_direction[i].y
                    related_move_score = correlation * move_score[j]
                    related_split_score = 0.0
                    if correlation >= 0.9:
                        related_split_score = split_score[j]
                    elif split_score[j] < 0:
                        related_split_score = split_score[j]
                    if correlation < 0.0 and attention_my_balls[j]['radius'] < attention_other_balls[j]['radius'] and \
                            (attention_my_balls[j]['position'] - attention_other_balls[j]['position']).length() < 1.5 * \
                            attention_my_balls[j]['radius'] + 2.1 * attention_other_balls[j]['radius']:
                        related_split_score = - 2.0 * attention_my_balls[j]['radius'] * attention_my_balls[j][
                            'radius'] / 0.2

                    if attention_my_balls[j]['radius'] > attention_other_balls[j]['radius']:
                        key = util.item_to_str(attention_other_balls[j])
                        rec = other_record.get(key, None)
                        if rec is None:
                            other_record[key] = [related_move_score, related_split_score]
                        else:
                            other_record[key][0] = max(other_record[key][0], related_move_score)
                            if related_split_score < 0.0:
                                other_record[key][1] = min(other_record[key][1], related_split_score)

                    else:
                        key = util.item_to_str(attention_my_balls[j])
                        rec = my_record.get(key, None)
                        if rec is None:
                            my_record[key] = [related_move_score, related_split_score]
                        else:
                            my_record[key][0] = max(my_record[key][0], related_move_score)
                            if related_split_score < 0.0:
                                my_record[key][1] = min(my_record[key][1], related_split_score)

            # selected atk
            src_to_center_dir = my_center - attention_my_balls[i]['position']
            tar_dir = attention_other_balls[i]['position'] - attention_my_balls[i]['position']
            same_dir = src_to_center_dir.x * tar_dir.x + src_to_center_dir.y + tar_dir.y > 0
            if False and attention_my_balls[i]['radius'] > attention_other_balls[i]['radius'] and (not same_dir):
                final_move_score[i] = 0
            else:
                final_move_score[i] += sum([rec[0] for rec in my_record.values()])
                final_move_score[i] += sum([rec[0] for rec in other_record.values()])

            final_split_score[i] += sum([rec[1] for rec in my_record.values()])
            final_split_score[i] += sum([rec[1] for rec in other_record.values()])

        # final selection
        if len(move_score) > 0:
            best_move_id = final_move_score.index(max(final_move_score))
            mv_my_ball = attention_my_balls[best_move_id]
            mv_target_ball = attention_other_balls[best_move_id]
            mv_score = final_move_score[best_move_id]
            mv_direction = attention_direction[best_move_id]

            best_split_id = final_split_score.index(max(final_split_score))
            sp_my_ball = attention_my_balls[best_split_id]
            sp_target_ball = attention_other_balls[best_split_id]
            sp_score = final_split_score[best_split_id]
            sp_direction = attention_direction[best_split_id]

            if mv_score >= sp_score:
                final_action = -1
                final_direction = mv_direction
                final_score = mv_score
                lead_my_ball = mv_my_ball
                if mv_my_ball['radius'] < mv_target_ball['radius']:
                    mb = mv_my_ball
                    ob = mv_target_ball
                    if self.cur_time < 120 and self.player_v[int(self.name)] < 10000:
                        final_direction = self.smart_escape_pd(mb, final_direction, enemy_clone_balls,
                                                               ob, my_clone_balls, thorns)
                else:
                    help_ball = copy.deepcopy(mv_target_ball)
            else:
                final_action = 4
                final_direction = sp_direction
                final_score = sp_score
                lead_my_ball = sp_my_ball

        return lead_my_ball, final_direction, final_action, final_score, help_ball

    def process_clone_balls(self, clone_balls):
        my_clone_balls = []
        others_clone_balls = []
        team_clone_balls = []
        for clone_ball in clone_balls:
            if clone_ball['player'] == self.name:
                my_clone_balls.append(clone_ball)
            elif clone_ball['team'] != self.team:
                others_clone_balls.append(clone_ball)
            else:
                team_clone_balls.append(clone_ball)
        my_clone_balls.sort(key=lambda a: a['radius'], reverse=True)
        others_clone_balls.sort(key=lambda a: a['radius'], reverse=True)
        team_clone_balls.sort(key=lambda a: a['radius'], reverse=True)

        return my_clone_balls, others_clone_balls, team_clone_balls

    def process_thorns_balls(self, thorns_balls, my_clone_balls, mode=0):
        """

        :param thorns_balls:
        :param my_clone_balls:
        :param mode:0-normal 1:avoid_same_thorn
        :return:
        """
        target_thorns_ball = None
        source_my_ball = None
        target_id = -1
        source_id = -1
        thorn_score = 0.0

        action_type = -1
        direction = Vector2(0.1, 0.1).normalize()
        score_rec = []

        for tb in range(0, len(thorns_balls)):
            for mb in range(0, len(my_clone_balls)):
                if thorns_balls[tb]['radius'] < my_clone_balls[mb]['radius']:
                    thorns_ball_volume = thorns_balls[tb]['radius'] * thorns_balls[tb]['radius']
                    distance = (thorns_balls[tb]['position'] - my_clone_balls[mb]['position']).length() - \
                               my_clone_balls[mb]['radius']
                    spd = 500. / (my_clone_balls[mb]['radius'] + 10.0)
                    eat_time = max(0.1, distance) / spd
                    score = thorns_ball_volume / eat_time

                    if mode == 0:
                        if score > thorn_score:
                            thorn_score = score
                            target_id = tb
                            source_id = mb
                    else:
                        player_id = my_clone_balls[mb]['player']
                        score_rec.append([int(player_id), tb, score])

        if target_id != -1:
            source_my_ball = my_clone_balls[source_id]
            target_thorns_ball = thorns_balls[target_id]
            direction = (target_thorns_ball['position'] - source_my_ball['position']).normalize()

        if mode == 0:
            return source_my_ball, direction, action_type, thorn_score
        else:
            return score_rec

    def process_thorns_balls_np(self, total_clone, thorns, mode=0):
        if thorns is None:
            return None, None, None, 0.0

        player_id = int(self.name)
        team_id = int(self.team)
        my_clone = total_clone[total_clone[:, -2] == player_id].reshape(-1, 5)
        enemy_clone = total_clone[total_clone[:, -1] != team_id].reshape(-1, 5)
        n_my_cl = my_clone.shape[0]

        th_score_rec = []
        for idx, my_cl in enumerate(my_clone):
            my_cl_x, my_cl_y, my_cl_r = my_cl[:3]
            my_spd = util.get_spd(my_cl_r)

            thorn_eat = thorns[my_cl_r > thorns[:, 2]]
            if thorn_eat.size == 0:
                continue

            for th in thorn_eat:
                th_x, th_y, th_r = th
                th_v = th_r * th_r

                dt_x = th[0] - my_cl[0]
                dt_y = th[1] - my_cl[1]
                dis = math.sqrt(math.pow(dt_x, 2) + math.pow(dt_y, 2))

                my_eat_info = self.chase_analyzer.chase_analyze(my_cl, th)
                my_eat_time = my_eat_info.spend_time
                th_score_raw = th_v / my_eat_time
                enemy_can_eat = enemy_clone[enemy_clone[:, 2] > th_r]

                if enemy_can_eat.size == 0 or dis > min(100, 2.2 * my_cl_r):
                    th_score = th_score_raw
                else:
                    enemy_dis = util.get_dis(enemy_can_eat[:, 0], enemy_can_eat[:, 1], th_x, th_y)
                    enemy_spd = util.get_spd(enemy_can_eat[:, 2])
                    enemy_spend_time = enemy_dis / enemy_spd
                    enemy_cl = enemy_can_eat[np.argmin(enemy_spend_time)]
                    enemy_player_id = enemy_cl[-2]

                    enemy_player_clone = self.player_clone_np[enemy_player_id]
                    enemy_idx = 0

                    for j, temp_cl in enumerate(enemy_player_clone):
                        if enemy_cl[0] == temp_cl[0] and enemy_cl[1] == temp_cl[1]:
                            enemy_idx = j

                    n_enemy_cl = enemy_player_clone.shape[0]

                    enemy_eat_info = self.chase_analyzer.chase_analyze(enemy_cl, th)
                    enemy_eat_time = enemy_eat_info.spend_time

                    # faster
                    if my_eat_time < enemy_eat_time + 0.1:
                        fast_cl = my_cl
                        slow_cl = enemy_cl
                        n_fast_cl = n_my_cl
                        n_slow_cl = n_enemy_cl
                        slow_idx = enemy_idx
                    # slower
                    else:
                        fast_cl = enemy_cl
                        slow_cl = my_cl
                        n_fast_cl = n_enemy_cl
                        n_slow_cl = n_my_cl
                        slow_idx = idx

                    fast_info = self.chase_analyzer.chase_analyze(fast_cl, th)
                    slow_info = self.chase_analyzer.chase_analyze(slow_cl, th)
                    fast_cl_x, fast_cl_y, fast_cl_r = fast_cl[:3]
                    fast_cl_v = fast_cl_r * fast_cl_r
                    fast_final_x, fast_final_y = fast_info.merge_cl[:2]
                    fast_spend_time = fast_info.spend_time

                    slow_cl_x, slow_cl_y, slow_cl_r = slow_cl[:3]
                    slow_cl_v = slow_cl_r * slow_cl_r
                    slow_final_x, slow_final_y = slow_info.merge_cl[:2]
                    slow_spend_time = slow_info.spend_time

                    slow_moment_x = slow_cl_x + (slow_final_x - slow_cl_x) * (fast_spend_time / slow_spend_time)
                    slow_moment_y = slow_cl_y + (slow_final_y - slow_cl_y) * (fast_spend_time / slow_spend_time)
                    slow_dis_moment = util.get_dis(slow_moment_x, slow_moment_y, fast_final_x, fast_final_y)
                    collide_before_eat_th = slow_dis_moment < max(fast_cl_r, slow_cl_r)

                    split_n = min(16 - n_fast_cl, 10)
                    merge_v = fast_cl_v + th_r * th_r
                    split_r = min(math.sqrt(merge_v / (split_n + 1)), 20)
                    split_v = split_r * split_r
                    middle_v = merge_v - split_v * split_n

                    if collide_before_eat_th:
                        if fast_cl_r > slow_cl_r:
                            faster_win = True
                        else:
                            faster_win = False
                    # can not fansha:no split
                    elif split_n == 0 and merge_v > slow_cl_v:
                        faster_win = True
                    # can not fansha: slower too small
                    elif split_n > 0 and (slow_cl_v < split_v or slow_cl_v + split_v < middle_v / 2):
                        faster_win = True
                    # may fansha
                    else:
                        # fansha 1
                        dis_free_move = 15 + 500. / (10 + slow_cl_r)
                        if slow_dis_moment < slow_cl_r + dis_free_move:
                            faster_win = False
                        # fansha 2
                        elif slow_dis_moment <= slow_cl_r * 2.12 + dis_free_move:
                            if slow_idx + n_slow_cl < 16:
                                faster_win = False
                            else:
                                faster_win = True
                        # no fight
                        else:
                            faster_win = True

                    if faster_win:
                        if my_eat_time < enemy_eat_time:
                            th_score = th_score_raw
                        else:
                            th_score = 0.0
                    else:
                        if my_eat_time < enemy_eat_time:
                            th_score = 0.0
                        else:
                            th_score = th_score_raw
                        # # no fansha
                        # if my_eat_time < enemy_eat_time + 0.1:
                        #     if slow_dis_moment < slow_cl_r + 15 + 500./(10+slow_cl_r):
                        #         th_score = 0.0
                        #     elif enemy_idx + n_enemy_cl < 16 and slow_dis_moment <= slow_cl_r * 2.2 + 15 + 500./(10+slow_cl_r):
                        #         th_score = 0.0
                        #     else:
                        #         th_score = th_score_raw
                        # # slower
                        # else:
                        #     if slow_dis_moment < slow_cl_r + 15:
                        #         th_score = th_score_raw * 1.5
                        #     elif idx + n_my_cl < 16 and slow_dis_moment <= slow_cl_r * 2.12 + 15:
                        #         th_score = th_score_raw
                        #     else:
                        #         th_score = 0.0

                action = -1
                direction = Vector2(dt_x, dt_y)
                th_score_rec.append([my_cl, direction, action, th_score])

        th_score = [r[-1] for r in th_score_rec]
        if th_score:
            th_score_max = max(th_score)
            best_rec_idx = th_score.index(th_score_max)
            res = th_score_rec[best_rec_idx]
            return res
        else:
            return None, None, None, 0.0

    # for i in range(split_num):
    #     angle = 2*math.pi*(i+1)/split_num
    #     unit_x = math.cos(angle)
    #     unit_y = math.sin(angle)
    #     vel = Vector2(self.split_vel_init*unit_x, self.split_vel_init*unit_y)
    #     acc = - Vector2(self.split_acc_init*unit_x, self.split_acc_init*unit_y)
    #     around_position = self.position + Vector2((self.radius+around_radius)*unit_x, (self.radius+around_radius)*unit_y)
    #     around_vels.append(vel)
    #     around_accs.append(acc)
    #     around_positions.append(around_position)
    def process_thorns_balls_pro(self, total_clone, thorns):
        if thorns is None or self.weak:
            return None, None, None, 0.0

        player_id = int(self.name)
        team_id = int(self.team)
        my_clone = total_clone[total_clone[:, -2] == player_id].reshape(-1, 5)
        enemy_clone = total_clone[total_clone[:, -1] != team_id].reshape(-1, 5)
        n_my_cl = my_clone.shape[0]

        if n_my_cl > 1:
            my_clone = my_clone[np.argsort(-my_clone[:, 2])]

        th_score_rec = []
        th_ignore = []
        for idx, my_cl in enumerate(my_clone):
            my_cl_x, my_cl_y, my_cl_r = my_cl[0:3]
            my_cl_v = my_cl_r * my_cl_r

            for th_idx, th in enumerate(thorns):
                if th_idx in th_ignore:
                    continue
                th_x, th_y, th_r = th
                th_v = th_r * th_r

                dt_x = th[0] - my_cl[0]
                dt_y = th[1] - my_cl[1]
                my_cl_to_th_dis = math.sqrt(math.pow(dt_x, 2) + math.pow(dt_y, 2))

                my_eat_info = self.chase_analyzer.chase_analyze(my_cl, th)
                my_eat_time = my_eat_info.spend_time
                th_score_raw = th_v / my_eat_time if my_cl_r > th_r else 0.0

                # ignore too far
                if n_my_cl == 1 and th_score_raw > 0.0:
                    th_too_far = False
                    for t in thorns:
                        t_x, t_y, t_r = t
                        t_v = t_r * t_r
                        dt_v = t_v - my_cl_v
                        if dt_v <= 0.0:
                            continue
                        farm_time = dt_v / self.avg_farm_spd
                        t_eat_info = self.chase_analyzer.chase_analyze(my_cl, t)
                        t_eat_time = t_eat_info.spend_time
                        farm_score = (t_v + dt_v) / (farm_time * 2 + t_eat_time)
                        if farm_score > th_score_raw:
                            th_too_far = True
                            break
                    if th_too_far:
                        continue

                # ignore hide in corder
                in_corner = False
                if math.sqrt(2) * th_r < my_cl_r * (math.sqrt(2) - 1):
                    th_pos = Vector2(th_x, th_y)
                    for corner_pos in [Vector2(0., 0.), Vector2(0., 1000.), Vector2(1000., 0.), Vector2(1000., 1000.)]:
                        dis_to_corner = (th_pos - corner_pos).length()
                        if dis_to_corner < my_cl_r * (math.sqrt(2) - 1):
                            in_corner = True
                            break

                # only consider my clones can split and n_my_cl < 16
                if my_cl_r < th_r or in_corner:
                    continue

                enemy_can_eat = enemy_clone[enemy_clone[:, 2] > th_r]
                # no conflict
                if n_my_cl > 1 or enemy_can_eat.size == 0:
                    th_score = th_score_raw
                # single cl
                elif n_my_cl == 1:
                    enemy_to_th_dis = util.get_dis(enemy_can_eat[:, 0], enemy_can_eat[:, 1], th_x, th_y)
                    enemy_spd = util.get_spd(enemy_can_eat[:, 2])
                    enemy_spend_time = enemy_to_th_dis / enemy_spd
                    enemy_spend_time_min = np.min(enemy_spend_time)
                    enemy_cl = enemy_can_eat[np.argmin(enemy_spend_time)]
                    enemy_cl_r = enemy_cl[2]
                    enemy_cl_to_th_dis = util.get_dis(enemy_cl[0], enemy_cl[1], th_x, th_y)

                    # if can eat ignore long distance influence
                    my_cl_danger_dis = 2.2 * enemy_cl_r + my_cl_r + 6 + 15
                    enemy_cl_danger_dis = 2.12 * my_cl_r + enemy_cl_r + 9 + 15

                    # no conflict
                    if enemy_cl_to_th_dis > my_cl_danger_dis and my_cl_to_th_dis > enemy_cl_danger_dis:
                        th_score = th_score_raw
                    else:
                        # if can eat ignore long distance influence
                        enemy_cl_v = enemy_cl_r * enemy_cl_r
                        enemy_player_id = enemy_cl[-2]

                        enemy_player_clone = self.player_clone_np[enemy_player_id]
                        n_enemy_cl = enemy_player_clone.shape[0]
                        enemy_idx = 0

                        for j, temp_cl in enumerate(enemy_player_clone):
                            if enemy_cl[0] == temp_cl[0] and enemy_cl[1] == temp_cl[1]:
                                enemy_idx = j

                        enemy_eat_info = self.chase_analyzer.chase_analyze(enemy_cl, th)
                        enemy_eat_time = enemy_eat_info.spend_time

                        # faster
                        if my_eat_time < enemy_eat_time:
                            fast_cl = my_cl
                            slow_cl = enemy_cl
                            n_fast_cl = n_my_cl
                            n_slow_cl = n_enemy_cl
                            slow_idx = enemy_idx
                        # slower
                        else:
                            fast_cl = enemy_cl
                            slow_cl = my_cl
                            n_fast_cl = n_enemy_cl
                            n_slow_cl = n_my_cl
                            slow_idx = idx

                        fast_info = self.chase_analyzer.chase_analyze(fast_cl, th)
                        slow_info = self.chase_analyzer.chase_analyze(slow_cl, th)
                        fast_cl_x, fast_cl_y, fast_cl_r = fast_cl[:3]
                        fast_cl_v = fast_cl_r * fast_cl_r
                        fast_final_x, fast_final_y = fast_info.merge_cl[:2]
                        fast_spend_time = fast_info.spend_time

                        slow_cl_x, slow_cl_y, slow_cl_r = slow_cl[:3]
                        slow_cl_v = slow_cl_r * slow_cl_r
                        slow_final_x, slow_final_y = slow_info.merge_cl[:2]
                        slow_spend_time = slow_info.spend_time

                        slow_moment_x = slow_cl_x + (slow_final_x - slow_cl_x) * (fast_spend_time / slow_spend_time)
                        slow_moment_y = slow_cl_y + (slow_final_y - slow_cl_y) * (fast_spend_time / slow_spend_time)
                        slow_dis_moment = util.get_dis(slow_moment_x, slow_moment_y, fast_final_x, fast_final_y)
                        collide_before_eat_th = slow_dis_moment < max(fast_cl_r, slow_cl_r)

                        split_n = min(16 - n_fast_cl, 10)
                        merge_v = fast_cl_v + th_r * th_r
                        split_r = min(math.sqrt(merge_v / (split_n + 1)), 20)
                        split_v = split_r * split_r
                        middle_v = merge_v - split_v * split_n

                        faster_win = False
                        bigger_win = slow_cl_v < split_v or slow_cl_v + split_v < middle_v / 2
                        # can not fansha:no split
                        if split_n == 0 and merge_v > slow_cl_v:
                            faster_win = True
                        # can not fansha: slower too small
                        elif split_n > 0 and bigger_win:
                            faster_win = True
                        # may fansha
                        else:
                            # fansha 1
                            split_free_move = 15 + fast_cl_r + split_r
                            slower_free_move = 500. / (10 + slow_cl_r)
                            collide_eat_faster = slow_dis_moment < slow_cl_r + split_free_move + slower_free_move
                            split_eat_faster = slow_dis_moment < slow_cl_r * 2.12 + split_free_move + slower_free_move and slow_cl_r > 10.
                            if (not collide_eat_faster) and (not split_eat_faster):
                                faster_win = True

                        thorn_near_border = (not 0.0 <= th_x - 2 * my_cl_r <= 1000.) or (not 0.0 <= th_y - 2 * my_cl_r <= 1000.)

                        if faster_win:
                            if my_eat_time < enemy_eat_time:
                                th_score = th_score_raw
                            else:
                                th_score = 0.0
                        else:
                            enemy_to_th_dt_x = th_x - enemy_cl[0] + 1e-4
                            enemy_to_th_dt_y = th_y - enemy_cl[1] + 1e-4
                            enemy_to_th_dis = np.sqrt(np.power(enemy_to_th_dt_x, 2) + np.power(enemy_to_th_dt_y, 2))
                            enemy_to_th_dir_x = enemy_to_th_dt_x / enemy_to_th_dis
                            enemy_to_th_dir_y = enemy_to_th_dt_y / enemy_to_th_dis
                            if my_cl_r > th_r:
                                fake_r = max(my_cl_r, th_r) * 1.5
                            else:
                                fake_r = th_r

                            if my_cl_r < enemy_cl_r:
                                tar_pos_x = th_x + enemy_to_th_dir_x * enemy_to_th_dis
                                tar_pos_y = th_y + enemy_to_th_dir_y * enemy_to_th_dis
                            else:
                                tar_pos_x = enemy_cl[0]
                                tar_pos_y = enemy_cl[1]

                            th_to_my_cl_ang = util.vector2angle(my_cl[0] - th[0] + 1e-3, my_cl[1] - th[1])
                            my_cl_to_th_ang = util.vector2angle(th[0] - my_cl[0] + 1e-3, th[1] - my_cl[1])

                            rad_range = np.arccos(fake_r / max(fake_r, my_cl_to_th_dis))
                            ang_range = rad_range * 180. / np.pi

                            th_to_tar_ang = util.vector2angle(tar_pos_x - th[0], tar_pos_y - th[1])
                            dt_ang = abs(th_to_tar_ang - th_to_my_cl_ang)
                            # go qie xian
                            if dt_ang > ang_range:
                                ang_1 = th_to_my_cl_ang - ang_range
                                ang_2 = th_to_my_cl_ang + ang_range

                                if ang_1 < 0:
                                    ang_1 += 360.
                                if ang_2 > 360.:
                                    ang_2 -= 360.

                                dt_ang_1 = min(abs(th_to_tar_ang - ang_1), abs(360. - (th_to_tar_ang - ang_1)))
                                dt_ang_2 = min(abs(th_to_tar_ang - ang_2), abs(360. - (th_to_tar_ang - ang_2)))
                                if dt_ang_1 < dt_ang_2:
                                    ang_select = my_cl_to_th_ang + (90. - ang_range)
                                else:
                                    ang_select = my_cl_to_th_ang - (90. - ang_range)
                                rad_select = ang_select / 180. * np.pi
                                dt_x = np.cos(rad_select)
                                dt_y = np.sin(rad_select)
                            else:
                                dt_x = tar_pos_x - my_cl[0]
                                dt_y = tar_pos_y - my_cl[1]
                            th_score = th_score_raw
                            if thorn_near_border:
                                th_score = 0.0
                else:
                    # update 1
                    # my cl close to th
                    my_cl_pos = Vector2(my_cl_x, my_cl_y)
                    my_cl_spd = util.get_spd(my_cl_r)
                    th_pos = Vector2(th_x, th_y)
                    dir_th_to_my = (my_cl_pos - th_pos).normalize()
                    fake_pos = th_pos + dir_th_to_my * my_cl_r

                    split_n = min(16 - n_my_cl, 10)
                    merge_v = my_cl_v + th_r * th_r
                    split_r = min(math.sqrt(merge_v / (split_n + 1)), 20)
                    split_v = split_r * split_r
                    middle_v = merge_v - split_v * split_n

                    no_eat = False
                    for enemy_cl in enemy_clone:
                        enemy_cl_x = enemy_cl[0]
                        enemy_cl_y = enemy_cl[1]
                        enemy_cl_pos = Vector2(enemy_cl_x, enemy_cl_y)
                        enemy_cl_r = enemy_cl[2]
                        enemy_player = int(enemy_cl[3])
                        enemy_cl_v = enemy_cl_r * enemy_cl_r
                        if enemy_cl_v < split_v:
                            continue
                        enemy_cl_spd = util.get_spd(enemy_cl_r)
                        dis_enemy_to_fake = (fake_pos - enemy_cl_pos).length()
                        dis_enemy_to_th = max(0.1, (enemy_cl_pos - th_pos).length() - enemy_cl_r)
                        dis_my_to_th = max(0.1, (my_cl_pos - th_pos).length() - my_cl_r)
                        t_my_to_th = dis_my_to_th / my_cl_spd
                        t_enemy_to_th = dis_enemy_to_th / enemy_cl_spd

                        if t_my_to_th >= t_enemy_to_th:
                            continue

                        enemy_total_clone = self.player_clone[enemy_player]
                        n_enemy_cl = len(enemy_total_clone)
                        enemy_idx = 0
                        for j, temp_cl in enumerate(enemy_total_clone):
                            if enemy_cl_pos == temp_cl['position']:
                                enemy_idx = j
                                break

                        enemy_can_split = enemy_idx == 0 and enemy_idx + n_enemy_cl < 16 and enemy_cl_r > 10.
                        split_dead = enemy_cl_v > my_cl_v / 2 and enemy_cl_v / 2 + split_v > middle_v
                        # move_dead = enemy_cl_v + split_v > middle_v
                        if enemy_can_split and dis_enemy_to_fake < 2.12 * enemy_cl_r and split_dead:
                            no_eat = True
                            print('no eat' + str(self.name) + str(self.cur_time))
                            break
                        # elif dis_enemy_to_th < 30. and move_dead:
                        #     no_eat = True
                        #     break

                    if no_eat:
                        # th_score = th_score_raw
                        th_score = 0.0
                        th_ignore.append(th_idx)
                    else:
                        th_score = th_score_raw
                action = -1
                direction = Vector2(dt_x, dt_y)
                th_score_rec.append([my_cl, direction, action, th_score])

        th_score = [r[-1] for r in th_score_rec]
        if th_score:
            th_score_max = max(th_score)
            best_rec_idx = th_score.index(th_score_max)
            res = th_score_rec[best_rec_idx]
            return res
        else:
            return None, None, None, 0.0

    def process_gather_thorns(self, my_clone_balls, team_balls, thorns, food_balls):
        thorns_gather = []
        balls_valid = []
        mb_valid = []
        tb_valid = []
        gather_target = None
        reward_max = 0
        my_cl_closet = my_clone_balls[0]
        mine_large_radius = max(my_clone_balls[0]['radius'], team_balls[0]['radius'])
        mb_closest_dis = 200
        num_all_team_balls = len(my_clone_balls) + len(team_balls)
        team_almost_full = False
        thorns_around = 0
        lack_of_food = len(food_balls) < 80
        # thorns around
        for thorn in thorns:
            thorns_around_temp = 0
            dis_to_mb = (my_clone_balls[0]['position'] - thorn['position']).length()
            if dis_to_mb < 120:
                thorns_around += 1
        for i in range(0, 3):
            team_num = self.player_n[int(self.team) * 3 + i]
            if team_num >= 15:
                team_almost_full = True
        if team_almost_full and thorns_around >= 5:
            detect_dis_max = 200
        elif num_all_team_balls > 30 and thorns_around >= 3:
            detect_dis_max = 150
        elif lack_of_food:
            detect_dis_max = 60
        else:
            detect_dis_max = 10
        # Detect gather thorns
        for thorn in thorns:
            dis_to_thorns = (my_clone_balls[0]['position'] - thorn['position']).length()
            mb_smaller = my_clone_balls[0]['radius'] < thorn['radius']
            team_smaller = team_balls[0]['radius'] < thorn['radius']
            if mb_smaller and team_smaller and dis_to_thorns < detect_dis_max:
                thorns_gather.append(thorn)
        # Find Possibility
        if thorns_gather:
            for tb_target in thorns_gather:
                together_volume = 0
                mb_far = my_clone_balls[0]
                mb_far_dis = 0
                tb_far = team_balls[0]
                tb_far_dis = 0
                for mb in my_clone_balls:
                    dis_to_thorns = (mb['position'] - tb_target['position']).length()
                    if dis_to_thorns < detect_dis_max:
                        mb_valid.append(mb)
                    if dis_to_thorns < mb_far_dis:
                        mb_far = mb
                        mb_far_dis = dis_to_thorns
                for team in team_balls:
                    dis_to_thorns = (team['position'] - tb_target['position']).length()
                    if dis_to_thorns < detect_dis_max:
                        tb_valid.append(team)
                    if dis_to_thorns < tb_far_dis:
                        tb_far = team
                        tb_far_dis = dis_to_thorns
                not_enough_num = len(mb_valid) <= 1 or len(tb_valid) <= 1
                if not_enough_num:
                    break
                else:
                    no_coop = tb_valid[0]['radius'] < mb_valid[-1]['radius'] or \
                          mb_valid[0]['radius'] < tb_valid[-1]['radius']
                    if no_coop:
                        break
                if len(mb_valid) == 1 and len(tb_valid) > 1:
                    balls_valid.append(mb_valid[0])
                    for tb in tb_valid[1:]:
                        balls_valid.append(tb)
                    if tb_far in balls_valid:
                        balls_valid.remove(tb_far)
                elif len(mb_valid) > 1 and len(tb_valid) == 1:
                    balls_valid.append(tb_valid[0])
                    for mb in mb_valid[1:]:
                        balls_valid.append(mb)
                    if mb_far in balls_valid:
                        balls_valid.remove(mb_far)
                elif len(mb_valid) > 1 and len(tb_valid) > 1:
                    for mb in mb_valid[1:]:
                        balls_valid.append(mb)
                    for tb in tb_valid[1:]:
                        balls_valid.append(tb)
                    if mb_far in balls_valid:
                        balls_valid.remove(mb_far)
                    if tb_far in balls_valid:
                        balls_valid.remove(tb_far)
                # Find Closest Ball
                for ball in balls_valid:
                    together_volume += ball['radius'] * ball['radius']
                    dis_to_tb = (ball['position'] - tb_target['position']).length()
                    if ball and ball['player'] == self.name and dis_to_tb < mb_closest_dis:
                        if mb_closest_dis < dis_to_tb:
                            mb_closest_dis = dis_to_tb
                            my_cl_closet = ball
                # Together
                if together_volume > tb_target['radius'] * tb_target['radius']:
                    reward = tb_target['radius'] * tb_target['radius']
                    if reward > reward_max:
                        reward_max = reward
                        gather_target = tb_target
        # Final decision
        if gather_target:
            if len(mb_valid) > 0 and len(tb_valid) > 0 and mb_valid[0]['radius'] < tb_valid[0]['radius']:
                direction = (tb_valid[0]['position'] - my_cl_closet['position']).normalize()
            else:
                direction = (gather_target['position'] - my_cl_closet['position']).normalize()
            time_spent = mb_closest_dis / 500 * (my_cl_closet['radius'] + 10)
            th_score = reward_max / time_spent
            return my_cl_closet, direction, -1, th_score
        else:
            return None, None, None, 0

    '''
    def edge_direction_process(self, direction, lead_ball):
        center_position = Vector2(500.0, 500.0)
        center_distance = (lead_ball['position'] - center_position).length()
        center_direction = (lead_ball['position'] - center_position).normalize()
        radius = lead_ball['radius']
        new_direction = direction
        correlation = center_direction.x * direction.x + center_direction.y * direction.y

        if correlation > 0.0 and center_distance + radius > 495.0: # moving to the edge, need to modify
            new_y = 1.0
            new_x = - center_direction.y / center_direction.x * (new_y - center_direction.y) + center_direction.x
            new_direction = Vector2(new_x - center_direction.x, new_y - center_direction.y).normalize()
            if new_direction.x * direction.x + new_direction.y * direction.y < 0: # change direction to tangent of the center circle
                new_direction = - new_direction
        return new_direction
    '''

    def process_food_balls(self, food_balls, my_clone_ball):
        target_food_ball = None
        target_id = -1
        target_distance = 0.0
        acc_score = 0.0
        food_score = 0.0

        action_type = -1
        direction = Vector2(0.1, 0.1).normalize()
        for fb in range(0, len(food_balls)):
            distance = (food_balls[fb]['position'] - my_clone_ball['position']).length()
            direction = (food_balls[fb]['position'] - my_clone_ball['position']).normalize()
            correlation = self.last_direction.x * direction.x + self.last_direction.y * direction.y + 1
            if correlation * correlation / distance > acc_score:
                acc_score = correlation * correlation / distance
                target_distance = distance
                target_id = fb

        if target_id != -1:
            target_food_ball = food_balls[target_id]
            eat_time = max(0.1, (target_distance - my_clone_ball['radius']) * (my_clone_ball['radius'] + 10.0) / 500.0)
            food_score = 4 / eat_time
            direction = (target_food_ball['position'] - my_clone_ball['position']).normalize()
        return target_food_ball, direction, action_type, food_score

    def process_food_balls_pd(self, food_balls, my_clone_ball, team_clone_balls, my_clone_balls, other_clone_balls,
                              rect):
        target_food_ball = None
        target_id = -1
        acc_score = 0.0
        food_score = 0.0

        action_type = -1
        direction = Vector2(0.1, 0.1).normalize()
        # Rect stats
        view_size_x = rect[2] - rect[0]
        view_size_y = rect[3] - rect[1]
        # Area Detection
        area_div = 4
        total_num = pow(area_div, 2)
        num_valid = 0
        area_size = Vector2(view_size_x, view_size_y) / area_div
        # Memory
        fd_index = [-1] * len(food_balls)
        area_id = []
        area_fd_num = []
        area_center = []
        # Initialize squares
        for i_x in range(0, area_div):
            for i_y in range(0, area_div):
                _index = i_x * area_div + i_y
                _center = Vector2(rect[0], rect[1]) + Vector2(i_x * area_size.x, i_y * area_size.y) + 0.5 * area_size
                area_center.append(_center)
                area_id.append(_index)
                area_fd_num.append(0)
        # Fill in areas
        for fb_id in range(0, len(food_balls)):
            fb = food_balls[fb_id]
            fd_pos_local = fb['position'] - Vector2(rect[0], rect[1])
            i_x = np.clip(int(fd_pos_local.x / area_size.x), 0, area_div - 1)
            i_y = np.clip(int(fd_pos_local.y / area_size.y), 0, area_div - 1)
            _index = i_x * area_div + i_y
            # different score
            dis_to_center = (fb['position'] - Vector2(500, 500)).length() - my_clone_ball['radius'] - 5
            if self.cur_time < 120 and self.merge_cd > 15:
                punish_min = 475
                punish_max = 700
            else:
                punish_min = 550
                punish_max = 700
            # num_add = (700 - dis_to_center) / 200 + 0.1
            num_add = (punish_max - dis_to_center) / (punish_max - punish_min) + 0.2
            area_fd_num[_index] += np.clip(num_add, 0, 1)
            fd_index[fb_id] = _index
        # Ignore areas around team or opponents
        for tb_id in range(0, len(team_clone_balls)):
            tb = team_clone_balls[tb_id]
            dis_min = 300
            for mb in my_clone_balls:
                if tb['radius'] < mb['radius']:
                    dis_away = 200
                else:
                    dis_away = 100
                dis = (mb['position'] - tb['position']).length() - tb['radius']
                dis_min = min(dis_min, dis)
                if dis_min < dis_away:
                    tb_pos_close = tb['position'] + (my_clone_ball['position'] - tb['position']).normalize() * tb[
                        'radius']
                    td_pos_local = tb_pos_close - Vector2(rect[0], rect[1])
                    i_x = np.clip(int(td_pos_local.x / area_size.x), 0, area_div - 1)
                    i_y = np.clip(int(td_pos_local.y / area_size.y), 0, area_div - 1)
                    _index = i_x * area_div + i_y
                    area_fd_num[_index] *= 0.5

        dis_min_ob = 200
        for ob_id in range(0, len(other_clone_balls)):
            ob = other_clone_balls[ob_id]
            if self.merge_cd > 12:
                dis_away = ob['radius'] + 60
            elif self.merge_cd > 6:
                dis_away = ob['radius'] + 30
            else:
                dis_away = ob['radius'] + 15
            for mb in my_clone_balls:
                if ob['radius'] > mb['radius']:
                    dis_mb_center = (mb['position'] - Vector2(500, 500)).length()
                    dis_ob_center = (mb['position'] - Vector2(500, 500)).length()
                    if dis_mb_center > dis_ob_center:
                        dis = (mb['position'] - ob['position']).length() - ob['radius']
                        dis_min_ob = min(dis_min_ob, dis)
            if dis_min_ob < dis_away:
                ob_pos_close = ob['position'] + (my_clone_ball['position'] - ob['position']).normalize() * ob['radius']
                od_pos_local = ob_pos_close - Vector2(rect[0], rect[1])
                i_x = np.clip(int(od_pos_local.x / area_size.x), 0, area_div - 1)
                i_y = np.clip(int(od_pos_local.y / area_size.y), 0, area_div - 1)
                _index = i_x * area_div + i_y
                area_fd_num[_index] *= 0.8
        # Detect direction size
        size_dir_max = my_clone_ball['radius']
        size_dirs = []
        for _id in area_id:
            _center = area_center[_id]
            _dir = _center - my_clone_ball['position']
            size_dir = self.process_vector_size(my_clone_ball, my_clone_balls, _dir)
            size_dirs.append(size_dir)
            if size_dir > size_dir_max:
                size_dir_max = size_dir
        for _id in area_id:
            if size_dirs[_id] < size_dir_max * 0.5:
                area_fd_num[_id] *= 0.66
            elif size_dirs[_id] < size_dir_max * 0.25:
                area_fd_num[_id] *= 0.33
        # Sort areas
        for i in range(1, total_num):
            if area_fd_num[i] > area_fd_num[i - 1]:
                area_fd_num[i - 1], area_fd_num[i] = area_fd_num[i], area_fd_num[i - 1]
                area_id[i - 1], area_id[i] = area_id[i], area_id[i - 1]
            if area_fd_num[i - 1] > 0:
                num_valid += 1
        # Define good id
        begin_spilt = self.merge_cd > 18
        good_balls = []
        if begin_spilt or len(food_balls) < 120:
            area_scores = area_fd_num[int(num_valid * (1 / 3))]
        else:
            area_scores = area_fd_num[int(num_valid * (2 / 3))]
        # Delete loose food balls
        for i in range(0, len(food_balls)):
            fb_area = fd_index[i]
            if area_fd_num[fb_area] >= area_scores:
                good_balls.append(food_balls[i])
        # Go for food
        mb_cornered = (my_clone_ball['position'] - Vector2(500, 500)).length() > 525

        if begin_spilt:
            cor_value = 1.2
        elif dis_min_ob < 50 or mb_cornered:
            cor_value = 1.7
        else:
            cor_value = 2.2
        # Detect if just split
        for fb in range(0, len(good_balls)):
            distance = (food_balls[fb]['position'] - my_clone_ball['position']).length()
            direction = (food_balls[fb]['position'] - my_clone_ball['position']).normalize()

            correlation = cor_value * (self.last_direction.x * direction.x + self.last_direction.y * direction.y) + 1
            if correlation * correlation / distance > acc_score:
                acc_score = correlation * correlation / distance
                target_id = fb
        if target_id != -1:
            # get eat time
            eat_time = 100
            for mb in my_clone_balls:
                target_distance = (good_balls[target_id]['position'] - mb['position']).length() - mb['radius']
                eat_time_mb = max(0.1, target_distance * (mb['radius'] + 10.0) / 500.0)
                if eat_time_mb < eat_time:
                    eat_time = eat_time_mb
            # get infos
            target_food_ball = food_balls[target_id]
            direction = (target_food_ball['position'] - my_clone_ball['position']).normalize()
            food_score = 4 / eat_time
            # Recalculate Food Score
            if len(my_clone_balls) > 1:
                size_dir_final = self.process_vector_size(my_clone_ball, my_clone_balls, direction)
                size_multiple = size_dir_final / my_clone_ball['radius']
                food_bonus = pow(size_multiple, 0.4)
                food_score = min(20.0, 4 * food_bonus / eat_time)
        return target_food_ball, direction, action_type, food_score

    def process_good_fb(self, food_balls, team_clone_balls, other_clone_balls, my_clone_ball, rect):
        good_balls = []
        # Area Detection
        area_div = 3
        total_num = 9
        mid_id = 4
        num_valid = 0
        view_size = rect[2] - rect[0]
        # Memory
        has_close_fb = False
        fd_index_far = [4] * len(food_balls)
        area_id = []
        area_fd_far_num = []
        area_size = []
        # Initialize squares
        dis_far = my_clone_ball['radius'] + 40
        dis_near = my_clone_ball['radius'] + 24
        edge_left_far = my_clone_ball['position'].x - dis_far
        edge_right_far = my_clone_ball['position'].x + dis_far
        edge_up_far = my_clone_ball['position'].y - dis_far
        edge_down_far = my_clone_ball['position'].y + dis_far
        edge_left_near = my_clone_ball['position'].x - dis_near
        edge_right_near = my_clone_ball['position'].x + dis_near
        edge_up_near = my_clone_ball['position'].y - dis_near
        edge_down_near = my_clone_ball['position'].y + dis_near
        # Define area size ratio
        length_corner = 0.5 * view_size - dis_far
        length_edge = 2 * dis_far
        area_ratio = length_corner / length_edge
        # Initialize areas
        for i_x in range(0, area_div):
            for i_y in range(0, area_div):
                _index = i_x * area_div + i_y
                area_id.append(_index)
                area_fd_far_num.append(0)
                if i_x == 1 or i_y == 1:
                    if i_x == 1 and i_y == 1:
                        _size = 1
                    else:
                        _size = area_ratio
                else:
                    _size = area_ratio * area_ratio
                area_size.append(_size)
        # Fill in areas
        for fb_id in range(0, len(food_balls)):
            fb = food_balls[fb_id]
            # Detect fb far
            if fb['position'].x < edge_left_far:
                i_x_far = 0
            elif edge_right_far < fb['position'].x:
                i_x_far = 2
            else:
                i_x_far = 1
            if edge_down_far < fb['position'].y:
                i_y_far = 0
            elif fb['position'].y < edge_up_far:
                i_y_far = 2
            else:
                i_y_far = 1
            # Calculate score
            _index = i_x_far * area_div + i_y_far
            area_fd_far_num[_index] += 1 / area_size[_index]
            fd_index_far[fb_id] = _index
            # Detect Close Food balls
            if edge_left_near < fb['position'].x < edge_right_near \
                    and edge_down_near < fb['position'].y < edge_up_near:
                has_close_fb = True
        # Detect Solo mode
        if has_close_fb or area_fd_far_num[mid_id] >= 2 / area_size[mid_id]:
            good_balls = food_balls
        # Avoid team balls
        for tb in team_clone_balls:
            dis_tb = (tb['position'] - my_clone_ball['position']).length()
            if dis_tb > 120:
                continue
            else:
                if tb['position'].x <= edge_left_far:
                    i_x = 0
                elif tb['position'].x >= edge_right_far:
                    i_x = 2
                else:
                    i_x = 1
                if tb['position'].y >= edge_down_far:
                    i_y = 0
                elif tb['position'].y <= edge_up_far:
                    i_y = 2
                else:
                    i_y = 1
            _index = i_x * area_div + i_y
            area_fd_far_num[_index] *= 0.3
        # Avoid other balls
        for ob in other_clone_balls:
            dis_ob = (ob['position'] - my_clone_ball['position']).length()
            if dis_ob > 100:
                continue
            else:
                if ob['position'].x <= edge_left_far:
                    i_x = 0
                elif ob['position'].x >= edge_right_far:
                    i_x = 2
                else:
                    i_x = 1
                if ob['position'].y >= edge_down_far:
                    i_y = 0
                elif ob['position'].y <= edge_up_far:
                    i_y = 2
                else:
                    i_y = 1
            _index = i_x * area_div + i_y
            area_fd_far_num[_index] *= 0.75
        # Sort areas
        for i in range(1, total_num):
            if area_fd_far_num[i] > area_fd_far_num[i - 1]:
                area_fd_far_num[i - 1], area_fd_far_num[i] = area_fd_far_num[i], area_fd_far_num[i - 1]
                area_id[i - 1], area_id[i] = area_id[i], area_id[i - 1]
            if area_fd_far_num[i - 1] > 0:
                num_valid += 1
        # Delete loose food
        if area_fd_far_num[4] == 0:
            ignore_num = int(num_valid * 0.5)
        else:
            ignore_num = int(num_valid - 1)
        area_scores = area_fd_far_num[ignore_num]
        for fb_id in range(0, len(food_balls)):
            fb_area = fd_index_far[fb_id]
            if area_fd_far_num[fb_area] > area_scores or fb_area == 4:
                good_balls.append(food_balls[fb_id])

        return good_balls

    def process_vector_size(self, eater_ball, my_clone_balls, vector_dir):
        projection_max = 0
        projection_min = 1000
        size = eater_ball['radius']
        size_dir = Vector2(-1 * vector_dir.y, vector_dir.x).normalize()
        for mb in my_clone_balls:
            if mb is not eater_ball:
                dif = mb['position'] - eater_ball['position']
                projection = dif.x * size_dir.x + dif.y * size_dir.y
                if projection < projection_min:
                    projection_min = projection
                if projection > projection_max:
                    projection_max = projection
        if projection_max - projection_min > size:
            size = projection_max - projection_min

        return size

    def process_food_eater(self, view_center, my_clone_balls):
        dis_to_center_min = 150
        radius_min = 30
        mb_chaser = my_clone_balls[-1]
        for mb in my_clone_balls:
            dis_to_center = (mb['position'] - view_center).length()
            radius = mb['radius']
            if dis_to_center < dis_to_center_min:
                dis_to_center_min = dis_to_center
                radius_min = radius
                mb_chaser = mb
            elif radius < radius_min * 0.5 and dis_to_center * 0.5 < dis_to_center_min:
                dis_to_center_min = dis_to_center
                radius_min = radius
                mb_chaser = mb

            return mb_chaser

    def process_food_balls_np(self, food_balls, my_clone_ball):
        action_type = -1

        ak = util.ActionKit()
        spd_x = self.last_spd[0]
        spd_y = self.last_spd[1]
        select_food, select_clone, food_score = ak.find_food(food_balls, my_clone_ball, spd_x, spd_y,
                                                             consider_vel_direction=True)
        if select_food is None or select_clone is None:
            return None, None, None, 0.0

        dt_x = select_food[0] - select_clone[0]
        dt_y = select_food[1] - select_clone[1]
        force_x, force_y = ak.give_force_to_pos(dt_x, dt_y, spd_x, spd_y)
        direction = Vector2(force_x, force_y)

        return select_food, direction, action_type, food_score

        target_food_ball = util.item_process_to_np(select_food)
        return target_food_ball, direction, action_type, food_score

    def preprocess(self, overlap):
        new_overlap = {}
        for k, v in overlap.items():
            if k == 'clone':
                new_overlap[k] = []
                for index, vv in enumerate(v):
                    tmp = {'position': Vector2(float(vv[0]), float(vv[1])), 'radius': float(vv[2]),
                           'player': str(int(vv[-2])),
                           'team': str(int(vv[-1]))}
                    new_overlap[k].append(tmp)
            else:
                new_overlap[k] = []
                for index, vv in enumerate(v):
                    tmp = {'position': Vector2(float(vv[0]), float(vv[1])), 'radius': float(vv[2])}
                    new_overlap[k].append(tmp)
        return new_overlap

    def evaluate_merge_time(self, my_clones):
        n_my_cl = len(my_clones)
        if n_my_cl <= 1:
            return 0.0
        center = util.get_center(my_clones)

    def smart_escape_pd(self, mb_main, cur_escape_dir, other_clone_balls, ob_main, my_clone_balls, thorns):
        # Memory
        if self.cur_time > 84:
            asd = 0
        ob_main_player = ob_main['player']
        mb_center = my_clone_balls[0]
        ob_second = None
        ob_second_threat = 0
        # Conditions
        trapped = False
        can_merge = self.merge_cd < 6 and len(my_clone_balls) > 1
        # Solve mb center
        mb_rect = [1000, 1000, 0, 0]
        for mb in my_clone_balls:
            if mb['position'].x - mb['radius'] < mb_rect[0]:
                mb_rect[0] = mb['position'].x - mb['radius']
            if mb['position'].x + mb['radius'] > mb_rect[2]:
                mb_rect[2] = mb['position'].x + mb['radius']
            if mb['position'].y - mb['radius'] < mb_rect[1]:
                mb_rect[1] = mb['position'].y - mb['radius']
            if mb['position'].y + mb['radius'] > mb_rect[3]:
                mb_rect[3] = mb['position'].y + mb['radius']
        mb_rect_center = Vector2(0.5 * (mb_rect[0] + mb_rect[2]), 0.5 * (mb_rect[1] + mb_rect[3]))
        dis_to_center_min = 200
        for mb in my_clone_balls:
            dis_to_center = (mb['position'] - mb_rect_center).length()
            if dis_to_center < dis_to_center_min:
                dis_to_center_min = dis_to_center
                mb_center = mb
        mb_center_pos = mb_center['position']
        # Solve edge and corner
        risk_dis = 450.0 - mb_center['radius']
        my_edge = [int((mb_center_pos.x - 500) / risk_dis), int((mb_center_pos.y - 500) / risk_dis)]
        center_to_mb = mb_center['position'] - Vector2(500, 500)
        dis_to_center = center_to_mb.length() + mb_center['radius']
        mb_cornered = center_to_mb.length() + mb_center['radius'] > 500
        if mb_cornered:
            my_corner = [-1 + 2 * int(center_to_mb.x > 0), -1 + 2 * int(center_to_mb.y > 0)]
        else:
            my_corner = [0, 0]
        my_corner_pos = Vector2(500 + my_corner[0] * 500, 500 + my_corner[1] * 500)
        # Solve Conflict
        ob_to_mb = (mb_main['position'] - ob_main['position']).normalize()
        ob_to_cb = (mb_center['position'] - ob_main['position']).normalize()
        if np.cross(ob_to_cb, ob_to_mb) < -0.5:
            escape_conflict = True
        else:
            escape_conflict = False
        # Solve Trapped
        mb_far = None
        mb_far_dis = 0
        main_dis = (ob_main['position'] - mb_center['position']).length() - ob_main['radius']
        # main_time = main_dis / 500 * (ob_main['radius'] + 10)
        # main_score = mb_main['radius'] * mb_main['radius'] / main_time
        for mb in my_clone_balls:
            dis_to_main = (mb_main['position'] - mb['position']).length()
            if 120 > dis_to_main > mb_far_dis:
                mb_far_dis = dis_to_main
                mb_far = mb
        if mb_far is None:
            mb_far = mb_main
        for ob in other_clone_balls:
            same_player = ob['player'] == ob_main_player
            ob_too_far = (mb_center['position'] - ob['position']).length() - ob['radius'] > 120
            if ob['radius'] < mb_far['radius'] or ob_too_far or same_player or mb_far is None:
                continue
            ob_dir = (mb_center['position'] - ob['position']).normalize()
            ob_dis = (mb_center['position'] - ob['position']).length() - ob['radius']
            # ob_time = ob_dis / 500 * (ob_main['radius'] + 10)
            # ob_score = mb_far['radius'] * mb_far['radius'] / ob_time
            ob_threat = min(main_dis / ob_dis, 1)
            is_ob_threat = np.cross(ob_dir, cur_escape_dir) < -0.5
            if is_ob_threat and ob_threat > ob_second_threat:
                ob_second_threat = ob_threat
                ob_second = ob
                if ob_second_threat > 0.5:
                    trapped = True
        # Solve ob
        ob_main_dir = (mb_main['position'] - ob_main['position']).normalize()
        ob_main_to_edge = [int(ob_main_dir.x * 1.7), int(ob_main_dir.y * 1.7)]
        ob_45_degree = ob_main_to_edge[0] != 0 and ob_main_to_edge[1] != 0
        # Solve Edge to mid
        if trapped:
            ob_second_dir = (mb_main['position'] - ob_second['position']).normalize()
            ob_second_to_edge = [int(ob_second_dir.x * 1.7), int(ob_second_dir.y * 1.7)]
            trap_along_edge_x = ob_main_to_edge[0] == -ob_second_to_edge[0] and my_edge[0] == 0 and my_edge[1] != 0
            trap_along_edge_y = ob_main_to_edge[1] == -ob_second_to_edge[1] and my_edge[1] == 0 and my_edge[0] != 0
            trap_along_edge = trap_along_edge_x or trap_along_edge_y
        else:
            trap_along_edge = False

        # Edge Situations
        escape_to_corner_merge = trapped and can_merge and mb_cornered and my_corner != [0, 0]
        if ob_second:
            corner_has_shelter = my_clone_balls[0]['radius'] > 2 * ob_main['radius'] and \
                                 my_clone_balls[0]['radius'] > 2 * ob_second['radius']
        else:
            main_to_center = (mb_main['position'] - Vector2(500, 500)).length()
            big_to_center = (my_clone_balls[0]['position'] - Vector2(500, 500)).length() - my_clone_balls[0]['radius']
            corner_has_shelter = my_clone_balls[0]['radius'] > 2 * ob_main['radius'] and main_to_center < big_to_center
        escape_to_corner_shelter = mb_cornered and corner_has_shelter
        escape_to_corner = escape_to_corner_merge or escape_to_corner_shelter
        mid_escape_trap = trapped and my_edge == [0, 0]
        edge_to_mid = trapped and trap_along_edge
        edge_escape = not trapped and not escape_conflict and my_edge != [0, 0]
        edge_along_escape = edge_escape and (my_corner == [0, 0] or my_edge == ob_main_to_edge)
        edge_to_edge = edge_escape and my_corner != [0, 0] and my_edge != ob_main_to_edge
        edge_to_edge_multi = edge_to_edge and len(my_clone_balls) > 3 and not ob_45_degree
        edge_to_edge_single = edge_escape and dis_to_center > 600 and len(my_clone_balls) == 1
        # Define escape direction
        if escape_to_corner:
            final_direction = (my_corner_pos - mb_center_pos).normalize()
        elif mid_escape_trap:
            escape_dir_main = (mb_center_pos - ob_main['position']).normalize()
            escape_dir_second = (mb_center_pos - ob_second['position']).normalize()
            final_direction = (escape_dir_main + escape_dir_second * ob_second_threat).normalize()
        elif edge_to_mid:
            final_direction = my_edge[0] * Vector2(-1, 0) + my_edge[1] * Vector2(0, -1)
        elif edge_along_escape:
            if my_edge[0] != 0:
                final_direction = Vector2(0, cur_escape_dir.y).normalize()
            elif my_edge[1] != 0:
                final_direction = Vector2(cur_escape_dir.x, 0).normalize()
            else:
                final_direction = cur_escape_dir
        elif edge_to_edge_multi or edge_to_edge_single:
            if ob_main_to_edge[0] == 0 or ob_main_to_edge[1] == 0:
                to_edge = ob_main_to_edge
                final_direction = Vector2(to_edge[0] - my_edge[0], to_edge[1] - my_edge[1]).normalize()
            else:
                if abs(mb_center_pos.x) > abs(mb_center_pos.y):
                    to_edge = [mb_center_pos.x / abs(mb_center_pos.x), 0]
                else:
                    to_edge = [0, mb_center_pos.y / abs(mb_center_pos.y)]
                if to_edge != my_edge:
                    final_direction = Vector2(to_edge[0] - my_edge[0], to_edge[1] - my_edge[1]).normalize()
                else:
                    final_direction = Vector2(to_edge[0] - my_corner[0], to_edge[1] - my_corner[1]).normalize()
        elif escape_conflict:
            my_volume = self.player_v[int(self.name)]
            rest_volume = my_volume - my_clone_balls[0]['radius'] * my_clone_balls[0]['radius']
            if rest_volume > my_volume * 0.66:
                rest_escape_dir = (mb_center['position'] - ob_main['position']).normalize()
                final_direction = rest_escape_dir
            else:
                final_direction = cur_escape_dir
        else:
            final_direction = cur_escape_dir

        return final_direction

    def be_surround(self):
        pass

    def detect_merge_clone(self):
        """
        to fight cooperte
        :return:
        """
        fake_merge_clone_rec = []
        for name in range(0, 12):
            team = name % 3
            if team == self.team:
                continue

            last_enemy_clone = self.last_player_clone[name]
            cur_enemy_clone = self.player_clone[name]
            if len(last_enemy_clone) == 1 or len(cur_enemy_clone) == 1:
                continue

            last_rect = self.get_bounding_box(last_enemy_clone)
            cur_rect = self.get_bounding_box(cur_enemy_clone)

            last_v = sum([b['radius'] * b['radius'] for b in last_enemy_clone])
            cur_v = sum([b['radius'] * b['radius'] for b in cur_enemy_clone])

            if last_rect[0] > cur_rect[0] and last_rect[1] > cur_rect[1] and last_rect[2] < cur_rect[2] and last_rect[
                3] < cur_rect[3]:
                is_merging = True
            else:
                is_merging = False
            if not is_merging or (last_v <= cur_v * 0.8):
                continue

            fake_v = cur_v * 0.9
            fake_r = math.sqrt(fake_v)
            center = util.get_center(cur_enemy_clone)
            for n in range(team * 3, team * 3 + 3):
                if n == name:
                    continue
                for b in self.player_clone[n]:
                    b_pos = b['position']
                    b_r = b['radius']
                    b_v = b_r * b_r
                    dis = (b_pos - center).length()
                    if dis < fake_r:
                        fake_v += b_v

            fake_r = math.sqrt(fake_v)
            # rect smaller
            fake_b = dict()
            fake_b['radius'] = fake_r
            fake_b['position'] = center
            fake_b['team'] = team
            fake_b['player'] = name
            fake_merge_clone_rec.append(fake_b)
            print("合体 name:%s fake_r:%s" % name, fake_r)
        return fake_merge_clone_rec

    def get_bounding_box(self, clones):
        rect = [0., 0., 0., 0.]
        for cl in clones:
            pos = cl['position']
            rect[0] = min(rect[0], pos.x)
            rect[1] = min(rect[1], pos.y)
            rect[2] = max(rect[2], pos.x)
            rect[3] = max(rect[3], pos.y)
        return rect

