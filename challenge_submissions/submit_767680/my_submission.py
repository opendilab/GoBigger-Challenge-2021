import os
import random
import logging
import copy
import queue
from pygame.math import Vector2
from math import sqrt,exp
import numpy as np
from easydict import EasyDict

class BaseAgent:
    '''
    Overview:
        The base class of all agents
    '''
    def __init__(self):
        pass

    def step(self, obs):
        raise NotImplementedError

class MyModel(BaseAgent):
    '''
    Overview:
        A simple script bot
    '''

    def __init__(self, team_name,player_name,level=5,all_vision=False):
        self.team = team_name
        self.name = player_name
        self.actions_queue = queue.Queue()
        self.last_clone_num = 1
        self.last_total_size = 0
        self.step_count = 0
        self.speed = None
        self.split = False
        self.level = level
        self.recombine_age = 0
        self.merging = -1
        self.all_vision = all_vision


    def step(self, obs):
        self.recombine_age -= 0.2

        self.step_count += 1
        if self.actions_queue.qsize() > 0:
            return self.actions_queue.get()
        if not self.all_vision:
            obs = obs['overlap']
        overlap = self.preprocess(obs)
        food_balls = overlap['food']
        thorns_balls = overlap['thorns']
        spore_balls = overlap['spore']
        clone_balls = overlap['clone']

        my_clone_balls, team_clone_balls,others_clone_balls = self.process_clone_balls_with_team(clone_balls)

        if len(my_clone_balls)>self.last_clone_num:
            self.last_clone_num = len(my_clone_balls)

        radius_overall = sqrt(sum([x['radius']**2 for x in my_clone_balls]))
        centroid = Vector2(0, 0)
        weight = 0
        for i in range(len(my_clone_balls)):
            centroid += my_clone_balls[i]['position'] * my_clone_balls[i]['radius'] ** 2
            weight += my_clone_balls[i]['radius'] ** 2
        centroid = centroid / weight
        if self.speed == None:
            self.speed = Vector2(0, 0)
            self.last_pos = centroid
        else:
            self.speed = centroid - self.last_pos
            self.last_pos = centroid

        # 先考虑有对手存在的情况下
        # TODO 如何解析目前安全度
        # 先考虑是否安全

        safe = True
        if len(others_clone_balls)>0:
            safe = False
            balls1 = copy.deepcopy(my_clone_balls)
            balls2 = copy.deepcopy(others_clone_balls)
            e = self.eat_or_be_eat(balls1,balls2)
            if e==0:
                safe = True

        # 在考虑已经安全的基础上，使用split策略
        if safe:
            max_value = -1
            split_direct = None

            for i in range(min(len(my_clone_balls),16-len(my_clone_balls))):
                for j in range(len(others_clone_balls)):
                    dis = others_clone_balls[j]['position'] - my_clone_balls[i]['position']
                    dis = dis.length()
                    if others_clone_balls[j]['radius']+1 < my_clone_balls[i]['radius']/sqrt(2) and dis - my_clone_balls[i]['radius']*3/2*sqrt(2) - 6 < 0 :
                        if others_clone_balls[j]['radius']>max_value:
                            max_value = others_clone_balls[j]['radius']
                            split_direct =  others_clone_balls[j]['position'] - my_clone_balls[i]['position']


            if split_direct and max_value**2 > weight*0.1+50 and len(my_clone_balls)<16:
                # print(f"can split at {self.step_count}")
                split_direct = split_direct.normalize()

                balls1 = copy.deepcopy(my_clone_balls)
                balls2 = copy.deepcopy(others_clone_balls)

                s = 0
                l = len(balls1)
                while len(balls1)<16:
                    if s==l:
                        break
                    balls1[s]['radius'] /= sqrt(2)
                    ball = copy.deepcopy(balls1[s])
                    ball['position'] += split_direct * ball['radius'] * 2
                    balls1.append(ball)
                    s+=1

                safe = self.eat_or_be_eat(balls1,balls2)
                if safe == 0:
                    # print(f"splited at {self.step_count}")
                    action = [split_direct.x, split_direct.y, 1]
                    return action




        direct = Vector2((random.random() - 0.5) / 10000, (random.random() - 0.5) / 10000)
        for i in range(len(my_clone_balls)):
            for j in range(len(food_balls)):
                v = (food_balls[j]['position'] - my_clone_balls[i]['position'])
                v += Vector2((random.random() - 0.5) / 1000, (random.random() - 0.5) / 1000)
                l = v.length() - my_clone_balls[i]['radius']
                l = max(l,1)
                v = v.normalize()
                atr = 4 * v * self.f1(l)
                direct += atr
            for j in range(len(thorns_balls)):
                if my_clone_balls[i]['radius'] > thorns_balls[j]['radius']:
                    v = (thorns_balls[j]['position'] - my_clone_balls[i]['position'])
                    v += Vector2((random.random() - 0.5) / 1000, (random.random() - 0.5) / 1000)
                    l = v.length() - my_clone_balls[i]['radius']
                    v = v.normalize()
                    atr = thorns_balls[j]['radius'] ** 2 * v * self.f1(l)
                    direct += atr
            for j in range(len(others_clone_balls)):
                if my_clone_balls[i]['radius'] > others_clone_balls[j]['radius']:
                    v = others_clone_balls[j]['position'] - my_clone_balls[i]['position']
                    v += Vector2((random.random() - 0.5) / 1000, (random.random() - 0.5) / 1000)
                    l = v.length() - my_clone_balls[i]['radius']
                    v = v.normalize()
                    atr = others_clone_balls[j]['radius'] ** 2 * v * self.f1(l)
                    direct += atr
                elif my_clone_balls[i]['radius'] > others_clone_balls[j]['radius']/sqrt(2):
                    v = my_clone_balls[i]['position'] - others_clone_balls[j]['position']
                    v += Vector2((random.random() - 0.5) / 1000, (random.random() - 0.5) / 1000)
                    l = v.length() - others_clone_balls[j]['radius']
                    v = v.normalize()
                    atr = my_clone_balls[i]['radius'] ** 2 * v * self.f1(l)
                    direct += atr
                else:
                    v = my_clone_balls[i]['position'] - others_clone_balls[j]['position']
                    v += Vector2((random.random() - 0.5) / 1000, (random.random() - 0.5) / 1000)
                    l = v.length() - others_clone_balls[j]['radius']*sqrt(2)*3/2
                    l = l if l>0 else v.length() - others_clone_balls[j]['radius']
                    v = v.normalize()
                    atr = my_clone_balls[i]['radius'] ** 2 * v * self.f1(l)
                    direct += atr


        acceleration = direct.normalize()
        # 减小自旋概率
        if my_clone_balls[0]['radius'] ** 2 < 30:
            acceleration += self.speed
        direction = acceleration

        action_type = -1
        self.actions_queue.put([direction.x, direction.y, action_type])

        action_ret = self.actions_queue.get()
        return action_ret



    def cos_Vector2(self,v1,v2):
        return  v1*v2/(v1.length()*v2.length())

    def f1(self,l):
        if l==0:
            l = 0.01
        return 1/l**2

    def eat_or_be_eat(self,balls1,balls2):

        while len(balls1)>0 and len(balls2)>0:
            # 选择最近的一对
            min_dis = 1e10
            pick = [-1, -1]
            for i in range(len(balls1)):
                for j in range(len(balls2)):
                    dis = (balls1[i]['position'] - balls2[j]['position']).length()
                    if dis<min_dis:
                        pick =[i,j]
                        min_dis = dis
            if balls1[pick[0]]['radius']>balls2[pick[1]]['radius']:
                balls1[pick[0]]['radius'] = sqrt(balls1[pick[0]]['radius']**2 + balls2[pick[1]]['radius']**2)
                balls1[pick[0]]['position'] = balls2[pick[1]]['position']
                balls2.pop(pick[1])
            else:
                balls2[pick[1]]['radius'] = sqrt(balls1[pick[0]]['radius'] ** 2 + balls2[pick[1]]['radius'] ** 2)
                balls2[pick[1]]['position'] = balls1[pick[0]]['position']
                balls1.pop(pick[0])
        if len(balls2) == 0:
            return 0
        else:
            return 1


    def process_clone_balls(self, clone_balls):
        my_clone_balls = []
        others_clone_balls = []
        for clone_ball in clone_balls:
            if clone_ball['player'] == self.name:
                my_clone_balls.append(copy.deepcopy(clone_ball))
        my_clone_balls.sort(key=lambda a: a['radius'], reverse=True)

        for clone_ball in clone_balls:
            if clone_ball['player'] != self.name:
                others_clone_balls.append(copy.deepcopy(clone_ball))
        others_clone_balls.sort(key=lambda a: a['radius'], reverse=True)
        return my_clone_balls, others_clone_balls

    def process_clone_balls_with_team(self, clone_balls):
        my_clone_balls = []
        team_clone_balls = []
        others_clone_balls = []

        for clone_ball in clone_balls:
            if clone_ball['player'] == self.name:
                my_clone_balls.append(copy.deepcopy(clone_ball))
            elif clone_ball['team'] == self.team:
                team_clone_balls.append(copy.deepcopy(clone_ball))
            else:
                others_clone_balls.append(copy.deepcopy(clone_ball))

        my_clone_balls.sort(key=lambda a: a['radius'], reverse=True)
        team_clone_balls.sort(key=lambda a: a['radius'], reverse=True)
        others_clone_balls.sort(key=lambda a: a['radius'], reverse=True)

        return my_clone_balls, team_clone_balls,others_clone_balls
    def judge_split_value(self,my_clone_balls,other_clone_balls):
        reward = 0
        for i in range(len(my_clone_balls)):

            for j in range(len(other_clone_balls)):
                distance = (my_clone_balls[i]['position'] - other_clone_balls[j]['position']).length()
                if my_clone_balls[i]['radius'] > other_clone_balls[j]['radius']:
                    reward += other_clone_balls[j]['radius'] / distance ** 2
                if my_clone_balls[i]['radius'] < other_clone_balls[j]['radius']:
                    reward -= distance * my_clone_balls[i]['radius'] / distance ** 2
        return  reward

    def process_thorns_balls(self, thorns_balls, my_max_clone_ball):
        min_distance = 10000
        min_thorns_ball = None
        for thorns_ball in thorns_balls:
            if thorns_ball['radius'] < my_max_clone_ball['radius']:
                distance = (thorns_ball['position'] - my_max_clone_ball['position']).length()
                if distance < min_distance:
                    min_distance = distance
                    min_thorns_ball = copy.deepcopy(thorns_ball)
        return min_distance, min_thorns_ball

    def process_food_balls(self, food_balls, my_max_clone_ball):
        min_distance = 10000
        min_food_ball = None
        for food_ball in food_balls:
            distance = (food_ball['position'] - my_max_clone_ball['position']).length()
            if distance < min_distance:
                min_distance = distance
                min_food_ball = copy.deepcopy(food_ball)
        return min_distance, min_food_ball

    def preprocess(self, overlap):
        new_overlap = {}
        for k, v in overlap.items():
            if k == 'clone':
                new_overlap[k] = []
                for index, vv in enumerate(v):
                    tmp = {}
                    tmp['position'] = Vector2(vv[0], vv[1])
                    tmp['radius'] = vv[2]
                    tmp['player'] = str(int(vv[-2]))
                    tmp['team'] = str(int(vv[-1]))
                    new_overlap[k].append(tmp)
            else:
                new_overlap[k] = []
                for index, vv in enumerate(v):
                    tmp = {}
                    tmp['position'] = Vector2(vv[0], vv[1])
                    tmp['radius'] = vv[2]
                    new_overlap[k].append(tmp)
        return new_overlap

    def preprocess_tuple2vector(self, overlap):
        new_overlap = {}
        for k, v in overlap.items():
            new_overlap[k] = []
            for index, vv in enumerate(v):
                new_overlap[k].append(vv)
                new_overlap[k][index]['position'] = Vector2(*vv['position'])
        return new_overlap

    def add_noise_to_direction(self, direction, noise_ratio=0.1):
        direction = direction + Vector2(((random.random() * 2 - 1) * noise_ratio) * direction.x,
                                        ((random.random() * 2 - 1) * noise_ratio) * direction.y)
        return direction

    def to_eat_food(self):
        pass


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

    def __init__(self, team_name, player_names, level=6,all_vision=False):
        super(MySubmission, self).__init__(team_name, player_names)
        self.agents = {}
        self.all_vision = all_vision
        for player_name in self.player_names:
            self.agents[player_name] = MyModel(team_name=team_name, player_name=player_name, level=level,all_vision=self.all_vision)

    def get_actions(self, obs):

        global_state, player_states = obs
        actions = {}
        if self.all_vision:
            overlap = {
                'food': [],
                'thorns': [],
                'spore': [],
                'clone': []
            }
            for _, v in player_states.items():
                overlap_ = v['overlap']
                overlap['food'].extend(overlap_['food'])
                overlap['thorns'].extend(overlap_['thorns'])
                overlap['spore'].extend(overlap_['spore'])
                overlap['clone'].extend(overlap_['clone'])


            for player_name, agent in self.agents.items():
                action = agent.step(overlap)
                actions[player_name] = action
        else:
            for player_name, agent in self.agents.items():
                action = agent.step(player_states[player_name])
                actions[player_name] = action
            return actions
        return actions

