"""
author:yuandong
version:4.1
update: 
在version 3.9的基础上 Times:100 ruleAgent:0.61 rule_agentV37:0.11 rule_agentV37:0.17 rule_agentV37:0.11


"""

import os
import random
import logging
import copy
import queue
import math

from dataclasses import dataclass,field
from pygame.math import Vector2
from typing import List

from .base_agent import BaseAgent

@dataclass
class info:
    food: List[dict]= field(default_factory=list) 
    spore: List[dict]= field(default_factory=list)
    thorns : List[dict]= field(default_factory=list)
    my_clone_balls: List[dict] = field(default_factory=list)
    my_team_clone_balls: List[dict] = field(default_factory=list)
    others_clone_balls: List[dict]= field(default_factory=list)

    cur_speed: Vector2 = Vector2(0,0) #当前的速度
    last_position: Vector2 = Vector2(0,0) #
    cur_step: int = 0
    action_state : str = 'none' #记录当前的状态

class RuleAgent(BaseAgent):
    '''
    Overview:
        
    '''
    def __init__(self, name=None):
        self.name = name
        self.actions_queue = queue.Queue()
        
        self.team_name = None
        self.my_info = info()
        
    def step(self, obs):
        return self.process(obs)

    def process(self, obs):
        ## 如果队列中有动作，执行
        if self.actions_queue.qsize() > 0:
            return self.actions_queue.get()

        ## 对obs进行处理
        self.team_name = obs['team_name']
        overlap = obs['overlap']
        overlap = self.preprocess_overlap(overlap)
        
        self.process_clone_balls(overlap['clone'])
        self.my_info.food = overlap['food']
        self.my_info.spore = overlap['spore']
        self.my_info.thorns = overlap['thorns']

        #计算当前速度
        if self.my_info.cur_step>0:
            self.my_info.cur_speed = (self.my_info.my_clone_balls[0]['position']-self.my_info.last_position)/0.25
        self.my_info.last_position = self.my_info.my_clone_balls[0]['position']
        self.my_info.cur_step+=1

        # if self.my_info.action_state == 'eject2Centered':
        #     self.eject2Center()
            
        #如果我的小于其他人的：逃跑
        if self.run_away(): 
            self.my_info.action_state='none'
            pass
        #如果比其他的大则追击
        elif self.attack():
            self.my_info.action_state='none'
            pass
        elif self.my_info.action_state == 'eject2Centered':
            self.eject2Center()
        # if self.attack():
        #     pass
        # elif self.run_away():
        #     pass
        # elif self.team_work():
            # pass
        elif self.eat_food():
            pass
        else:
            self.actions_queue.put([None,None,-1])
        action_ret = self.actions_queue.get()

        return action_ret

    def process_vector(self,v1,v2):
        # reflet=(v1.dot(v2)*v1)/(v1.length()*v1.length())
        return v1
        # if v1.dot(v2)>0:
            # return v1
        # else:
            # return v1-reflet

    def process_clone_balls(self, clone_balls):
        """
        处理overlap中clone balls的信息,返回安装radius排序后的
        my_clone_balls: 玩家自己的clone ball列表
        my_team_clone_balls: 玩家队伍除了自己以外的clone balll列表
        others_clone_balls: 其他队伍的clone balll列表
        """
        my_clone_balls,my_team_clone_balls,others_clone_balls = [],[],[]

        for clone_ball in clone_balls:# 遍历视野内所有的分身球
            if clone_ball['player'] == self.name:# 找到属于自己的分身球
                my_clone_balls.append(copy.deepcopy(clone_ball))
            elif clone_ball['team']==self.team_name: #属于自己队伍的分身球
                my_team_clone_balls.append(copy.deepcopy(clone_ball))
            else:
                others_clone_balls.append(copy.deepcopy(clone_ball))

        # 按半径从大到小进行排序
        my_clone_balls.sort(key=lambda a: a['radius'], reverse=True) 
        my_team_clone_balls.sort(key=lambda a: a['radius'], reverse=True) 
        others_clone_balls.sort(key=lambda a: a['radius'], reverse=True)

        # print(type(my_clone_balls[0]))
        self.my_info.my_clone_balls=my_clone_balls
        self.my_info.my_team_clone_balls=my_team_clone_balls
        self.my_info.others_clone_balls=others_clone_balls
        
        return my_clone_balls,my_team_clone_balls,others_clone_balls

    def process_thorns_balls(self, thorns_balls, my_max_clone_ball):
        """
            return:
                min_thorns_ball: 距离当前最大的球的最近的刺
                min_distance: min_thorns_ball的距离
        """
        min_distance = 10000
        min_thorns_ball = None
        for thorns_ball in thorns_balls:
            if thorns_ball['radius'] < my_max_clone_ball['radius']: #如果我当前最大的球的半径大于刺的半径
                distance = (thorns_ball['position'] - my_max_clone_ball['position']).length() #我与刺的距离
                if distance < min_distance:
                    min_distance = distance
                    min_thorns_ball = copy.deepcopy(thorns_ball)
        return min_distance, min_thorns_ball

    def preprocess_overlap(self, overlap):
        """
            处理obs中的overlap信息,将每个种类的每一个生成一个字典。
            clone包含'position','radius','player','team'信息
            其他包含'position','radius'信息
        """
        new_overlap = {}
        for k, v in overlap.items():
            if k =='clone':
                new_overlap[k] = []
                for index, vv in enumerate(v):
                    tmp={}
                    tmp['position'] = Vector2(vv[0],vv[1]) 
                    tmp['radius'] = vv[2]
                    tmp['player'] = str(int(vv[-2]))
                    tmp['team'] = str(int(vv[-1]))
                    new_overlap[k].append(tmp)
            else:
                new_overlap[k] = []
                for index, vv in enumerate(v):
                    tmp={}
                    tmp['position'] = Vector2(vv[0],vv[1])
                    tmp['radius'] = vv[2]
                    new_overlap[k].append(tmp)
        return new_overlap

    #####################################################################################################
    #####################################################################################################
    ###########################################高级动作###################################################
    #####################################################################################################
    #####################################################################################################


    def eject2Center(self):
        """"high-level operation:Eject towards the center"""
        if len(self.my_info.my_clone_balls)==1 or len(self.my_info.my_clone_balls)<4 or self.my_info.my_clone_balls[1]['radius']<27:
            # math.sqrt(self.my_info.my_clone_balls[0]['radius']**2)*0.5>math.sqrt(sum(ball['radius']**2 for ball in self.my_info.my_clone_balls[1:]))\
                
            self.my_info.action_state = 'none'
            return False
        self.my_info.action_state = 'eject2Centered'
        self.actions_queue.put([None, None, 2]) # 使用停止技能
        self.actions_queue.put([None, None, -1]) # 不操作，等待球球聚集
        self.actions_queue.put([None, None, -1])
        self.actions_queue.put([None, None, -1])
        self.actions_queue.put([None, None, -1])
        self.actions_queue.put([None, None, -1])
        self.actions_queue.put([None, None, -1])
        self.actions_queue.put([None, None, 0]) # 使用吐孢子球技能
        self.actions_queue.put([None, None, 0])
        self.actions_queue.put([None, None, 0])
        self.actions_queue.put([None, None, 0])
        self.actions_queue.put([None, None, 0])
        self.actions_queue.put([None, None, 0])
        self.actions_queue.put([None, None, 0])
        self.actions_queue.put([None, None, 0])
        return True

    def run_away(self):
        """run away from"""
        my_clone_balls = self.my_info.my_clone_balls
        others_clone_balls = self.my_info.others_clone_balls
        thorns_balls = self.my_info.thorns
        used_ball=[]
        # 1.首先判断是否进行中吐来聚集，
        
        # 如果我最小的小于其他球最大的，我最大的大于其他球最大的
        # 如果我的体积和大于其他球
        # math.sqrt(sum([ball['radius']**2 for ball in my_clone_balls]))*0.75 > math.sqrt(others_clone_balls[0]['radius']**2):
        
        # if len(others_clone_balls) > 0 and len(my_clone_balls)>0 and \
        #         ((my_clone_balls[-1]['radius'] <others_clone_balls[0]['radius'] and my_clone_balls[0]['radius']>others_clone_balls[0]['radius']) or \
        #         (my_clone_balls[0]['radius']<others_clone_balls[0]['radius'] and math.sqrt(sum([ball['radius']**2 for ball in my_clone_balls]))*0.7 > math.sqrt(others_clone_balls[0]['radius']**2))):
            # self.my_info.action_state = 'eject2Center'
        # if len(others_clone_balls) > 0 and len(my_clone_balls)>0 and \
        #         (my_clone_balls[-1]['radius'] <others_clone_balls[0]['radius'] and my_clone_balls[0]['radius']>others_clone_balls[0]['radius']):
        #     return self.eject2Center()


        # 计算所有敌对球（大于我球的）对我的球的作用力
        # 进入进攻范围的才计算入

        run_away_directions=[]
        
        for other_ball in others_clone_balls:
            for my_ball in my_clone_balls:
                if other_ball['radius']>my_ball['radius']:
                    dist=(my_ball['position'] - other_ball['position']).length()
                    if (dist < other_ball['radius']*math.sqrt(2)*4+1.5 and other_ball['radius']*math.sqrt(2)/2>my_ball['radius']) or \
                        (dist < other_ball['radius']*math.sqrt(2)*2 and other_ball['radius']>my_ball['radius']):
                        direction = (other_ball['radius']*my_ball['radius']/(dist-other_ball['radius']))*((my_ball['position'] - other_ball['position']).normalize())
                        run_away_directions.append(direction)
                        used_ball.append(my_ball)

        # 
        if len(run_away_directions)>0:
            ##加入吸引向量
            attract_directions=[]
            for other_ball in others_clone_balls:
                for my_ball in my_clone_balls:
                    if other_ball['radius']<my_ball['radius']*0.85:# and my_ball not in used_ball:
                        dist=(my_ball['position'] - other_ball['position']).length()
                        # if dist<my_ball['radius']*6:
                        direction = (other_ball['radius']*my_ball['radius']/(dist-my_ball['radius']))*((other_ball['position'] - my_ball['position']).normalize())
                        attract_directions.append(direction)
            
            ##加入food的向量
            food = self.my_info.food
            spore = self.my_info.spore
            food_directions=[]
            for other_ball in food:
                for my_ball in my_clone_balls:
                    # if not my_ball in used_ball:
                    ball_dist = (my_ball['position']-other_ball['position']).length()       
                    direction = (other_ball['radius']/(ball_dist-my_ball['radius']))*((other_ball['position'] - my_ball['position']).normalize())
                    food_directions.append(direction)


            # for thorns_ball in thorns_balls:
            #     for my_ball in my_clone_balls:
            #         if thorns_ball['radius'] < my_ball['radius']:
            #             ball_dist = (my_ball['position']-thorns_ball['position']).length()
            #             direction = (thorns_ball['radius']/(ball_dist-my_ball['radius']))*((thorns_ball['position'] - my_ball['position']).normalize())
            #             food_directions.append(direction)

            final_direction = Vector2(0,0)
            run_away_direct = Vector2(0,0)
            attract_direct = Vector2(0,0)
            food_direct = Vector2(0,0)
            for direct in run_away_directions:
                run_away_direct+=direct
            
            for direct in attract_directions:
                attract_direct+=direct
            if attract_direct.length()>0:
                attract_direct = self.process_vector(attract_direct,run_away_direct)

            for direct in food_directions:
                food_direct+=direct
            if food_direct.length()>0:
                food_direct = self.process_vector(food_direct,run_away_direct)
            
            if len(food_directions)>0 and len(attract_directions)>0:
                # final_direction=(run_away_direct.normalize()+0.1*food_direct.normalize()+0.2*attract_direct.normalize()).normalize()
                final_direction = ((run_away_direct+attract_direct).normalize()+0.2*food_direct.normalize()).normalize()
            elif len(food_directions)>0:
                final_direction=(run_away_direct.normalize()+0.2*food_direct.normalize()).normalize()
            elif len(attract_directions)>0:
                final_direction=(run_away_direct+attract_direct).normalize()
            else:
                final_direction=run_away_direct.normalize()
            
            
            if my_clone_balls[0]['position'].x-my_clone_balls[0]['radius']<0+5:#左边界
                if my_clone_balls[0]['position'].y-my_clone_balls[0]['radius']<0+5:#左下边界
                    final_direction = Vector2(0,1) if abs(final_direction.x) > abs(final_direction.y) else Vector2(1,0)
                elif my_clone_balls[0]['position'].y+my_clone_balls[0]['radius']>1000-5:#左上边界
                    final_direction = Vector2(0,-1) if abs(final_direction.x) > abs(final_direction.y) else Vector2(1,0)
                else:
                    final_direction = Vector2(0,1 if final_direction.y>0 else -1).normalize()
            elif my_clone_balls[0]['position'].x+my_clone_balls[0]['radius']>1000-5:#右边界
                if my_clone_balls[0]['position'].y-my_clone_balls[0]['radius']<0+5:#右下边界
                    final_direction = Vector2(0,1) if abs(final_direction.x) > abs(final_direction.y) else Vector2(-1,0)
                elif my_clone_balls[0]['position'].y+my_clone_balls[0]['radius']>1000-5:#右上边界
                    final_direction = Vector2(0,-1) if abs(final_direction.x) > abs(final_direction.y) else Vector2(-1,0)
                else:
                    final_direction = Vector2(0,1 if final_direction.y>0 else -1).normalize()
            elif my_clone_balls[0]['position'].y-my_clone_balls[0]['radius']<0+5:#下边界
                if my_clone_balls[0]['position'].x-my_clone_balls[0]['radius']<0+5:#左下边界
                    final_direction = Vector2(0,1) if abs(final_direction.x) > abs(final_direction.y) else Vector2(1,0)
                elif my_clone_balls[0]['position'].x+my_clone_balls[0]['radius']>1000-5:#右下边界
                    final_direction = Vector2(0,1) if abs(final_direction.x) > abs(final_direction.y) else Vector2(-1,0)
                else:
                    final_direction = Vector2(1 if final_direction.x>0 else -1,0).normalize()
            elif my_clone_balls[0]['position'].y+my_clone_balls[0]['radius']>1000-5:#上边界
                if my_clone_balls[0]['position'].x-my_clone_balls[0]['radius']<0+5:#左上边界
                    final_direction = Vector2(0,-1) if abs(final_direction.x) > abs(final_direction.y) else Vector2(1,0)
                elif my_clone_balls[0]['position'].x+my_clone_balls[0]['radius']>1000-5:#右上边界
                    final_direction = Vector2(0,-1) if abs(final_direction.x) > abs(final_direction.y) else Vector2(-1,0)
                else:
                    final_direction = Vector2(1 if final_direction.x>0 else -1,0).normalize()
            action_type = -1
            self.actions_queue.put([final_direction.x, final_direction.y, action_type])
            return True

        elif len(others_clone_balls) > 0 and len(my_clone_balls)>0 and \
                ((my_clone_balls[-1]['radius'] <others_clone_balls[0]['radius'] and my_clone_balls[0]['radius']>others_clone_balls[0]['radius']) or
                (len(my_clone_balls) >= 9 and my_clone_balls[4]['radius'] > 14)):
            return self.eject2Center()

        return False
    
    def eat_food(self):
        my_clone_balls = self.my_info.my_clone_balls
        my_team_clone_balls = self.my_info.my_team_clone_balls
        food = self.my_info.food
        thorns_balls = self.my_info.thorns
        others_clone_balls = self.my_info.others_clone_balls
        spore = self.my_info.spore
        # my_team_clone_balls = self.my_info.my_team_clone_balls
        
        #吃食物
        all_directions=[]
        for food_ball in food:
            for my_ball in my_clone_balls:
                ball_dist = (my_ball['position']-food_ball['position']).length()        
                direction = (food_ball['radius']/(ball_dist-my_ball['radius']))*((food_ball['position'] - my_ball['position']).normalize())
                all_directions.append(direction)

        if sum(ball['radius']**2 for ball in my_clone_balls)<10000:
            for thorns_ball in thorns_balls:
                for my_ball in my_clone_balls:
                    if thorns_ball['radius'] < my_ball['radius']:
                        ball_dist = (my_ball['position']-thorns_ball['position']).length()
                        direction = (thorns_ball['radius']/(ball_dist-my_ball['radius']))*((thorns_ball['position'] - my_ball['position']).normalize())
                        all_directions.append(direction)
        else:
            for thorns_ball in thorns_balls:
                for my_ball in my_clone_balls:
                    if thorns_ball['radius'] < my_ball['radius']:
                        ball_dist = (my_ball['position']-thorns_ball['position']).length()
                        direction = (thorns_ball['radius']/(ball_dist-my_ball['radius']))*((my_ball['position'] - thorns_ball['position']).normalize())
                        all_directions.append(direction)
        # 吃其它小球
        for other_ball in others_clone_balls:
            for my_ball in my_clone_balls:
                if other_ball['radius'] < my_ball['radius']*0.85:
                    ball_dist = (my_ball['position']-other_ball['position']).length()
                    # if ball_dist<my_ball['radius']*6:
                    direction = (other_ball['radius']/(ball_dist-my_ball['radius']))*((other_ball['position'] - my_ball['position']).normalize())
                    all_directions.append(direction)

        if len(all_directions)>0:
            final_direction = Vector2(0,0)
            for direct in all_directions:
                final_direction+=direct
            final_direction=final_direction.normalize()
            # if len(my_clone_balls)==1:
            #     action_type = 4
            # else:
            action_type = -1
            self.actions_queue.put([final_direction.x, final_direction.y, action_type])
            return True
        return False
    
    def attack(self):
        """
        攻击,如果可以攻击返回True,否则返回False
        我们只对可以在一次分裂或者两次分裂可以吃的的进行攻击
        """
        my_clone_balls = self.my_info.my_clone_balls
        others_clone_balls = self.my_info.others_clone_balls
        throns = self.my_info.thorns

        ## 不进行attack的条件
        if len(others_clone_balls)==0: #or my_clone_balls[0]['radius']*math.sqrt(2)*0.5<others_clone_balls[0]['radius']:
            return False

        target_1 = None
        target_2 = None
        target_1_my = None
        target_2_my = None
        for idx,other_ball in enumerate(others_clone_balls):
            for my_ball in my_clone_balls:
                ball_dist = (other_ball['position']-my_ball['position']).length()
                if my_ball['radius']*(math.sqrt(2)/2)*0.9 > other_ball['radius'] and my_ball['radius']*(math.sqrt(2)/2)*0.4<other_ball['radius'] and ball_dist<my_ball['radius']*math.sqrt(2)+1.5: #一次分裂可以吃掉
                    
                    flag=True
                    if idx>0:
                        for other_ball_tmp in others_clone_balls[:idx]:
                            dist_ball_ball = (other_ball['position']-other_ball_tmp['position']).length()
                            if other_ball_tmp['radius']>my_ball['radius']*(math.sqrt(2)/2) and (dist_ball_ball<other_ball_tmp['radius']*math.sqrt(2)*1.5+1.5 or ball_dist<other_ball_tmp['radius']*math.sqrt(2)*1.5+1.5):
                                flag=False
                                break
                        
                    if flag:
                        target_1 = copy.deepcopy(other_ball)
                        target_1_my = copy.deepcopy(my_ball)
                        break
        
        if target_1:
            direction = (target_1['position'] - target_1_my['position']).normalize()
            action_type = 4
            self.actions_queue.put([direction.x, direction.y, action_type])
            return True
        return False

    def team_work(self):
        my_team_clone_balls = self.my_info.my_team_clone_balls
        my_clone_balls = self.my_info.my_clone_balls
        if len(my_team_clone_balls)==0 or (len(my_team_clone_balls)==1 and len(my_clone_balls)==1):
            return False
        
        all_directions=[]
        for my_ball in my_clone_balls:
            for team_ball in my_team_clone_balls:
                direction = (team_ball['position'] - my_ball['position']).normalize()
                all_directions.append(direction)
        
        final_direction = Vector2(0,0)
        for direct in all_directions:
            final_direction+=direct
        final_direction=final_direction.normalize()
        action_type = -1
        self.actions_queue.put([final_direction.x, final_direction.y, action_type])
        return True
        

    # def attack(self):
    #     """
    #     攻击,如果可以攻击返回True,否则返回False
    #     我们只对可以在一次分裂或者两次分裂可以吃的的进行攻击
    #     """
    #     my_clone_balls = self.my_info.my_clone_balls
    #     others_clone_balls = self.my_info.others_clone_balls
    #     throns = self.my_info.thorns

    #     ## 不进行attack的条件
    #     if len(others_clone_balls)==0: #or my_clone_balls[0]['radius']*math.sqrt(2)*0.5<others_clone_balls[0]['radius']:
    #         return False

    #     target_1 = None
    #     target_2 = None
    #     target_1_my = None
    #     target_2_my = None
    #     for other_ball in others_clone_balls:
    #         for my_ball in my_clone_balls:
    #             ball_dist = (other_ball['position']-my_ball['position']).length()
    #             if my_ball['radius']*(math.sqrt(2)/2)*0.9 > other_ball['radius'] and my_ball['radius']*(math.sqrt(2)/2)*0.4<other_ball['radius'] and ball_dist<my_ball['radius']*math.sqrt(2)/2+15: #一次分裂可以吃掉
    #                 target_1 = copy.deepcopy(other_ball)
    #                 target_1_my = copy.deepcopy(my_ball)
    #                 break
        
    #     if target_1:
    #         direction = (target_1['position'] - target_1_my['position']).normalize()
    #         action_type = 4
    #         self.actions_queue.put([direction.x, direction.y, action_type])
    #         return True
    #     return False