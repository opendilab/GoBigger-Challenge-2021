import math
import os
import random
import logging
import copy
import queue
from pygame.math import Vector2
from gobigger.agents.base_agent import BaseAgent
# from .base_agent import BaseAgent

tabu_size=1
position_size=11
class BotAgent(BaseAgent):
    '''
    Overview:
        A simple script bot
    '''

    def __init__(self, name=None, level=3):
        self.name = name
        self.actions_queue = queue.Queue()
        self.last_clone_num = 1
        self.last_total_size = 0
        self.level = level
        self.my_score=0
        # self.team_score=0
        self.position_times=[]

    def step(self, obs):
        global_obs,obs=obs
        # self.team_score=global_obs['leaderboard'][str(int(self.name)//3)]
        ally_info,others_clone_balls=self.process_ally_balls(obs)
        obs=obs[self.name]

        overlap = obs['overlap']
        overlap = self.preprocess(overlap)
        food_balls = overlap['food']
        thorns_balls = overlap['thorns']
        spore_balls = overlap['spore']
        clone_balls = overlap['clone']
        food_balls.extend(spore_balls)

        my_clone_balls, ally_clone_balls, near_other_balls = self.process_clone_balls_1(clone_balls)
        my_total_score=sum(my_ball['radius']**2 for my_ball in my_clone_balls)

        for my_ball in my_clone_balls:
            for position, time in self.position_times:
                if (my_ball['position'] - position).length() < 6.5:
                    my_ball['time'] = time
                    break
            else:
                my_ball['time'] = 101
            my_ball['time'] = max(0, my_ball['time'] - 1)

        self.position_times = [(my_ball['position'], my_ball['time']) for my_ball in my_clone_balls]

        if abs(self.last_total_size-my_clone_balls[0]['radius'])<0.01:
            self.stay_same_times+=1
        else:
            self.stay_same_times=0

        if self.stay_same_times>5 and self.last_total_size<10 and len(my_clone_balls)==1:
            self.stay_same_times = 0
            # print(fr'={self.name}===================')
            self.actions_queue.put([None, None, 2])
            self.actions_queue.put([None, None, 2])
            self.actions_queue.put([None, None, 2])
            self.actions_queue.put([None, None, 2])
            self.actions_queue.put([None, None, 2])
            action_ret = self.actions_queue.get()
            return action_ret
        self.last_total_size=my_clone_balls[0]['radius']

        direction_attact=self.attact(my_clone_balls,others_clone_balls,thorns_balls)
        if direction_attact and len(my_clone_balls)<16:
            direction=direction_attact.normalize()
            action_type = 1
            # print(f'{self.name}nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnattack')
            self.actions_queue.queue.clear()
            self.actions_queue.put([direction.x, direction.y, action_type])
            action_ret = self.actions_queue.get()
            return action_ret

        direction_attact=self.attact2(my_clone_balls,others_clone_balls,thorns_balls)
        if direction_attact and len(my_clone_balls)<16:
            direction=direction_attact.normalize()
            action_type = 1
            # print(f'{self.name}nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnattack')
            self.actions_queue.queue.clear()
            self.actions_queue.put([direction.x, direction.y, action_type])
            action_ret = self.actions_queue.get()
            return action_ret

        # if self.actions_queue.qsize() > 0:
        #     return self.actions_queue.get()

        direction,danger = self.APF(my_clone_balls, others_clone_balls)
        if direction.length()>=1:
            direction=direction.normalize()
            action_type = -1
            self.actions_queue.queue.clear()
            self.actions_queue.put([direction.x, direction.y, action_type])
            action_ret = self.actions_queue.get()
            return action_ret

        direction0=self.near_thorns_balls(thorns_balls, my_clone_balls, others_clone_balls)
        if direction0:
            action_type = -1
            self.actions_queue.queue.clear()
            self.actions_queue.put([direction0.x, direction0.y, action_type])
            action_ret = self.actions_queue.get()
            return action_ret
        elif self.actions_queue.qsize() > 0:
            return self.actions_queue.get()

        elif direction.length()<0.5 and danger and len(my_clone_balls)>3:
            self.actions_queue.queue.clear()
            self.actions_queue.put([None, None, 2])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            action_ret = self.actions_queue.get()
            return action_ret

        for other_ball in others_clone_balls:
            if  800<my_total_score/6<other_ball['radius']**2<0.47*my_total_score and (other_ball['position']-my_clone_balls[0]['position']).length()<2*math.sqrt(0.9*my_total_score):
                self.actions_queue.put([None, None, 2])
                self.actions_queue.put([None, None, -1])
                self.actions_queue.put([None, None, -1])
                self.actions_queue.put([None, None, -1])
                self.actions_queue.put([None, None, -1])
                self.actions_queue.put([None, None, -1])
                self.actions_queue.put([None, None, -1])
                self.actions_queue.put([None, None, 0])
                self.actions_queue.put([None, None, 0])
                self.actions_queue.put([None, None, 0])
                self.actions_queue.put([None, None, 0])
                self.actions_queue.put([None, None, 0])
                self.actions_queue.put([None, None, 0])
                self.actions_queue.put([None, None, 0])
                self.actions_queue.put([None, None, 0])
                action_ret = self.actions_queue.get()
                # print('/////////////////////////////')
                return action_ret

        if (len(my_clone_balls) >= 9 and my_clone_balls[4]['radius'] > 20) or (my_clone_balls[0]['radius']**2<my_total_score/4 and self.my_score>5000) or \
                (len(my_clone_balls)>1 and (my_clone_balls[0]['position']-my_clone_balls[1]['position']).length()<1.2*my_clone_balls[0]['radius']):
            self.actions_queue.put([None, None, 2])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, -1])
            self.actions_queue.put([None, None, 0])
            self.actions_queue.put([None, None, 0])
            self.actions_queue.put([None, None, 0])
            self.actions_queue.put([None, None, 0])
            self.actions_queue.put([None, None, 0])
            self.actions_queue.put([None, None, 0])
            self.actions_queue.put([None, None, 0])
            self.actions_queue.put([None, None, 0])
            action_ret = self.actions_queue.get()
            return action_ret
        else:
            direction = self.APF2(direction,my_clone_balls, others_clone_balls, ally_info, food_balls, thorns_balls)
            action_type=-1
            self.actions_queue.put([direction.x, direction.y, action_type])
            action_ret = self.actions_queue.get()
            return action_ret


    def process_clone_balls_1(self, clone_balls):
        my_clone_balls = []
        ally_clone_balls=[]
        others_clone_balls = []
        for clone_ball in clone_balls:
            if clone_ball['player'] == self.name:
                my_clone_balls.append(copy.deepcopy(clone_ball))
            elif clone_ball['team'] == str(int(self.name) // 3):
                ally_clone_balls.append(copy.deepcopy(clone_ball))
            else:
                others_clone_balls.append(copy.deepcopy(clone_ball))

        my_clone_balls.sort(key=lambda a: a['radius'], reverse=True)
        ally_clone_balls.sort(key=lambda a: a['radius'], reverse=True)
        others_clone_balls.sort(key=lambda a: a['radius'], reverse=True)
        # others_clone_balls.sort(key=lambda a: (my_clone_balls[0]['position'] - a['position']).length()-2*a['radius'], reverse=False)
        # threaten_balls=[]
        # for other_ball in others_clone_balls:
        #     for my_ball in my_clone_balls[0:int(2*len(my_clone_balls)/3)]:
        #         if my_ball['radius']<other_ball['radius']:
        return my_clone_balls,ally_clone_balls, others_clone_balls


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


    def covered(self,my_ball,other_ball,my_clone_balls):
        neighbor_balls=[]
        other_ball_newradius=math.sqrt(my_ball['radius']**2+other_ball['radius']**2/2)
        for my_ball0 in my_clone_balls:
            if my_ball0['radius']<other_ball_newradius and (my_ball0['position']-my_ball['position']).length()<other_ball_newradius:
                neighbor_balls.append(my_ball0)
        tmp=other_ball['radius']**2/2
        for neighbor_ball in neighbor_balls:
            tmp+=neighbor_ball['radius']**2
        new_radius=math.sqrt(tmp)

        for my_big_ball in my_clone_balls:
            if 0.71*my_big_ball['radius']>new_radius:
                if my_big_ball['radius']*2.12>(my_big_ball['position']-my_ball['position']).length()+0.71*other_ball['radius']:
                    # print('==========',self.name)
                    return True
            else:
                return False
        return False
    def APF(self,my_clone_balls,other_clone_balls):


        danger=False
        rep = Vector2(0, 0)  # 所有障碍物总斥力
        for my_ball in my_clone_balls:
            for other_ball in other_clone_balls:
                t_vec = my_ball['position'] - other_ball['position']

                if my_ball['radius']>other_ball['radius'] or t_vec.length()>20+2.5*other_ball['radius']:
                    pass
                elif t_vec.length()>20+other_ball['radius'] and self.covered(my_ball,other_ball,my_clone_balls):
                    pass
                else:
                    direction=t_vec.normalize()
                    rep_tmp=direction * 10000 * (2.5 / (t_vec.length()-20)- 1.0 / other_ball['radius']) / (t_vec.length())*(other_ball['radius']/my_clone_balls[0]['radius'])**2
                    rep_tmp=rep_tmp * my_ball['radius'] ** 2 / self.my_score * 10
                    if my_ball['radius']<0.71*other_ball['radius'] and \
                            (other_ball['position']-my_ball['position']).length()<10+2.12*other_ball['radius']: #3倍根号2
                        rep_tmp=rep_tmp*3

                        new_radius = math.sqrt(my_ball['radius'] ** 2 + 0.5 * other_ball['radius'] ** 2)
                        for my_ball_1 in my_clone_balls:
                            if my_ball_1 !=my_ball and 0.71*new_radius >my_ball_1['radius'] and (my_ball['position']-my_ball_1['position']).length()<10+2.12*new_radius:
                                rep+=rep_tmp*(my_ball_1['radius']/my_ball['radius'])
                                # print('xxxxxxxxxxxxasfasfasd',self.name)
                    # avg_radius=math.sqrt(sum(tmp_ball['radius']**2 for tmp_ball in my_clone_balls)/len(my_clone_balls))
                    # if my_ball['radius']<0.71*other_ball['radius'] and my_ball['radius']>avg_radius*0.8 and \
                    #         (other_ball['position']-my_ball['position']).length()<20+2.212*other_ball['radius']: #3倍根号2
                    #     rep_tmp=rep_tmp*len(my_clone_balls)
                    #     print('xxxxxxxxxxxxasfasfasd')

                    rep +=rep_tmp
                    if rep_tmp.length()>=5:
                        danger=True
        return rep,danger

    def APF2(self,rep,my_clone_balls,other_clone_balls,ally_info,food_balls,thorns_balls):
        food_balls.extend(thorns_balls)
        neighbor_food_balls=[]
        att=Vector2(0.01, 0.01) #食物球,队友引力

        #队友合并
        min_time=10
        direct=None
        for my_ball in my_clone_balls:
            for ally_ball in ally_info:
                if ally_ball[2] !=self.name and len(my_clone_balls)>4 and ally_ball[3]>2000 and self.my_score>2000:
                    dis=(ally_ball[1]-my_ball['position']).length()
                    time=(dis-max(my_ball['radius'],ally_ball[0]))/((500/(10+my_ball['radius']))+(500/(10+ally_ball[0])))
                    if dis<1.5*(my_ball['radius']+ally_ball[0]) and time<min_time:
                        min_time=time
                        direct=(ally_ball[1]-my_ball['position']).normalize()
        if direct:
            return direct



        # if len(my_clone_balls)>=15 and my_clone_balls[-1]['radius']<ally_info[0][0]:
        #     direct=ally_info[0][1]-my_clone_balls[-1]['position']
        #     att+=direct.normalize()*math.sqrt(direct.length())

        # if (ally_info[0][1]-my_clone_balls[0]['position']).length()-my_clone_balls[0]['radius']>350 and 3000<self.my_score<6000:
        #     direct=ally_info[0][1]-my_clone_balls[0]['position']
        #     att+=direct.normalize()*math.sqrt(direct.length())


        # if len(my_clone_balls)>=15:
        #     if self.name != ally_info[0][2] and my_clone_balls[-1]['radius']<ally_info[0][0]:
        #         direct=ally_info[0][1]-my_clone_balls[-1]['position']
        #         att+=direct.normalize()*math.sqrt(direct.length())
        #     elif my_clone_balls[-1]['radius']<ally_info[1][0]:
        #         direct=ally_info[1][1]-my_clone_balls[-1]['position']
        #         att+=direct.normalize()*math.sqrt(direct.length())
        # elif (ally_info[0][1]-my_clone_balls[0]['position']).length()-my_clone_balls[0]['radius']>400 and 30<my_clone_balls[0]['radius']<80:
        #     direct=ally_info[0][1]-my_clone_balls[0]['position']
        #     att+=direct.normalize()*math.sqrt(direct.length())
        # elif self.name != ally_info[0][2] and ally_info[0][3] > 5000:
        #     self_score=[info[3] for info in ally_info if info[2]==self.name][0]
        #     if self_score>2000:
        #         direct = ally_info[0][1] - my_clone_balls[0]['position']
        #         att += direct.normalize() * math.sqrt(direct.length())

        for my_ball in my_clone_balls:
            for food_ball in food_balls:
                x,y=food_ball['position'][0],food_ball['position'][1]
                dis=min(x+y,1000-x+y,1000-y+x,2000-x-y)
                if my_ball['radius']>=food_ball['radius'] and dis>my_ball['radius']:
                    t_vec = food_ball['position'] - my_ball['position']
                    if food_ball['radius']>5 and len(my_clone_balls)<16 and t_vec.length()<50+my_ball['radius']/10 and \
                            (not self.is_thorns_ball_safe(my_ball, food_ball, other_clone_balls,16-len(my_clone_balls),my_clone_balls)): #如果荆棘球不安全，会产生斥力?
                        if t_vec.length()>10+my_ball['radius']:
                            continue
                        else:
                            t_vec=-t_vec
                            att += t_vec.normalize() * (food_ball['radius'] ** 2) / t_vec.length()*10
                    else:
                        att+=t_vec.normalize()*(food_ball['radius']**2)/t_vec.length()
                        if t_vec.length()<100+my_ball['radius']/10 and food_ball not in neighbor_food_balls:
                            neighbor_food_balls.append(food_ball)
            #     elif dis>my_ball['radius'] and 0.8*my_ball['radius']>food_ball['radius']:
            #         t_vec = food_ball['position'] - my_ball['position']
            #         att += t_vec.normalize() * (food_ball['radius'] ** 2) *min(t_vec.length(),50)/ 5000*(my_ball['radius']/food_ball['radius'])**2
            #         # print('222222222222222222')
            # for other_bal l in other_clone_balls:
            #     if my_ball['radius']>other_ball['radius']:
            #         t_vec = other_ball['position'] - my_ball['position']
            #         dis_x=1000-other_ball['position'][0]-other_ball['radius'] if t_vec[0]>0 else other_ball['position'][0]-other_ball['radius']
            #         dis_y=1000-other_ball['position'][1]-other_ball['radius'] if t_vec[1]>0 else other_ball['position'][1]-other_ball['radius']
            #
            #         time=(dis_x+dis_y-my_ball['radius'])/(250/(10+my_ball['radius']))
            #         if t_vec.length()<100+my_ball['radius']/10 and time<50 and other_ball['radius']>50:
            #             print('------------------------------------',time,self.name,other_ball['player'])
            #
            #             att += t_vec.normalize() * (other_ball['radius'] ** 2 / 5) / time
            #             neighbor_food_balls.append(other_ball)
        try:
            direction_goal=(rep+att).normalize()
        except:
            direction_goal =att.normalize()
            print(f'att:{att}')
            print(f'rep:{rep}')
        best_direction=copy.deepcopy(direction_goal)
        min_div=100000

        for my_ball in my_clone_balls:
            for food_ball in neighbor_food_balls:
                if my_ball['radius']>food_ball['radius']:

                    copy_balls=copy.deepcopy(my_clone_balls)
                    for copy_ball in copy_balls:
                        copy_ball['position']=copy_ball['position']+(food_ball['position']-my_ball['position'])*(10+my_ball['radius'])/(10+copy_ball['radius'])
                    direction,danger=self.APF(copy_balls,other_clone_balls)
                    if direction.length()>2 or danger:
                        continue

                    t_vec=food_ball['position']-my_ball['position']
                    diverse=((direction_goal-t_vec.normalize()).length()+0.5)*(t_vec.length()-my_ball['radius'])/(250/(10+my_ball['radius']))/food_ball['radius']
                    if diverse<min_div:
                        min_div=diverse
                        best_direction=t_vec.normalize()
        if best_direction==direction_goal:
            for my_ball in my_clone_balls:
                for food_ball in neighbor_food_balls:
                    if my_ball['radius'] > food_ball['radius']:
                        t_vec = food_ball['position'] - my_ball['position']
                        diverse = ((direction_goal - t_vec.normalize()).length() + 0.5) * (
                                    t_vec.length() - my_ball['radius']) / (250 / (10 + my_ball['radius'])) / food_ball[
                                      'radius']
                        if diverse < min_div:
                            min_div = diverse
                            best_direction = t_vec.normalize()
        return best_direction

    def near_thorns_balls(self,thorns_balls,my_clone_balls,others_clone_balls):
        min_div=3
        best_direction=None
        for my_ball in my_clone_balls:
            for thorns_ball in thorns_balls:
                if my_ball['radius']>thorns_ball['radius'] and self.is_thorns_ball_safe(my_ball,thorns_ball,others_clone_balls,16-len(my_clone_balls),my_clone_balls):
                    t_vec=thorns_ball['position']-my_ball['position']
                    diverse=(t_vec.length()-my_ball['radius'])/(250/(10+my_ball['radius']))
                    if diverse<min_div:
                        min_div=diverse
                        best_direction=t_vec.normalize()
        return best_direction


    def is_thorns_ball_safe(self,my_ball,thorns_ball,other_balls,len,my_clone_balls):
        for other_ball in other_balls:
            my_ball_tmp=copy.deepcopy(my_ball)
            my_ball_tmp['radius']=math.sqrt(my_ball_tmp['radius']**2+thorns_ball['radius']**2)
            if self.covered(my_ball_tmp,other_ball,my_clone_balls):
                continue

            new_radius_sqr=(my_ball['radius']**2+thorns_ball['radius']**2)/min(10,(len+1))
            new_radius1_sqr=my_ball['radius']**2+thorns_ball['radius']**2-400*min(9,len)
            new_radius=math.sqrt(max(new_radius1_sqr,new_radius_sqr))
            if other_ball['radius']*0.71>new_radius:
                enemy_radius=max(math.sqrt(other_ball['radius']**2/2+min(400,new_radius_sqr)*min(9,len)/3),other_ball['radius'])
            else:
                enemy_radius=other_ball['radius']
            if (thorns_ball['position'] - my_ball['position']).length()-my_ball['radius']>(thorns_ball['position'] - other_ball['position']).length()-10-2.12*other_ball['radius'] and \
                    new_radius<enemy_radius and other_ball['radius']>10:
                return False
        return True

    def process_ally_balls(self,obs):
        ally_info=[]
        others_clone_balls = []
        other_ball_position = []
        for name,obs_player in obs.items():
            # if name != self.name:
            overlap = obs_player['overlap']
            overlap = self.preprocess(overlap)
            ally_clone_balls_obs = overlap['clone']
            ally_clone_balls=[]
            totol_score=0
            for clone_ball in ally_clone_balls_obs:
                if clone_ball['player'] == name:
                    ally_clone_balls.append(copy.deepcopy(clone_ball))
                    totol_score+=clone_ball['radius']**2
                elif clone_ball['team'] != str(int(self.name) // 3) and clone_ball['position'] not in other_ball_position:
                    others_clone_balls.append(copy.deepcopy(clone_ball))
                    other_ball_position.append(clone_ball['position'])
            ally_clone_balls.sort(key=lambda a: a['radius'], reverse=True)
            ally_info.append([ally_clone_balls[0]['radius'],ally_clone_balls[0]['position'],name,totol_score])
            if name==self.name:
                self.my_score=totol_score
        ally_info.sort(key=lambda a:a[3],reverse=True)
        others_clone_balls.sort(key=lambda a: a['radius'], reverse=True)

        for i in range(len(others_clone_balls)-1):
            for j in range(i+1,len(others_clone_balls)):
                other_ball=others_clone_balls[i]
                other_ball1=others_clone_balls[j]
                if (other_ball['position']-other_ball1['position']).length()<1.1*other_ball['radius']:
                    other_ball['radius']=math.sqrt(other_ball['radius']**2+other_ball1['radius']**2)

        return ally_info,others_clone_balls


    def is_safe(self,my_ball,other_balls,direction,culed_reward_pos):
        loss=0
        reward=0
        new_position = my_ball['position'] + direction * (10 + 1.41 * my_ball['radius'])
        for other_ball in other_balls:
            if  my_ball['radius']<1.01*other_ball['radius'] and (other_ball['position']-my_ball['position']).length()<15+2.2*other_ball['radius']:
                loss += my_ball['radius'] ** 2 / 2
            if 0.7*my_ball['radius']<other_ball['radius'] and \
                    (new_position-other_ball['position']).length()<other_ball['radius']:
                loss+=my_ball['radius']**2/2
            elif my_ball['radius']<1.01*other_ball['radius'] and (other_ball['position']-new_position).length()<15+2.2*other_ball['radius']:
                loss += my_ball['radius'] ** 2 / 2
            elif  0.7*my_ball['radius']>other_ball['radius'] and (new_position-other_ball['position']).length()<0.7*my_ball['radius'] and other_ball['position'] not in culed_reward_pos:
                reward+=other_ball['radius']**2
                culed_reward_pos.append(other_ball['position'])

            loss=min(loss,my_ball['radius'] ** 2)
            # if 0.7*my_ball['radius']>other_ball['radius']:
            #     return loss
        return loss,reward

    def attact(self,my_clone_balls,other_clone_balls,thorns_balls):
        for other_ball in other_clone_balls:
            for i in range(max(min(len(my_clone_balls),16-len(my_clone_balls)),0)):
                if 0.7*my_clone_balls[i]['radius']>other_ball['radius'] and 0.05*self.my_score<other_ball['radius']**2 and \
                        (my_clone_balls[i]['position'] - other_ball['position']).length() <= max(0,15-(500 / (10 + other_ball['radius'])))+ 2.12*my_clone_balls[i]['radius']:
                    flag=False
                    if len(my_clone_balls)<8:
                        new_position = my_clone_balls[i]['position'] + (other_ball['position'] - my_clone_balls[i]['position']).normalize() * (1.41 * my_clone_balls[i]['radius'] + 5)
                        for throns_ball in thorns_balls:
                            if (throns_ball['position']-new_position).length()<0.71*my_clone_balls[i]['radius']+5:
                                # print('[[[[[[[[[[[[[[[[[[[[[[[[[[[[[')
                                flag=True
                                break
                    if flag:
                        continue

                    direction=(other_ball['position'] - my_clone_balls[i]['position']).normalize()
                    reward=other_ball['radius']**2
                    loss=0
                    culed_reward_pos=[other_ball['position']]
                    for j in range(min(len(my_clone_balls),16-len(my_clone_balls))):
                        loss1,reward1=self.is_safe(my_clone_balls[j],other_clone_balls,direction,culed_reward_pos)
                        loss+=loss1
                        reward+=reward1
                        if reward-loss<0.05*self.my_score:
                            break
                    else:
                        return direction
        return False

    def attact1(self,my_clone_balls,other_clone_balls,thorns_balls):
        if len(my_clone_balls)<=15 and my_clone_balls[0]['radius']>50:
            for other_ball in other_clone_balls:
                if 0.49*my_clone_balls[0]['radius']>other_ball['radius'] and 0.3*my_clone_balls[0]['radius']<other_ball['radius'] and\
                        (my_clone_balls[0]['position'] - other_ball['position']).length() <= (1.41+1.5) *my_clone_balls[0]['radius']:
                    direction=(other_ball['position'] - my_clone_balls[0]['position']).normalize()

                    position1=my_clone_balls[0]['position']+direction*1.41 *my_clone_balls[0]['radius']
                    if len(my_clone_balls)>7:
                        for j in range(1,len(my_clone_balls)):
                            if (my_clone_balls[j]['position']-position1).length()<0.71*my_clone_balls[0]['radius']  and my_clone_balls[j]['time']==0:
                                break
                        else:
                            continue
                    #判断是否有荆棘球
                    if self.attact_thorns(position1, my_clone_balls[0]['radius'],len(my_clone_balls),thorns_balls):
                        continue

                    my_new_clone_balls=copy.deepcopy(my_clone_balls)
                    for i in range(min(len(my_clone_balls), 16 - len(my_clone_balls))):
                        my_new_clone_balls[i]['radius']=0.71*my_new_clone_balls[i]['radius']
                        new_clone_tmp=copy.deepcopy(my_new_clone_balls[i])
                        new_clone_tmp['position']=new_clone_tmp['position']+direction*1.41 *new_clone_tmp['radius']
                        new_clone_tmp['radius']=new_clone_tmp['radius']+0.1 #保证新分裂的排第一个
                        my_new_clone_balls.append(new_clone_tmp)
                    my_new_clone_balls.sort(key=lambda a: a['radius'], reverse=True)

                    reward=other_ball['radius']**2
                    loss=0
                    culed_reward_pos=[other_ball['position']]
                    for j in range(min(len(my_clone_balls),16-len(my_clone_balls))):
                        loss1,reward1=self.is_safe(my_clone_balls[j],other_clone_balls,direction,culed_reward_pos)
                        loss+=loss1
                        reward+=reward1
                    for j in range(max(1,min(len(my_new_clone_balls), 16 - len(my_new_clone_balls)))):
                        loss1,reward1=self.is_safe(my_new_clone_balls[j], other_clone_balls, direction,culed_reward_pos)
                        loss += loss1
                        reward += reward1
                    if reward-loss<0.05*self.my_score:
                        continue
                    else:
                        return direction
        return False
    def attact2(self,my_clone_balls,other_clone_balls,thorns_balls):
        if (len(my_clone_balls)<=7 or (7<len(my_clone_balls)<16 and my_clone_balls[15-len(my_clone_balls)]['radius']<10)) and my_clone_balls[0]['radius']>50:
            for other_ball in other_clone_balls:
                if 0.49*my_clone_balls[0]['radius']>other_ball['radius'] and 0.3*my_clone_balls[0]['radius']<other_ball['radius'] and\
                        (my_clone_balls[0]['position'] - other_ball['position']).length() <= (1.41+1.5) *my_clone_balls[0]['radius']:
                    direction=(other_ball['position'] - my_clone_balls[0]['position']).normalize()

                    position1=my_clone_balls[0]['position']+direction*1.41 *my_clone_balls[0]['radius']
                    # if len(my_clone_balls)>7:
                    #     for j in range(1,len(my_clone_balls)):
                    #         if (my_clone_balls[j]['position']-position1).length()<0.71*my_clone_balls[0]['radius']  and my_clone_balls[j]['time']==0:
                    #             break
                    #     else:
                    #         continue
                    #判断是否有荆棘球
                    if self.attact_thorns(position1, my_clone_balls[0]['radius'],len(my_clone_balls),thorns_balls):
                        continue

                    my_new_clone_balls=copy.deepcopy(my_clone_balls)
                    for i in range(min(len(my_clone_balls), 16 - len(my_clone_balls))):
                        my_new_clone_balls[i]['radius']=0.71*my_new_clone_balls[i]['radius']
                        new_clone_tmp=copy.deepcopy(my_new_clone_balls[i])
                        new_clone_tmp['position']=new_clone_tmp['position']+direction*1.41 *new_clone_tmp['radius']
                        new_clone_tmp['radius']=new_clone_tmp['radius']+0.1 #保证新分裂的排第一个
                        my_new_clone_balls.append(new_clone_tmp)
                    my_new_clone_balls.sort(key=lambda a: a['radius'], reverse=True)

                    reward=other_ball['radius']**2
                    loss=0
                    culed_reward_pos=[other_ball['position']]
                    for j in range(min(len(my_clone_balls),16-len(my_clone_balls))):
                        loss1,reward1=self.is_safe(my_clone_balls[j],other_clone_balls,direction,culed_reward_pos)
                        loss+=loss1
                        reward+=reward1
                    for j in range(max(1,min(len(my_new_clone_balls), 16 - len(my_new_clone_balls)))):
                        loss1,reward1=self.is_safe(my_new_clone_balls[j], other_clone_balls, direction,culed_reward_pos)
                        loss += loss1
                        reward += reward1
                    if reward-loss<0.05*self.my_score:
                        continue
                    else:
                        return direction
        return False
    def attact_thorns(self,new_position,radius,len,thorns_balls):
        if len<8:
            for thorns_ball in thorns_balls:
                if (thorns_ball['position']-new_position).length()<radius:
                    return True
        return False
