import os
import random
import logging
import copy
import queue
import math
from collections import defaultdict
from random import uniform
from math import sqrt

from pygame.math import *
from gobigger.utils import *

# 有威胁不吃刺!! 尽量用小球吃, 然后吃到刺, 丢个大球
# 利用寻路跑路


# 自己 分裂, 别人也分裂, 没考虑

# 合球的条件, 百分比啥的

tmpflag =False
Y2 = (205, 233, 38)
BLUE = (0, 0, 255)


sign = lambda x: (1, -1)[x<0]


CO_START_S2B = 1000
S1_FOOD_THORN = 1500 
S2_KILL_THORN = 5000


def splitForward(ball):
    return 30*0.1 + 27*0.1 + ball['vmax']*0.2  # 略少一点

def inRect(vec,rect):
    return vec.x>=rect[0] and vec.x<=rect[2] and vec.y>=rect[1] and vec.y<=rect[3]
       
def instantEat(a, b):
    if a['radius'] > b['radius']:
        p = b['position'] - a['position']
        if p.length() < a['radius']:
            return True
    return False

def splitCanEat(a, b):
    if a['radius'] > b['radius']:
        p = b['position'] - a['position']
        if p.length() < a['radius'] + a['vmax'] -b['vmax'] + 7:
            return True
    return False    
    

def centerHasBigger(srcBall, destP, catchballs, maxGoT=3, aheadT=1, goballR=None, infos=None):
    for catchball in catchballs:
        if isInter(srcBall, destP, catchball, maxGoT=maxGoT,aheadT=aheadT, goballR=goballR, infos=infos):
            if infos:
                infos.append((1, srcBall['position'], destP, GRAY))
                infos.append((1, srcBall['position'], catchball['position'], BLACK))
            return catchball
    
    return None

def get_foot(p, a, b):
    ap = p - a
    ab = b - a
    result = a + ap.dot(ab)/ab.dot(ab) * ab
    return result

# 主球去一个点, 然后判断其他的会不会有危险
def gothrough(mainball, destP, myballs, destballs ,aheadT=0.6, infos=None):
    
    dir = destP - mainball['position'] 
    t = dir.length() / mainball['vmax']
    
    dangers = []
    myballs_indanger = []
    
    # if infos is not None:
    #     addPathInfo(infos, mainball, destP)
        
    for myball in myballs:
        myballdestP = myball['position'] + dir.normalize()*(t*myball['vmax'])
        
        catcher = centerHasBigger(myball, myballdestP, destballs, aheadT=aheadT)
        if catcher:
            if myball not in myballs_indanger:
                myballs_indanger.append(myball)
                if catcher not in dangers:   
                    dangers.append(catcher)
                    
                    
        if infos is not None:
            # addPathInfo(infos, myball, myballdestP)
            for ball in dangers:
                infos.append((88,ball['position'], ball['radius']-2, Y2 ))
            

    return myballs_indanger, dangers

def gothroughHasThorn(mainball, destP, myballs, destballs , infos=None):
    
    dir = destP - mainball['position'] 
    t = dir.length() / mainball['vmax']
    
    dangers = []
    myballs_indanger = []
    
    if infos:
        infos.append((0, destP, 4, RED))
        # addPathInfo(infos, mainball, destP)
        
    for myball in myballs:
        
        P0 = myball['position']
        vec = dir.normalize()*(t*myball['vmax'])
        myballdestP = P0 + vec
        
        # if infos is not None:
        #     addPathInfo(infos, myball, myballdestP)
        
        for destball in destballs:
            if pathHas(myball, myballdestP, destball):
                return True

    return False

def addGoDebug(infos, myball, myballdestP):
    infos.append((11, myball['position'], myballdestP, YELLOW))
    
    div = 20
    other_balls = []
    for i in range(50):
        for j in range(50):
            
            tmp={}
            tmp['position'] = Vector2(div*i,div*j)
            r = math.sqrt(2500)
            tmp['radius'] = r
            tmp['size'] = r*r
            tmp['vmax'] = cal_vel_max(r)
            
            newsize =  tmp['size']/2
            newR = math.sqrt(newsize)
            tmp['splitR'] =  newR
            tmp['splitD'] =  newR * 2 + 15
            tmp['sporeD'] =  tmp['radius'] + 40
            tmp['splitE'] =  newR * 2 + 15 + newR
            other_balls.append(tmp)
            
            if tmp['size']>= 100:
                tmp['willsplit'] = True
            else:
                tmp['willsplit'] = False
    
            if isInter(myball, myballdestP, tmp):
                infos.append((0, tmp['position'], r, RED))

def addPathInfo(infos, myball, myballdestP, color=Y2):
    
    P0 = myball['position']
    dir = myballdestP - P0
    
    if dir.length()==0:
        return
    
    PR = dir.normalize()*myball['radius']
    Pplus = Vector2(-PR.y, PR.x)+P0
    Pneg = Vector2(PR.y, -PR.x)+P0
    
    destPlus = Pplus + dir
    destNeg = Pneg + dir
    
    infos.append((1, Pplus, destPlus, color))
    infos.append((1, Pneg, destNeg, color))

def isInter(goball, destP, catcher,  maxGoT=3, aheadT=1, goballR=None, goMerge=False, infos=None):
    size = goball['size']
    if goballR is not None:
        size = goballR*goballR
    else:
        goballR = goball['radius']
        
    if size >= catcher['size']:
        return False
    
    
    # 在之间 , 不split
    a = goball['position']
    b = destP
    cp = catcher['position']
    
    if (a-b).length()<=0:
        return False
    
    destGo = b - a
    CatherGo = cp - a
    angle = destGo.angle_to(CatherGo)
    if angle < 0:
        angle += 360
    
    if abs(angle-180)< 60:
        # if infos is not None:
        #     infos.append((1, a, b, RED))
        #     infos.append((1, cp, b, RED))
        return False
    
    
    r = get_foot(cp, a, b)

    rx = r.x 
    ax = a.x
    bx = b.x
    
    if ax==bx:
        rx = r.y
        ax = a.y
        bx = b.y
    
    p = (rx-ax)/(bx-ax)
    L = (a-b).length()
     
    VA = goball['vmax']
    if goMerge:
        VA = mergeSpd(goball)
    VB = catcher['vmax']
    R = catcher['radius']
    
    t = L/VA
    if t> maxGoT and maxGoT>0:
        t = maxGoT
    
    B = r
    if p>=1:
        p = 1
        B = destP
    elif p<=0:
        p = 0
        B = a
    
    # 算走到目标的时间
    dis = (cp - B).length() - catcher['radius']
    enet = dis / catcher['vmax']
    if enet < t*p + aheadT:
        return True
    
    if catcher['willsplit']:
        if catcher['splitR'] >= goballR:
            split_dis = (cp - B).length() - catcher['splitE'] 
            enet = split_dis / catcher['vmax']
        
        if enet < t*p + aheadT:
            return True
    return False  

# 上面的简单版, 用于静止的东西
def pathHas(goball, destP, catcher, useSize=False):
    size = goball['size']
    
    if useSize:
        if size > catcher['size']:
            return False
    
    # 在之间 , 不split
    a = goball['position']
    b = destP
    cp = catcher['position']
    
    if (a-b).length()<=0:
        return False
    
    r = get_foot(cp, a, b)

    rx = r.x 
    ax = a.x
    bx = b.x
    
    if ax==bx:
        rx = r.y
        ax = a.y
        bx = b.y
    
    p = (rx-ax)/(bx-ax)
    L = (a-b).length()
     
    B = r
    if p>=1:
        B = destP
    if p<=0:
        return False
    
    # 算走到目标的时间
    dis = (cp - B).length()
    has = dis < goball['radius'] + catcher['radius']
    return has
     
# 过于严厉
def canCatch(a, b, t, goball, catcher):
    
    if goball['size'] >= catcher['size']:
        return False
    
    catcher_pos = catcher['position']

    VB = catcher['vmax']
    R = catcher['radius']
    
    # 再算会不会 split
    other_ball = catcher
    if other_ball['splitR'] >= goball['radius']:
        if other_ball['willsplit']:
            for i in range(20):
                tmp = i/19
                P = (b-a)*tmp + a
                
                dis = (catcher_pos-P).length() - other_ball['splitR'] - other_ball['splitD']
                enet = dis/VB
            
                if enet < t:
                    return True
        
    
    for i in range(20):
        tmp = i/19
        P = (b-a)*tmp + a
        
        dis = (catcher_pos-P).length() - R
        enet = dis/VB
        
        if enet <= t:
            return True
        
        pass

    return False  
         
def getDirTo(srcBall, dest):
    return getDir(srcBall['position'], dest, srcBall['vmax'] , srcBall['v'])
# 修正速度过载
def getDir(pos, dest, vel_max, curVel):
    direction = (dest - pos).normalize()*vel_max - curVel
    # print("d", direction, "p", pos, "dest", dest, curVel)
    return direction

def cal_vel_max(radius, vel_max=25):
    return vel_max*20/(radius+10)

def hasSplitEat(balls, oldBalls):
    
    if len(balls) <= len(oldBalls):
        return False
        
    if len(balls) < 2:
        return False
    
    newBalls = []
    hasBallSplit = False
    for ball in balls:
        if ball['last'] is not None:
            r = ball['last']['size'] /ball['size'] 
            if r<2.1 and r>1.9:
                return True
        else:
            newBalls.append(ball)
        
    return False

def cal_centroid(balls):
    x = 0
    y = 0
    total_size = 0
    for ball in balls:
        x += ball['size'] * ball['position'].x
        y += ball['size'] * ball['position'].y
        total_size += ball['size']
    return Vector2(x, y) / total_size


# 不准
def mergeTotalT(balls):
    maxR = balls[0]['radius']
    center = cal_centroid(balls)

    totalsize = sum([a['size'] for a in balls])
    
    totalT = 0
    for ball in balls:
        dis = (ball['position'] - center).length()-maxR*(1-ball['size']/totalsize)
        t = dis/mergeSpd(ball)
        if t > totalT:
            totalT = t
        
    return totalT

#
def mergeballsCenterIndanger(balls, otherballs, infos=None):
    dangers = []
    myballs_indanger = []
    
    if len(balls) == 1:
        return myballs_indanger, dangers
    
    center = cal_centroid(balls)
        
    for myball in balls:
        myballdestP = center
        
        if infos is not None:
            addPathInfo(infos, myball, myballdestP)
            
        for destball in otherballs:
            if isInter(myball, myballdestP, destball, goMerge=True):
                if myball not in myballs_indanger:
                    myballs_indanger.append(myball)
                if destball not in dangers:   
                    dangers.append(destball)

    return myballs_indanger, dangers

def mergeballsCenterHasThorn(balls, otherballs, infos=None):
    
    if len(balls) == 1:
        return False
    
    center = cal_centroid(balls)
        
    for myball in balls:
        myballdestP = center
        
        if infos is not None:
            addPathInfo(infos, myball, myballdestP)
            
        for destball in otherballs:
            if pathHas(myball, myballdestP, destball):
                return True

    return False

def mergeTime(ball1,ball2):
    dis = (ball1['position'] - ball2['position']).length() - max(ball1['radius'], ball2['radius'])
    return dis/(mergeSpd(ball1)+mergeSpd(ball2)) + 1.5 # stops 时间啥的

def smallToBigMergetTime(ball1,ball2):
    d = (ball1['position'] - ball2['position']).length()
    v = max(ball1['vmax'], ball2['vmax'])
    V = min(ball1['vmax'], ball2['vmax'])
    r = min(ball1['radius'], ball2['radius'])
    R = max(ball1['radius'], ball2['radius'])
    x =  (V*(d -2*R + r) + v*R )/(v-V+0.0000000001) 
    t = (x-R)/(V+0.0000000001) 
    return t

def mergeSpd(ball):
    return 40/math.sqrt(ball['radius'])

def isSameDir(v1,v2):
    return v1.x * v2.x + v1.y * v2.y > 0

def getRect(balls1):
    xmin = 9999
    xmax = 0
    ymin = 9999
    ymax = 0
    
    for ball in balls1:
        if ball['position'].x - ball['radius']  < xmin:
            xmin = ball['position'].x - ball['radius']
        if ball['position'].x + ball['radius'] > xmax:
            xmax = ball['position'].x + ball['radius']
        if ball['position'].y - ball['radius'] < ymin:
            ymin = ball['position'].y - ball['radius']
        if ball['position'].y + ball['radius'] > ymax:
            ymax = ball['position'].y + ball['radius']
            
    return xmin, xmax, ymin, ymax

def getSurround(balls1, balls2):
    xmin, xmax, ymin, ymax = getRect(balls1)
    
    for ball in balls2:
        if ball['position'].x < xmax and ball['position'].x > xmin and ball['position'].y < ymax and ball['position'].y > ymin:
            return ball
        
    return None

def manhatondis(b1, b2):
    return abs(b1['position'].x -  b2['position'].x)+abs(b1['position'].y -  b2['position'].y)

def calcSpd(lastballs, nowballs, act_per_sec = 5):
    for nowball in nowballs:
        last = None
        mindis = 1000000000000
        
        if lastballs is None:
            nowball['vpf'] = nowball['v'] = Vector2(0,0)
            nowball['age'] = 0
            nowball['last'] = None
        else:
            for lastball in lastballs:
                
                if nowball['team'] != lastball['team']:
                    continue
                if nowball['player'] != lastball['player']:
                    continue    
                
                xdis = nowball['position'].x -  lastball['position'].x   
                ydis = nowball['position'].y -  lastball['position'].y   
                dis = xdis*xdis + ydis*ydis
                if  dis < mindis:
                    mindis = dis
                    last = lastball
            
            nowball['last'] = last # 这个可能有问题
            if last is  None:
                nowball['vpf'] = nowball['v'] = Vector2(0,0)
                nowball['age'] = 0
            else:
                lastball  = last       
                
                if "matchs" not in last: # TODO  还要引入 吐球方向, 太烦了
                    last['matchs'] = []
                
                xdis = nowball['position'].x -  lastball['position'].x   
                ydis = nowball['position'].y -  lastball['position'].y           
                nowball['v'] = Vector2(xdis,ydis)*act_per_sec
                
                vmax = max(nowball['vmax'], 30)
                if nowball['v'].length()>vmax:
                    nowball['v'] = nowball['v'].normalize()*vmax
                
                nowball['vpf'] = Vector2(xdis,ydis)
                
                if 'used' in last:
                    nowball['age'] = 0
                else:
                    last['used'] = True
                    nowball['age'] = last['age'] + 0.2
        
        nowball['vl'] = nowball['v'].length()   
        nowball['vpfl'] = nowball['vpf'].length()    

def totalSize(my_clone_balls):
    size = 0
    for ball in my_clone_balls:
        size += ball['size']
    
    return size
  
def in_coner(w, bigR, small, smallR):
    if smallR/bigR >=0.3:
        return False
    
    if small.x < bigR and small.y  < bigR:
        return (Vector2(bigR,bigR) - small).length() >= bigR
    if small.x > w - bigR and small.y < bigR:
        return (Vector2(w - bigR, bigR) -small).length() >= bigR
    if small.x < bigR and small.y > w - bigR:
        return (Vector2(bigR, w - bigR) -small).length() >= bigR
    if small.x > w - bigR and small.y > w - bigR:
        return (Vector2(w - bigR, w - bigR) -small).length() >= bigR
    return False

def inCircle(p, o, r):
    return (p-o).length() < r

def update_centers(data_set, assignments, old_centers):
    
    new_means = defaultdict(list)
    centers = []
    r_arr = []
    
    
    for i in range(len(old_centers)):
        mean = Vector2(0,0)
        s = 0
        r = 0
        
        points = []
        
        for j in range(len(data_set)):
            assignment = assignments[j]
            p = data_set[j]

            if assignment == i:
                points.append(p)
                
        
        if len(points)> 0:
            for p in points:  
                mean += p['size']*p['position']
                s += p['size']
            
            mean = mean/s
            r = sqrt(s)
            s = s/len(points)
        else:
            mean = old_centers[i]
            
        r_arr.append(r)
        centers.append(mean)

    return centers, r_arr

def assign_points(data_points, centers, r_arr):

    assignments = []
    for point in data_points:
        shortest = 99999999999999  # positive infinity
        shortest_index = 0
        
        for i in range(len(centers)):
            val =  (point['position'] - centers[i]).length() - point['radius'] #- r_arr[i] #distance 
            # print(i,val)
            if val < shortest:
                shortest = val
                shortest_index = i
                
        assignments.append(shortest_index)
    # print(assignments)
    return assignments

def generate_k(data_set, k):
    centers = []
    min_max = defaultdict(int)

    for point in data_set:
        pos = point['position']
        
        i = 'x'
        val = pos.x
        min_key = 'min_' + i
        max_key = 'max_' + i
        if min_key not in min_max or val < min_max[min_key]:
            min_max[min_key] = val
        if max_key not in min_max or val > min_max[max_key]:
            min_max[max_key] = val
            
        i = 'y'
        val = pos.y
        min_key = 'min_' + i
        max_key = 'max_' + i
        if min_key not in min_max or val < min_max[min_key]:
            min_max[min_key] = val
        if max_key not in min_max or val > min_max[max_key]:
            min_max[max_key] = val
            

    xdiv = (min_max['max_x'] - min_max['min_x'])/4
    ydiv = (min_max['max_y'] - min_max['min_y'])/4
    
    for _k in range(k):
        pos = Vector2(0,0)
        
        i = 'x'
        min_val = min_max['min_' + i]
        max_val = min_max['max_' + i]
        pos.x = min_val +  (max_val-min_val)/(k+1)*(_k+1)     #(uniform(min_val, max_val))
        
        i = 'y'
        min_val = min_max['min_' + i]
        max_val = min_max['max_' + i]
        pos.y =  min_val +  (max_val-min_val)/(k+1)*(_k+1)     #(uniform(min_val, max_val))

        centers.append(pos)
        
    if xdiv!=0 and ydiv !=0:
        if k == 3:
            centers = []
            centers.append( Vector2(min_max['min_x']+2*xdiv, min_max['min_y']+1*ydiv) )
            centers.append( Vector2(min_max['min_x']+1*xdiv, min_max['min_y']+3*ydiv) )
            centers.append( Vector2(min_max['min_x']+3*xdiv, min_max['min_y']+3*ydiv) )
        if k == 4 or k==5:
            centers = []
            centers.append( Vector2(min_max['min_x']+1*xdiv, min_max['min_y']+1*ydiv) )
            centers.append( Vector2(min_max['min_x']+1*xdiv, min_max['min_y']+3*ydiv) )
            centers.append( Vector2(min_max['min_x']+3*xdiv, min_max['min_y']+1*ydiv) )  
            centers.append( Vector2(min_max['min_x']+3*xdiv, min_max['min_y']+3*ydiv) )    
        if k == 5:
            centers.append( Vector2(min_max['min_x']+2*xdiv, min_max['min_y']+2*ydiv) ) 

    # print(centers)
    return centers

def k_means(dataset, k):
    # print("==========================================", k)
    k_points = generate_k(dataset, k)
    r_arr = [0] * k
    assignments = assign_points(dataset, k_points, r_arr)
    old_centers = k_points
    old_assignments = None
    while assignments != old_assignments:
        new_centers, r_arr = update_centers(dataset, assignments, old_centers)
        old_assignments = assignments
        old_centers = new_centers
        assignments = assign_points(dataset, new_centers, r_arr)
    
    new_centers, r_arr = update_centers(dataset, assignments, old_centers)
    
    while 0 in r_arr:
        idx = 0
        for i in range(len(r_arr)):
            if r_arr[i] == 0:
                idx = i
                break
        r_arr.remove(0)
        del new_centers[idx] 
        
        for i in range(len(assignments)):
            if assignments[i] > idx:
                assignments[i] -= 1
                
    nl = list(zip(new_centers, r_arr))
    nl.sort(key=lambda a: a[1], reverse=True)
    
    # print('============')
    # print(assignments)
    # print(r_arr)
    # print(nl)
    
    _new_centers = [a[0] for a in nl]
    _r_arr = [a[1] for a in nl]
    
    # print(_new_centers)
    for i in range(len(assignments)):
        assignments[i] =   _new_centers.index(new_centers[assignments[i]])
        
    # print(assignments)    
    return assignments, _new_centers, _r_arr

def wss(assignments, dataset, new_centers, r_arr):
    w = 0
    for i in range(len(assignments)):
        ball = dataset[i]
        idx =  assignments[i]
        p = ball['position'] - new_centers[idx]
        dis = p.length() - r_arr[idx]
        if dis <0:
            dis = 0
        w += dis*dis
    return w

def judge_K(balls):
    
    assignments = []
    new_centers = []
    r_arr = []
    
    if len(balls) == 1:
        return 1, [0], [balls[0]['position']] , [balls[0]['radius']]
    
    if len(balls) == 2:
        dis = (balls[0]['position'] - balls[1]['position']).length()
        if dis > (balls[0]['radius'] + balls[1]['radius'])*2:
            return 2, [0,1], [balls[0]['position'], balls[1]['position']], [balls[0]['radius'], balls[1]['radius']]
        else:
            total_size, r, mass_p, mass_v =  centerInfo(balls)
            return 1, [0,0], [mass_p], [r]
    
    assignments_a = []
    new_centers_a = []
    r_arr_a = []
    
    wss_a = [] 
    K = min(len(balls),5)
    
    # print("========================")
    
    for k in range(1, K+1):
        assignments, new_centers, r_arr = k_means(balls, k)
        assignments_a.append(assignments)
        new_centers_a.append(new_centers)
        r_arr_a.append(r_arr)
        
        ws = wss(assignments, balls, new_centers, r_arr)
        wss_a.append(ws)
        
        # print(k ,ws, assignments )
        
    calcKMax = 1
    for k in range(1, K+1):
        assignments = assignments_a[k-1]
        # print(assignments, wss_a[k-1])
        calcKMax = max(max(assignments)+1,calcKMax)
    
    # print("cr", cr)
    best = calcKMax
    
    # cr = balls[0]['radius']+balls[1]['radius']
    # if len(balls) == 2:
    #     if wss_a[k-1] <= cr:
    #         best = 1
    #     else:
    #         best = 2
    # else:
        
    for k in range(2, calcKMax):
        ratio = (wss_a[k-2] - wss_a[k-1])/(wss_a[k-2]+0.00000001)
        
        # print(k, wss_a[k-1], ratio)
        if ratio<0.33:
            best = k-1
            break
    
    # if best > len(assignments_a) or best > len(new_centers_a) or best > len(r_arr_a):
    #     print(best)
    return best, assignments_a[best-1], new_centers_a[best-1], r_arr_a[best-1]
  
def convertAss(assignments, k):
    _assignments = {}
    for i in range(k):
        _assignments[i] = []
    for id in range(len(assignments)):
        _assignments[assignments[id]].append(id) 
    return _assignments
    
def eatThornNeedMerge(thorns_balls, my_clone_balls, totalSize):
    
    srcBall = None
    needTogether = False
    dest_balls = thorns_balls
    
    min_dest_ball = None
    
    # 如果没有能直接吃的
    min_distance = 1000000
    for dest_ball in dest_balls:
        if dest_ball['size'] + 1 < totalSize:   #  合的时候有衰减
            needTogether = True
            srcBall = my_clone_balls[0]
            distance = (dest_ball['position'] - srcBall['position']).length()
            if distance < min_distance:
                min_distance = distance
                min_dest_ball = (dest_ball)
        
    return srcBall, min_dest_ball, needTogether, min_distance
    
def balls_by_team_player(clone_balls,team_name):
    
    my_balls = {}
    other_balls = []
    other_team_balls = {}
    other_team_players = {}
 
    clone_balls.sort(key=lambda a: a['radius'], reverse=True)
    
    for clone_ball in clone_balls:
        tname = clone_ball['team']
        pname = clone_ball['player']
        tpn = tname+"_"+pname
        if tname == team_name:
            if pname not in my_balls:
                my_balls[pname] = []
                
            my_balls[pname].append(clone_ball)
        else:
            other_balls.append(clone_ball)
            
            if tname not in other_team_players:
                other_team_players[tname] = {}
            if pname not in other_team_players[tname]:
                other_team_players[tname][pname] = []
                
            if tname not in other_team_balls:
                other_team_balls[tname] = []    
            
            other_team_balls[tname].append(clone_ball)
            other_team_players[tname][pname].append(clone_ball)
            
    
    return my_balls, other_balls, other_team_balls, other_team_players

def preprocess(overlap):
    new_overlap = {}
    for k, v in overlap.items():
        if k =='clone':
            new_overlap[k] = []
            for index, vv in enumerate(v):
                tmp={}
                tmp['position'] = Vector2(vv[0],vv[1])
                tmp['radius'] = vv[2]
                tmp['size'] = vv[2]*vv[2]
                tmp['vmax'] = cal_vel_max(vv[2])
                if len(vv)>5:
                    tmp['spd'] = Vector2(vv[3],vv[4])
                tmp['player'] = str(int(vv[-2]))
                tmp['team'] = str(int(vv[-1]))
                
                newsize =  tmp['size']/2
                newR = math.sqrt(newsize)
                tmp['splitR'] =  newR
                tmp['splitD'] =  newR * 2 + 15
                tmp['splitP'] =  newR * 2 + splitForward(tmp)
                tmp['splitE'] =  newR * 2 + 15 + newR
                tmp['sporeD'] =  tmp['radius'] + 40
                
                new_overlap[k].append(tmp)
        else:
            new_overlap[k] = []
            for index, vv in enumerate(v):
                tmp={}
                tmp['position'] = Vector2(vv[0],vv[1])
                tmp['radius'] = vv[2]
                tmp['size'] = vv[2]*vv[2]
                new_overlap[k].append(tmp)
    return new_overlap

def addSplitInfo(balls, teamsplit):
    
    num = len(balls)
    
    for ball in balls:
        ball['cansplit'] = False
        if ball['size']>=100:
            if num + 1<=16:
                num += 1
                ball['cansplit'] = True
        
        if ball['cansplit'] and teamsplit:
            ball['willsplit'] = True
        else:
            ball['willsplit'] = False

def getballSplitMin(balls):
    
    sizes = []
    for ball in balls:
        sizes.append(ball['size'])
    
    split  = 0
    while len(sizes) < 16 and sizes[0]>=100:
        
        split += 1
        oldsize = sizes
        sizes = []
        
        for i in range(len(oldsize)):
            s = oldsize[i]
            
            if s<100:
                break
            
            sizes.append(s/2)
            sizes.append(s/2)

            if i< len(oldsize) and  len(sizes) + len(oldsize) - i>16:
                sizes.extend(oldsize[i+1:])
                break
            
        sizes.sort(reverse=True)
    
    return sizes, split

def getballSplit(balls, destSize):
    
    sizes = []
    for ball in balls:
        sizes.append(ball['size'])
    
    if sizes[-1] < destSize:
        return True, sizes, 0
    
    split  = 0
    while len(sizes) < 16 and sizes[0]>=100:
        
        split += 1
        oldsize = sizes
        sizes = []
        
        for i in range(len(oldsize)):
            s = oldsize[i]
            
            if s<100:
                break
            
            sizes.append(s/2)
            sizes.append(s/2)

            if i< len(oldsize) and  len(sizes) + len(oldsize) - i>16:
                sizes.extend(oldsize[i+1:])
                break
            
        sizes.sort(reverse=True)
        
        if sizes[-1] < destSize:
            return True, sizes, split
        
    
    return False, sizes, split

def centerInfo(balls):
    total_size = 0
    for ball in balls:
        total_size += ball['size']
    mass_p = Vector2(0,0)
    for ball in balls:
        mass_p += ball['position']* ball['size']/total_size
    mass_v = Vector2(0,0)
    for ball in balls:
        mass_v += ball['v']* ball['size']/total_size
    
    return total_size, math.sqrt(total_size), mass_p, mass_v

def cos(d1,d2):
    if d1.length() == 0 or d2.length() == 0:
        return 0
    angle = d1.angle_to(d2)
    angle_rad = math.radians(angle)
    return math.cos(angle_rad)

# 考虑了双方当前速度方向的相遇, a到b的时间, 考虑了方向, 用于 近距离 
def a2bT(dis, ball, destball, dir):
    
    cosA = cos(ball['v'], dir)
    cosB = cos(destball['v'], dir)
    v = ball['vmax'] +  destball['vmax']*cosB 
        
    v = max(v, 0.001)
    t = max(dis/v, 0.01) + 0.1*(1-cosA)       
    return t 

#  TODO 弄上安全范围 , 然后还要判断对面是不是一家的  tsq tmp
def findSplitEat(my_clone_balls, others_teams_clone_balls, w):
    
    eat_dest = None
    
    if len(my_clone_balls) >=16:
        return  None, None
    
    for i in range(len(my_clone_balls)) :
        
        if i+1 + len(my_clone_balls) > 16:
            break
        if eat_dest is not None:
            break
        
        myball = my_clone_balls[i]
        myballP = myball['position']
        myballR = myball['radius']
        myballV = myball['v']
        myballVl = myball['vl']
        
        newsize =  myball['size']/2
        # newR = math.sqrt(newsize)
        splitR = myball['splitR']
        splitD = myball['splitD']
        
        for ball in others_teams_clone_balls:
            
            dest = ball
            destSize = dest['size']
            destP = dest['position'] + dest['v']*0.2 # 待调
            if newsize > destSize + 10: # 冗余
                
                if in_coner(w, myballR, destP, dest['radius']):
                    # print(" 在角落没法吃======")
                    continue
                
                dir = destP - myballP
                dir = dir.normalize()
                newP = myballP + dir * splitD # (newR * 2 + 20) 
                d = newP - destP
                
                if d.length() <  splitR:
                    
                    cansplit = True
                    # 还要判断分裂后, 会不会被对方其他的直接吃掉  , 
                    if False:
                        for other_ball2 in others_teams_clone_balls:
                            size2 = other_ball2['size']
                            if size2 <= destSize:
                                break
                            
                            if newsize < size2:
                                d2 = newP - other_ball2['position']
                                if d2.length() <= other_ball2['radius'] + 10:
                                    # danger
                                    cansplit = False
                                    break
                                
                    if cansplit:
                        eat_dest = destP
                        direction = dir
                        break 

    return eat_dest, myball

def keyv(ClassName, pv):
    dict = {attr: ClassName.__dict__[attr] for attr in ClassName.__dict__ if not callable(getattr(ClassName, attr)) and not attr.startswith('__')}
    for k, v in dict.items():
        if pv==v:
            return k

class ACT:
    NO = -1
    Spore_Mov = 0
    Split_Mov = 1
    Stop2Merge = 2
    Spore = 3
    Split = 4
    pass

class PolicyAct:
    
    NO = -1
    SINGLE = 888
    
    SOLO = 0
    SOLO_NO_CHASE = 100 
        
    Eat_Free = 1   # 吃刺 first
    Move_First = 4
    Eat_Spores = 8
    Eat_Fix = 9
    Eat_Front = 10
    
    
    Eat_Thorn_to_dest = 40  #
    
    Eat_other_No_thorn = 50
    SendSelfToMain = 60
    
    Go_To_Pos = 20
    Eat_Spore_With_Dest = 110
    Eat_Team = 130
    
class TeamAct:
    
    No_Team = -1
    First_Free = 0  # Init 
    First_Move_Together = 10
    Self_Merge = 20
    
    Patrol_Init = 300
    Patrol_Move = 310 
    Patrol_Split = 320
    Patrol_Merge = 330
    
    Kill = 400
    
    Split_Eat = 350
    MergerToP2 = 360
 
class Intent:  
    
    T_Eat = 1
    T_S_Eat = 2
    T_SS_Eat = 3
    
    T_Mov = 10
    T_Eat_Thorn = 11
    T_Food = 12
    
    T_Merge = 20
    T_Merge_S2B = 21
    
    T_Merget_Eat = 22
    
    T_Escape = 30
    T_Escape_Split = 31
    
    T_Go_After = 40
    T_Go_A_AllOk = 41
    T_Go_A_SomeOk = 42
    
    T_eat_team = 110
    
    
    def __init__(self):
        
        self.type = None
        self.destball = None
        self.destPos = None
        self.srcPos = None
        self.dir = None
        self.myball = None
        self.score = 0
        self.t = 999999
        
        self.go_after_t = Intent.T_Go_A_AllOk
        

class MyBotAgent:
    
    def __init__(self, name=None):
        self.name = name
        self.actions_queue = queue.Queue()
        
        self.must_act_que = queue.Queue()
        
        self.last_clone_num = 1
        self.last_total_size = 0
        self.team_name = None
        
        self.last_complex = None
        self.tog_eat_thorn = None
        
        self.to_eat_other = None
        
        self.split_radius_min = 100
        
        self.eatAreaCenter = None
        self.goto_dest = None
        self.eatTeam = []
        self.main_ball = None
        
        
        self.allballEatThorn = True
        
        self.stopTick = 0
        
        
        self.destAgent = None
        
        self.cur_act = PolicyAct.SINGLE
        
        
        self.merge_idx = 0
        self.needMerge = False
        
        self.needSplitDir = None 
        
        self.cur_intent = None
        self.last_intent = None
        
        self.max_threat_s  =0
        
        self.debugidx = 0
        self.debugInfos = []
        
    # 算质心 啥的    
    def process(self):
        
        overlap = self.overlap
        food_balls = overlap['food']
        thorns_balls = overlap['thorns']
        spore_balls = overlap['spore']
        clone_balls = overlap['clone']
        
        # 
        addSplitInfo(self.myballs, True)
        
        self.total_size = totalSize(self.myballs)
        self.mass_r = math.sqrt(self.total_size)
        self.max_r = self.myballs[0]['radius']
        self.min_r = self.myballs[-1]['radius']
        self.max_pos = self.myballs[0]['position']
        
        self.mass_p = Vector2(0,0)
        for ball in self.myballs:
            self.mass_p += ball['position']* ball['size']/self.total_size
        self.mass_v = Vector2(0,0)
        for ball in self.myballs:
            self.mass_v += ball['v']* ball['size']/self.total_size
            
        best , assignments, new_centers, r_arr = judge_K(self.myballs)
        _assignments = convertAss(assignments, len(new_centers))
        self.assignments = _assignments
        self.centers = new_centers
        self.r_arr = r_arr
        
    def eatSmall(self, destballs, dir=None):
        
        direction = None
        
        my_clone_balls = self.myballs
        
        bestIntent, allintents = self.find_eat_balls(destballs, my_clone_balls, self.W, maxDis=50, follow_dir=dir) 
        
        if bestIntent is not None and bestIntent.score>0:
            
            min_food_ball = bestIntent.destball
            srcBall = bestIntent.myball
            if srcBall["radius"] < 8 and len(my_clone_balls)<6:
                direction = getDirTo(srcBall, min_food_ball['position'])  # 快速
            else:
                direction = (min_food_ball['position'] - srcBall['position']).normalize()

        
            intent = bestIntent
            intent.dir  = direction
            intent.type = Intent.T_Food
            self.cur_intent = intent
            
            self.infos.append((88, min_food_ball['position'], 5, YELLOW))
            
            return [direction.x, direction.y,  ACT.NO]
        
        return None
    
    def eatFoodAndSpores(self, dir=None):
        food_balls = self.overlap['food']
        spore_balls = self.overlap['spore']
        destballs = food_balls + spore_balls

        return self.eatSmall(destballs, dir=dir)
        
    def eatFood(self, dir=None):
        food_balls = self.overlap['food']
        return self.eatSmall(food_balls, dir=dir)
    
    def eatSpores(self, dir=None):
        spore_balls = self.overlap['spore']
        return self.eatSmall(spore_balls, dir=dir)

    def eatThorn(self, dir=None):
        #
        thorns_balls = self.overlap['thorns']
        
        mycloneballs = self.myballs
        if not self.allballEatThorn:
            mycloneballs = self.myballs[1:]
        
        bestIntent, allintents = self.find_eat_balls(thorns_balls, mycloneballs, self.W,  infos=self.infos, follow_dir=dir) 
        
        for intent in allintents:
            
            if intent.score <= 0:
                continue
        
            dest_thorn = intent.destball
            srcBall = intent.myball
            
            self.infos.append((88, dest_thorn['position'], dest_thorn['radius'], YELLOW))
            # self.infos.append((1, srcBall['position'], min_dest_thorn['position'], GREEN))
            
            destPos = dest_thorn['position']
            
            # 小的吃刺, 不小心碰到大的
            if not self.allballEatThorn:
                bigOk = True
                ball0 = self.myballs[0]
                ball0destP = intent.dir.normalize()*intent.t*ball0['vmax'] + ball0['position']
                
                for thorn in thorns_balls:
                    addPathInfo(self.infos, ball0, ball0destP, color=RED)
                    self.infos.append((8, ball0destP, ball0['radius'],RED))
                    if pathHas(ball0, ball0destP, thorn):
                        bigOk = False
                        break
                
                if not bigOk:
                    continue      
            
            
            intent.type = Intent.T_Eat_Thorn
            self.cur_intent = intent
            
            return [intent.dir.x, intent.dir.y,  ACT.NO]
            
        return None
    
    
    def find_eat_balls(self, dest_balls, my_clone_balls, width, maxDis=500, follow_dir=None, infos=None):
        
        # 方向不一样, 则可能没有
        max_score = -99999999999
        if follow_dir is not None:
            max_score = 0
            
        min_dest_ball = None
        srcBall = my_clone_balls[0]
        
        allintents = []
        bestIntent = None
        
        
        for myball in my_clone_balls:
            mypos =  myball['position']
            myVl = myball['vl']
            myballR = myball['radius']
            for dest_ball in dest_balls:
                dest_pos = dest_ball['position']
                

                if dest_ball['radius'] >= myball['radius']:
                    continue
                
                dir = dest_pos - mypos
                distance = dir.length() - myball['radius'] 
                
                if distance>maxDis:
                    continue
                
                if in_coner(width, myballR, dest_pos, dest_ball['radius']):
                    continue
                
                
                t = distance /  myball['vmax']  
                # 如果方向差的多, 则加时间
                c = cos(myball['v'], dir)
                if c<0:
                    t += (-c)*0.2
                
                if t<=0:
                    t= 0.001
                
                s = dest_ball['radius']
                
                myballs_indanger, dangers = gothrough(myball, dest_ball['position'], self.myballs, self.other_balls, aheadT=t, infos=infos)
                for subball in myballs_indanger:
                    s -= subball['size']* 1.5
                
                if self.eatAreaCenter is not None:
                    bigdestDis = (self.eatAreaCenter - dest_ball['position']).length()
                    if bigdestDis > 400: #离目标太远
                        continue
                    t = t + bigdestDis/myball['vmax']
                
                
                if s>0:
                    s = s/t
                
                if follow_dir is not None:
                    s = s * cos(follow_dir,dir)
                

                
                # print(distance,myball['vmax'], s)
                
                
                intent = Intent()
                intent.destPos = dest_ball['position']
                intent.destball = dest_ball
                intent.t = t
                intent.myball = myball
                intent.score =s
                intent.dir = dir
                
                # debug1.append(s)
                # debug2.append(dest_ball)
                # debug3.append(myball)
                
                allintents.append(intent)
                
                if s > max_score:
                    bestIntent = intent
                    max_score = s
        
        allintents.sort(key=lambda a: a.score, reverse=True)
        
        infos = self.infos
        if infos is not None and len(allintents)>0: 
            scores = [int.score for int in allintents]       
            maxs = max(scores)
            mins = min(scores)
            if maxs==mins:
                maxs=mins+1
            
            for intent in allintents:
                s = intent.score
                rgb = int(255*  (s-mins)/(maxs-mins))
                infos.append((88, intent.destPos, intent.destball['radius'] , (0,rgb,0))) 
        
        return bestIntent, allintents  

    def findMinTeamBall(self , dest_balls, my_clone_balls, width):
        
        # 方向不一样, 则可能没有
        score = -99999999999
            
        min_dest_ball = None
        srcBall = my_clone_balls[0]
        
        if len(dest_balls) == 1:
            return score, srcBall, min_dest_ball  
        
        for myball in my_clone_balls:
            mypos =  myball['position']
            myVl = myball['vl']
            myballR = myball['radius']
            for dest_ball in dest_balls:
                dest_pos = dest_ball['position']
                
                if in_coner(width, myballR, dest_pos, dest_ball['radius']):
                    continue
                
                if dest_ball['radius'] >= myball['radius']:
                    continue
                
                # 吃队友, 考虑危险, 这个待 优化 TODO
                # if centerHasBigger(myball,dest_ball['position'], self.other_balls,myballR):
                #     continue
                
                dir = dest_pos - mypos
                distance = dir.length() - myball['radius'] 
                t = distance /  myball['vmax']  
                
                s = 1/t
                s = s * dest_ball['radius']
                
                # debug1.append(s)
                # debug2.append(dest_ball)
                # debug3.append(myball)
                
                if s > score:
                    score = s
                    min_dest_ball = (dest_ball)
                    srcBall = myball

        return score, srcBall, min_dest_ball   
    
    def eatTeamBalls(self):
        score, srcBall, min_dest = self.findMinTeamBall(self.eatTeam, self.myballs, self.W) 
        
        if min_dest is not None : # 有一个目标
            
            # self.infos.append((88, min_dest['position'], 5, GREEN))
            # self.infos.append((1, srcBall['position'], min_dest['position'], GREEN))
            
            destPos = min_dest['position']
            direction = (min_dest['position'] - srcBall['position']).normalize()
            
            intent = Intent()
            intent.type = Intent.T_eat_team
            intent.destPos = destPos
            self.cur_intent = intent
            
            return [direction.x, direction.y,  ACT.NO]
   
    def isAllPointCenterMass(self, p=0.9):
        
        if len(self.myballs)== 1:
            return True
        
        total = 0
        for ball in self.myballs:
            dir = self.mass_p - ball['position']
            angle = dir.angle_to(ball['v']) 
            
            # self.infos.append((1,  ball['position'], self.mass_p, YELLOW))
            
            angle = abs(angle)
            if angle>180:
                angle = 360 -angle
            total += abs(angle)/180
            
            rgb = abs(angle)/180*255
            # self.infos.append((1,  ball['position'], ball['position']+ball['v']*1000, (rgb,rgb,rgb)))
            
            #     return False
        re = total/(len(self.myballs))
        # print("dirs", re)    
        return  1-re > p
    
    def isAllPointCenter(self, p=0.9):
        
        if len(self.myballs)== 1:
            return True
        
        total = 0
        for ball in self.myballs:
            if ball == self.myballs[0]:
                continue
            dir = ball['position'] - self.myballs[0]['position']
            angle = dir.angle_to(ball['v']) 
            
            # self.infos.append((1,  ball['position'], self.mass_p, YELLOW))
            
            angle = abs(angle)
            if angle>180:
                angle = 360 -angle
            total += abs(angle)/180
            
            rgb = abs(angle)/180*255
            # self.infos.append((1,  ball['position'], ball['position']+ball['v']*100, (rgb,rgb,rgb)))
            
            #     return False
        re = total/(len(self.myballs)-1)
        # print("dirs", re)    
        return  re > p

    def isCentered(self, mainP=0.5, per=0.75, num=8):
        # 判断是不是
        if len(self.myballs) == 1:
            return True
        
        if len(self.myballs) > num:
            return False
        
        # print ("0:" , self.myballs[0]['size']/self.total_size)
        if self.myballs[0]['size']/self.total_size < mainP:
            return False
        
        size = 0
        for ball in self.myballs:
            
            centerdir = ball['position'] - self.mass_p
            if centerdir.length() < self.mass_r:
                size += ball['size'] 
        
        # print ("in:" , size/self.total_size)
        if size/self.total_size >= per:
            return True
        
        return False

               
    # 没考虑 附近有敌人的情况. 因为合球的时候, 会不动 
    def calc_merge(self):
        
        # 外围出现了 敌方比较大的, 可能需要合球
        indangers = []
        dangerDests = []
        dt = 8
        for myball in self.myballs:
              
            if myball['size']/self.total_size<0.09:
                continue
                
            for dest in self.other_balls: 
                dis = (myball['position'] - dest['position']).length()
                if dest['size']> myball['size'] and dis / dest['vmax'] < dt:
                    if myball not in indangers:
                        indangers.append(myball)
                        if dest not in dangerDests:
                            dangerDests.append(dest)

        
        dangerDests.sort(key=lambda a: a['size'], reverse=True)
        for ball in dangerDests:
            self.infos.append((8, ball['position'], ball['radius'], BLUE))
        for ball in indangers:
            self.infos.append((8, ball['position'], ball['radius'], BLUE))
        
        
        if len(self.myballs) == 1: # 直接停
            if self.merge_idx>0:
                self.merge_idx = 0
                self.needMerge = False
            
            
            if len(indangers)==0:
                if self.myballs[0]['size'] > 800:
                    self.needSplitDir = Vector2(0,0)
                pass
                        
            return 
        
        ball0 = self.myballs[0]
        
        # 计算
        canMergeP = 0
        canSize = 0
        
        outerOKp = 0
        outerOkSize = 0
        outerOkNum = 0
        outerCanEatThorn = 0
        outerMaxDis =  0
        outerMaxBall = self.myballs[1] 
        
        minWaitT = 20  
        
        for i in range(0, len(self.myballs)):
            
            ball = self.myballs[i]
            
            if i>0 and ball['size']>=90:
                outerCanEatThorn += 1
                tmpdis = (ball['position'] - ball0['position']).length()
                if tmpdis > outerMaxDis:
                    outerMaxDis = tmpdis
                    outerMaxBall = ball
            
            if ball['age'] >= 20:
                canSize += ball['size']
                if i>0:
                    outerOkSize += ball['size']
                    outerOkNum += 1
            else:
                wt = 20 - ball['age']
                if wt < minWaitT:
                    minWaitT = wt
                    
        canMergeP = canSize/(self.total_size)
        outerOKp = outerOkSize/(self.total_size-ball0['size'])
        

        
        # 现在合球, 会不会遇到 敌人
        mergeDanger = False 
        myballs_indanger, dangers =  mergeballsCenterIndanger(self.myballs, self.other_balls)
        
        if len(myballs_indanger)>0:
            mergeDanger = True         
            

            
       
        # 判断要不要合!!       self.total_size < S1_FOOD_THORN
        
        
        # 平时,如果自己的球很多, 得到
        
        
        # 如果即将吃刺, 则不合!! TODO
        
         
        # 如果 出现了小的 在自己中间, 则合
        # 如果出现了大的在自己中间, 则跑, 要侧跑
        #======== 要不要合
        self.needMerge = False 
        if self.merge_idx == 0:
            if len(indangers) > 0: # 有
                # 敌人如果最大的超过自己的中位数, 则合球
                if dangerDests[0]['size'] > self.myballs[int(len(self.myballs)/2)]['size']:
                    # print("敌人如果最大的超过自己的中位数, 则合球")
                    self.needMerge = True
            else:
                
                if self.total_size < S1_FOOD_THORN:
                    # 比较小不用合
                    pass
                else:
                    # 
                    # 如果外面有几个可以吃刺的, 且其中有一些可以合的. 则合
                    # if len(self.myballs)>4 and outerOKp>0.4 and outerCanEatThorn>4:
                    #     self.needMerge = True 
                    
                    # 直接看最远的
                    # if  outerMaxDis>300 and :
                    #     print("太远, 需要合球")
                    #     self.needMerge = True 
                    # else: #能朝外多少
                    #     dir = outerMaxBall['position'] - ball0['position']
                    #     otherP =ball0['position']+ (dir).normalize()*300
                    #     self.infos.append((11, ball0['position'], otherP, RED))
                    pass
            # =======
            if self.needMerge:
                # 现在合球, 会不会吃到刺
                thorns_balls = self.overlap['thorns']
                hasThorn = mergeballsCenterHasThorn(self.myballs, thorns_balls, self.infos)         
                if hasThorn:
                    # print("============== 有刺 !!")
                    self.needMerge = False
                
                
                if mergeDanger:
                    # print(" 继续合有危险 !!")
                    for ball in dangers:
                        self.infos.append((8, ball['position'], 5, RED))
                    self.needMerge = False 

                # if self.last_intent is not None and self.last_intent.type == Intent.T_Eat_Thorn:
                #     center = cal_centroid(self.myballs)
                #     if (self.last_intent.destPos - center).length()<100:
                #         self.needMerge = False 
                     
                        
                
         
        # 判断怎么合 以及怎么 结束
        # 如果当前合球有危险, 则暂停
        # 如果合球能打败这个危险,则继续
        
        # TODO VIP 判断危险性!!!
        
        # print("canMergeP ", canMergeP, "outerOKp", outerOKp, "minWaitT", minWaitT)
        # ===========判断停止
        if self.merge_idx>0:
            # if self.canMergeP <  0.7
            if outerOKp < 0.2 and minWaitT>2.5:
                self.merge_idx = 0
                # print("没有太多值得合的了, 结束合球")
                self.needMerge = False
                pass   
            
            
            if len(indangers) == 0: #周围没敌人
                # 这个时候不用合到一个, 有几个可以吃外面刺的就行
                
                if outerCanEatThorn<4:
                    self.merge_idx = 0
                    self.needMerge = False
                    # print("周围没敌人 , 有几个可以吃外面刺的就行, 结束合球")
               
                pass
                
        # # merg的危险性
        # if len(self.myballs) >=2:
            
        #     if self.total_size > 1000:
            
        #         if len(self.centers)> 1:
        #             # 两个中心且离得较远
        #             maxdis = 0
                    
        #             center1 = self.centers[0]
        #             for center2 in self.centers:
        #                 if self.centers[0] == center2:
        #                     continue
        #                 dis = (center1 - center2).length()
        #                 if dis > maxdis:
        #                     maxdis =  dis
        #             if dis >  self.r_arr[0]*6:
        #                 self.needMerge = True  
                    
        #             # # 两个中心且离得较远
        #             # if self.r_arr[1]*self.r_arr[1] / (self.r_arr[0]*self.r_arr[0]) > 0.3:
        #             #     return True

        #         # if len(self.myballs)>12:
        #         #     return True

        #         # if self.size_indanger> self.total_size*0.1:
        #         #     self.needMerge = True  
    
 
                
            # 如果没啥危险了, 可以走出
            
            # if self.size_indanger < self.total_size *0.05:
            #     self.merge_idx = None
            
            # if self.size_indanger< self.total_size*0.1:
            #     self.merge_idx = None
     
    
    def mergeAct(self):
        # 如果当前可以合 , 则继续合, 如果没有 intent了,也不继续合了
            
        if self.merge_idx == 0:
            if self.needMerge:
                self.merge_idx  = 1  # 开始
            
            
        if self.merge_idx >0: 
            #TODO 聚类, K是大于2类的那种
            # d = ball1['position']  - ball2['position']  
            # dis = d.length()
                
            # if dis > 2.5 *(ball2['radius']) and ball2['size']/ball1['size']<0.3:
            #     intent = Intent()
            #     intent.type = Intent.T_Merge_S2B
            #     intent.dir = ball1['position'] -  ball2['position']  
            #     intent.srcPos =  ball2['position']  
            #     intent.destPos = ball1['position']
            #     return intent
            
            intent = Intent()
            intent.type = Intent.T_Merge
            
            self.cur_intent = intent
                
             # 中间有个吐 TODO  不能全 stop , 特别慢   
            if intent.type == Intent.T_Merge_S2B:
                # self.infos.append((11, intent.srcPos, intent.destPos, Y2))
                return [intent.dir.x, intent.dir.y,  ACT.NO]
            else:
                if self.debugidx <3:
                    act = [None,None, ACT.Stop2Merge] 
                elif self.debugidx % 20 == 1:
                    # print("=== t ", self.global_state['last_time'], act, mergeTotalT(self.myballs))
                    act = [None,None, ACT.Stop2Merge]  
                else:
                    act = [None,None, -1]  
                
                
                self.merge_idx += 1    
                return act
        
     
     
    def calcMassDir(self):
        
        bigDir = Vector2(0,0)
        escapeDir = Vector2(0,0)
        atkDir = Vector2(0,0)
        
        numBig = 0
        numSame = 0
        numSmall = 0
        
        # 质心
        for t in self.other_team_player_info:
            team = self.other_team_player_info[t]
            for p in team:
                player = team[p]
                
                total_size = player['total_size']
                mass_p = player['mass_p']
                mass_v = player['mass_v']
                mass_r = player['mass_r']
                
                destR = mass_r
                dir = mass_p - self.mass_p 
                dis = dir.length()
                if dis > destR*8 and dis > self.mass_r*8:
                    # self.infos.append((8, mass_p, destR, WHITE))
                    pass
                else:
                    
                    bigDir += (self.mass_r - destR) /dis * dir.normalize() * self.mass_r
                    
                    if total_size  >= self.total_size*1.2:
                        numBig += 1
                    elif total_size< self.total_size*1.2 and total_size> self.total_size*0.8:
                        numSame += 1
                    else:
                        numSmall += 1
                    
                    if total_size >= self.total_size:
                        escapeDir += (self.mass_r - destR) /dis * dir.normalize() * self.mass_r
                        
                    #     if dis < destR*2:
                    #         self.infos.append((8, mass_p, destR, RED))
                    #     else:
                    #         self.infos.append((8, mass_p, destR, YELLOW))
                    # else:
                        
                    #     if total_size < self.total_size/2:
                    #         self.infos.append((8, mass_p, destR, BLACK))
                    #     else:
                    #         self.infos.append((8, mass_p, destR, (0,0,255)))
         
        # self.infos.append((1, self.mass_p, bigDir*100+self.mass_p, YELLOW))
    
   
    # 用于 离得很近的逃窜   myball 与 dest 的 碰撞时间 ,是不是小于 dt
    def esc_intent(self, myball, dest, dt=1, splitDt=1, preT=0.8):
        intent = None

        myballP = myball['position']
        myballR = myball['radius']
        myballS = myball['size']
        myballV = myball['v']
        myballVl = myball['vl']
        
        destR = dest['radius']
        destSize = dest['size']
        destP = dest['position']
        destV = dest['v']
        destVmax = dest['vmax']

        dir = myballP - destP
        dis = dir.length()
                
        if myballS < destSize:
            
            if centerHasBigger(dest, myball['position'], self.myballs, aheadT=0.2):
                return None
            
            destsplitR = dest['splitR']
            destsplitE = dest['splitE']
            splitdir = myballP - destP
            splitDis = splitdir.length()
            
            t = a2bT(splitDis-destsplitE, dest, myball, splitdir)
            if dest['willsplit'] and myballR < destsplitR and t <splitDt:
               
                if centerHasBigger(dest, myball['position'], self.myballs, aheadT=0.2, goballR=destsplitR*destsplitR):
                    return None
                
                destPn = dest['v']*min(t, preT) + destP
                dirPn = myballP - destPn
                
                intent = Intent()
                intent.type = Intent.T_Escape_Split
                intent.destball = dest      
                intent.myball =  myball
                intent.destPos = destPn   
                intent.dir =  dirPn
                intent.srcPos = myballP
                intent.t = t
            
            else:
                t = a2bT(dis-destR, dest, myball, dir)
                destPn = dest['v']*min(t, preT) + destP
                dirPn = myballP - destPn
                
                if t < dt:
                    intent = Intent()
                    intent.type = Intent.T_Escape
                    intent.destball = dest      
                    intent.myball =  myball
                    intent.destPos = destPn   
                    intent.dir = dirPn
                    intent.srcPos = myballP
                    intent.t = t
                    
        return intent
    
    # 一两帧的被吃
    def in_danger(self, myball, destP , other_balls=None):
        if other_balls is None:
            other_balls = self.other_balls
        
        return centerHasBigger(myball,destP, other_balls, maxGoT=0.4, aheadT=0.4)
    
    
    
    def getSplitNewBalls(self, dir, eatball=None):
        newballs = []    
        eated = []
        if eatball is not None:
            eated.append(eatball)
        
        for subball in self.myballs:
            if subball['willsplit']:
                
                newball = subball.copy()
                newball['radius'] = subball['splitR']
                newball['size'] = subball['splitR']*subball['splitR']
                newball['vmax'] = cal_vel_max(newball['radius'])
                
                newP = newball['position'] + dir.normalize()*newball['vmax']*0.2
                newball['newP'] = newP
                newballs.append(newball)
                
                newball2 = newball.copy()
                newball2['position'] = subball['position'] + dir.normalize()*subball['splitP']
                newP = dir.normalize()*subball['splitD'] + subball['position']
                newball2['newP'] = newP
                newballs.append(newball2)
                
                for sub_destball in self.other_balls:
                    if sub_destball in eated:
                        continue
                    if splitCanEat(newball2, sub_destball): # 其他的也可以吃别人
                        newball2['size'] += sub_destball['size']
                        newball2['radius'] = math.sqrt(newball2['size'])
                        newball2['vmax'] = cal_vel_max(newball2['radius'])
                        eated.append(sub_destball)
                
            else:
                newP = subball['position'] + dir.normalize()*newball['vmax']*0.2
                subball['newP'] = newP
                newballs.append(subball)
                
        return newballs, eated
    
    
    def splitHasDanger(self, dir, eatball=None):
        
        ballsIndanger = []
        
        newballs, eated = self.getSplitNewBalls(dir, eatball=eatball)   
        otherballs = []
        for ball in self.other_balls:
            if ball not in eated:
                otherballs.append(ball)
            pass
        
        for newball in newballs:
            if self.in_danger(newball, newball['newP'], other_balls=otherballs):
                ballsIndanger.append(newball)

        return newballs, ballsIndanger, eated
                
    
    # 个体的躲避 , 用上寻路 
    # 舍弃一些小的
    def escape_intents(self):

        for myball in self.myballs:
            # myball 的最大威胁
            max_threat = None
            max_threat_t = 999999999
            # 
            for dest in self.other_balls:
                intent  = self.esc_intent(myball, dest, dt=4, splitDt=3)
                if intent is not None and intent.t < max_threat_t:
                    max_threat_t = intent.t
                    max_threat = intent
                    
            if max_threat is not None:
                self.infos.append((1, myball['position'], max_threat.destPos, RED))
                max_threat.score = max_threat.myball['size']/(max_threat.t+0.01)
                self.EscapeIntents.append(max_threat)
        
        self.EscapeIntents.sort(key=lambda a: a.score, reverse=True)
           
    def escape(self):
        if len(self.EscapeIntents) > 0:
            intent = self.EscapeIntents[0]
            direction = intent.dir
            self.cur_intent = intent
            return [direction.x, direction.y,  ACT.NO] 
        
        return None
       
    # 单个的, 适合离得比较近, 简单情况 , 已经考虑了风险
    # score 分数 要修正
    def atk_intents(self):
        # 先吃 离得很近的, 随时可以吃掉的, 在没有吃和追逐目标的时候, 看大方向
        PRE_T = 2
        # 找能吃的
        for i in range(len(self.myballs)):
            myball = self.myballs[i]
            myballP = myball['position']
            myballR = myball['radius']
            myballS = myball['size']
            myballV = myball['v']
            myballVmax = myball['vmax']
            myballVl = myball['vl']
            
            splitR = myball['splitR']
            splitE = myball['splitE']
            myballcansplit = myball['willsplit']
            
            # 如果质量很小的, 占比很小, 可以不看
            if myballR < 5: # 吃对方没意义
                continue
            
            toEatIntent = None
            toEatScore = 0
            
            goafterIntent = None
            goafterScore = 0
            
            # 先吃 离得很近的, 随时可以吃掉的
            # 从最近的找起
            for ball in self.other_balls:
                
                dest = ball
                destR = dest['radius']
                destSize = dest['size']
                destP = dest['position']
                destV = dest['v']
                
                dir = destP - myballP
                dis = dir.length()
                t0 = dis/myball['vmax']
                dp0 = destP + destV* min(t0,PRE_T) 
                t = a2bT(dis-myballR, myball, dest, dp0-myballP)
                
                
                # if destSize<80 and destSize / self.total_size <0.03: # 太小不如吃刺
                #     continue
                if destSize < 50:
                    continue
                
                if myballS < destSize + 9:
                    continue
                
                if dis > 400:
                    continue
                
                destPn = destP + destV* min(PRE_T,t)
                dirN = destPn - myballP
                
                splitdir = destP - myballP
                splitDis = splitdir.length() + (2+cos(splitdir,destV))*0.2*dest['vmax']
                
                
                # self.infos.append((88, destPn, 5, RED))
                # self.infos.append((1, myballP, destPn, RED))

                if t <= 0.4:
                    
                    if in_coner(self.W, myballR, destP, destR):
                        continue
                    
                    # self.infos.append((11, myballP, destP, YELLOW))
                    intent = Intent()
                    intent.type = Intent.T_Eat
                    intent.destball = ball      
                    intent.myball =  myball
                    intent.destPos = destPn   
                    intent.dir =  dirN
                    
                    
                    intent.t = t
                    
                    score = destSize  # 直接吃
                    
                    myballs_indanger, dangers = gothrough(myball, destPn, self.myballs, self.other_balls, aheadT=t, infos=self.infos)
                    for subball in myballs_indanger:
                        score -= subball['size']*1.1
                    
                    if score > 0:
                        score = score/t
                    intent.score = score
                    if score > toEatScore:
                        toEatScore = score
                        toEatIntent = intent
                        
                elif myballcansplit  and splitR > destR  and splitDis <  splitE :
                    
                    if in_coner(self.W, splitR, destP, destR):
                        continue
                    
                    if centerHasBigger(myball, dest['position'], self.other_balls, aheadT=0.6, goballR=splitR*splitR):
                        continue
                    
                    intent = Intent()
                    intent.type = Intent.T_S_Eat
                    intent.destball = ball      
                    intent.myball =  myball
                    intent.destPos = destP   
                    intent.dir =  dir
                    
                    t = 0.2
                    intent.t = t
                    # 
                    score = 0

                    newballs, ballsIndanger, eated = self.splitHasDanger(dir, eatball=dest)  
                    for newball in newballs:
                        self.infos.append((88,newball['position'], newball['radius'],WHITE))
                    
                    for ball in ballsIndanger:
                        score -= ball['size']*1.1
                    
                    for ball in eated:
                        score += ball['size']
                    
                    if score>0:
                        score = score / t
                    intent.score = score
                    
                    if score > toEatScore:
                        toEatScore = score
                        toEatIntent = intent 
                        
                else: # 可以追
                    # 这个要 A*, 以及拦截
                    # 二分不能吃就算了, 以后改4分
                    if in_coner(self.W, splitR, destP, destR):
                        continue
                    
                    if myballS > destSize * 1.1:
                        
                        # 在外面放弃
                        pret = dis/dest['vmax']
                        pret = min(pret, PRE_T) # 远程追的pred
                        
                        dest_goafter = destP
                        if destV.length()>0:
                            dest_goafter = destP + destV.normalize()*dest['vmax']*pret
                        goafter_dir = dest_goafter - myballP
                        
                        
                        intent = Intent()
                        intent.type = Intent.T_Go_After
                        intent.destball = ball      
                        intent.myball =  myball
                        intent.destPos = dest_goafter   
                        intent.dir =  goafter_dir  
                        
                        
                        t = (dis-myballR)/myball['vmax']
                        intent.t = t
                        
                        #   
                        score = destSize      
                        loss = 0               
                        myballs_indanger, dangers = gothrough(myball, dest_goafter, self.myballs, self.other_balls)
                        for ballindanger in myballs_indanger:
                            loss += ballindanger['size']
                        
                        score -= loss*2
                        
                        score = score / t
                        intent.score = score
                        #
                        if score > goafterScore:
                            goafterScore = score
                            goafterIntent = intent
            
            if toEatIntent is not None: 
                self.toEatIntents.append(toEatIntent)
                if toEatIntent.type == Intent.T_S_Eat:
                    self.infos.append((11, toEatIntent.myball['position'], toEatIntent.destPos, Y2))
                else:
                    self.infos.append((1, toEatIntent.myball['position'], toEatIntent.destPos, GREEN))
            
            if goafterIntent is not None: 
                
                rect = self.player_states['rectangle']
                if inRect(goafterIntent.destPos,rect):
                    self.GoAfterIntents.append(goafterIntent)
                    self.infos.append((1, goafterIntent.myball['position'], goafterIntent.destPos, BLUE))
                    
        
        # for intent in self.toEatIntents:
        #     if intent.type == Intent.T_Eat and intent.score>0:
        #         self.infos.append((8, intent.destPos, intent.destball['radius']/2, YELLOW ))
                
        #     if intent.type == Intent.T_S_Eat and intent.score>0:
        #         self.infos.append((8, intent.destball['position'], intent.destball['radius']/2, RED))
        #         self.infos.append((8, intent.destPos, intent.destball['radius']/2, RED))
        
        
        self.toEatIntents.sort(key=lambda a: a.score, reverse=True)
        self.toEatIntents = [a for a in self.toEatIntents if a.score>0]
        
        #  TODO  如果对方很大, 自己很大, 但自己分散, 可能需要合球
        for intent in self.toEatIntents:
            # 先看对面势力
            team = self.other_team_player_info[intent.destball['team']]
            player = team[intent.destball['player']]
            
            total_size = player['total_size']
            mass_p = player['mass_p']
            mass_v = player['mass_v']
            mass_r = player['mass_r']
            
            # 如果大于, 还需要分裂吃, 就尽量别去
         
        # for intent in self.GoAfterIntents:
        #     self.infos.append((11, intent.srcPos, intent.destPos, GREEN))
        # for intent in self.EscapeIntents:
        #     self.infos.append((11, intent.srcPos, 2*intent.srcPos-intent.destPos  , RED))

        self.GoAfterIntents.sort(key=lambda a: a.score, reverse=True)

    def eat_others(self):
        
        if len(self.toEatIntents) > 0:
            intent = self.toEatIntents[0]
            direction = intent.dir
            
            self.cur_intent = intent
            
            if intent.type == Intent.T_Eat:
                return [direction.x, direction.y,  ACT.NO] 
            if intent.type == Intent.T_S_Eat:
                return [direction.x, direction.y,  ACT.Split_Mov] 
        return None 
    
    def tst_go_inter(self):
        
        ball0 = self.myballs[0]
        p = ball0['position']
        
        destP = Vector2(400, 400)
        self.infos.append((1, p, destP, YELLOW))
        
        dir = destP - p
        self.must_act_que.put([dir.x,dir.y,-1])
        
        div = 10
        
        other_balls = []
        for i in range(100):
            for j in range(100):
                
                tmp={}
                tmp['position'] = Vector2(div*i,div*j)
                r = math.sqrt(1200)
                tmp['radius'] = r
                tmp['size'] = r*r
                tmp['vmax'] = cal_vel_max(r)
                
                newsize =  tmp['size']/2
                newR = math.sqrt(newsize)
                tmp['splitR'] =  newR
                tmp['splitD'] =  newR * 2 + 15
                tmp['splitP'] =  newR * 2 + splitForward(tmp)
                tmp['sporeD'] =  tmp['radius'] + 40
                tmp['splitE'] =  newR * 2 + 15 + newR
                other_balls.append(tmp)
                
                if tmp['size']>= 100:
                    tmp['willsplit'] = True
        
        
        ##
        myballs_indanger, dangers = gothrough(ball0, destP, self.myballs, other_balls)
        for ball in dangers:
            self.infos.append((0, ball['position'], ball['radius'], RED))
            
        
        
        pass
    
    def tst_go_inter1(self):
        
        ball0 = self.myballs[0]
        p = ball0['position']
        X = p.x - 500
        Y = p.y - 300
        
        destP = Vector2(p.x-200, p.y-100)
        self.infos.append((1, p, destP, YELLOW))
        
        div = 20
        
        other_balls = []
        for i in range(40):
            for j in range(20):
                
                tmp={}
                tmp['position'] = Vector2(X+div*i,Y+div*j)
                r = math.sqrt(1200)
                tmp['radius'] = r
                tmp['size'] = r*r
                tmp['vmax'] = cal_vel_max(r)
                
                newsize =  tmp['size']/2
                newR = math.sqrt(newsize)
                tmp['splitR'] =  newR
                tmp['splitD'] =  newR * 2 + 15
                tmp['sporeD'] =  tmp['radius'] + 40
                tmp['splitE'] =  newR * 2 + 15 + newR
                other_balls.append(tmp)
                
        
        
        addSplitInfo(other_balls, True)
        
        for ball in other_balls:
            if isInter(ball0, destP, ball):
                self.infos.append((0, ball['position'], r, RED))
            # else:
            #     self.infos.append((0, ball['position'], r, GREEN))
                
        
        
        
        pass
     

    def preAI(self):
        rect = self.player_states['rectangle']
        self.infos.append(("rect", ((rect[0],rect[1]),(rect[2]-rect[0],rect[3]-rect[1])), RED))
        
        self.last_intent = self.cur_intent
        
        # self.infos.append((8, self.mass_p, math.sqrt(self.total_size), GREEN))
        self.toEatIntents = []     
        self.GoAfterIntents = []
        self.EscapeIntents = []
        
        # 没用上
        # self.calcMassDir()
       
        #KM
           
        #=== 个体 
        self.atk_intents()
        self.escape_intents()
        self.calc_merge()
        
        
        
        pass
    
    def ai(self):
        
        
        
        # self.tst_go_inter()
        
        if self.must_act_que.qsize() > 0:
            return self.must_act_que.get()
        
        # 合作 
        
        # 4分  四分吃球 
        
        # 大方向 , 去未知区域  合球后, 视领域小 等..
        
        # 吃孢子优化 zone
        
        # 还要离得远点就跑..!!!!
        
        # ban zone 
        # A*
        
        
        
        
        # 流程
        # 看清局势
        # 躲比较优先
        # 找 各种 可能的动作 
        # 看看有没有可以吃的
        # 直接可以,  判断 有没有危险
        # 不可以判断要不要追, 要不要合
        # 判断要不要躲 , 评估怎么躲
        # 吃刺
        
        # tsq 判断外面局势, 只要大多数比外面大 , 就不需要合球
        # 每个核心, 都比威胁的大, 就不用合
        # 现在是 有个专门的不吃刺,  
        # 3个 solo的时候, 不需要这个

        # if self.cur_intent is not None:
        #     print(self.name, " to ",keyv(Intent ,self.cur_intent.type)) 
        # else:
        #     print(" tsq no intent!!")
            

        # TODO 吃刺的时候, 有可能为了方向, 离得很近的也不吃, 要改 , 得看能不能很快吃到对方, 以及能不能很快吃到刺
        
        # TODO
        # 合球只要合到比最大的大,  当很少的时候
        # 写个判断安全的, 找出危险的那些球
        # 保证 危险的球 微不足道 或者 可以被合
        
        

        
        
        # merge 条件 TODO
        # 把这个条件写在外面 , 需要变量太多
        # 得看外面的形式, 如果都比自己大, 并且自己合起来能反超, 则合 
        # 如果合起来不能反超, 则逃
        
        
        
        
        # TODO 全部小的, 去吃
        # 合球 太慢了, 不如逃跑
        # corner 杀

        if self.actions_queue.qsize() > 0:
            return self.actions_queue.get()
        
        # 
        if self.cur_act == PolicyAct.SINGLE:
            
            act = None
            
            

            
            # # 持续合球
            # if act is None:  
            #     act = self.mergeAct()
            # if self.needSplitDir is not None:
            #     act = [self.needSplitDir.x, self.needSplitDir.y,  ACT.Split]
            #     self.needSplitDir = None
                
            # 大方向立即
            
            # 立刻吃的, 已经考虑躲避   
            if act is None:
                act = self.eat_others()
                

                
            # merge goafter  大方向 吃刺 需要协调  

            # 可能需要寻路, 这个需要 danger grid , 并且 自己的比较聚拢, 负责 难
            # 先要自己聚拢
            if self.total_size < S1_FOOD_THORN:
                self.allballEatThorn = True
                self.eatAreaCenter = None
                
                
                if act is None:    
                    act = self.eatThorn() 
                # 需要立刻躲避
                if act is None:    
                    act = self.escape() 
                # if act is None:    
                #     act = self.mergeAct()
                if act is None:
                    act = self.eatFoodAndSpores()
            else:
                
                self.allballEatThorn = True  
                self.eatAreaCenter = None
                if len(self.myballs)>1 and len(self.myballs)<16:
                    self.allballEatThorn = False 
                if len(self.myballs)>1:
                    self.eatAreaCenter = self.myballs[0]['position']
                
                # if act is None:    
                #     act = self.mergeAct()
                if act is None:
                    # corner 
                    if len(self.GoAfterIntents)>0:
                        for intent in self.GoAfterIntents:
                            self.cur_intent = intent
                            act = [self.GoAfterIntents[0].dir.x, self.GoAfterIntents[0].dir.y, -1]
                            break
                if act is None:   
                    act = self.eatThorn() 
                # 需要立刻躲避
                if act is None:    
                    act = self.escape() 
                if act is None:
                    act = self.eatFoodAndSpores()
                
            
            if act is not None:
                return act

        if self.cur_act == PolicyAct.Eat_Thorn_to_dest:
            act = None
            
            # 需要立刻躲避
            if act is None:    
                act = self.escape() 
            if act is None:   
                act = self.eatThorn() 
            if act is None:
                act = self.eatFoodAndSpores()    
            if act is not None:
                return act    

        return  [0,0,-1]
    

class AgentInfo:
    
    def __init__(self):
        self.teamname = None
        self.hasspliteat = False


class TeamAI:
    
    def __init__(self, team_name, player_names):
        
        self.team_name = team_name
        self.player_names = player_names
        self.player_num = len(self.player_names)
        
        self.agents = []
        for player_name in self.player_names:
            agent = MyBotAgent(name=player_name)
            agent.team_name = self.team_name
            self.agents.append(agent)
            
        self.lastBalls = []
        
        
        self.co_stage = 0
        
        self.co_ids = []
        self.cur_big = -1
        self.cur_small = -1
        self.gank_id = -1
        
        
        
        self.patrol_id = -1    
        self.patrol_points = []
        
        self.infos = []
        
        self.agentInfos = {}
        
        
        self.smSplit = None
        self.smEat = None
        self.smDir = None
        self.smIdx = 0
     
    def showSplitDebugInfo(self, balls):
        
        for i in range(len(balls)):
            myball = balls[i]
            myballP = myball['position']
            myballR = myball['radius']
            myballV = myball['v']
            myballVl = myball['vl']
            
            splitR = myball['splitR'] 
            splitE = myball['splitE'] 
            sporeD = myball['sporeD'] 
            
            
            # if myball['cansplit']:
                # self.infos.append((0, myballP, sporeD, YELLOW))
                # self.infos.append((0, myballP, splitD, RED))
                # self.infos.append((8, myballP, splitR, PURPLE))
                # self.infos.append((1, myballP, myballP+myballV, BLACK)) #  速度
                
    
    def team_ai(self):
        
        dinfo = "stage  "+ keyv(TeamAct, self.co_stage)  +" ; "
        for agent in self.agents:
            dinfo +=  keyv(PolicyAct ,agent.cur_act) + ', '
        # print(dinfo)
        
        
        #  时间以及 规模的策略不一样
        last_time =  self.global_state['last_time']
        if last_time<10:
            # 抓紧发展, 特别好的才追
            # 
            pass
        elif last_time<20:
            # 中期
            
            pass
        else:
            # 
            if self.ranks.index(self.team_name) == 0 and  self.leaderboard[self.team_name]> 0.5* self.global_state['total_score']:
                pass
            else:
                pass
                #
                #
        
        
        
        if len(self.agents) == 1:
            self.co_stage = -1  # 测试 eat free
        self.co_stage = -1   # debug
        
        
        
        # 
        # 重新发育, 如果两个都小于300
        if self.co_stage > TeamAct.First_Free:
            if self.agents[self.cur_big].total_size <20 or self.agents[self.cur_small].total_size <20:
                self.co_stage = TeamAct.First_Free
                self.agents[self.cur_big].bigdest = None
                self.agents[self.cur_small].bigdest = None
                self.agents[self.cur_big].cur_act = PolicyAct.Eat_Free
                self.agents[self.cur_small].cur_act = PolicyAct.Eat_Free
        
        
        # 先判断起手  3个都判断, 防止两个离得特别远的过去
        if self.co_stage == TeamAct.First_Free:
            ok = 0
            for agent in self.agents:
                if agent.total_size >= 1000:
                    ok += 1
            
            if ok >= len(self.player_names):
                self.co_stage = TeamAct.First_Move_Together
        
        
        # 找两个先靠近
        if self.co_stage == TeamAct.First_Move_Together: 
            
            # 先找起手两个最近的
            if len(self.co_ids) == 0:
                mind = 8888888
                co_i = 0
                for i in range(self.n):
                    p1 = self.mass_ps[i]
                    p2 = self.mass_ps[(i+1)%self.n]
                    d = p1-p2  
                    if d.length() < mind:
                        mind = d.length()
                        co_i = i

                self.co_ids.append(co_i)
                self.co_ids.append((co_i+1)%self.n)
        
        

        # AB2C
        # 走过去碰不到敌人的判断
        #  大少小一 快合
        
    def pre(self, obs):
        global_state, player_states = obs
        
        self.global_state = global_state
        self.player_states = player_states
        
        
        # 总体解析
        self.W = global_state['border'][0]
        
        patrol_d = self.W /4
        self.patrol_points = [Vector2(patrol_d, patrol_d), #Vector2(patrol_d*2,patrol_d),
                              Vector2(patrol_d*3,patrol_d),#Vector2(patrol_d*3,patrol_d*2),
                              Vector2(patrol_d*3,patrol_d*3),#Vector2(patrol_d*2,patrol_d*3),
                              Vector2(patrol_d,patrol_d*3),#Vector2(patrol_d,patrol_d*2)
                              ]
        
        
        self.leaderboard = leaderboard = global_state['leaderboard']
        leaderarr = list(zip(leaderboard.keys(), leaderboard.values()))
        leaderarr.sort(key=lambda a: a[1], reverse=True)
        self.ranks = [a[0] for a in  leaderarr]
        
        total_score = sum(leaderboard.values())
        global_state['total_score'] = total_score
        # print("----total_score----", total_score, ranks, leaderboard)
        
        
        for teamname in leaderboard.keys():
            if teamname not in self.agentInfos:
                agentinfo = AgentInfo()
                agentinfo.teamname = teamname
                self.agentInfos[teamname] = agentinfo
        
         
        
        self.overlaps = []
        
        curBalls = []
        ballHash = {}
        
        for i in range(self.player_num):
            player_name = self.player_names[i]
            agent = self.agents[i]
            
            agent.infos = self.infos
            agent.W = self.W
            
            obs = player_states[player_name]
            overlap = obs['overlap']
            overlap = preprocess(overlap)
            self.overlaps.append(overlap)
            
            agent.overlap = overlap
            agent.global_state = global_state
            agent.player_states = obs
            agent.ranks = self.ranks
            
            
        # 先算V
        for overlap in self.overlaps:
            clone_balls = overlap['clone']
            for ball in clone_balls:
                key = ball['team']+ball['player']+ str(ball['position'])
                if key not in ballHash:
                    ballHash[key] = ball
                    curBalls.append(ball)    
        
        calcSpd(self.lastBalls, curBalls)    
        
        
        for teamname in leaderboard.keys():
            teamballs = [ball for ball in curBalls if ball['team']==teamname]
            old_teamballs = [ball for ball in self.lastBalls if ball['team']==teamname]
            
            playernames =set([ball['player'] for ball in teamballs]) 
            for pn in playernames:
                playerballs = [ball for ball in teamballs if ball['player']==pn]
                old_playerballs = [ball for ball in old_teamballs if ball['player']==pn]
                
                if hasSplitEat(playerballs, old_playerballs):
                    self.agentInfos[teamname].hasspliteat = True
                    # print("===================   ", teamname)
                
        
        
        my_balls, other_balls, other_team_balls, other_team_players = balls_by_team_player(curBalls, self.team_name)
        self.my_balls = my_balls
        self.other_balls = other_balls
        self.other_team_balls = other_team_balls
        self.other_team_players = other_team_players
        
        self.lastBalls = curBalls
        
        other_team_info = {}
        other_team_player_info = {}
        
        for t in self.other_team_players:
            other_team_info[t]  = dict(
                hasspliteat = self.agentInfos[t].hasspliteat, 
                rank = self.ranks.index(t),
            )
            
        for t in self.other_team_players:
            other_team_player_info[t] = {}
            team = self.other_team_players[t]
            for p in team:
                player = team[p]
                
                # 算能不能裂
                addSplitInfo(player, True) #self.agentInfos[t].hasspliteat)
                #
                total_size, mass_r, mass_p, mass_v = centerInfo(player)
                #
                best, assignments, new_centers, r_arr = judge_K(player)
                _assignments = convertAss(assignments, len(new_centers))
                
                other_team_player_info[t][p] = dict(
                    total_size = total_size,
                    mass_p = mass_p,
                    mass_v = mass_v,
                    mass_r = mass_r,
                    assignments = _assignments,
                    new_centers = new_centers,
                    r_arr = r_arr,
                    balls = player,
                )
                
        
        for i in range(self.player_num):
            player_name = self.player_names[i]
            agent = self.agents[i]
            
            agent.myballs = my_balls[player_name]
            agent.other_balls = other_balls
            agent.other_team_balls = other_team_balls
            agent.other_team_players = other_team_players
            agent.other_team_info = other_team_info
            agent.other_team_player_info = other_team_player_info
            
            agent.process()
            pass
        
        mass_ps = [agent.mass_p for agent in self.agents]
        self.mass_ps = mass_ps
        
        
        self.showSplitDebugInfo(self.other_balls)
        
        # if self.cur_big!= -1:  # center
        #     self.infos.append((8,mass_ps[self.cur_big], math.sqrt(self.agents[self.cur_big].total_size)/2, WHITE))
        #     self.infos.append((8,mass_ps[self.cur_small], math.sqrt(self.agents[self.cur_small].total_size)/2, Y2))
         
    def get_actions(self, obs):
        
        self.infos = []
        
        self.pre(obs)
        
        # 默认  
        actions = {}  
        for i in range(self.player_num):
            player_name = self.player_names[i]
            actions[player_name] = [0,0,-1]
            agent = self.agents[i]
            agent.preAI()
        
        
        self.team_ai()
        

        
        for i in range(self.player_num):
            player_name = self.player_names[i]
            agent = self.agents[i]
            
            action = agent.ai()
            actions[player_name] = action
        
        # print("actions", actions)
        return actions


if __name__ == '__main__':
    from tst_simple import *
    
    launch_a_game()
