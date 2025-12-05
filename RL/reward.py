import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# Env state 
# info = {
#     "x_pos",  # (int) The player's horizontal position in the level.
#     "y_pos",  # (int) The player's vertical position in the level.
#     "score",  # (int) The current score accumulated by the player.
#     "coins",  # (int) The number of coins the player has collected.
#     "time",   # (int) The remaining time for the level.
#     "flag_get",  # (bool) True if the player has reached the end flag (level completion).
#     "life"   # (int) The number of lives the player has left.
# }


# simple actions_dim = 7 
# SIMPLE_MOVEMENT = [
#     # ["NOOP"],       # Do nothing.
#     ["right"],      # Move right.
#     ["right", "A"], # Move right and jump.
#     ["right", "B"], # Move right and run.
#     ["right", "A", "B"], # Move right, run, and jump.
#     ["A"],          # Jump straight up.
#     ["left"],       # Move left.
#     # ["right", "A", "A", "B"],
#     # ["right", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A"],
# ]
#-----------------------------------------------------------------------------
#獎勵函數
'''
get_coin_reward         : 根據硬幣數量變化提供額外獎勵

'''
'''
環境資訊 (info)
1."x_pos": 水平位置，用於判斷角色的前進情況
2."y_pos": 垂直位置，用於分析跳躍或下落行為
3."score": 玩家目前的遊戲分數
4."coins": 收集到的硬幣數量
5."time": 剩餘時間
5."flag_get": 是否到達終點旗幟（遊戲完成）
6."life": 玩家剩餘的生命數
'''

#===============to do===============================請自定義獎勵函數 至少7個(包含提供的)
#例子:用來獎勵玩家蒐集硬幣的行為
# 一顆金幣會拿200分
def get_coin_reward(info, reward, prev_info):
    #寫下蒐集到硬幣會對應多少獎勵
    total_reward = reward                                         #獲得目前已有的獎勵數量
    coin_reward = 0
    if info['coins'] - prev_info['coins'] == 1 and info['score'] - prev_info['score'] == 200:
        coin_reward = 2000
    total_reward += coin_reward     #這裡是定義，如果玩家有蒐集到硬幣，則獎勵加10(這裡是可以自己去定義獎勵要給多少的)
    return total_reward, coin_reward

#用來鼓勵玩家進行跳躍或高度變化(因為有時前方有障礙物 會被卡住)
def distance_y_offset_reward(info, reward, prev_info):
    total_reward = reward
    y_offset_reward = 0
    abs_y_change = abs(info['y_pos'] - prev_info['y_pos'])
    if info['y_pos'] >= prev_info['y_pos']: # 現在比之前高(跳躍中)
        y_offset_reward = abs_y_change * 2
    else: # 現在比之前矮(落下中)
        y_offset_reward = abs_y_change * -1
    total_reward += y_offset_reward
    return total_reward, y_offset_reward

# #用來鼓勵玩家前進，懲罰原地停留或後退
def distance_x_offset_reward(info, reward, prev_info):
    total_reward = reward
    x_offset_reward = 0
    abs_x_change = abs(info['x_pos'] - prev_info['x_pos'])
    if info['x_pos'] > prev_info['x_pos']: # 現在比之前右邊(前進)
        x_offset_reward = abs_x_change * 2
    elif info['x_pos'] < prev_info['x_pos']: # 現在比之前左邊(後退)
        x_offset_reward = abs_x_change * -1
    else: # 不動
        x_offset_reward = -2
    total_reward += x_offset_reward
    return total_reward, x_offset_reward

# #用來鼓勵玩家提高分數（例如擊敗敵人)
# 一隻怪物會拿100、200、300...分
def monster_score_reward(info, reward, prev_info):
    total_reward = reward
    score_reward = 0
    if (info['score'] - prev_info['score']) % 100 == 0 and info['coins'] == prev_info['coins'] and (info['score'] - prev_info['score']) != 1000: #代表踩死一隻怪
        score_reward = 1000
    total_reward += score_reward
    return total_reward, score_reward

# #用來鼓勵玩家完成關卡（到達終點旗幟）
def final_flag_reward(info, reward, prev_info):
    total_reward = reward
    flag_reward = 0
    if info['flag_get'] and not prev_info['flag_get']:
        flag_reward = 50000  # 如果成功碰到旗子，加 50000 分
    total_reward += flag_reward
    return total_reward, flag_reward

# 鼓勵馬力歐吃變大香菇
def become_man_reward(info, reward, prev_info):
    total_reward = reward
    man_reward = 0
    if info['score'] - prev_info['score'] == 1000:
        man_reward += 10000
    total_reward += man_reward
    return total_reward, man_reward

# 鼓勵馬力歐跳過第三個水管(高水管)
def pipe_jump_reward(info, reward, prev_info):
    total_reward = reward
    pipe_reward = 0
    
    # 假設水管的位置範圍
    pipe_x_start = 722
    pipe_y_top = 145
    
    # 判斷 Mario 是否接近水管
    if pipe_x_start - 25 <= info['x_pos'] <= pipe_x_start + 25:
        # 鼓勵跳躍行為
        if info['y_pos'] > pipe_y_top and prev_info['y_pos'] < pipe_y_top:  # y_pos 超過水管高度
            pipe_reward += 5000  # 鼓勵跳躍
        else:
            pipe_reward -= 2  # 懲罰未跳過
            
    total_reward += pipe_reward
    return total_reward, pipe_reward

# 鼓勵馬力歐跳過第四個水管(高水管)
def pipe_jump_reward2(info, reward, prev_info):
    total_reward = reward
    pipe_reward = 0
    
    # 假設水管的位置範圍
    pipe_x_start = 898
    pipe_y_top = 145
    
    # 判斷 Mario 是否接近水管
    if pipe_x_start - 25 <= info['x_pos'] <= pipe_x_start + 25:
        # 鼓勵跳躍行為
        if info['y_pos'] > pipe_y_top and prev_info['y_pos'] < pipe_y_top:  # y_pos 超過水管高度
            pipe_reward += 5000  # 鼓勵跳躍
        else:
            pipe_reward -= 2  # 懲罰未跳過
            
    total_reward += pipe_reward
    return total_reward, pipe_reward
#===============to do==========================================