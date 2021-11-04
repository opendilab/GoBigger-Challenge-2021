import os
import sys
import logging
import importlib
import time
import argparse
import requests
import subprocess
from tqdm import tqdm
import traceback

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['SDL_AUDIODRIVER'] = 'dsp'

from gobigger.server import Server
from gobigger.render import EnvRender

logging.basicConfig(level=logging.DEBUG)


def test():
    server = Server(dict(
            team_num=4,
            player_num_per_team=3,
            match_time=30,
    ))
    render = EnvRender(server.map_width, server.map_height)
    server.set_render(render)
    server.start()
    agents = []
    team_player_names = server.get_team_names()
    team_names = list(team_player_names.keys())
    for index in range(server.team_num):
        try:
            p = importlib.import_module('my_submission.my_submission')
            agents.append(p.MySubmission(team_name=team_names[index], 
                                         player_names=team_player_names[team_names[index]]))
        except Exception as e:
            logging.error(''.join(traceback.format_tb(e.__traceback__)))
            logging.error(sys.exc_info()[0])
            logging.error('You must implement `MySubmission` in my_submission.py !')
            exit()
    
    for i in tqdm(range(30*server.action_tick_per_second)):
        obs = server.obs()
        global_state, player_states = obs
        actions = {}
        for agent in agents:
            agent_obs = [global_state, {
                player_name: player_states[player_name] for player_name in agent.player_names
            }]
            actions.update(agent.get_actions(agent_obs))
        finish_flag = server.step(actions=actions)
        if finish_flag:
            logging.debug('Game Over!')
            break
    server.close()
    logging.debug('Success!')

def postprocess():
    logging.debug('tar zcf my_submission.tar.gz my_submission/')
    output = subprocess.getoutput('tar zcf my_submission.tar.gz my_submission/')
    assert os.path.isfile('my_submission.tar.gz')
    logging.debug('###################################################################')
    logging.debug('#                                                                 #')
    logging.debug('#   Now you can upload my_submission.tar.gz as your submission.   #')
    logging.debug('#                                                                 #')
    logging.debug('###################################################################')

if __name__ == '__main__':
    test()
    postprocess()
