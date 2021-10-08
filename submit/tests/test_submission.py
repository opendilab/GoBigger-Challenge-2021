import logging
import pytest
import uuid
from pygame.math import Vector2
import time
import random
import numpy as np
import cv2
import pygame

from gobigger.agents import BotAgent
from gobigger.utils import Border
from gobigger.server import Server
from gobigger.render import EnvRender
from gobigger.submit import RandomSubmission, BotSubmission

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.unittest
class TestSubmission:

    def test_random_submission(self):
        server = Server(dict(
            team_num=1, # 队伍数量
            player_num_per_team=2, # 每个队伍的玩家数量
        ))
        render = EnvRender(server.map_width, server.map_height)
        server.set_render(render)
        server.start()
        names = server.get_team_names()
        random_submission = RandomSubmission(
            team_name=list(names.keys())[0],
            player_names=list(names.values())[0]
        )
        for _ in range(10):
            obs = server.obs()
            actions = random_submission.get_actions(obs)
            server.step(actions=actions)
        server.close()

    def test_bot_submission(self):
        server = Server(dict(
            team_num=1, # 队伍数量
            player_num_per_team=2, # 每个队伍的玩家数量
        ))
        render = EnvRender(server.map_width, server.map_height)
        server.set_render(render)
        server.start()
        names = server.get_team_names()
        bot_submission = BotSubmission(
            team_name=list(names.keys())[0],
            player_names=list(names.values())[0]
        )
        for _ in range(10):
            obs = server.obs()
            actions = bot_submission.get_actions(obs)
            server.step(actions=actions)
        server.close()
