import random
from .co1 import TeamAI

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
        self.teamAI = TeamAI(team_name, player_names)

    def get_actions(self, obs):
        global_state, player_states = obs
        actions = {}


        actions = self.teamAI.get_actions(obs)
        return actions

