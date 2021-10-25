import random


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


class RandomSubmission(BaseSubmission):

    def __init__(self, team_name, player_names):
        super(RandomSubmission, self).__init__(team_name, player_names)

    def get_actions(self, obs):
        global_state, player_states = obs
        actions = {}
        for player_name, _ in player_states.items():
            action = [random.uniform(-1, 1), random.uniform(-1, 1), -1]
            actions[player_name] = action
        return actions

