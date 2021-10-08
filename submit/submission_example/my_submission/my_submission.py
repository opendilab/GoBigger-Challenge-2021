import random

from gobigger.submit import BaseSubmission


class MySubmission(BaseSubmission):

    def __init__(self, team_name, player_names):
        super(MySubmission, self).__init__(team_name, player_names)

    def get_actions(self, obs):
        global_state, player_states = obs
        actions = {}
        for player_name, _ in player_states.items():
            action = [random.uniform(-1, 1), random.uniform(-1, 1), -1]
            actions[player_name] = action
        return actions

