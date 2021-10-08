from gobigger.submit import BaseSubmission
from gobigger.agents import BotAgent


class BotSubmission(BaseSubmission):

    def __init__(self, team_name, player_names):
        super(BotSubmission, self).__init__(team_name, player_names)
        self.agents = {}
        for player_name in self.player_names:
            self.agents[player_name] = BotAgent(name=player_name)

    def get_actions(self, obs):
        global_state, player_states = obs
        actions = {}
        for player_name, agent in self.agents.items():
            action = agent.step(player_states[player_name])
            actions[player_name] = action
        return actions
