# GoBigger Challenge 2021

[en](https://github.com/opendilab/GoBigger-Challenge-2021/blob/main/README.md) / [中文](https://github.com/opendilab/GoBigger-Challenge-2021/blob/main/README_zh.md)

## Overview

2021 "Go-Bigger: Multi-agent Decision Intelligence Challenge" (hereinafter referred to as the "challenge") is jointly organized by Shanghai Artificial Intelligence Laboratory and Sensetime Research Institute, and is jointly supported by the Global Artificial Intelligence Academic Alliance of Universities and the Student Union of the Chinese Academy of Sciences. The scientific and technological competition for global technology developers and school students aims to explore the research of multi-agent games and promote the training of technical talents in decision-making intelligence related fields, so as to create "global leading", "original" and "open" decision-making AI open source technology ecology.

## Challenge Description

This challenge is divided into two stages. The first stage is to help contestants have a more correct understanding of the performance of the submitted agent. In the second stage, the final ranking of the participating teams will be conducted.

## Specific Task

This challenge uses [Go-Bigger](https://github.com/opendilab/GoBigger) as the game environment. Go-Bigger is a multi-players competitive environment. For more details, please refer to the Go-Bigger documentation. In the match, each team participating in the challenge controls one team in the game (each team consists of multiple players). Contest participating teams need to submit an agent to control a certain team in the match and the players it contains, and obtain higher scores through teamwork, thereby achieving a higher ranking in the match.

## Submission

Here in [submit](https://github.com/opendilab/GoBigger-Challenge-2021/blob/main/submit), we provide examples of submission for all teams in our challenge. We also provide [BaseSubmission](https://github.com/opendilab/GoBigger-Challenge-2021/blob/main/submit/base_submission.py), and participants should implements their owned submission based on the code.

```python
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
```

Note that all submission should extend with `BaseSubmission`. We will provide `team_name` and `player_names` for each submission as their basic parameters. `team_names` means the name of team which this submission controls. We also know that there are several players in a team, which is relative with the `player_names` in the parameters. We will call `get_actions()` when we try to get actions from this submission. So that participants should implements `get_actions()` in their submission. This function will receive `obs` as its parameters, which is similar with what we provide in [tutorial](https://opendilab.github.io/GoBigger/tutorial/space.html#space). For example, submissions will get `obs` as following:

```python
global_state, player_state = obs
```

`global_state` in details:

```python
{
    'border': [map_width, map_height], # the map size
    'total_time': match_time, # the duration of a game
    'last_time': last_time,   # the length of time a game has been played
    'leaderboard': {
        team_name: team_size
    } # the team with its size in this game
}
```

Participants can find their `team_name` in submission matched with the `team_name` in `leaderboard`.

`player_state` in details:

```python
{
    player_name: {
        'feature_layers': list(numpy.ndarray), # features of player
        'rectangle': [left_top_x, left_top_y, right_bottom_x, right_bottom_y], # the vision's position in the map
        'overlap': {
            'food': [{'position': position, 'radius': radius}, ...], # the length of food is not sure
            'thorns': [{'position': position, 'radius': radius}, ...], # the length of food is not sure
            'spore': [{'position': position, 'radius': radius}, ...], # the length of food is not sure
            'clone': [{'position': position, 'radius': radius, 'player': player_name, 'team': team_name}, ...], # the length of food is not sure
        }, # all balls' info in vision
        'team_name': team_name, # the team which this player belongs to 
    }
}
```

However, we will only provide the submission with the `player_state` matched with its players. That means, if `player_a` and `player_b` (both are player name) are in the team belongs to this submission, and `player_c` not belongs to this team, participants will only get `player_a` and `player_b` in the submission.

After getting the `obs`, submissions should return `actions` in `get_actions()`. `actions` should look like:

```python
{
    player_a: actions_a,
    player_b: actions_b
}
```

Remember that both `player_a` and `player_b` should be the name in your submission's `player_names`. And `actions_a` should be a list, which contains there items, which are the same with what we propose in [action-space](https://opendilab.github.io/GoBigger/tutorial/space.html#action-space).


### Examples and Test

We provide [RandomSubmission](https://github.com/opendilab/GoBigger-Challenge-2021/blob/main/submit/random_submission.py) and [BotSubmission](https://github.com/opendilab/GoBigger-Challenge-2021/blob/main/submit/bot_submission.py). `RandomSubmission` provide actions randomly, and `BotSubmission` provide actions based on a script. Both of them could be an example of your submission. More details in code.

We also provide an example for the pipeline of the submission. Please refer to [submission_example](https://github.com/opendilab/GoBigger-Challenge-2021/blob/main/submit/submission_example/) for more details. You can also develop your agent in this directory. Once you finish your `my_submission.py`, you can call `python -u test.py` to check your submission and finally get the `.tar.gz` file to upload.


### Supplements

If you want to add other things in your submission, such as model checkpoints or any other materials, please place them in `./supplements` and tar them with submission. 


### Finally

You should place all your code and materials under `my_submission/`. Use `tar zcf submission.tar.gz my_submission/` to get your final submission files. The final `submission.tar.gz` should be:

```
    - my_submission
    | - __init__.py
    | - my_submission.py
    | - supplements/
        | - checkpoints or other materials
```

Attention: `__init__.py` should be an empty file.

## Resources

* [Challenge Settings](https://github.com/opendilab/GoBigger-Challenge-2021/blob/main/challenge_settings.md)
* [Evaluation](https://github.com/opendilab/GoBigger-Challenge-2021/blob/main/evaluation.md)



