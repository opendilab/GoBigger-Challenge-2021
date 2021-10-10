## Challenge Settings

### Basic Game Settings
In this challenge, your agent control your team against other 3 teams and your goal is to eat and go bigger than others. There are a few points to note:

1. Each game will last 10 minutes (3000 steps).

2. The map size will be 1000 * 1000.

3. The game mode is FFA (Free For All), which means that everyone may take part in a game.

4. Usually there are 3 players in a team, and your agent need to control all 3 players in your team. That means teamwork is important in this challenge if you want to get a higher score.

5. We will maintain a league system to score each agent in this challenge.

### More Game Settings

#### Food

At the beginning of the challenge, there will be 2000 food balls in the map. We will replenish 30 food balls every 2 seconds until the number of food balls reach the upper limit, which is set 2500. Each food ball will be a circle with radius 2, and a food ball will remain in our map until someone eat it.

#### Thorns

Same as food balls, 15 thorns balls will be initialized in the map. And we will add two more thorns balls each 2 seconds. Make sure that the number of thorns balls will not greater than 20, which is the upper limit in this challenge. The radius of each thorns ball will randomly decided from 12 to 20.

#### Clone

You should note that each team consists of 3 players, and each player can have a maximum of 16 clone balls. Each clone ball will have a maximum speed limit, which depends on it size. The larger balls move slower. We make sure that the radius of each clone ball will be 3 at least and 100 at most. When the radius of a clone ball is more than 10, it can split or eject. After split, a clone ball will enter a coll-down period (20 seconds), which means it can not merge with other clone ball until the period ends.

#### Spore

Spore balls can be eat by any other clone balls larger than itself. We make sure that all spore balls have the same radius of 3.
