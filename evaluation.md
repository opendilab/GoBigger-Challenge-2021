# Evaluation

In order to easily evaluate the level of competition between different players, we use a rating system to quantify the player's skill level. 
  
At the same time, in order to ensure the fairness of the game and the stability of the player's rating value, we designed a matchmaking mechanism based on the [TrueSkill](https://trueskill.org/) rating system.

In order to ensure the normal display of the formula, please use Chrome and install [MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related). 

## Rating System Introduction

The TrueSkill rating system is a scoring system based on Bayesian inference, developed by Microsoft Research. And it is mainly used in multiplayer games. It takes into account the uncertainty of individual players' level, and comprehensively considers each player's winning rate and possible level fluctuations.

The TrueSkill rating system assumes that the player's rating can be represented by a normal distribution, and the normal distribution can be fully described by two parameters: mean and variance. 

Suppose the rating value is $R$, 

$$ R = \mu - K*\sigma $$

$K$ is a manually set parameter. The larger the value of $K$, the more conservative the score of the system. For example $K=3$, the system is 99%(three-sigma rule）sure that the player's skill is actually higher than what is displayed as their rank.

Take two players as an example, update their rating value according to the outcome of the game,

$$ \mu _{winner}\leftarrow \mu _{winner}+\frac{\sigma _{winner}^{2}}{c}*\nu (\frac{\mu _{winner}-\mu _{loser}}{c},\frac{\varepsilon }{c}) $$

$$ \mu _{loser}\leftarrow \mu _{loser}-\frac{\sigma _{loser}^{2}}{c}*\nu (\frac{\mu _{winner}-\mu _{loser}}{c},\frac{\varepsilon }{c}) $$

$$ \sigma _{winner}^{2}\leftarrow \sigma _{winner}^{2}*\left [ 1-\frac{\sigma _{winner}^{2}}{c}\ast \omega (\frac{\mu _{winner}-\mu _{loser}}{c},\frac{\varepsilon }{c}) \right ] $$

$$ \sigma _{loser}^{2}\leftarrow \sigma _{loser}^{2}*\left [ 1-\frac{\sigma _{loser}^{2}}{c}\ast \omega (\frac{\mu _{winner}-\mu _{loser}}{c},\frac{\varepsilon }{c}) \right ] $$

$$ c^{2}=2\beta ^{2}+\sigma_{winner}^{2}+\sigma_{loser}^{2} 
\beta ^{2}=\left ( \frac{\sigma}{2} \right )^{2} $$


Player rating value as a Gaussian distribution which starts from $\mathbb{N} (\mu ,\sigma ^{2})$. Winner Player is assumed to exhibit a performance $p_{winner}\sim \mathbb{N} (p_{winner}; \mu_{winner}, \sigma_{winner}^{2})$. Loser Player is assumed to exhibit a performance $p_{loser}\sim \mathbb{N}(p_{loser}; \mu_{loser},\sigma_{loser}^{2})$. Where $\nu$ and $\omega$ are the correction term for the mean and variance of a (doubly) truncated Gaussian.And see [TrueSkill paper](https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/) for details. The $\nu$ and the $\omega$ determine update range of the rating value. If there is a strength gap between the skills of the two players, but the two eventually draw, the range of changes in their rating value will increase. If there is a strength gap between the skills of the two players and the result is a win, it is in line with expectations, and the update range becomes smaller. Otherwise, it does not meet the expectations and the update range is larger.The $\omega$ is always greater than 0, and the $\nu$ can be positive or negative. Parameter $\varepsilon$ indicates the margin that can be tolerated in the draw.

In short, $\mu$ increases when the player wins. On the contrary, if the player loses, it will decrease. But whether win or lose, $\sigma$ is decreasing, so there may be situations where games lose but $\mu$ increases.

## Matchmaking Mechanisms Introduction

In order to ensure the fairness of the game and obtain the player's rating value more accurately, we have designed a matching mechanism based on the TrueSkill rating system. In order to ensure the balance of players' matches, we select the player with the least number of matches each time. At the same time, we use quality as the indicator for selecting opponents.

### Matches less first, quality as prob (MLF-quality)

It means, first select the player with the least number of matches. And then, use its draw probability(also called quality) with the remaining players as the probability of selection. Specific steps are as follows,

1. Initialize the rating value as a Gaussian distribution which starts from $\mathbb{N} (\mu ,\sigma ^{2})$.
2. Select a player with the least number of matches.
3. Calculate the draw probability with the remaining players by formula, such as two players.

$$ quality_{draw}(\beta^{2},\mu_{i},\mu_{j},\sigma_{i},\sigma_{j})=\sqrt{\frac{2\beta^{2}}{2\beta^{2}+\sigma_{i}^{2}+\sigma_{j}^{2}}}  \cdot exp(-\frac{(\mu_{i}-\mu_{j})^{2}}{2(2\beta^{2}+\sigma_{i}^{2}+\sigma_{j}^{2})}) $$

  * Player $i$ is assumed to exhibit a performance $p_{i}\sim \mathbb{N}(p_{i};\mu_{i},\sigma_{i}^{2})$
  * Player $j$ is assumed to exhibit a performance $p_{j}\sim \mathbb{N}(p_{j};\mu_{j},\sigma_{j}^{2})$ where $\beta ^{2}=\left ( \frac{\sigma}{2} \right )^{2}$

4. Normalize the draw probability so that the sum is 1.
5. Use the draw probability as the probability of player selection.
6. Repeat step 2-5 until $\sigma$ is less than a certain value (It means playes rating value is stability).

In our league, the type of matches is free-for-all (Player vs. ALL). When calculating the quality of matches, we take a head-to-head (1 vs. 1) method to calculate the draw probability, and use this as the probability of selection.

## Parameter Settings

In our rating system and matchmaking mechanisms, $\mu=1000$, $\sigma=8.333$.










