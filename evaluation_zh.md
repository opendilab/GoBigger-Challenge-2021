## 评估方法

为了方便评估不同玩家之间的竞技水平，我们使用评分系统(天梯榜)来量化玩家的技能水平。
  
同时，为了保证游戏的公平性和玩家技能水平的稳定性，我们设计了基于【TrueSkill】(https://trueskill.org/)评分系统的匹配机制。


### 评分系统介绍

TrueSkill评分系统是微软研究院开发的基于贝叶斯推理的评分系统，其主要用于多人游戏。由于个别玩家技能水平存在不确定性，该系统综合考虑了每个玩家的胜率和可能的分值波动。

TrueSkill评分系统假设玩家的评分符合正态分布，并用两个参数来描述：均值和方差。

假设玩家天梯分数为$R$, 

$$ R = \mu - K*\sigma $$

$K$是可调节的参数，$K$值越大，评分系统将会越保守。 

以两个玩家为例，根据比赛结果更新它们的天梯得分，

$$ \mu _{winner}\leftarrow \mu _{winner}+\frac{\sigma _{winner}^{2}}{c}*\nu (\frac{\mu _{winner}-\mu _{loser}}{c},\frac{\varepsilon }{c}) $$

$$ \mu _{loser}\leftarrow \mu _{loser}-\frac{\sigma _{loser}^{2}}{c}*\nu (\frac{\mu _{winner}-\mu _{loser}}{c},\frac{\varepsilon }{c}) $$

$$ \sigma _{winner}^{2}\leftarrow \sigma _{winner}^{2}*\left [ 1-\frac{\sigma _{winner}^{2}}{c}\ast \omega (\frac{\mu _{winner}-\mu _{loser}}{c},\frac{\varepsilon }{c}) \right ] $$

$$ \sigma _{loser}^{2}\leftarrow \sigma _{loser}^{2}*\left [ 1-\frac{\sigma _{loser}^{2}}{c}\ast \omega (\frac{\mu _{winner}-\mu _{loser}}{c},\frac{\varepsilon }{c}) \right ] $$

$$ c^{2}=2\beta ^{2}+\sigma_{winner}^{2}+\sigma_{loser}^{2} 
\beta ^{2}=\left ( \frac{\sigma}{2} \right )^{2} $$

玩家的天梯得分满足高斯分布$\mathbb{N} (\mu ,\sigma ^{2})$。赢的那一方天梯得分满足$p_{winner}\sim \mathbb{N} (p_{winner}; \mu_{winner}, \sigma_{winner}^{2})$.

输的一方天梯得分满足$p_{loser}\sim \mathbb{N}(p_{loser}; \mu_{loser},\sigma_{loser}^{2})$.

其中 $\nu$ 和 $\omega$ 是两个复杂的函数，其为有加法和乘法校正项的（双重）截断高斯的均值和方差。可以参考[TrueSkill 论文](https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/）了解详情。 $\nu$ 和 $\omega$ 决定了天梯分的更新范围。如果两个玩家存在实力差距，但最终两人平局，那么他们的天梯分变化幅度就会增加。如果两个玩家的存在实力差距，实力强的一方获胜，则符合预期，更新幅度变小。否则不符合预期，更新范围大。$\omega$ 总是大于0，$\nu$ 可以为正也可以为负。

参数 $\varepsilon$ 表示在判定游戏平局时可以容忍的分差。

简而言之，$\mu$ 在玩家获胜时增加。相反，如果玩家输了，它会减少。但是无论输赢，$\sigma$ 都在减少，所以可能会出现输了但 $\mu$ 增加的情况。

### 匹配机制介绍

为了保证游戏的公平性，并且更准确地获取玩家的评分值，我们设计了基于TrueSkill评分系统的匹配机制。为了保证玩家匹配的平衡性，我们每次都选择匹配次数最少的玩家。同时，为了保证比赛的公平性，我们以平局概率作为选择对手的指标。

#### 匹配次数最少优先，平局概率作为选择概率(Matches less first, quality as prob，MLF-quality)

其表示，首先选择匹配次数最少的玩家，然后计算该玩家与剩余玩家的平局概率（也称为质量）作为选择概率。

1. 初始化玩家天梯分数，其符合正态分布 $\mathbb{N} (\mu ,\sigma ^{2})$。
2. 选择匹配次数最少的玩家。
3. 计算该玩家与剩余玩家的平局概率。计算公式如下，以两个玩家为例，

$$ quality_{draw}(\beta^{2},\mu_{i},\mu_{j},\sigma_{i},\sigma_{j})=\sqrt{\frac{2\beta^{2}}{2\beta^{2}+\sigma_{i}^{2}+\sigma_{j}^{2}}}  \cdot exp(-\frac{(\mu_{i}-\mu_{j})^{2}}{2(2\beta^{2}+\sigma_{i}^{2}+\sigma_{j}^{2})}) $$

  * 玩家 $i$ 的天梯分数满足 $p_{i}\sim \mathbb{N}(p_{i};\mu_{i},\sigma_{i}^{2})$
  * 玩家 $j$ 的天梯分数满足 $p_{j}\sim \mathbb{N}(p_{j};\mu_{j},\sigma_{j}^{2})$ 
  * 其中 $\beta ^{2}=\left ( \frac{\sigma}{2} \right )^{2}$

4. 将平局概率进行归一化，使得其满足和概率和为1。
5. 按照归一化后的平局概率进行对手玩家的选择。
6. 重复2-5步直到$\sigma$收敛到固定值（这表示玩家的天梯得分处于稳定状态）。

在我们的评分系统中，匹配的方式是多人混战。当在计算玩家的平局概率时，我们采用1对1的匹配方式，并且将该平局概率作为选择对手的概率分布。

### 参数设置

在我们的评分系统和匹配机制中，设定的参数为 $\mu=1000$, $\sigma=8.333$.










