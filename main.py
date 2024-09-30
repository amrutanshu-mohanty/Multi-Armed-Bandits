# Write code which will run all the different bandit agents together and:
# 1. Plot a common cumulative regret curves graph
# 2. Plot a common graph of average reward curves

from epsilon_greedy import EpsilonGreedyAgent
from klucb import KLUCBAgent
from thompson import ThompsonSamplingAgent
from ucb import UCBAgent
from base import MultiArmedBandit
import numpy as np
import matplotlib.pyplot as plt

# Init Bandit
TIME_HORIZON = 10_000
bandit = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
epsilon_greedy_agent = EpsilonGreedyAgent(TIME_HORIZON, bandit)
kl_ucb_agent = KLUCBAgent(TIME_HORIZON, bandit)
ucb_agent = UCBAgent(TIME_HORIZON, bandit)
thompson_sampling_agent = ThompsonSamplingAgent(TIME_HORIZON, bandit)

# Loop
for i in range(TIME_HORIZON):
    epsilon_greedy_agent.give_pull()
    kl_ucb_agent.give_pull()
    ucb_agent.give_pull()
    thompson_sampling_agent.give_pull()

# Plotting Cumulative Regret Curves


# Plotting Cumulative Average Reward Curves

fig = plt.figure(figsize=(12,6))

ax1 = fig.add_subplot(121)
timesteps = np.arange(1, len(epsilon_greedy_agent.bandit.cumulative_regret_array) + 1)
regret_array1 = np.asarray(epsilon_greedy_agent.bandit.cumulative_regret_array)
ax1.plot(timesteps, regret_array1, linestyle='-', color='r', label='Epsilon Greedy')
timesteps = np.arange(1, len(ucb_agent.bandit.cumulative_regret_array) + 1)
regret_array2 = np.asarray(ucb_agent.bandit.cumulative_regret_array)
ax1.plot(timesteps, regret_array2, linestyle='-', color='b', label='UCB')
print(regret_array2.shape, regret_array1.shape)

timesteps = np.arange(1, len(kl_ucb_agent.bandit.cumulative_regret_array) + 1)
regret_array = kl_ucb_agent.bandit.cumulative_regret_array
ax1.plot(timesteps, regret_array, linestyle='-', color='g', label='KL UCB')
timesteps = np.arange(1, len(thompson_sampling_agent.bandit.cumulative_regret_array) + 1)
regret_array = thompson_sampling_agent.bandit.cumulative_regret_array
ax1.plot(timesteps, regret_array, linestyle='-', color='y', label='Thompson Sampling')

# ax1.plot(timesteps, regret_array1 - regret_array2)



ax2 = fig.add_subplot(122)
# Average out self.rewards
avg_rewards = [np.mean(epsilon_greedy_agent.rewards[0:T+1]) for T in range(epsilon_greedy_agent.time_to_run)]
timesteps = np.arange(1, len(epsilon_greedy_agent.rewards) + 1)
ax2.plot(timesteps, avg_rewards, linestyle='-', color='r', label='Epsilon Greedy')
timesteps = np.arange(1, len(ucb_agent.rewards) + 1) # basically defining a timestep for each pull
avg_rewards = [np.mean(ucb_agent.rewards[0:T+1]) for T in range(ucb_agent.time_to_run)]
ax2.plot(timesteps, avg_rewards, linestyle='-', color='b', label='UCB')
avg_rewards = [np.mean(kl_ucb_agent.rewards[0:T+1]) for T in range(kl_ucb_agent.time_to_run)]
ax2.plot(timesteps, avg_rewards, linestyle='-', color='g', label='KL UCB')
avg_rewards = [np.mean(thompson_sampling_agent.rewards[0:T+1]) for T in range(thompson_sampling_agent.time_to_run)]
ax2.plot(timesteps, avg_rewards, linestyle='-', color='y', label='Thompson Sampling')


# Formatting

ax1.set_title('Cumulative Regret Over Time', fontsize=16)
ax1.set_xlabel('Timesteps', fontsize=14)
ax1.set_ylabel('Cumulative Regret', fontsize=14)
ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
# ax1.set_yticks(np.arange(0, max(epsilon_greedy_agent.bandit.cumulative_regret_array) + 5, step=5))


ax2.set_title('Average Reward Over Time', fontsize=16)
ax2.set_xlabel('Timesteps', fontsize=14)
ax2.set_ylabel('Mean Reward Value upto timestep t', fontsize=14)
ax2.grid(True, which='both', linestyle='-', linewidth=0.5)

# Add legend
ax1.legend(loc='upper left', fontsize=12)
ax2.legend(loc='upper left', fontsize=12)

# Tight layout to ensure there's no clipping of labels
plt.tight_layout()

# Show plot
plt.show()