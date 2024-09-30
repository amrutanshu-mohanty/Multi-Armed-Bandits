import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt


class UCBAgent(Agent):
    # Add fields
    reward_memory : np.ndarray # A per arm value of how much reward was gathered
    count_memory : np.ndarray # An array of the number of times an arm is pulled 
    ucb : np.ndarray # An array of the upper confidence bounds for each arm at a given timestep

    def __init__(self, time_horizon, bandit:MultiArmedBandit,): 
        # Add fields
        super().__init__(time_horizon, bandit)
        self.bandit : MultiArmedBandit = bandit
        self.reward_memory = np.zeros(len(bandit.arms))
        self.count_memory = np.zeros(len(bandit.arms))
        self.time_step = 1 # starting from 1 to avoid issues with log
        self.ucb = np.zeros(len(bandit.arms))

    def give_pull(self):
        if self.time_step <= len(self.bandit.arms):
            # needed to initialize
            arm = self.time_step - 1
        else:
            arm = np.argmax(self.ucb)
        
        reward = self.bandit.pull(arm)
        self.reinforce(reward, arm)

    def reinforce(self, reward, arm):
        self.count_memory[arm]  += 1
        self.reward_memory[arm] += reward
        self.time_step += 1
        self.rewards.append(reward)

        if self.time_step <= len(self.bandit.arms):
            # needed to initialize
            self.ucb = (self.reward_memory[arm] / self.count_memory[arm]) + np.sqrt(2*np.log(self.time_step)/self.count_memory[arm])
        else:
            self.ucb = (self.reward_memory / self.count_memory) + np.sqrt(2*np.log(self.time_step)/self.count_memory)
        """
        # using numpy vector functions instead of if-else
        self.ucb = np.where(self.count_memory != 0, (self.reward_memory / self.count_memory) + np.sqrt(2*np.log(self.time_step)/self.count_memory), 0)
        """
        
    def plot_arm_graph(self):
        counts = self.count_memory
        indices = np.arange(len(counts))

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.bar(indices, counts, color='skyblue', edgecolor='black')

        # Formatting
        plt.title('Counts per Category', fontsize=16)
        plt.xlabel('Arm', fontsize=14)
        plt.ylabel('Pull Count', fontsize=14)
        plt.grid(axis='y', linestyle='-')  # Add grid lines for the y-axis
        plt.xticks(indices, [f'Category {i+1}' for i in indices], rotation=45, ha='right')
        # plt.yticks(np.arange(0, max(counts) + 2, step=2))

        # Annotate the bars with the count values
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=12, color='black')

        # Tight layout to ensure there's no clipping of labels
        plt.tight_layout()

        # Show plot
        plt.show()


# Code to test
if __name__ == "__main__":
    # Init Bandit
    TIME_HORIZON = 10_000
    bandit = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    agent = UCBAgent(TIME_HORIZON, bandit) ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()
