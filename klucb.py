import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt
"""
def KL_divergence(x : np.ndarray, y : np.ndarray):
    # we code it such that it can be applied to numpy arrays
    return np.sum(np.where(x != 0, x * np.log(x/y), 0))
"""

def KL_divergence(x : float, y : float):
    # print(x, " ", y)
    return x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))

class KLUCBAgent(Agent):
    # Add fields
    c : float
    reward_memory : np.ndarray # A per arm value of how much reward was gathered
    count_memory : np.ndarray # An array of the number of times an arm is pulled 
    kl_ucb : np.ndarray # An array storing the kl-ucb value for each arm at given time step
    # q_solve_binary_search_vectorized : int # Vectorized function to solve required equation
    def __init__(self, time_horizon, bandit:MultiArmedBandit, c = 3): 
        # Add fields
        super().__init__(time_horizon, bandit)
        self.c = c
        self.bandit : MultiArmedBandit = bandit
        self.reward_memory = np.zeros(len(bandit.arms))
        self.count_memory = np.zeros(len(bandit.arms))
        self.time_step = 0
        self.kl_ucb = np.zeros(len(bandit.arms))
        # self.q_solve_binary_search_vectorized = np.vectorize(self.q_solve_binary_search)

    def q_solve_binary_search(self, empirical_mean, t, u_t, c):
        left_pointer = empirical_mean
        right_pointer = 1
        mid_point = (left_pointer + right_pointer) / 2
        value = (np.log(t) + c*np.log(np.log(t))) / u_t
        # print(empirical_mean, " ", mid_point)
        
        estimated_value = KL_divergence(empirical_mean, mid_point)
        
        while abs(value - estimated_value) > 0.001:
            if estimated_value > value:
                right_pointer = mid_point
            else:
                left_pointer = mid_point
            mid_point = (left_pointer + right_pointer) / 2
            estimated_value = KL_divergence(empirical_mean, mid_point)
            #print(mid_point, value)
            # print(value," ",estimated_value)
        return mid_point

    def give_pull(self):
        # it is easier to do one round of round-robin for easy initialization, same works for ucb as well
        if self.time_step < len(self.bandit.arms):
            arm = self.time_step
        else:
            arm = np.argmax(self.kl_ucb)
        reward = self.bandit.pull(arm)
        self.reinforce(reward, arm) 

    def reinforce(self, reward, arm):
        self.count_memory[arm] += 1
        self.reward_memory[arm] += reward
        self.time_step += 1
        self.rewards.append(reward)
        if self.time_step - 1 < len(self.bandit.arms):
            # print(self.reward_memory, self.count_memory)
            return None
        # self.kl_ucb = self.q_solve_binary_search_vectorized((self.reward_memory/self.count_memory), self.time_step, self.count_memory, self.c)
        for i in range(len(self.bandit.arms)):
            # did this so that we don't face an issue of empirical mean = 0 or 1 in the KL divergence
            # Empirical mean will remain 0 or 1 for only so long and since we are setting the kl_ucb for them to be 1, they will be
            # picked next ensuring value dips at some point
            if(self.reward_memory[i]/self.count_memory[i] == 1 or self.reward_memory[i]/self.count_memory[i] == 0):
                self.kl_ucb[i] = 1
            else:
                self.kl_ucb[i] = self.q_solve_binary_search(self.reward_memory[i]/self.count_memory[i], self.time_step, self.count_memory[i], self.c)
            # print("Reward: ", self.reward_memory[i])
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
    agent = KLUCBAgent(TIME_HORIZON, bandit, 4.5) ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()
        # print(agent.kl_ucb)

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()
