import numpy as np
import os
import random

import grid_env

class Solve:
    def __init__(self, env: grid_env.GridEnv):
        self.gamma = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list
        self.state_value = np.ones(shape=self.state_space_size)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        
        self.example_dir = './images/mc_learning'
        if not os.path.exists(self.example_dir):
            os.makedirs(self.example_dir, exist_ok=True)
        
    def obtain_episode(self, policy, start_state, start_action, length):
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            _, reward, done, _, _ = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            # print ('next state: {}, next action: {}, reward: {}'.format(next_state, next_action, reward))
            next_action = np.random.choice(np.arange(len(policy[next_state])), p=policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})
        return episode

    def mc_basic(self, length=30, epochs=10):
        """
        Monte Carlo Basic Algorithm Implementation
        
        Purpose:
        --------
        This method performs the Monte Carlo basic algorithm to estimate the 
        action-value function (Q-values) and improve the policy iteratively.
        It generates episodes, computes returns, updates Q-values, and refines the policy.

        Parameters:
        -----------
        length: int
            The length of each episode (default: 30).
        epochs: int
            The number of iterations (epochs) to run the Monte Carlo algorithm (default: 10).

        Workflow:
        ---------
        1. Iterate for a specified number of epochs:
            a. For each state in the state space:
                i. For each action in the action space:
                    - Generate an episode starting from the state-action pair.
                    - Compute the return (discounted sum of rewards) for the episode.
                    - Update the Q-value for the state-action pair.
                ii. Find the optimal action (max Q-value) for the current state.
                iii. Update the policy to be greedy for the optimal action.
        2. After all epochs, calculate the state value function based on the 
           updated Q-values and policy.

        Notes:
        ------
        - Uses a greedy policy improvement strategy.
        - Supports stochastic policies by using soft probabilities during updates.
        - The algorithm assumes the environment is episodic.

        Output:
        -------
        - Updates `self.qvalue` (state-action values).
        - Updates `self.policy` (optimal policy).
        - Updates `self.state_value` (state values derived from the policy and Q-values).
        """
        for _ in range(epochs):
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    episode = self.obtain_episode(self.policy, state, action, length)
                    g = 0
                    for step in range(len(episode) - 1, -1, -1):
                        g = episode[step]['reward'] + self.gamma * g
                    self.qvalue[state][action] = g
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                self.policy[state] = np.zeros(shape=self.action_space_size)
                self.policy[state, action_star] = 1
                
        self.update_state_value()
            
            
    def mc_exploring_starts(self, length=10, tolerance=0.001, max_epochs=1000):
        """
        Monte Carlo Exploring Starts Implementation
        
        Purpose:
        --------
        Implements the Monte Carlo Exploring Starts algorithm, which starts each 
        episode with a randomly selected state-action pair to ensure adequate exploration.

        Parameters:
        -----------
        length: int, optional (default: 10)
            The length of each episode.
        tolerance: float, optional (default: 0.001)
            The stopping criterion for policy convergence (difference between successive policies).
        max_epochs: int, optional (default: 1000)
            The maximum number of epochs to run the algorithm.

        Notes:
        ------
        - Uses First-Visit Monte Carlo to estimate returns.
        - Updates Q-values and policy iteratively based on episodes starting from random state-action pairs.

        Output:
        -------
        - Updates `self.qvalue` (state-action values).
        - Updates `self.policy` (optimal policy).
        - Prints policy convergence norm at each epoch.
        """
        returns = [[[0] for _ in range(self.action_space_size)] for _ in range(self.state_space_size)]

        for epoch in range(max_epochs):
            previous_policy = self.policy.copy()

            visit_list = set()
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    # Generate an episode starting from the random state-action pair
                    episode = self.obtain_episode(self.policy, start_state=state, start_action=action, length=length)
                    g = 0
                    # Process the episode in reverse order to compute returns
                    for step in range(len(episode) - 1, -1, -1):
                        reward = episode[step]['reward']
                        state = episode[step]['state']
                        action = episode[step]['action']
                        g = self.gamma * g + reward

                        # First-visit check
                        if (state, action) not in visit_list:
                            visit_list.add((state, action))
                            returns[state][action].append(g)
                            self.qvalue[state][action] = np.mean(returns[state][action])
                    
            # Update policy based on the new Q-values
            for state in range(self.state_space_size):
                best_action = np.argmax(self.qvalue[state])
                self.policy[state] = np.zeros(self.action_space_size)
                self.policy[state][best_action] = 1

            # Check for policy convergence
            policy_diff = np.linalg.norm(self.policy - previous_policy, ord=1)
            print(f"Epoch {epoch + 1}, Policy Convergence Norm: {policy_diff}")
            if policy_diff < tolerance:
                break
            
        self.update_state_value()
        
    
    def mc_epsilon_greedy(self, length=100, tolerance=0.001, epsilon=0.1, epsilon_min=0.01, decay_rate=0.99, max_epochs=1000):
        """
        Monte Carlo Epsilon-Greedy with Every-Visit Method
        
        Purpose:
        --------
        Implements a Monte Carlo control algorithm with an epsilon-greedy exploration strategy
        to estimate Q-values and derive an optimal policy using the every-visit approach.

        Parameters:
        -----------
        length: int, optional (default: 10)
            The length of each episode.
        tolerance: float, optional (default: 0.001)
            The stopping criterion for policy convergence (difference between successive policies).
        max_epochs: int, optional (default: 1000)
            The maximum number of epochs to run the algorithm.
        epsilon: float, optional (default: 0.1)
            The exploration probability. At each step, the agent will choose a random action with
            probability `epsilon` and the greedy action with probability `1 - epsilon`.

        Notes:
        ------
        - Updates Q-values for all occurrences of state-action pairs in an episode.
        - Balances exploration and exploitation using the epsilon-greedy strategy.

        Output:
        -------
        - Updates `self.qvalue` (state-action values).
        - Updates `self.policy` (epsilon-greedy optimal policy).
        - Prints policy convergence norm at each epoch.
        """
        returns_sum = np.zeros((self.state_space_size, self.action_space_size))
        returns_count = np.zeros((self.state_space_size, self.action_space_size))
        
        for epoch in range(max_epochs):
            previous_policy = self.policy.copy()
            
            epsilon = max(epsilon_min, epsilon * (decay_rate ** epoch))
            
            state = random.choice(range(self.state_space_size))
            action = random.choice(range(self.action_space_size))
            episode = self.obtain_episode(self.policy, start_state=state, start_action=action, length=length)
            
            # Process the episode to compute returns for all occurrences
            g = 0
            for step in range(len(episode) - 1, -1, -1):
                reward = episode[step]['reward']
                state = episode[step]['state']
                action = episode[step]['action']
                g = self.gamma * g + reward

                # Every-visit update: add G to the returns sum and increment count
                returns_sum[state][action] += g
                returns_count[state][action] += 1
                self.qvalue[state][action] = returns_sum[state][action] / returns_count[state][action]

            # Update policy using epsilon-greedy strategy
            for state in range(self.state_space_size):
                best_action = np.argmax(self.qvalue[state])
                for action in range(self.action_space_size):
                    if action == best_action:
                        self.policy[state][action] = 1 - (self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        self.policy[state][action] = 1.0 * epsilon / self.action_space_size

            # Check for policy convergence
            policy_diff = np.linalg.norm(self.policy - previous_policy, ord=1)
            print(f"Epoch {epoch + 1}, Policy Convergence Norm: {policy_diff}, Epsilon: {epsilon}")
            if policy_diff < tolerance and epoch > 100:
                break

        self.update_state_value()

    
    def update_state_value(self):
        # Calculate state values after policy and Q-values are updated
        for state in range(self.state_space_size):
            state_value_cur = 0.
            for action in range(self.action_space_size):
                state_value_cur += self.qvalue[state][action] * self.policy[state][action]
            self.state_value[state] = state_value_cur
    
    
    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.action_to_direction[action],
                                             radius=policy * 0.1)
                

    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)
            
            
def mc_basic():
    env = grid_env.GridEnv(size=5, target=[2, 3],
                           forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
                           render_mode='')
    
    solver = Solve(env)
    solver.mc_basic()
    
    solver.show_policy()
    solver.show_state_value(solver.state_value, y_offset=0.25)
    solver.env.render()
    solver.env.render_.save_frame('./{}/mc_basic.jpg'.format(solver.example_dir))
    
    
def mc_exploring_starts():
    env = grid_env.GridEnv(size=5, target=[2, 3],
                           forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
                           render_mode='')
    
    solver = Solve(env)
    solver.mc_exploring_starts()
    
    solver.show_policy()
    solver.show_state_value(solver.state_value, y_offset=0.25)
    solver.env.render()
    solver.env.render_.save_frame('./{}/mc_exploring_starts.jpg'.format(solver.example_dir))
    
    
def mc_epsilon_greedy():
    env = grid_env.GridEnv(size=5, target=[2, 3],
                           forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
                           render_mode='')
    
    solver = Solve(env)
    solver.mc_epsilon_greedy(epsilon=0.05, length=10000)
    
    solver.show_policy()
    solver.show_state_value(solver.state_value, y_offset=0.25)
    solver.env.render()
    solver.env.render_.save_frame('./{}/mc_epsilon_greedy.jpg'.format(solver.example_dir))
    
    
if __name__ == '__main__':
    # mc_basic()
    mc_exploring_starts()
    mc_epsilon_greedy()