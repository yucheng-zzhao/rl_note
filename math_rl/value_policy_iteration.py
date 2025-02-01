import numpy as np
import os

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
        
        self.example_dir = './images/value_policy_iteration'
        if not os.path.exists(self.example_dir):
            os.makedirs(self.example_dir, exist_ok=True)
        
        
    def random_greed_policy(self):
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        for state_index in range(self.state_space_size):
            action = np.random.choice(range(self.action_space_size))
            policy[state_index, action] = 1
        return policy
    

    def policy_improvement(self, state_value):
        """
        Policy Improvement Algorithm
        
        Purpose:
        --------
        This function performs policy improvement based on the current state values. 
        It computes the action-value function (Q-values) for each state-action pair 
        using the provided state values and updates the policy to select actions 
        that maximize the Q-values (greedy policy improvement).

        Parameters:
        -----------
        state_value: np.ndarray
            A NumPy array representing the current state value function (V(s)) 
            for all states in the environment.

        Returns:
        --------
        - policy: np.ndarray
            The improved policy, represented as a deterministic policy where each 
            state selects the optimal action that maximizes the Q-value.
        - state_value_k: np.ndarray
            Updated state values based on the maximum Q-values for each state 
            (same as input state_value since it's copied and used internally).

        Workflow:
        ---------
        1. Initialize an empty policy array and a copy of the state values.
        2. For each state:
            a. Calculate the Q-values for all possible actions in the state using the 
               `calculate_qvalue` function.
            b. Update the state value (V(s)) to the maximum Q-value (greedy update).
            c. Identify the action that maximizes the Q-value.
            d. Update the policy to select this optimal action deterministically.
        3. Return the updated policy and state values.

        Notes:
        ------
        - This method ensures the policy is improved iteratively by selecting 
          actions that maximize the expected return.
        - Greedy policy improvement guarantees convergence to an optimal policy 
          when combined with policy evaluation.

        Example:
        --------
        If the state value function for a given state is [10, 20, 15] for three 
        actions, the policy will select the action with the highest value (20).
        """
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        state_value_k = state_value.copy()
        for state in range(self.state_space_size):
            qvalue_list = []
            # Compute Q-values for all possible actions in the current state
            for action in range(self.action_space_size):
                qvalue_list.append(self.calculate_qvalue(state, action, state_value.copy()))
            
            # Update the state value for the current state to the maximum Q-value (greedy approach)
            state_value_k[state] = max(qvalue_list)

            # Find the action that maximizes the Q-value
            action_star = qvalue_list.index(max(qvalue_list))

            # Update the policy for the current state to select the optimal action deterministically
            policy[state, action_star] = 1
        return policy, state_value_k
    
    
    def policy_evaluation(self, policy, tolerance=0.001, steps=10):
        """
        Policy Evaluation Algorithm

        Purpose:
        --------
        This function evaluates a given policy by computing the state value function (V(s)) 
        iteratively. It calculates the expected return for each state when actions are 
        selected according to the specified policy.

        Parameters:
        -----------
        policy: np.ndarray
            A NumPy array representing the policy, where `policy[s, a]` is the probability of 
            taking action `a` in state `s`.
        tolerance: float, optional (default: 0.001)
            The stopping criterion for convergence. The iteration stops when the 
            difference between successive state value estimates is less than this threshold 
            (measured using the L1 norm).
        steps: int, optional (default: 10)
            The maximum number of iterations allowed to compute the state values.

        Returns:
        --------
        - state_value_k: np.ndarray
            A NumPy array representing the evaluated state value function (V(s)) for all 
            states in the environment.

        Workflow:
        ---------
        1. Initialize:
            - `state_value_k`: A temporary state value array initialized to ones.
            - `state_value`: A state value array initialized to zeros.
        2. Iteratively update the state values until convergence or the maximum number of steps is reached:
            a. Copy the current state values to `state_value`.
            b. For each state:
                - Compute the expected value by summing over all actions weighted by the policy 
                  and their respective Q-values.
                - Update the temporary state value (`state_value_k`) for the current state.
        3. Return the updated state values (`state_value_k`).

        Notes:
        ------
        - This algorithm assumes a stationary policy and calculates the expected return for 
          each state by iterating over the Bellman expectation equation.
        - The convergence is guaranteed for finite state and action spaces.

        Example:
        --------
        Given a simple environment with states S = {0, 1}, actions A = {0, 1}, 
        and a policy that assigns equal probabilities to both actions:
        - The algorithm will iteratively compute V(s) for all states until it converges 
          to the true expected return under the policy.
        """
        state_value_k = np.ones(self.state_space_size)
        state_value = np.zeros(self.state_space_size)
        while np.linalg.norm(state_value_k - state_value, ord=1) > tolerance:
            state_value = state_value_k.copy()
            for state in range(self.state_space_size):
                value = 0
                for action in range(self.action_space_size):
                    value += policy[state, action] * self.calculate_qvalue(state_value=state_value_k.copy(),
                                                                           state=state,
                                                                           action=action)  
                state_value_k[state] = value
        return state_value_k
    
    

    def calculate_qvalue(self, state, action, state_value):
        qvalue = 0
        for i in range(self.reward_space_size):
            qvalue += self.reward_list[i] * self.env.Rsa[state, action, i]
        for next_state in range(self.state_space_size):
            qvalue += self.gamma * self.env.Psa[state, action, next_state] * state_value[next_state]
        return qvalue

    def value_iteration(self, tolerance=0.001, steps=1000):
        """
        Value Iteration Algorithm

        Purpose:
        --------
        This function implements the Value Iteration algorithm, a dynamic programming method
        to compute the optimal policy and state values for a Markov Decision Process (MDP).

        Parameters:
        -----------
        tolerance: float, optional (default: 0.001)
            The stopping criterion for convergence. The algorithm stops when the difference 
            between successive state value estimates is less than this threshold.
        steps: int, optional (default: 1000)
            The maximum number of iterations allowed to ensure convergence.

        Workflow:
        ---------
        1. Initialize `state_value_k` (state values) as a zero array.
        2. Iteratively update state values and policy until:
            a. The state value estimates converge (difference between iterations is less than `tolerance`).
            b. The number of iterations exceeds the maximum allowed steps.
        3. For each iteration:
            a. Update the current state values with `state_value_k.copy()`.
            b. Perform policy improvement to find the optimal policy based on the updated state values.
            c. Update `state_value_k` with the improved values.
        4. Return the number of remaining steps.

        Returns:
        --------
        steps: int
            The number of steps remaining after convergence or hitting the maximum limit.

        Notes:
        ------
        - This method uses a greedy policy improvement approach to iteratively improve the policy.
        - The state value convergence is measured using the L1 norm.
        - Ensures optimal policy and state values are derived for an episodic MDP.

        Output:
        -------
        - Updates `self.policy` to the optimal policy.
        - Updates `self.state_value` to the optimal state values.
        """
        state_value_k = np.zeros(self.state_space_size)
        while np.linalg.norm(state_value_k - self.state_value, ord=1) > tolerance and steps > 0:
            steps -= 1
            self.state_value = state_value_k.copy()
            self.policy, state_value_k = self.policy_improvement(state_value_k.copy())
        return steps
    
    
    def policy_iteration(self, tolerance=0.001, steps=1000):
        policy = self.random_greed_policy()
        while np.linalg.norm(policy - self.policy, ord=1) > tolerance and steps > 0:
            steps -= 1
            policy = self.policy.copy()
            self.state_value = self.policy_evaluation(self.policy.copy(), tolerance, steps)
            self.policy, _ = self.policy_improvement(self.state_value)
        return steps
    
    
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


def value_iteration():
    env = grid_env.GridEnv(size=5, target=[2, 3],
                           forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
                           render_mode='')
    
    solver = Solve(env)
    solver.value_iteration()
    
    solver.show_policy()
    solver.show_state_value(solver.state_value, y_offset=0.25)
    solver.env.render()
    solver.env.render_.save_frame('./{}/value_interation.jpg'.format(solver.example_dir))
    

def policy_iteration():
    env = grid_env.GridEnv(size=5, target=[2, 3],
                           forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]],
                           render_mode='')
    
    solver = Solve(env)
    solver.policy_iteration()
    
    solver.show_policy()
    solver.show_state_value(solver.state_value, y_offset=0.25)
    solver.env.render()
    solver.env.render_.save_frame('./{}/policy_interation.jpg'.format(solver.example_dir))
    
    
if __name__ == '__main__':
    value_iteration()
    
    policy_iteration()