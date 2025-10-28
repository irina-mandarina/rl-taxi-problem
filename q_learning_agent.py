"""
q_learning_agent module: implements Q-learning reinforcement learning agent.

this module provides:
    - QLearningAgent class for learning policies using Q-learning
    - support for multiple exploration strategies (epsilon-greedy, Boltzmann)
    - Q-table management and visualization support
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


class QLearningAgent:
    """
    implements Q-learning algorithm for discrete state-action environments.
    
    this class manages:
        - Q-table initialization and updates
        - exploration-exploitation strategies
        - training and evaluation phases
    """
    
    # class constants for exploration strategies
    EXPLORATION_EPSILON_GREEDY = "epsilon_greedy"
    EXPLORATION_BOLTZMANN = "boltzmann"
    
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        exploration_strategy: str = "epsilon_greedy",
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        boltzmann_temperature: float = 1.0,
    ):
        """
        initialize Q-learning agent.
        
        args:
            num_states: number of discrete states in the environment
            num_actions: number of discrete actions available
            learning_rate: alpha parameter for Q-value updates (0 to 1)
            discount_factor: gamma parameter for future rewards (0 to 1)
            exploration_strategy: "epsilon_greedy" or "boltzmann"
            epsilon: initial exploration rate for epsilon-greedy
            epsilon_decay: decay rate for epsilon per episode
            epsilon_min: minimum epsilon value
            boltzmann_temperature: temperature parameter for Boltzmann policy
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.boltzmann_temperature = boltzmann_temperature
        
        # initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions))
        
        # track training statistics
        self.episode_rewards = []
        self.episode_steps = []
    
    def select_action(
        self,
        state: int,
        training: bool = True,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        select action using the configured exploration strategy.
        
        args:
            state: current state index
            training: if False, use greedy policy (exploitation only)
            action_mask: optional mask for valid actions (1 = valid, 0 = invalid)
        
        returns:
            selected action index
        """
        if not training:
            # exploitation: greedy policy
            return self._select_greedy_action(state, action_mask)
        
        if self.exploration_strategy == self.EXPLORATION_EPSILON_GREEDY:
            return self._select_epsilon_greedy_action(state, action_mask)
        elif self.exploration_strategy == self.EXPLORATION_BOLTZMANN:
            return self._select_boltzmann_action(state, action_mask)
        else:
            raise ValueError(f"unknown exploration strategy: {self.exploration_strategy}")
    
    def _select_greedy_action(
        self,
        state: int,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        select action greedily using highest Q-value.
        
        args:
            state: current state index
            action_mask: optional mask for valid actions
        
        returns:
            action with highest Q-value (or masked highest Q-value)
        """
        q_values = self.q_table[state, :].copy()
        
        if action_mask is not None:
            # mask invalid actions by setting their Q-values to -inf
            q_values[action_mask == 0] = -np.inf
        
        return np.argmax(q_values)
    
    def _select_epsilon_greedy_action(
        self,
        state: int,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        select action using epsilon-greedy strategy.
        
        with probability epsilon, select random action; otherwise select greedy action.
        
        args:
            state: current state index
            action_mask: optional mask for valid actions
        
        returns:
            selected action index
        """
        if np.random.random() < self.epsilon:
            # exploration: random action
            if action_mask is not None:
                valid_actions = np.where(action_mask == 1)[0]
                return np.random.choice(valid_actions)
            else:
                return np.random.randint(0, self.num_actions)
        else:
            # exploitation: greedy action
            return self._select_greedy_action(state, action_mask)
    
    def _select_boltzmann_action(
        self,
        state: int,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        select action using Boltzmann (softmax) policy.
        
        probability of selecting action a is proportional to exp(Q(s,a)/T)
        where T is the temperature parameter.
        
        args:
            state: current state index
            action_mask: optional mask for valid actions
        
        returns:
            selected action index (sampled from Boltzmann distribution)
        """
        q_values = self.q_table[state, :].copy()
        
        # apply action mask if provided
        if action_mask is not None:
            q_values[action_mask == 0] = -np.inf
        
        # compute Boltzmann probabilities
        # subtract max for numerical stability
        q_values_normalized = q_values - np.max(q_values[~np.isinf(q_values)])
        
        # compute softmax probabilities
        exp_q = np.exp(q_values_normalized / self.boltzmann_temperature)
        
        # set probabilities of masked actions to 0
        if action_mask is not None:
            exp_q[action_mask == 0] = 0
        
        probabilities = exp_q / np.sum(exp_q)
        
        # sample action from probability distribution
        return np.random.choice(self.num_actions, p=probabilities)
    
    def update_q_value(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """
        update Q-value using Q-learning update rule.
        
        Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        
        args:
            state: current state index
            action: action taken
            reward: reward received
            next_state: resulting next state
            done: whether episode terminated
        """
        if done:
            # terminal state has no future reward
            target_q_value = reward
        else:
            # estimate future reward using max Q-value of next state
            max_next_q = np.max(self.q_table[next_state, :])
            target_q_value = reward + self.discount_factor * max_next_q
        
        # compute temporal difference error
        td_error = target_q_value - self.q_table[state, action]
        
        # update Q-value
        self.q_table[state, action] += self.learning_rate * td_error
    
    def decay_exploration(self) -> None:
        """decay exploration rate (epsilon) after each episode."""
        if self.exploration_strategy == self.EXPLORATION_EPSILON_GREEDY:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def record_episode_stats(self, episode_reward: float, episode_steps: int) -> None:
        """
        record episode statistics for analysis.
        
        args:
            episode_reward: total reward accumulated in episode
            episode_steps: number of steps taken in episode
        """
        self.episode_rewards.append(episode_reward)
        self.episode_steps.append(episode_steps)
    
    def get_q_table(self) -> np.ndarray:
        """return the current Q-table."""
        return self.q_table.copy()
    
    def set_q_table(self, q_table: np.ndarray) -> None:
        """set the Q-table (useful for loading trained agents)."""
        self.q_table = q_table.copy()
    
    def get_training_statistics(self) -> Dict[str, List[float]]:
        """
        return collected training statistics.
        
        returns:
            dictionary containing episode_rewards and episode_steps lists
        """
        return {
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
        }
