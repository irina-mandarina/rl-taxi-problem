"""
experiment_runner module: handles training and evaluation of Q-learning agents.

this module provides:
    - train_agent function for running training episodes
    - evaluate_agent function for running evaluation episodes
    - experiment configuration and execution
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Optional
from q_learning_agent import QLearningAgent


def train_agent(
    agent: QLearningAgent,
    env: gym.Env,
    num_episodes: int,
    max_steps_per_episode: int = 200,
) -> Dict[str, List[float]]:
    """
    train the agent on the environment for specified number of episodes.
    
    args:
        agent: QLearningAgent instance to train
        env: gymnasium environment
        num_episodes: number of training episodes
        max_steps_per_episode: maximum steps allowed per episode
    
    returns:
        dictionary containing training statistics (rewards, steps)
    """
    for episode_idx in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        episode_step_count = 0
        done = False
        
        while not done and episode_step_count < max_steps_per_episode:
            # get action mask from environment (if available)
            action_mask = info.get("action_mask", None)
            
            # select action using exploration strategy
            action = agent.select_action(state, training=True, action_mask=action_mask)
            
            # take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # update Q-value
            agent.update_q_value(state, action, reward, next_state, done)
            
            # update state and accumulate reward
            state = next_state
            episode_reward += reward
            episode_step_count += 1
        
        # record episode statistics
        agent.record_episode_stats(episode_reward, episode_step_count)
        
        # decay exploration rate
        agent.decay_exploration()
    
    return agent.get_training_statistics()


def evaluate_agent(
    agent: QLearningAgent,
    env: gym.Env,
    num_episodes: int,
    max_steps_per_episode: int = 200,
) -> Tuple[List[float], List[int]]:
    """
    evaluate the agent on the environment using greedy policy (no exploration).
    
    args:
        agent: trained QLearningAgent instance
        env: gymnasium environment
        num_episodes: number of evaluation episodes
        max_steps_per_episode: maximum steps allowed per episode
    
    returns:
        tuple of (episode_rewards, episode_steps) lists
    """
    evaluation_rewards = []
    evaluation_steps = []
    
    for episode_idx in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        episode_step_count = 0
        done = False
        
        while not done and episode_step_count < max_steps_per_episode:
            # get action mask
            action_mask = info.get("action_mask", None)
            
            # select greedy action (no exploration)
            action = agent.select_action(state, training=False, action_mask=action_mask)
            
            # take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # update state and accumulate reward
            state = next_state
            episode_reward += reward
            episode_step_count += 1
        
        evaluation_rewards.append(episode_reward)
        evaluation_steps.append(episode_step_count)
    
    return evaluation_rewards, evaluation_steps


def run_experiment(
    experiment_config: Dict,
    num_train_episodes: int = 1000,
    num_eval_episodes: int = 100,
    max_steps_per_episode: int = 200,
) -> Dict:
    """
    run a complete training experiment with given configuration.
    
    args:
        experiment_config: dictionary with agent and environment configuration
        num_train_episodes: number of training episodes
        num_eval_episodes: number of evaluation episodes
        max_steps_per_episode: maximum steps per episode
    
    returns:
        dictionary containing training and evaluation results
    """
    # create environment
    env_kwargs = experiment_config.get("env_kwargs", {})
    env = gym.make("Taxi-v3", **env_kwargs)
    
    # create agent
    agent_kwargs = experiment_config.get("agent_kwargs", {})
    agent = QLearningAgent(
        num_states=500,  # Taxi-v3 has 500 states
        num_actions=6,   # Taxi-v3 has 6 actions
        **agent_kwargs
    )
    
    # training phase
    print(f"training agent: {experiment_config.get('name', 'experiment')}")
    train_agent(agent, env, num_train_episodes, max_steps_per_episode)
    
    # evaluation phase
    print(f"evaluating agent: {experiment_config.get('name', 'experiment')}")
    eval_rewards, eval_steps = evaluate_agent(
        agent, env, num_eval_episodes, max_steps_per_episode
    )
    
    # compile results
    training_stats = agent.get_training_statistics()
    results = {
        "name": experiment_config.get("name", "experiment"),
        "config": experiment_config,
        "agent": agent,
        "training_rewards": training_stats["episode_rewards"],
        "training_steps": training_stats["episode_steps"],
        "evaluation_rewards": eval_rewards,
        "evaluation_steps": eval_steps,
        "final_q_table": agent.get_q_table(),
    }
    
    env.close()
    return results
