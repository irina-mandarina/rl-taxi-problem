"""
main module: orchestrates Q-learning experiments on Taxi-v3 environment.

this module:
    - configures multiple experiments with different hyperparameters
    - runs training and evaluation for all experiments
    - generates comprehensive visualizations and analysis
"""

import os
import numpy as np
from experiment_runner import run_experiment
from visualization import (
    plot_learning_curves,
    plot_hyperparameter_sensitivity,
    plot_q_table_heatmap,
    plot_policy_grid,
    plot_exploration_comparison,
)


SHOW_PLOTS = False

def build_experiment_configs() -> list:
    """
    build list of experiment configurations to test different hyperparameters.
    
    Configurations systematically explore:
    - Learning rate (alpha): [0.01, 0.1, 0.5]
    - Exploration decay: [0.98, 0.995, 0.99]
    - Boltzmann temperature: [0.5, 1.0, 2.0]
    - Discount factor (gamma): [0.9, 0.99, 0.999]
    
    returns:
        list of experiment configuration dictionaries
    """
    
    # baseline epsilon-greedy with standard hyperparameters
    baseline_eg = {
        "name": "baseline (eps-greedy, alpha=0.1, gamma=0.99)",
        "env_kwargs": {},
        "agent_kwargs": {
            "learning_rate": 0.1,
            "discount_factor": 0.99,
            "exploration_strategy": "epsilon_greedy",
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
        },
    }
    
    # high learning rate
    high_lr = {
        "name": "high learning rate (alpha=0.5)",
        "env_kwargs": {},
        "agent_kwargs": {
            "learning_rate": 0.5,
            "discount_factor": 0.99,
            "exploration_strategy": "epsilon_greedy",
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
        },
    }
    
    # low learning rate
    low_lr = {
        "name": "low learning rate (alpha=0.01)",
        "env_kwargs": {},
        "agent_kwargs": {
            "learning_rate": 0.01,
            "discount_factor": 0.99,
            "exploration_strategy": "epsilon_greedy",
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
        },
    }
    
    # low epsilon decay (slower exploration decay)
    slow_epsilon_decay = {
        "name": "slow eps decay (decay=0.98)",
        "env_kwargs": {},
        "agent_kwargs": {
            "learning_rate": 0.1,
            "discount_factor": 0.99,
            "exploration_strategy": "epsilon_greedy",
            "epsilon": 1.0,
            "epsilon_decay": 0.98,
            "epsilon_min": 0.01,
        },
    }
    
    # high epsilon decay (faster exploration decay)
    fast_epsilon_decay = {
        "name": "fast eps decay (decay=0.99)",
        "env_kwargs": {},
        "agent_kwargs": {
            "learning_rate": 0.1,
            "discount_factor": 0.99,
            "exploration_strategy": "epsilon_greedy",
            "epsilon": 1.0,
            "epsilon_decay": 0.99,
            "epsilon_min": 0.01,
        },
    }
    
    # Boltzmann strategy with low temperature
    boltzmann_low_temp = {
        "name": "Boltzmann (T=0.5, low temperature)",
        "env_kwargs": {},
        "agent_kwargs": {
            "learning_rate": 0.1,
            "discount_factor": 0.99,
            "exploration_strategy": "boltzmann",
            "boltzmann_temperature": 0.5,
        },
    }
    
    # Boltzmann strategy with medium temperature
    boltzmann_med_temp = {
        "name": "Boltzmann (T=1.0, medium temperature)",
        "env_kwargs": {},
        "agent_kwargs": {
            "learning_rate": 0.1,
            "discount_factor": 0.99,
            "exploration_strategy": "boltzmann",
            "boltzmann_temperature": 1.0,
        },
    }
    
    # Boltzmann strategy with high temperature
    boltzmann_high_temp = {
        "name": "Boltzmann (T=2.0, high temperature)",
        "env_kwargs": {},
        "agent_kwargs": {
            "learning_rate": 0.1,
            "discount_factor": 0.99,
            "exploration_strategy": "boltzmann",
            "boltzmann_temperature": 2.0,
        },
    }
    
    # high discount factor
    high_gamma = {
        "name": "high discount factor (gamma=0.999)",
        "env_kwargs": {},
        "agent_kwargs": {
            "learning_rate": 0.1,
            "discount_factor": 0.999,
            "exploration_strategy": "epsilon_greedy",
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
        },
    }
    
    # low discount factor
    low_gamma = {
        "name": "low discount factor (gamma=0.9)",
        "env_kwargs": {},
        "agent_kwargs": {
            "learning_rate": 0.1,
            "discount_factor": 0.9,
            "exploration_strategy": "epsilon_greedy",
            "epsilon": 1.0,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
        },
    }
    
    return [
        baseline_eg,
        high_lr,
        low_lr,
        slow_epsilon_decay,
        fast_epsilon_decay,
        boltzmann_low_temp,
        boltzmann_med_temp,
        boltzmann_high_temp,
        high_gamma,
        low_gamma,
    ]


def main():
    """main entry point for experiments."""
    
    print("=" * 80)
    print("Q-LEARNING ON TAXI-V3 ENVIRONMENT")
    print("=" * 80)
    print()
    
    # create output directory for figures
    output_dir = "figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # build experiment configurations
    print("building experiment configurations...")
    experiment_configs = build_experiment_configs()
    print(f"created {len(experiment_configs)} experiment configurations\n")
    
    # run all experiments
    print("running experiments (this may take a few minutes)...")
    print("-" * 80)
    all_results = []
    for idx, config in enumerate(experiment_configs, 1):
        print(f"[{idx}/{len(experiment_configs)}] running: {config['name']}")
        result = run_experiment(
            config,
            num_train_episodes=1000,
            num_eval_episodes=100,
            max_steps_per_episode=200,
        )
        all_results.append(result)
    
    print("-" * 80)
    print("\nall experiments completed!\n")
    
    # ========================================================================
    # FIGURE 1: LEARNING CURVES COMPARISON
    # ========================================================================
    print("generating Figure 1: learning curves comparison...")
    plot_learning_curves(
        all_results,
        figure_number=1,
        save_path=os.path.join(output_dir, "figure_1_learning_curves.png"),
    )
    
    # ========================================================================
    # FIGURE 2: HYPERPARAMETER SENSITIVITY ANALYSIS
    # ========================================================================
    print("generating Figure 2: hyperparameter sensitivity analysis...")
    plot_hyperparameter_sensitivity(
        all_results,
        figure_number=2,
        save_path=os.path.join(output_dir, "figure_2_sensitivity.png"),
    )
    
    # ========================================================================
    # FIGURE 3: EPSILON-GREEDY LEARNING CURVES
    # ========================================================================
    print("generating Figure 3: epsilon-greedy variants comparison...")
    epsilon_greedy_results = [r for r in all_results if "eps-greedy" in r["name"] or "eps decay" in r["name"]]
    if epsilon_greedy_results:
        plot_exploration_comparison(
            epsilon_greedy_results,
            figure_number=3,
            save_path=os.path.join(output_dir, "figure_3_epsilon_greedy.png"),
        )
    
    # ========================================================================
    # FIGURE 4: BOLTZMANN POLICY COMPARISON
    # ========================================================================
    print("generating Figure 4: Boltzmann policy comparison...")
    boltzmann_results = [r for r in all_results if "Boltzmann" in r["name"]]
    if boltzmann_results:
        plot_exploration_comparison(
            boltzmann_results,
            figure_number=4,
            save_path=os.path.join(output_dir, "figure_4_boltzmann.png"),
        )
    
    # ========================================================================
    # FIGURE 5: Q-TABLE ANALYSIS (BEST PERFORMING AGENT)
    # ========================================================================
    print("generating Figure 5: Q-table analysis...")
    best_result = max(all_results, key=lambda r: np.mean(r["evaluation_rewards"]))
    plot_q_table_heatmap(
        best_result["final_q_table"],
        figure_number=5,
        save_path=os.path.join(output_dir, "figure_5_q_table_heatmap.png"),
    )
    
    # ========================================================================
    # FIGURE 6: OPTIMAL POLICY VISUALIZATION
    # ========================================================================
    print("generating Figure 6: optimal policy grid visualization...")
    plot_policy_grid(
        best_result["final_q_table"],
        figure_number=6,
        save_path=os.path.join(output_dir, "figure_6_policy_grid.png"),
    )
    
    # Display summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print()
    
    print("BEST PERFORMING CONFIGURATION:")
    print(f"  name: {best_result['name']}")
    mean_eval_reward = np.mean(best_result["evaluation_rewards"])
    std_eval_reward = np.std(best_result["evaluation_rewards"])
    print(f"  mean evaluation reward: {mean_eval_reward:.2f} ± {std_eval_reward:.2f}")
    print()
    
    print("TOP 5 CONFIGURATIONS BY EVALUATION PERFORMANCE:")
    sorted_results = sorted(
        all_results,
        key=lambda r: np.mean(r["evaluation_rewards"]),
        reverse=True,
    )
    for rank, result in enumerate(sorted_results[:5], 1):
        mean_reward = np.mean(result["evaluation_rewards"])
        std_reward = np.std(result["evaluation_rewards"])
        print(f"  {rank}. {result['name']}")
        print(f"     mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    print()
    print("EXPLORATION STRATEGY COMPARISON:")
    
    epsilon_greedy_rewards = [
        np.mean(r["evaluation_rewards"])
        for r in all_results
        if "eps-greedy" in r["name"] or "eps decay" in r["name"]
    ]
    boltzmann_rewards = [
        np.mean(r["evaluation_rewards"])
        for r in all_results
        if "Boltzmann" in r["name"]
    ]
    
    if epsilon_greedy_rewards:
        avg_eg = np.mean(epsilon_greedy_rewards)
        std_eg = np.std(epsilon_greedy_rewards)
        print(f"  eps-greedy: {avg_eg:.2f} ± {std_eg:.2f}")
    
    if boltzmann_rewards:
        avg_bm = np.mean(boltzmann_rewards)
        std_bm = np.std(boltzmann_rewards)
        print(f"  Boltzmann: {avg_bm:.2f} ± {std_bm:.2f}")
    
    print()
    print("=" * 80)
    print(f"all figures saved to: {os.path.abspath(output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
