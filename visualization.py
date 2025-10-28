"""
visualization module: creates plots for Q-learning analysis.

this module provides:
    - plot_learning_curves for training analysis
    - plot_q_table_heatmap for Q-value visualization
    - plot_policy_grid for optimal policy visualization
    - compute_moving_average for smoothing curves
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')


# visualization constants
FIGURE_DPI = 100
FIGURE_SIZE_LARGE = (14, 10)
FIGURE_SIZE_MEDIUM = (10, 6)
TAXI_GRID_SIZE = 5
TAXI_ACTIONS = {
    0: "south",
    1: "north",
    2: "east",
    3: "west",
    4: "pickup",
    5: "dropoff",
}
TAXI_ACTION_ARROWS = {
    0: "↓",
    1: "↑",
    2: "→",
    3: "←",
    4: "P",
    5: "D",
}
SHOW_PLOTS = False  


def compute_moving_average(values: List[float], window_size: int = 50) -> List[float]:
    """
    compute moving average of a list of values.
    
    args:
        values: list of values to smooth
        window_size: size of moving average window
    
    returns:
        list of smoothed values
    """
    if len(values) < window_size:
        return values
    
    moving_avg = []
    for idx in range(len(values) - window_size + 1):
        avg = np.mean(values[idx:idx + window_size])
        moving_avg.append(avg)
    
    return moving_avg


def plot_learning_curves(
    experiments_results: List[Dict],
    figure_number: int,
    save_path: Optional[str] = None,
) -> None:
    """
    plot learning curves showing average reward per episode.
    
    args:
        experiments_results: list of experiment result dictionaries
        figure_number: figure number for labeling
        save_path: optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LARGE, dpi=FIGURE_DPI)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_results)))
    
    for exp_idx, result in enumerate(experiments_results):
        rewards = result["training_rewards"]
        moving_avg = compute_moving_average(rewards, window_size=50)
        
        # plot raw rewards (light, transparent)
        axes[0].plot(
            rewards,
            alpha=0.2,
            color=colors[exp_idx],
            linewidth=0.5,
        )
        
        # plot moving average (bold)
        axes[0].plot(
            range(len(moving_avg)),
            moving_avg,
            label=result["name"],
            color=colors[exp_idx],
            linewidth=2,
        )
    
    axes[0].set_xlabel("training episode", fontsize=11)
    axes[0].set_ylabel("episode reward", fontsize=11)
    axes[0].set_title("learning curves (with moving average)", fontsize=12)
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    
    # evaluation results
    for exp_idx, result in enumerate(experiments_results):
        eval_rewards = result["evaluation_rewards"]
        axes[1].scatter(
            [exp_idx] * len(eval_rewards),
            eval_rewards,
            alpha=0.5,
            s=50,
            color=colors[exp_idx],
        )
        
        # plot mean and std
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        axes[1].errorbar(
            exp_idx,
            mean_reward,
            yerr=std_reward,
            fmt="D",
            color=colors[exp_idx],
            markersize=8,
            capsize=5,
            capthick=2,
        )
    
    axes[1].set_xticks(range(len(experiments_results)))
    axes[1].set_xticklabels(
        [r["name"] for r in experiments_results],
        rotation=45,
        ha="right",
    )
    axes[1].set_ylabel("episode reward", fontsize=11)
    axes[1].set_title("evaluation results (mean ± std)", fontsize=12)
    axes[1].grid(True, alpha=0.3, axis="y")
    
    fig.suptitle(
        f"Figure {figure_number}: Q-Learning Performance Comparison\n"
        f"Experimental Settings: 10 configurations (ε-greedy decay 0.98-0.995, Boltzmann T=0.5-2.0, α=0.01-0.5, γ=0.9-0.999)\n"
        f"Training: 1000 episodes, Evaluation: 100 episodes (greedy policy)",
        fontsize=12,
        fontweight="bold",
    )
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else None


def plot_hyperparameter_sensitivity(
    experiments_results: List[Dict],
    figure_number: int,
    save_path: Optional[str] = None,
) -> None:
    """
    plot sensitivity analysis for hyperparameters.
    
    args:
        experiments_results: list of experiment result dictionaries
        figure_number: figure number for labeling
        save_path: optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_LARGE, dpi=FIGURE_DPI)
    axes = axes.flatten()
    
    # extract hyperparameters and results
    learning_rates = []
    epsilon_decays = []
    final_rewards_lr = []
    final_rewards_ed = []
    
    for result in experiments_results:
        config = result["config"].get("agent_kwargs", {})
        eval_rewards = result["evaluation_rewards"]
        mean_eval_reward = np.mean(eval_rewards)
        
        if "learning_rate" in config:
            learning_rates.append(config["learning_rate"])
            final_rewards_lr.append(mean_eval_reward)
        
        if "epsilon_decay" in config:
            epsilon_decays.append(config["epsilon_decay"])
            final_rewards_ed.append(mean_eval_reward)
    
    # plot learning rate sensitivity
    if learning_rates:
        axes[0].plot(learning_rates, final_rewards_lr, "o-", linewidth=2, markersize=8)
        axes[0].set_xlabel("learning rate (alpha)", fontsize=11)
        axes[0].set_ylabel("mean evaluation reward", fontsize=11)
        axes[0].set_title("sensitivity to learning rate", fontsize=11)
        axes[0].grid(True, alpha=0.3)
    
    # plot epsilon decay sensitivity
    if epsilon_decays:
        axes[1].plot(epsilon_decays, final_rewards_ed, "o-", linewidth=2, markersize=8)
        axes[1].set_xlabel("epsilon decay rate", fontsize=11)
        axes[1].set_ylabel("mean evaluation reward", fontsize=11)
        axes[1].set_title("sensitivity to epsilon decay", fontsize=11)
        axes[1].grid(True, alpha=0.3)
    
    # training speed analysis
    axes[2].clear()
    for result in experiments_results:
        training_rewards = result["training_rewards"]
        moving_avg = compute_moving_average(training_rewards, window_size=50)
        axes[2].plot(moving_avg, label=result["name"], linewidth=1.5, alpha=0.8)
    
    axes[2].set_xlabel("training episode", fontsize=11)
    axes[2].set_ylabel("episode reward", fontsize=11)
    axes[2].set_title("convergence speed comparison", fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    # final performance boxplot
    axes[3].clear()
    eval_data = [result["evaluation_rewards"] for result in experiments_results]
    axes[3].boxplot(eval_data)
    axes[3].set_xticklabels([r["name"] for r in experiments_results], rotation=45, ha="right")
    axes[3].set_ylabel("episode reward", fontsize=11)
    axes[3].set_title("final performance distribution", fontsize=11)
    axes[3].grid(True, alpha=0.3, axis="y")
    
    fig.suptitle(
        f"Figure {figure_number}: Hyperparameter Sensitivity Analysis\n"
        f"Experimental Settings: α sensitivity [0.01, 0.1, 0.5], ε-decay [0.98, 0.995, 0.99], γ [0.9, 0.99, 0.999]",
        fontsize=12,
        fontweight="bold",
    )
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else None


def decode_taxi_state(state: int) -> Tuple[int, int, int, int]:
    """
    decode taxi state into position and locations.
    
    taxi-v3 encodes state as:
    ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
    
    args:
        state: encoded state index
    
    returns:
        tuple of (taxi_row, taxi_col, passenger_location, destination)
    """
    destination = state % 4
    state //= 4
    passenger_location = state % 5
    state //= 5
    taxi_col = state % 5
    taxi_row = state // 5
    
    return taxi_row, taxi_col, passenger_location, destination


def plot_q_table_heatmap(
    q_table: np.ndarray,
    figure_number: int,
    save_path: Optional[str] = None,
) -> None:
    """
    plot Q-table as heatmap showing max Q-values across actions.
    Illustrates "craving" concept: high Q-values at states near rewarding outcomes.
    
    args:
        q_table: Q-table numpy array (states x actions)
        figure_number: figure number for labeling
        save_path: optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LARGE, dpi=FIGURE_DPI)
    
    # compute max Q-value for each state
    max_q_values = np.max(q_table, axis=1)
    
    # plot histogram of Q-values
    axes[0].hist(max_q_values, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("max Q-value", fontsize=11)
    axes[0].set_ylabel("number of states", fontsize=11)
    axes[0].set_title("distribution of maximum Q-values per state", fontsize=11)
    axes[0].grid(True, alpha=0.3, axis="y")
    
    # plot heatmap of Q-values for each state
    # reshape to visualize in grid form
    state_indices = np.arange(len(max_q_values))
    scatter = axes[1].scatter(
        state_indices,
        np.ones_like(state_indices),
        c=max_q_values,
        cmap="RdYlGn",
        s=100,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[1].set_xlim(-10, len(max_q_values) + 10)
    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_xlabel("state index", fontsize=11)
    axes[1].set_title("Q-values across all states (craving effect)", fontsize=11)
    axes[1].set_yticks([])
    
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label("max Q-value", fontsize=10)
    
    fig.suptitle(
        f"Figure {figure_number}: Q-Table Analysis - Craving Visualization\n"
        f"Experimental Settings: Best agent (ε-greedy, α=0.1, decay=0.995, γ=0.99)\n"
        f"Interpretation: High Q-values near destinations show agent learned reward anticipation",
        fontsize=12,
        fontweight="bold",
    )
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else None


def plot_policy_grid(
    q_table: np.ndarray,
    figure_number: int,
    save_path: Optional[str] = None,
) -> None:
    """
    plot optimal policy on taxi grid with action arrows.
    
    args:
        q_table: Q-table numpy array
        figure_number: figure number for labeling
        save_path: optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=FIGURE_DPI)
    
    # taxi locations
    locations = {
        0: (0, 0, "Red"),
        1: (0, 4, "Green"),
        2: (4, 0, "Yellow"),
        3: (4, 3, "Blue"),
    }
    location_colors = {0: "red", 1: "green", 2: "yellow", 3: "blue"}
    
    # draw grid
    for i in range(TAXI_GRID_SIZE + 1):
        ax.axhline(i - 0.5, color="black", linewidth=2)
        ax.axvline(i - 0.5, color="black", linewidth=2)
    
    # mark special locations
    for loc_idx, (row, col, name) in locations.items():
        circle = plt.Circle((col, row), 0.15, color=location_colors[loc_idx], alpha=0.7)
        ax.add_patch(circle)
        ax.text(col, row, name[0], ha="center", va="center", fontsize=8, fontweight="bold")
    
    # for each taxi position, show the optimal action
    for taxi_row in range(TAXI_GRID_SIZE):
        for taxi_col in range(TAXI_GRID_SIZE):
            # only visualize for passenger at location 0, destination at 3
            passenger_location = 0
            destination = 3
            
            state = ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
            
            if state < len(q_table):
                # get greedy action
                action = np.argmax(q_table[state, :])
                q_value = np.max(q_table[state, :])
                
                # plot arrow or action indicator
                action_symbol = TAXI_ACTION_ARROWS.get(action, "?")
                color_intensity = min(1.0, max(0.2, q_value / 20))  # normalize for coloring
                
                ax.text(
                    taxi_col,
                    taxi_row,
                    action_symbol,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=plt.cm.RdYlGn(color_intensity),
                )
    
    ax.set_xlim(-1, TAXI_GRID_SIZE)
    ax.set_ylim(-1, TAXI_GRID_SIZE)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(TAXI_GRID_SIZE))
    ax.set_yticks(range(TAXI_GRID_SIZE))
    ax.set_xlabel("column", fontsize=11)
    ax.set_ylabel("row", fontsize=11)
    ax.set_title(
        "optimal policy visualization\n(passenger at Red, destination at Blue)",
        fontsize=12,
    )
    
    # legend
    legend_text = "action symbols: ↑=north, ↓=south, ←=west, →=east, P=pickup, D=dropoff"
    ax.text(
        0.5,
        -0.1,
        legend_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    
    fig.suptitle(
        f"Figure {figure_number}: Optimal Policy Grid Visualization\n"
        f"Experimental Settings: Best agent (ε-greedy, α=0.1, decay=0.995, γ=0.99)\n"
        f"Task: Passenger at Red (0,0) → Destination Blue (4,3)",
        fontsize=12,
        fontweight="bold",
    )
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else None


def plot_exploration_comparison(
    experiments_results: List[Dict],
    figure_number: int,
    save_path: Optional[str] = None,
) -> None:
    """
    compare different exploration strategies.
    
    args:
        experiments_results: list of experiment result dictionaries
        figure_number: figure number for labeling
        save_path: optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LARGE, dpi=FIGURE_DPI)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_results)))
    
    # plot 1: learning curves
    for exp_idx, result in enumerate(experiments_results):
        rewards = result["training_rewards"]
        moving_avg = compute_moving_average(rewards, window_size=50)
        axes[0].plot(
            range(len(moving_avg)),
            moving_avg,
            label=result["name"],
            color=colors[exp_idx],
            linewidth=2,
        )
    
    axes[0].set_xlabel("training episode", fontsize=11)
    axes[0].set_ylabel("episode reward (moving avg)", fontsize=11)
    axes[0].set_title("learning curves comparison", fontsize=11)
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    
    # plot 2: evaluation comparison
    x_positions = np.arange(len(experiments_results))
    mean_rewards = [np.mean(r["evaluation_rewards"]) for r in experiments_results]
    std_rewards = [np.std(r["evaluation_rewards"]) for r in experiments_results]
    
    axes[1].bar(
        x_positions,
        mean_rewards,
        yerr=std_rewards,
        capsize=5,
        color=colors,
        alpha=0.8,
        edgecolor="black",
    )
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels([r["name"] for r in experiments_results], rotation=45, ha="right")
    axes[1].set_ylabel("mean evaluation reward", fontsize=11)
    axes[1].set_title("final performance comparison", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")
    
    # Determine if this is epsilon-greedy or Boltzmann comparison
    if "Boltzmann" in experiments_results[0].get("name", ""):
        caption_suffix = "Boltzmann Temperature Sensitivity (T=0.5, 1.0, 2.0)"
    else:
        caption_suffix = "Epsilon-Greedy Decay Rate Comparison (decay=0.98, 0.995, 0.99)"
    
    fig.suptitle(
        f"Figure {figure_number}: Exploration Strategy Comparison\n"
        f"Experimental Settings: {caption_suffix}\n"
        f"Base parameters: α=0.1, γ=0.99",
        fontsize=12,
        fontweight="bold",
    )
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else None

