"""Composite evaluation-result visualizations."""

import numpy as np
import matplotlib.pyplot as plt

from ..metrics import get_collision_mask
from .trajectory import (plot_2d_projection, plot_3d_trajectory,
                         plot_collision_markers, plot_position_time,
                         plot_speed_comparison)

def plot_combined_trajectories(combined_cases, show_flag=True,
                               show_collision_points=True):
    """
    Plot 3D trajectories from multiple test cases in a single figure (3x3 grid).

    Args:
        combined_cases: List of 9 case data dictionaries, each containing:
            - idx: case index
            - original: original trajectory positions (seq_len, 3)
            - guided: CBF-guided sampled trajectory positions (seq_len, 3)
            - history: history trajectory positions (history_len, 3)
            - target: target waypoint position (3,)
            - obstacles: list of obstacle dictionaries
            - hist_style: history style name
            - pred_style: prediction style name
        show_flag: Whether to display the plot or save to file
    """
    num_cases = len(combined_cases)
    if num_cases == 0:
        print("No cases to plot combined trajectories.")
        return

    # Define grid layout (3x3)
    rows = 3
    cols = 3

    # Create large figure
    fig = plt.figure(figsize=(18, 18))
    fig.suptitle('AeroTrajGen Obstacle-Aware Trajectory Generation - 9 Test Cases (CBF-Guided vs Unguided)',  fontsize=16, fontweight='bold', y=0.98)

    # Color scheme for different trajectory types
    HISTORY_COLOR = 'magenta'
    ORIGINAL_COLOR = 'blue'
    UNGUIDED_COLOR = 'pink'
    GUIDED_COLOR = 'green'
    TARGET_COLOR = 'gold'
    OBSTACLE_COLOR = 'red'

    for idx, case in enumerate(combined_cases):
        row = idx // cols
        col = idx % cols
        subplot_idx = row * cols + col + 1

        ax = fig.add_subplot(rows, cols, subplot_idx, projection='3d')

        # Plot history trajectory if available
        if case['history'] is not None and len(case['history']) > 0:
            ax.plot(case['history'][:, 0], case['history'][:, 1], case['history'][:, 2],
                   color=HISTORY_COLOR, linewidth=2, alpha=0.8, label='History')
            # Mark history end point
            ax.scatter(case['history'][-1, 0], case['history'][-1, 1], case['history'][-1, 2],
                      color=HISTORY_COLOR, s=50, marker='o', edgecolors='black', alpha=0.8)

        # Plot original trajectory
        ax.plot(case['original'][:, 0], case['original'][:, 1], case['original'][:, 2],
               color=ORIGINAL_COLOR, linewidth=2, alpha=0.9, label='Original')

        # Plot unguided sampled trajectory
        ax.plot(case['unguided'][:, 0], case['unguided'][:, 1], case['unguided'][:, 2],
               color=UNGUIDED_COLOR, linewidth=1.5, alpha=0.8, linestyle='-', marker='o', markersize=2, label='Unguided')

        # Plot CBF-guided sampled trajectory
        ax.plot(case['guided'][:, 0], case['guided'][:, 1], case['guided'][:, 2],
               color=GUIDED_COLOR, linewidth=1.5, alpha=0.8, linestyle='-', marker='o', markersize=2, label='CBF Guided')

        # Mark actual collision samples. Each time step is marked at most once,
        # even when the point lies inside multiple overlapping obstacles.
        unguided_collision_mask = get_collision_mask(case['unguided'], case['obstacles'])
        guided_collision_mask = get_collision_mask(case['guided'], case['obstacles'])
        if show_collision_points:
            plot_collision_markers(
                ax, case['unguided'], unguided_collision_mask, (0, 1, 2),
                'crimson', 'X', 'Unguided Collision')
            plot_collision_markers(
                ax, case['guided'], guided_collision_mask, (0, 1, 2),
                'gold', 'P', 'Guided Collision')

        # # Mark start and end points of guided trajectory
        # ax.scatter(case['guided'][0, 0], case['guided'][0, 1], case['guided'][0, 2],
        #           color=GUIDED_COLOR, s=60, marker='^', edgecolors='black', alpha=0.9, label='Start')
        # ax.scatter(case['guided'][-1, 0], case['guided'][-1, 1], case['guided'][-1, 2],
        #           color=GUIDED_COLOR, s=60, marker='s', edgecolors='black', alpha=0.9, label='End')

        # Plot target waypoint
        if case['target'] is not None:
            ax.scatter(case['target'][0], case['target'][1], case['target'][2],
                      color=TARGET_COLOR, s=150, marker='*', edgecolors='black', linewidth=2,
                      label='Target')

        # Plot obstacles
        if case['obstacles']:
            for obstacle in case['obstacles']:
                center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
                radius = obstacle['radius']

                # Create sphere surface for 3D visualization
                u = np.linspace(0, 2 * np.pi, 15)
                v = np.linspace(0, np.pi, 10)
                x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

                ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.25, color=OBSTACLE_COLOR)

        # Set axis labels and title
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)

        # Create title with style information
        title_text = f"Case {case['idx']}"
        if 'hist_style' in case and 'pred_style' in case:
            title_text += f"\n{case['hist_style']} → {case['pred_style']}"
        title_text += (
            f"\nCollision points U/G: "
            f"{unguided_collision_mask.sum()}/{guided_collision_mask.sum()}"
        )
        ax.set_title(title_text, fontsize=11, fontweight='bold')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Set equal aspect ratio for better visualization (approximate)
        # Get bounds from all trajectories in this subplot
        all_points = np.vstack([case['original'], case['guided']])
        if case['history'] is not None:
            all_points = np.vstack([all_points, case['history']])

        center_xyz = np.mean(all_points, axis=0)
        max_range = np.max(all_points.max(axis=0) - all_points.min(axis=0)) / 2.0

        ax.set_xlim(center_xyz[0] - max_range, center_xyz[0] + max_range)
        ax.set_ylim(center_xyz[1] - max_range, center_xyz[1] + max_range)
        ax.set_zlim(center_xyz[2] - max_range, center_xyz[2] + max_range)

        # Add legend only for the first subplot to avoid clutter
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Display or save
    if show_flag:
        plt.show()
    # else:
    filename = "Figs/combined_9_test_cases.svg"
    plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Combined plot saved to {filename}")

def plot_test_results(original, sampled_unguided_denorm, sampled_guided_denorm, history, target,
                      obstacles=None, show_flag=True, step_idx=0,
                      history_style=None, pred_style=None, style_names=None,
                      show_collision_points=True):
    """
    Plot test results including original, reconstructed (unguided), and guided samples.
    Supports 3D, 2D projections, time-series plots, and style information display.

    Args:
        original: Original trajectory (1, seq_len, state_dim)
        sampled_unguided_denorm: Unguided sampled trajectory (1, seq_len, state_dim)
        sampled_guided_denorm: Guided sampled trajectory (1, seq_len, state_dim)
        history: History segment (1, history_len, state_dim)
        target: Target waypoint (1, 3)
        obstacles: List of obstacle dictionaries
        show_flag: Whether to display plot or save to file
        step_idx: Sample index for filename
        history_style: Style index of history segment (int or tensor)
        pred_style: Style index of prediction segment (int or tensor)
        style_names: Dictionary mapping style indices to names
    """
    # Precompute all data at once to avoid repeated operations
    original_pos = original[0, :, 1:4].detach().cpu().numpy()
    reconstructed_pos = sampled_unguided_denorm[0, :, 1:4].detach().cpu().numpy()
    sampled_pos = sampled_guided_denorm[0, :, 1:4].detach().cpu().numpy()

    # Extract speeds
    original_speed = original[0, :, 0].detach().cpu().numpy()
    reconstructed_speed = sampled_unguided_denorm[0, :, 0].detach().cpu().numpy()
    sampled_speed = sampled_guided_denorm[0, :, 0].detach().cpu().numpy()

    time_steps = np.arange(len(original_pos))
    history_pos = history[0, :, 1:4].detach().cpu().numpy() if history is not None else None
    target_pos = target[0, :].detach().cpu().numpy() if target is not None else None

    # Process style information
    history_style_name = "Unknown"
    pred_style_name = "Unknown"
    if style_names is not None:
        if history_style is not None:
            hist_idx = history_style.item() if hasattr(history_style, 'item') else history_style
            history_style_name = style_names.get(hist_idx, f"Style_{hist_idx}")
        if pred_style is not None:
            pred_idx = pred_style.item() if hasattr(pred_style, 'item') else pred_style
            pred_style_name = style_names.get(pred_idx, f"Style_{pred_idx}")

    # Create figure with optimized layout (added one more subplot for style info)
    fig = plt.figure(figsize=(24, 18))

    # Main title with style information
    title_text = f'AeroTrajGen Trajectory Generation Results (Test Sample {step_idx})'
    if history_style is not None or pred_style is not None:
        title_text += f'\nHistory Style: {history_style_name} | Prediction Style: {pred_style_name}'
    unguided_collision_count = get_collision_mask(reconstructed_pos, obstacles).sum()
    guided_collision_count = get_collision_mask(sampled_pos, obstacles).sum()
    title_text += (
        f'\nCollision points — Unguided: {unguided_collision_count} | '
        f'Guided: {guided_collision_count}'
    )
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)

    # Define consistent styling
    STYLES = {
        'history': {'color': 'magenta', 'linewidth': 2, 'alpha': 0.8, 'marker': 'o', 'markersize': 3},
        'original': {'color': 'blue', 'linewidth': 2, 'alpha': 0.9},
        'reconstructed': {'color': 'red', 'linewidth': 1.5, 'alpha': 0.8, 'linestyle': '-.', 'marker': '.'},
        'sampled': {'color': 'green', 'linewidth': 1.5, 'alpha': 0.8, 'linestyle': '-.', 'marker': '.'},
        'target': {'color': 'yellow', 's': 50, 'marker': '*', 'edgecolors': 'black', 'linewidth': 1}
    }

    # 1. 3D trajectory plot
    ax1 = fig.add_subplot(241, projection='3d')
    plot_3d_trajectory(
        ax1, original_pos, reconstructed_pos, sampled_pos, history_pos,
        target_pos, obstacles, STYLES,
        show_collision_points=show_collision_points)

    # 2-4. 2D Projections
    projections = [
        (242, 'X-Y Projection', 0, 1, 'X', 'Y'),
        (243, 'X-Z Projection', 0, 2, 'X', 'Z'),
        (244, 'Y-Z Projection', 1, 2, 'Y', 'Z')
    ]

    for subplot_idx, title, dim1, dim2, xlabel, ylabel in projections:
        ax = fig.add_subplot(subplot_idx)
        plot_2d_projection(ax, original_pos, reconstructed_pos, sampled_pos, history_pos,
                          target_pos, obstacles, STYLES, dim1, dim2, title, xlabel, ylabel,
                          show_collision_points=show_collision_points)

    # 5-7. Position over time
    positions = [
        (245, 'X Position Over Time', 0, 'X Position'),
        (246, 'Y Position Over Time', 1, 'Y Position'),
        (247, 'Z Position Over Time', 2, 'Z Position')
    ]

    for subplot_idx, title, dim, ylabel in positions:
        ax = fig.add_subplot(subplot_idx)
        plot_position_time(ax, time_steps, original_pos, reconstructed_pos, sampled_pos,
                          history_pos, dim, title, ylabel, STYLES)

    # 8. Speed comparison
    ax8 = fig.add_subplot(248)
    plot_speed_comparison(ax8, time_steps, original_speed, reconstructed_speed, sampled_speed, STYLES)

    # # 9. Error analysis
    # ax9 = fig.add_subplot(349)
    # plot_error_analysis(ax9, time_steps, original_pos, reconstructed_pos, sampled_pos)

    # # 10. Style information display
    # ax10 = fig.add_subplot(3, 4, 10)
    # plot_style_information(ax10, history_style_name, pred_style_name, style_names,
    #                        history_pos, original_pos, history_style, pred_style)

    # # 11. Trajectory statistics
    # ax11 = fig.add_subplot(3, 4, 11)
    # plot_trajectory_statistics(ax11, original_pos, reconstructed_pos, sampled_pos)

    # # 12. Style distribution or additional info
    # ax12 = fig.add_subplot(3, 4, 12)
    # plot_connection_info(ax12, history_pos, original_pos, history_len=history_pos.shape[0] if history_pos is not None else 0)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    if show_flag:
        plt.show()
    else:
        filename = f"Figs/test_sample_{step_idx:03d}_results.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
        plt.close()
