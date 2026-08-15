"""Maneuver-style and trajectory-statistics visualizations."""

import numpy as np
import matplotlib.pyplot as plt

def plot_action_comparison(all_trajectories, target, obstacles, history, mean_state, std_state, show_flag, sample_idx=1):
    """Plot comparison of trajectories generated with different actions/styles"""

    num_styles = len(all_trajectories)
    cols = min(4, num_styles)
    rows = (num_styles + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    fig.suptitle(f'Effect of Different Maneuver Styles on Trajectory Generation (Sample {sample_idx})',
                 fontsize=16, fontweight='bold')

    # Color map for different styles
    colors = plt.cm.tab20(np.linspace(0, 1, num_styles))

    for idx, (style_name, trajectory) in enumerate(all_trajectories.items()):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')

        # Plot trajectory
        ax.plot(trajectory[:, 1], trajectory[:, 2], trajectory[:, 3],
                color=colors[idx], linewidth=2, alpha=0.8, label=style_name)

        # Mark start and end points
        ax.scatter(trajectory[0, 1], trajectory[0, 2], trajectory[0, 3],
                  color=colors[idx], s=50, marker='o', edgecolors='black', label='Start')
        ax.scatter(trajectory[-1, 1], trajectory[-1, 2], trajectory[-1, 3],
                  color=colors[idx], s=50, marker='s', edgecolors='black', label='End')

        # Plot obstacles
        if obstacles:
            for obstacle in obstacles:
                center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
                radius = obstacle['radius']

                # Create sphere
                u = np.linspace(0, 2 * np.pi, 10)
                v = np.linspace(0, np.pi, 10)
                x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='red')

        # Plot target
        ax.scatter(target[0], target[1], target[2], color='gold', s=100,
                  marker='*', edgecolors='black', linewidth=2, label='Target')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{style_name}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Only show legend for first subplot to avoid clutter
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    if show_flag:
        plt.show()
    else:
        filename = f"Figs/action_comparison_sample_{sample_idx:03d}.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight')
        plt.close()

def plot_style_statistics(all_trajectories, show_flag, sample_idx=1):
    """Plot statistical comparison of different styles"""

    num_styles = len(all_trajectories)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Style Statistics Comparison (Sample {sample_idx})', fontsize=14, fontweight='bold')

    style_names = list(all_trajectories.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, num_styles))

    # 1. Trajectory length (total distance traveled)
    ax1 = axes[0, 0]
    lengths = []
    for style_name, traj in all_trajectories.items():
        positions = traj[:, 1:4]  # x, y, z
        diffs = np.diff(positions, axis=0)
        total_length = np.sum(np.linalg.norm(diffs, axis=1))
        lengths.append(total_length)

    bars = ax1.bar(style_names, lengths, color=colors)
    ax1.set_ylabel('Total Path Length')
    ax1.set_title('Trajectory Length by Style')
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    for bar, length in zip(bars, lengths):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{length:.1f}',
                ha='center', va='bottom', fontsize=8)

    # 2. Maximum altitude (Z)
    ax2 = axes[0, 1]
    max_altitudes = []
    for style_name, traj in all_trajectories.items():
        max_z = np.max(traj[:, 3])  # Z is index 3 (after speed at index 0, x,y,z at 1,2,3)
        max_altitudes.append(max_z)

    bars = ax2.bar(style_names, max_altitudes, color=colors)
    ax2.set_ylabel('Maximum Altitude (Z)')
    ax2.set_title('Maximum Altitude by Style')
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    for bar, alt in zip(bars, max_altitudes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{alt:.1f}',
                ha='center', va='bottom', fontsize=8)

    # 3. Speed profile (mean speed)
    ax3 = axes[1, 0]
    mean_speeds = []
    for style_name, traj in all_trajectories.items():
        speeds = traj[:, 0]  # Speed is at index 0
        mean_speed = np.mean(speeds)
        mean_speeds.append(mean_speed)

    bars = ax3.bar(style_names, mean_speeds, color=colors)
    ax3.set_ylabel('Mean Speed')
    ax3.set_title('Mean Speed by Style')
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    for bar, speed in zip(bars, mean_speeds):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{speed:.1f}',
                ha='center', va='bottom', fontsize=8)

    # 4. Curvature (how winding the path is)
    ax4 = axes[1, 1]
    curvatures = []
    for style_name, traj in all_trajectories.items():
        positions = traj[:, 1:4]
        if len(positions) >= 3:
            # Approximate curvature using three points
            curv = []
            for i in range(1, len(positions) - 1):
                v1 = positions[i] - positions[i-1]
                v2 = positions[i+1] - positions[i]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                    curv.append(angle)
            if curv:
                curvatures.append(np.mean(curv))
            else:
                curvatures.append(0)
        else:
            curvatures.append(0)

    bars = ax4.bar(style_names, curvatures, color=colors)
    ax4.set_ylabel('Mean Turning Angle (rad)')
    ax4.set_title('Path Curvature by Style')
    ax4.tick_params(axis='x', rotation=45, labelsize=8)

    plt.tight_layout()

    if show_flag:
        plt.show()
    else:
        filename = f"Figs/style_statistics_sample_{sample_idx:03d}.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight')
        plt.close()

def plot_style_information(ax, history_style_name, pred_style_name, style_names,
                           history_pos, original_pos, history_style, pred_style):
    """
    Plot style information for history and prediction segments.

    Args:
        ax: Matplotlib axis
        history_style_name: Name of history style
        pred_style_name: Name of prediction style
        style_names: Dictionary of all style names
        history_pos: History positions array
        original_pos: Original prediction positions array
        history_style: History style index
        pred_style: Prediction style index
    """
    ax.axis('off')

    # Create style information text
    info_lines = []
    info_lines.append("=" * 40)
    info_lines.append("STYLE INFORMATION")
    info_lines.append("=" * 40)
    info_lines.append("")

    # History segment style
    if history_style is not None:
        hist_idx = history_style.item() if hasattr(history_style, 'item') else history_style
        info_lines.append(f"📜 HISTORY SEGMENT STYLE:")
        info_lines.append(f"   Index: {hist_idx}")
        info_lines.append(f"   Name:  {history_style_name}")
        if history_pos is not None:
            info_lines.append(f"   Length: {len(history_pos)} frames")
    else:
        info_lines.append(f"📜 HISTORY SEGMENT: None")

    info_lines.append("")
    info_lines.append("-" * 40)
    info_lines.append("")

    # Prediction segment style
    if pred_style is not None:
        pred_idx = pred_style.item() if hasattr(pred_style, 'item') else pred_style
        info_lines.append(f"🎯 PREDICTION SEGMENT STYLE:")
        info_lines.append(f"   Index: {pred_idx}")
        info_lines.append(f"   Name:  {pred_style_name}")
        info_lines.append(f"   Length: {len(original_pos)} frames")
    else:
        info_lines.append(f"🎯 PREDICTION SEGMENT: Unknown")

    info_lines.append("")
    info_lines.append("=" * 40)

    # Check if styles match
    if history_style is not None and pred_style is not None:
        hist_idx = history_style.item() if hasattr(history_style, 'item') else history_style
        pred_idx = pred_style.item() if hasattr(pred_style, 'item') else pred_style
        if hist_idx == pred_idx:
            info_lines.append("✓ Styles MATCH")
        else:
            info_lines.append("⚠ Styles DIFFERENT")

    # Color code based on style match
    if history_style is not None and pred_style is not None:
        hist_idx = history_style.item() if hasattr(history_style, 'item') else history_style
        pred_idx = pred_style.item() if hasattr(pred_style, 'item') else pred_style
        if hist_idx == pred_idx:
            box_color = 'lightgreen'
        else:
            box_color = 'lightyellow'
    else:
        box_color = 'lightgray'

    # Display text in a box
    info_text = '\n'.join(info_lines)
    ax.text(0.5, 0.5, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))

def plot_trajectory_statistics(ax, original_pos, reconstructed_pos, sampled_pos):
    """
    Plot trajectory statistics comparison.

    Args:
        ax: Matplotlib axis
        original_pos: Original trajectory positions
        reconstructed_pos: Reconstructed (unguided) positions
        sampled_pos: Sampled guided positions
    """
    # Compute statistics
    def compute_stats(pos):
        total_length = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
        max_height = np.max(pos[:, 2])
        mean_speed = np.mean(np.linalg.norm(np.diff(pos, axis=0), axis=1))
        return total_length, max_height, mean_speed

    orig_len, orig_height, orig_speed = compute_stats(original_pos)
    recon_len, recon_height, recon_speed = compute_stats(reconstructed_pos)
    sampled_len, sampled_height, sampled_speed = compute_stats(sampled_pos)

    # Create bar chart
    categories = ['Path Length', 'Max Height', 'Mean Speed']
    orig_values = [orig_len, orig_height, orig_speed]
    recon_values = [recon_len, recon_height, recon_speed]
    sampled_values = [sampled_len, sampled_height, sampled_speed]

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax.bar(x - width, orig_values, width, label='Original', color='blue', alpha=0.7)
    bars2 = ax.bar(x, recon_values, width, label='Unguided', color='red', alpha=0.7)
    bars3 = ax.bar(x + width, sampled_values, width, label='Guided', color='green', alpha=0.7)

    ax.set_ylabel('Value')
    ax.set_title('Trajectory Statistics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

def plot_connection_info(ax, history_pos, original_pos, history_len=0):
    """
    Plot connection information between history and prediction.

    Args:
        ax: Matplotlib axis
        history_pos: History positions array
        original_pos: Original prediction positions array
        history_len: Length of history segment
    """
    ax.axis('off')

    info_lines = []
    info_lines.append("=" * 40)
    info_lines.append("CONNECTION INFORMATION")
    info_lines.append("=" * 40)
    info_lines.append("")

    if history_pos is not None and len(history_pos) > 0 and len(original_pos) > 0:
        # Compute position jump at connection
        history_end = history_pos[-1]
        pred_start = original_pos[0]
        pos_jump = np.linalg.norm(pred_start - history_end)

        # Compute velocity continuity
        if len(history_pos) >= 2:
            history_vel = history_pos[-1] - history_pos[-2]
        else:
            history_vel = np.zeros(3)

        if len(original_pos) >= 2:
            pred_vel = original_pos[1] - original_pos[0]
        else:
            pred_vel = np.zeros(3)

        vel_jump = np.linalg.norm(pred_vel - history_vel)

        info_lines.append(f"History length: {len(history_pos)}")
        info_lines.append(f"Prediction length: {len(original_pos)}")
        info_lines.append("")
        info_lines.append(f"Position jump at connection:")
        info_lines.append(f"  {pos_jump:.4f}")
        info_lines.append("")
        info_lines.append(f"Velocity discontinuity:")
        info_lines.append(f"  {vel_jump:.4f}")

        # Evaluate connection quality
        info_lines.append("")
        if pos_jump < 0.01:
            info_lines.append("✓ Position: EXCELLENT")
        elif pos_jump < 0.1:
            info_lines.append("✓ Position: GOOD")
        else:
            info_lines.append("⚠ Position: POOR")

        if vel_jump < 0.1:
            info_lines.append("✓ Velocity: SMOOTH")
        elif vel_jump < 0.5:
            info_lines.append("✓ Velocity: ACCEPTABLE")
        else:
            info_lines.append("⚠ Velocity: ABRUPT")
    else:
        info_lines.append("No connection information")
        info_lines.append("(Missing history or prediction)")

    info_lines.append("")
    info_lines.append("=" * 40)

    info_text = '\n'.join(info_lines)
    ax.text(0.5, 0.5, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
