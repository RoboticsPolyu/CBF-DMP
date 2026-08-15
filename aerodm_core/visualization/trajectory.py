"""Low-level trajectory and obstacle plotting primitives."""

import numpy as np
import matplotlib.pyplot as plt

from ..metrics import get_collision_mask

def plot_trajectories_demo(demo_trajectories, rows=3, cols=6):
    """Utility to plot a grid of 3D trajectories for visualization."""
    num_trajectories = demo_trajectories.shape[0]
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(f'Sample of {num_trajectories} Generated Aerobatic Trajectories (3D View)', fontsize=20, fontweight='bold')

    for i in range(min(rows * cols, num_trajectories)):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        # Trajectory is [speed, x, y, z, attitude(6)]. Position is [1:4]
        trajectory = demo_trajectories[i, :, 1:4].numpy()

        # Custom coloring based on index for variation
        if i % 3 == 0:
            color = 'blue'
            marker_color = 'red'
        elif i % 3 == 1:
            color = 'green'
            marker_color = 'orange'
        else:
            color = 'purple'
            marker_color = 'cyan'

        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color, linewidth=2.5, alpha=0.8)
        # Mark every 10th point
        ax.scatter(trajectory[::10, 0], trajectory[::10, 1], trajectory[::10, 2],
                color=marker_color, s=20, alpha=0.6, marker='o')

        ax.set_title(f'Trajectory {i+1}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('X', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Remove fill for better 3D visualization
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to accommodate title and text
    plt.show()

def plot_collision_markers(ax, trajectory_pos, collision_mask, dimensions,
                           color, marker, label):
    """Mark collision samples on either a 2D or 3D trajectory axis."""
    collision_points = trajectory_pos[collision_mask]
    if len(collision_points) == 0:
        return

    scatter_kwargs = {
        'color': color,
        's': 10,
        'marker': marker,
        'edgecolors': 'pink',
        'linewidths': 0.3,
        'zorder': 0,
        'label': label,
    }
    if len(dimensions) == 3:
        ax.scatter(collision_points[:, dimensions[0]],
                   collision_points[:, dimensions[1]],
                   collision_points[:, dimensions[2]],
                   depthshade=False, **scatter_kwargs)
    else:
        ax.scatter(collision_points[:, dimensions[0]],
                   collision_points[:, dimensions[1]], **scatter_kwargs)

def plot_3d_trajectory(ax, original_pos, reconstructed_pos, sampled_pos, history_pos,
                       target_pos, obstacles, styles, bounds=None,
                       show_collision_points=True):
    """Plot 3D trajectory with obstacles"""
        # Plot trajectories
    if history_pos is not None:
        ax.plot(history_pos[:, 0], history_pos[:, 1], history_pos[:, 2],
                label='His', **styles['history'])
    ax.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2],
            label='Orin', **styles['original'])
    ax.plot(reconstructed_pos[:, 0], reconstructed_pos[:, 1], reconstructed_pos[:, 2],
            label='Unguided', **styles['reconstructed'])
    ax.plot(sampled_pos[:, 0], sampled_pos[:, 1], sampled_pos[:, 2],
            label='Guided', **styles['sampled'])

    if show_collision_points:
        unguided_collision_mask = get_collision_mask(reconstructed_pos, obstacles)
        guided_collision_mask = get_collision_mask(sampled_pos, obstacles)
        plot_collision_markers(
            ax, reconstructed_pos, unguided_collision_mask, (0, 1, 2),
            'crimson', 'X', 'Unguided Collision')
        plot_collision_markers(
            ax, sampled_pos, guided_collision_mask, (0, 1, 2),
            'gold', 'P', 'Guided Collision')

    # Plot target
    if target_pos is not None:
        ax.scatter(target_pos[0], target_pos[1], target_pos[2],
                  label='Tar', **styles['target'])

    # Plot obstacles
    obstacle_proxies = plot_3d_obstacles(ax, obstacles)

    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    if obstacle_proxies:
        handles.extend(obstacle_proxies)
        labels.extend([p.get_label() for p in obstacle_proxies])
    ax.legend(handles, labels, loc='upper right', fontsize=8)
    if bounds is not None:
        ax.set_xlim(bounds['xlim'])
        ax.set_ylim(bounds['ylim'])
        ax.set_zlim(bounds['zlim'])
    else:
        # Get bounds from all trajectories
        all_points = [original_pos]
        if reconstructed_pos is not None:
            all_points.append(reconstructed_pos)
        if sampled_pos is not None:
            all_points.append(sampled_pos)
        if history_pos is not None:
            all_points.append(history_pos)

        # Stack all points
        all_points_stacked = np.vstack(all_points)

        # Compute center and range
        center_xyz = np.mean(all_points_stacked, axis=0)
        max_range = np.max(all_points_stacked.max(axis=0) - all_points_stacked.min(axis=0)) / 2.0

        # Set equal aspect ratio bounds
        ax.set_xlim(center_xyz[0] - max_range, center_xyz[0] + max_range)
        ax.set_ylim(center_xyz[1] - max_range, center_xyz[1] + max_range)
        ax.set_zlim(center_xyz[2] - max_range, center_xyz[2] + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory')
    ax.grid(True, alpha=0.3)

def plot_3d_obstacles(ax, obstacles):
    """Plot 3D obstacles and return legend proxies"""
    if not obstacles:
        return []

    obstacle_proxies = []
    colors = plt.cm.Set3(np.linspace(0, 1, len(obstacles)))

    for i, obstacle in enumerate(obstacles):
        center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
        radius = obstacle['radius']

        # Create sphere
        u = np.linspace(0, 2 * np.pi, 12)  # Reduced resolution for performance
        v = np.linspace(0, np.pi, 8)
        obs_x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        obs_y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        obs_z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(obs_x, obs_y, obs_z, alpha=0.3, color=colors[i])

        if i == 0:
            from matplotlib.patches import Patch
            obstacle_proxies.append(Patch(color=colors[i], alpha=0.5, label='Obstacles'))

    return obstacle_proxies

def plot_2d_projection(ax, original_pos, reconstructed_pos, sampled_pos, history_pos,
                      target_pos, obstacles, styles, dim1, dim2, title, xlabel, ylabel,
                      show_collision_points=True):
    """Plot 2D projection with obstacles"""
    # Plot trajectories
    if history_pos is not None:
        ax.plot(history_pos[:, dim1], history_pos[:, dim2],
                label='History', **styles['history'])

    ax.plot(original_pos[:, dim1], original_pos[:, dim2],
            label='Original', **styles['original'])
    ax.plot(reconstructed_pos[:, dim1], reconstructed_pos[:, dim2],
            label='Reconstructed', **styles['reconstructed'])
    ax.plot(sampled_pos[:, dim1], sampled_pos[:, dim2],
            label='Sampled Guided', **styles['sampled'])

    if show_collision_points:
        unguided_collision_mask = get_collision_mask(reconstructed_pos, obstacles)
        guided_collision_mask = get_collision_mask(sampled_pos, obstacles)
        plot_collision_markers(
            ax, reconstructed_pos, unguided_collision_mask, (dim1, dim2),
            'crimson', 'X', 'Unguided Collision')
        plot_collision_markers(
            ax, sampled_pos, guided_collision_mask, (dim1, dim2),
            'gold', 'P', 'Guided Collision')

    # Plot target
    if target_pos is not None:
        ax.scatter(target_pos[dim1], target_pos[dim2],
                  label='Target', **styles['target'])

    # Plot obstacles
    plot_2d_obstacles(ax, obstacles, dim1, dim2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

def plot_2d_obstacles(ax, obstacles, dim1, dim2):
    """Plot 2D obstacles"""
    if not obstacles:
        return

    colors = plt.cm.Set3(np.linspace(0, 1, len(obstacles)))

    for i, obstacle in enumerate(obstacles):
        center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
        radius = obstacle['radius']

        circle = plt.Circle((center[dim1], center[dim2]), radius,
                          color=colors[i], alpha=0.4,
                          label=f'Obstacle {i+1}' if i < 3 else "")
        ax.add_patch(circle)

def plot_position_time(ax, time_steps, original_pos, reconstructed_pos, sampled_pos,
                       history_pos, dim, title, ylabel, styles):
    """Plot position over time for a specific dimension, including history."""
    # Plot history if available
    if history_pos is not None:
        history_time = np.arange(-len(history_pos), 0)
        ax.plot(history_time, history_pos[:, dim],
                label='History', **{k: v for k, v in styles['history'].items() if k != 'marker'})

    ax.plot(time_steps, original_pos[:, dim],
            label=f'Original', **{k: v for k, v in styles['original'].items() if k != 'marker'})
    ax.plot(time_steps, reconstructed_pos[:, dim],
            label=f'Reconstructed', **{k: v for k, v in styles['reconstructed'].items() if k != 'marker'})
    ax.plot(time_steps, sampled_pos[:, dim],
            label=f'Sampled Guided', **{k: v for k, v in styles['sampled'].items() if k != 'marker'})

    # Add vertical line at connection point
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Connection')

    ax.set_xlabel('Time Step')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

def plot_speed_comparison(ax, time_steps, original_speed, reconstructed_speed, sampled_speed, styles):
    """Plot speed comparison over time"""
    ax.plot(time_steps, original_speed,
            label='Original Speed', **{k: v for k, v in styles['original'].items() if k != 'marker'})
    ax.plot(time_steps, reconstructed_speed,
            label='Reconstructed Speed', **{k: v for k, v in styles['reconstructed'].items() if k != 'marker'})
    ax.plot(time_steps, sampled_speed,
            label='Sampled Guided Speed', **{k: v for k, v in styles['sampled'].items() if k != 'marker'})

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Speed')
    ax.legend(fontsize=8)
    ax.set_title('Speed Over Time')
    ax.grid(True, alpha=0.3)

def plot_error_analysis(ax, time_steps, original_pos, reconstructed_pos, sampled_pos):
    """Plot error analysis"""
    recon_error = np.linalg.norm(reconstructed_pos - original_pos, axis=1)
    sampled_error = np.linalg.norm(sampled_pos - original_pos, axis=1)

    ax.plot(time_steps, recon_error, 'r--', label='Unguided Error', linewidth=2, alpha=0.8)
    ax.plot(time_steps, sampled_error, 'g-.', label='Guided Error', linewidth=2, alpha=0.8)

    # Add mean lines with annotations
    mean_recon = np.mean(recon_error)
    mean_sampled = np.mean(sampled_error)

    ax.axhline(mean_recon, color='r', linestyle=':', alpha=0.7,
               label=f'Mean Unguided ({mean_recon:.2f})')
    ax.axhline(mean_sampled, color='g', linestyle=':', alpha=0.7,
               label=f'Mean Guided ({mean_sampled:.2f})')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('L2 Position Error')
    ax.legend(fontsize=8)
    ax.set_title('L2 Position Error w.r.t Original')
    ax.grid(True, alpha=0.3)
