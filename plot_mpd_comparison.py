import numpy as np
import matplotlib.pyplot as plt


def plot_mpd_comparison(original, standard_sample, mpd_sample, history, target, obstacles, show_flag, step_idx):
    """Plot comparison between standard sampling and MPD planning with actual metrics"""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'MPD Planning vs Standard Sampling (Test {step_idx})', fontsize=16, fontweight='bold')
    
    # Extract positions
    original_pos = original[0, :, 1:4].detach().cpu().numpy()
    standard_pos = standard_sample[0, :, 1:4].detach().cpu().numpy()
    mpd_pos = mpd_sample[0, :, 1:4].detach().cpu().numpy()
    target_pos = target[0].detach().cpu().numpy() if target is not None else None
    
    # Calculate actual metrics
    # 1. Collision avoidance metrics
    min_distances_standard = calculate_min_obstacle_distance(standard_pos, obstacles)
    min_distances_mpd = calculate_min_obstacle_distance(mpd_pos, obstacles)
    
    # 2. Goal achievement metrics
    goal_error_standard = calculate_goal_error(standard_pos, target_pos)
    goal_error_mpd = calculate_goal_error(mpd_pos, target_pos)
    
    # 3. Trajectory smoothness metrics
    smoothness_standard = calculate_trajectory_smoothness(standard_pos)
    smoothness_mpd = calculate_trajectory_smoothness(mpd_pos)
    
    # 4. Collision avoidance performance (0-1 scale, higher is better)
    collision_avoidance_standard = calculate_collision_avoidance_score(min_distances_standard, obstacles)
    collision_avoidance_mpd = calculate_collision_avoidance_score(min_distances_mpd, obstacles)
    
    # 5. Goal achievement performance (0-1 scale, higher is better)
    goal_achievement_standard = calculate_goal_achievement_score(goal_error_standard)
    goal_achievement_mpd = calculate_goal_achievement_score(goal_error_mpd)
    
    # 1. 3D trajectory comparison
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], 
             'b-', label='Original', linewidth=3, alpha=0.7)
    ax1.plot(standard_pos[:, 0], standard_pos[:, 1], standard_pos[:, 2], 
             'r--', label='Standard Sampling', linewidth=2, alpha=0.8)
    ax1.plot(mpd_pos[:, 0], mpd_pos[:, 1], mpd_pos[:, 2], 
             'g-.', label='MPD Planning', linewidth=2, alpha=0.8)
    
    # Plot target
    if target_pos is not None:
        ax1.scatter(target_pos[0], target_pos[1], target_pos[2], 
                   color='yellow', s=200, marker='*', edgecolors='black', 
                   linewidth=2, label='Target', zorder=10)
    
    # Plot obstacles
    for obstacle in obstacles:
        center = obstacle['center'].cpu().numpy()
        radius = obstacle['radius']
        
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='red')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('3D Trajectory Comparison')
    ax1.grid(True)
    
    # 2. Cost analysis
    ax2 = fig.add_subplot(232)
    costs = ['Standard', 'MPD Planning']
    
    x = np.arange(len(costs))
    width = 0.35
    
    ax2.bar(x - width/2, [collision_avoidance_standard, collision_avoidance_mpd], 
            width, label='Collision Avoidance', alpha=0.7, color='orange')
    ax2.bar(x + width/2, [goal_achievement_standard, goal_achievement_mpd], 
            width, label='Goal Achievement', alpha=0.7, color='blue')
    
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Performance Score (0-1)')
    ax2.set_title('Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(costs)
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Position error over time
    ax3 = fig.add_subplot(233)
    time_steps = np.arange(len(original_pos))
    standard_error = np.linalg.norm(standard_pos - original_pos, axis=1)
    mpd_error = np.linalg.norm(mpd_pos - original_pos, axis=1)
    
    ax3.plot(time_steps, standard_error, 'r--', label='Standard Error', linewidth=2)
    ax3.plot(time_steps, mpd_error, 'g-.', label='MPD Error', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('L2 Position Error')
    ax3.set_title('Trajectory Error Comparison')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Final position accuracy
    ax4 = fig.add_subplot(234)
    final_errors = [goal_error_standard, goal_error_mpd]
    
    bars = ax4.bar(costs, final_errors, color=['red', 'green'], alpha=0.7)
    ax4.set_ylabel('Final Position Error')
    ax4.set_title('Goal Achievement Accuracy')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, error in zip(bars, final_errors):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.3f}', ha='center', va='bottom')
    
    # 5. Obstacle clearance
    ax5 = fig.add_subplot(235)
    min_distances = [min_distances_standard, min_distances_mpd]
    
    bars = ax5.bar(costs, min_distances, color=['red', 'green'], alpha=0.7)
    
    # Calculate safety threshold (average obstacle radius)
    avg_radius = np.mean([obs['radius'] for obs in obstacles]) if obstacles else 1.0
    safety_threshold = avg_radius * 1.2  # 20% safety margin
    
    ax5.axhline(y=safety_threshold, color='orange', linestyle='--', 
                label=f'Safety Threshold ({safety_threshold:.2f})', linewidth=2)
    ax5.set_ylabel('Minimum Obstacle Distance')
    ax5.set_title('Collision Avoidance Performance')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, distance in zip(bars, min_distances):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{distance:.3f}', ha='center', va='bottom')
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    
    summary_text = (
        f"MPD Planning Summary:\n\n"
        f"• Obstacles: {len(obstacles)}\n"
        f"• Goal Error Reduction: {((goal_error_standard-goal_error_mpd)/(goal_error_standard*100+1e-6)):.1f}%\n"
        f"• Safety Improvement: {((min_distances_mpd-min_distances_standard)/(min_distances_standard*100+1e-6)):.1f}%\n"
        f"• Min Distance Standard: {min_distances_standard:.3f}\n"
        f"• Min Distance MPD: {min_distances_mpd:.3f}\n"
        f"• Goal Error Standard: {goal_error_standard:.3f}\n"
        f"• Goal Error MPD: {goal_error_mpd:.3f}"
    )
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    if show_flag:
        plt.show()
    else:
        filename = f"Figs/mpd_planning_test_{step_idx:03d}.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight')
        plt.close()

# Helper functions for metric calculations
def calculate_min_obstacle_distance(trajectory_pos, obstacles):
    """Calculate minimum distance to any obstacle along the trajectory"""
    if not obstacles:
        return float('inf')
    
    min_distance = float('inf')
    for obstacle in obstacles:
        center = obstacle['center'].cpu().numpy()
        radius = obstacle['radius']
        
        # Calculate distances from trajectory points to obstacle center
        distances = np.linalg.norm(trajectory_pos - center, axis=1)
        # Distance to obstacle surface (positive outside, negative inside)
        surface_distances = distances - radius
        
        # Find minimum surface distance (most critical point)
        trajectory_min_distance = np.min(surface_distances)
        min_distance = min(min_distance, trajectory_min_distance)
    
    return max(min_distance, 0)  # Ensure non-negative

def calculate_goal_error(trajectory_pos, target_pos):
    """Calculate error between final position and target"""
    if target_pos is None:
        return float('inf')
    
    final_pos = trajectory_pos[-1]
    return np.linalg.norm(final_pos - target_pos)

def calculate_trajectory_smoothness(trajectory_pos):
    """Calculate trajectory smoothness (lower is smoother)"""
    if len(trajectory_pos) <= 1:
        return 0.0
    
    # Use acceleration magnitude as smoothness metric
    velocities = np.diff(trajectory_pos, axis=0)
    accelerations = np.diff(velocities, axis=0)
    
    if len(accelerations) == 0:
        return 0.0
    
    return np.mean(np.linalg.norm(accelerations, axis=1))

def calculate_collision_avoidance_score(min_distance, obstacles, safety_factor=1.2):
    """Calculate collision avoidance score (0-1, higher is better)"""
    if not obstacles:
        return 1.0
    
    # Calculate average obstacle radius
    avg_radius = np.mean([obs['radius'] for obs in obstacles])
    safety_threshold = avg_radius * safety_factor
    
    if min_distance >= safety_threshold:
        return 1.0
    elif min_distance <= 0:
        return 0.0
    else:
        # Linear interpolation between 0 and safety_threshold
        return min_distance / safety_threshold

def calculate_goal_achievement_score(goal_error, tolerance=0.5):
    """Calculate goal achievement score (0-1, higher is better)"""
    if goal_error <= tolerance:
        return 1.0
    else:
        # Exponential decay beyond tolerance
        return np.exp(-goal_error / tolerance)