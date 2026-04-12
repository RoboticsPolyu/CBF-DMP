import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_circular_end_trajectories(
    num_trajectories=12,
    seq_len=60,
    start_point=None,
    end_circle_radius=10.0,
    end_circle_height=20.0,
    start_similar_length=10,
    smoothness_factor=0.8
):
    """
    Generate a cluster of trajectories with:
    1. Same starting point
    2. Similar first 10 points
    3. End points distributed on a circular ring
    4. Smooth transitions between trajectories
    
    Args:
    - num_trajectories: Number of trajectories
    - seq_len: Number of time steps per trajectory
    - start_point: Starting point coordinates (x, y, z), None for default
    - end_circle_radius: Radius of the circular ring for end points
    - end_circle_height: Height (Z) of the circular ring
    - start_similar_length: How many initial points should be similar
    - smoothness_factor: Smoothness of trajectories (0-1)
    
    Returns:
    - trajectories: Tensor of shape (num_trajectories, seq_len, 11)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set default starting point
    if start_point is None:
        start_point = torch.tensor([0.0, 0.0, 0.0])
    else:
        start_point = torch.tensor(start_point)
    
    # Initialize trajectory tensor
    trajectories = torch.zeros(num_trajectories, seq_len, 8, device=device)
    
    # Generate end points on a circular ring
    angles = torch.linspace(0, 2 * torch.pi, num_trajectories + 1)[:num_trajectories]
    end_points = torch.zeros(num_trajectories, 3, device=device)
    
    # End points evenly distributed on the circular ring
    for i, angle in enumerate(angles):
        end_points[i, 0] = end_circle_radius * torch.cos(angle)  # X
        end_points[i, 1] = end_circle_radius * torch.sin(angle)  # Y
        end_points[i, 2] = end_circle_height  # Z
    
    # Generate control points for each trajectory
    for i in range(num_trajectories):
        # First start_similar_length points are similar (use first trajectory's control points)
        if i == 0:
            # First trajectory: generate control points
            control_points = torch.zeros(4, 3, device=device)
            
            # Control point 1: Starting point
            control_points[0] = start_point
            
            # Control point 2: Near starting point (for smooth start)
            control_points[1] = start_point + torch.tensor([1.0, 0.5, 2.0])
            
            # Control point 3: Midpoint towards end
            mid_point = (start_point + end_points[i]) * 0.5
            control_points[2] = mid_point + torch.tensor([0.0, 0.0, 5.0])  # Slightly elevated
            
            # Control point 4: End point
            control_points[3] = end_points[i]
            
            first_control_points = control_points.clone()
        else:
            # Subsequent trajectories: use first trajectory's first two control points,
            # last two control points transition towards target end
            control_points = torch.zeros(4, 3, device=device)
            control_points[0] = first_control_points[0]  # Same start point
            control_points[1] = first_control_points[1]  # Similar second point
            
            # Middle control point gradually changes
            t = i / (num_trajectories - 1) if num_trajectories > 1 else 0
            control_points[2] = first_control_points[2] * (1 - t * 0.3) + end_points[i] * (t * 0.3)
            
            # End control point
            control_points[3] = end_points[i]
        
        # Generate smooth trajectory using cubic Bezier curve
        t_vals = torch.linspace(0, 1, seq_len, device=device)
        trajectory = torch.zeros(seq_len, 3, device=device)
        
        for j, t in enumerate(t_vals):
            # Cubic Bezier curve formula
            trajectory[j] = (1 - t)**3 * control_points[0] + \
                          3 * (1 - t)**2 * t * control_points[1] + \
                          3 * (1 - t) * t**2 * control_points[2] + \
                          t**3 * control_points[3]
        
        # Add noise to make trajectories more natural (but keep smooth)
        noise = torch.randn(seq_len, 3, device=device) * 0.1 * (1 - smoothness_factor)
        
        # First start_similar_length points have less noise (more similar)
        for k in range(min(start_similar_length, seq_len)):
            noise[k] *= 0.2  # Less noise for initial segment
        
        # trajectory += noise
        
        # Calculate velocity (position differences)
        velocity = torch.zeros(seq_len, 3, device=device)
        velocity[1:] = trajectory[1:] - trajectory[:-1]
        velocity[0] = velocity[1]  # First point velocity same as second
        
        # Calculate speed magnitude
        speed = torch.norm(velocity, dim=1, keepdim=True)
        
        # FIXED: Proper attitude calculation (roll, pitch, yaw approximation)
        # For aviation, we'll use a simplified approach:
        # - Yaw (ψ): direction in XY plane (arctan2(vy, vx))
        # - Pitch (θ): vertical angle (arcsin(vz/speed))
        # - Roll (φ): bank angle, we'll assume small bank proportional to curvature
        
        attitude = torch.zeros(seq_len, 3, device=device)
        
        for k in range(seq_len):
            if speed[k] > 0.01:  # Avoid division by zero
                v = velocity[k]
                spd = speed[k].item()
                
                # Yaw (ψ) - direction in XY plane
                yaw = torch.atan2(v[1], v[0])
                
                # Pitch (θ) - vertical angle
                pitch = torch.asin(v[2] / spd)
                
                # Roll (φ) - simplified: proportional to lateral acceleration
                # For smooth trajectories, use small roll angles
                if k > 0 and k < seq_len - 1:
                    # Estimate curvature from velocity change
                    acc = velocity[k] - velocity[k-1]
                    roll = torch.norm(acc[:2]) * 0.5  # Simplified roll
                else:
                    roll = torch.tensor(0.0, device=device)
                
                # Store attitude: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
                attitude[k, 0] = roll
                attitude[k, 1] = pitch
                attitude[k, 2] = yaw

            else:
                # Zero velocity: use neutral attitude
                attitude[k] = torch.tensor([0.0, 0.0, 0.0], device=device)
        
        # Fill trajectory data
        trajectories[i, :, 0] = speed.squeeze()  # Speed magnitude
        trajectories[i, :, 1:4] = trajectory  # Position (x, y, z)
        trajectories[i, :, 4:7] = attitude  # Attitude (6D)
        
        # Add style label (last dimension)
        trajectories[i, :, -1] = 0  # Cycle through 14 styles
    
    return trajectories.cpu()

def visualize_trajectory_cluster(trajectories, show_3d=True, show_2d=True):
    """
    Visualize the trajectory cluster
    
    Args:
    - trajectories: Trajectory data
    - show_3d: Whether to show 3D plot
    - show_2d: Whether to show 2D projections
    """
    num_trajectories = trajectories.shape[0]
    seq_len = trajectories.shape[1]
    
    # Create color mapping
    colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories))
    
    if show_3d:
        fig = plt.figure(figsize=(18, 12))
        
        # 1. 3D View
        ax1 = fig.add_subplot(331, projection='3d')
        for i in range(num_trajectories):
            traj = trajectories[i, :, 1:4].numpy()
            ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                    color=colors[i], alpha=0.7, linewidth=2)
        
        # Mark start and end points
        start_point = trajectories[0, 0, 1:4].numpy()
        ax1.scatter(start_point[0], start_point[1], start_point[2], 
                   c='red', s=200, marker='o', label='Start', zorder=5)
        
        # Mark end points on circular ring
        for i in range(num_trajectories):
            end_point = trajectories[i, -1, 1:4].numpy()
            ax1.scatter(end_point[0], end_point[1], end_point[2], 
                       c=colors[i], s=3, marker='*', zorder=5)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Trajectory Cluster')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    if show_2d:
        if not show_3d:
            fig = plt.figure(figsize=(15, 5))
            ax2 = plt.subplot(131)
            ax3 = plt.subplot(132)
            ax4 = plt.subplot(133)
        else:
            ax2 = plt.subplot(332)
            ax3 = plt.subplot(333)
            ax4 = plt.subplot(334)
        
        # 2. XY Projection
        for i in range(num_trajectories):
            traj = trajectories[i, :, 1:4].numpy()
            ax2.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.7, linewidth=2)
        
        ax2.scatter(start_point[0], start_point[1], c='red', s=200, marker='o', label='Start', zorder=5)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('XY Projection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # 3. XZ Projection
        for i in range(num_trajectories):
            traj = trajectories[i, :, 1:4].numpy()
            ax3.plot(traj[:, 0], traj[:, 2], color=colors[i], alpha=0.7, linewidth=2)
        
        ax3.scatter(start_point[0], start_point[2], c='red', s=200, marker='o', label='Start', zorder=5)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('XZ Projection')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. YZ Projection
        for i in range(num_trajectories):
            traj = trajectories[i, :, 1:4].numpy()
            ax4.plot(traj[:, 1], traj[:, 2], color=colors[i], alpha=0.7, linewidth=2)
        
        ax4.scatter(start_point[1], start_point[2], c='red', s=200, marker='o', label='Start', zorder=5)
        ax4.set_xlabel('Y')
        ax4.set_ylabel('Z')
        ax4.set_title('YZ Projection')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    if show_3d:
        # 5. Speed profiles
        ax5 = plt.subplot(335)
        for i in range(num_trajectories):
            speed = trajectories[i, :, 0].numpy()
            time_steps = np.arange(seq_len)
            ax5.plot(time_steps, speed, color=colors[i], alpha=0.7, linewidth=2)
        
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Speed')
        ax5.set_title('Speed Profiles')
        ax5.grid(True, alpha=0.3)
        
        # 6. Attitude visualization (roll, pitch, yaw)
        ax6 = plt.subplot(336)
        # Plot attitude for first trajectory as example
        roll = trajectories[0, :, 4].numpy()
        pitch = trajectories[0, :, 5].numpy()
        yaw = trajectories[0, :, 6].numpy()
        
        time_steps = np.arange(seq_len)
        ax6.plot(time_steps, roll, 'r-', label='Roll', alpha=0.8, linewidth=2)
        ax6.plot(time_steps, pitch, 'g-', label='Pitch', alpha=0.8, linewidth=2)
        ax6.plot(time_steps, yaw, 'b-', label='Yaw', alpha=0.8, linewidth=2)
        
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Angle (rad)')
        ax6.set_title('Attitude Angles (First Trajectory)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. First 10 steps (zoomed view)
        ax7 = plt.subplot(337)
        for i in range(num_trajectories):
            traj = trajectories[i, :10, 1:4].numpy()
            ax7.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.7, linewidth=2)
        
        ax7.set_xlabel('X')
        ax7.set_ylabel('Y')
        ax7.set_title('First 10 Steps (Similar Start)')
        ax7.grid(True, alpha=0.3)
        ax7.axis('equal')
        
        # 8. Angular rates
        ax8 = plt.subplot(338)
        roll_rate = trajectories[0, :, 7].numpy()
        pitch_rate = trajectories[0, :, 8].numpy()
        yaw_rate = trajectories[0, :, 9].numpy()
        
        ax8.plot(time_steps, roll_rate, 'r--', label='Roll Rate', alpha=0.8, linewidth=2)
        ax8.plot(time_steps, pitch_rate, 'g--', label='Pitch Rate', alpha=0.8, linewidth=2)
        ax8.plot(time_steps, yaw_rate, 'b--', label='Yaw Rate', alpha=0.8, linewidth=2)
        
        ax8.set_xlabel('Time Step')
        ax8.set_ylabel('Angular Rate (rad/step)')
        ax8.set_title('Angular Rates (First Trajectory)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Start point visualization
        ax9 = plt.subplot(339)
        for i in range(min(num_trajectories, 6)):  # Show first 6 trajectories
            traj = trajectories[i, :start_similar_length, 1:4].numpy()
            ax9.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.7, linewidth=2, label=f'Traj {i+1}')
        
        ax9.scatter(start_point[0], start_point[1], c='red', s=200, marker='o', label='Start', zorder=5)
        ax9.set_xlabel('X')
        ax9.set_ylabel('Y')
        ax9.set_title(f'First {start_similar_length} Steps Comparison')
        ax9.legend(fontsize=8, loc='upper right')
        ax9.grid(True, alpha=0.3)
        ax9.axis('equal')
    
    plt.tight_layout()
    plt.show()

def analyze_trajectory_cluster(trajectories, start_similar_length=10):
    """
    Analyze the trajectory cluster properties
    
    Args:
    - trajectories: Trajectory data
    - start_similar_length: Number of initial similar points
    """
    num_trajectories = trajectories.shape[0]
    seq_len = trajectories.shape[1]
    
    print(f"\n{'='*60}")
    print("TRAJECTORY CLUSTER ANALYSIS")
    print(f"{'='*60}")
    
    # Basic information
    print(f"Number of trajectories: {num_trajectories}")
    print(f"Sequence length: {seq_len}")
    print(f"State dimension: {trajectories.shape[2]}")
    
    # Starting point analysis
    start_points = trajectories[:, 0, 1:4].numpy()
    start_std = np.std(start_points, axis=0)
    print(f"\nStarting point:")
    print(f"  Mean: {np.mean(start_points, axis=0)}")
    print(f"  Standard deviation: {start_std}")
    print(f"  Max position difference: {np.max(np.abs(start_points - start_points[0]), axis=0)}")
    
    # First N points similarity analysis
    print(f"\nFirst {start_similar_length} points similarity:")
    first_n_points = trajectories[:, :start_similar_length, 1:4].numpy()
    
    # Calculate mean position for each trajectory over first N points
    mean_positions = np.mean(first_n_points, axis=1)
    mean_std = np.std(mean_positions, axis=0)
    print(f"  Mean positions std: {mean_std}")
    
    # Calculate pairwise distances between trajectories
    pairwise_distances = []
    for i in range(num_trajectories):
        for j in range(i+1, num_trajectories):
            # Average distance between corresponding points
            dist = np.mean(np.linalg.norm(
                first_n_points[i] - first_n_points[j], axis=1
            ))
            pairwise_distances.append(dist)
    
    print(f"  Average pairwise distance (first {start_similar_length} pts): {np.mean(pairwise_distances):.4f}")
    print(f"  Min/Max pairwise distance: {np.min(pairwise_distances):.4f}/{np.max(pairwise_distances):.4f}")
    
    # End points analysis
    end_points = trajectories[:, -1, 1:4].numpy()
    print(f"\nEnd points (circular ring distribution):")
    print(f"  Radius (mean): {np.mean(np.linalg.norm(end_points[:, :2], axis=1)):.2f}")
    print(f"  Height (mean Z): {np.mean(end_points[:, 2]):.2f}")
    
    # Calculate angular distribution
    angles = np.arctan2(end_points[:, 1], end_points[:, 0])
    print(f"  Angular spread: {np.rad2deg(np.max(angles)-np.min(angles)):.1f}°")
    
    # Speed analysis
    speeds = trajectories[:, :, 0].numpy()
    print(f"\nSpeed statistics:")
    print(f"  Mean speed: {np.mean(speeds):.4f}")
    print(f"  Speed std: {np.std(speeds):.4f}")
    print(f"  Min/Max speed: {np.min(speeds):.4f}/{np.max(speeds):.4f}")
    
    # Attitude analysis
    attitudes = trajectories[:, :, 4:10].numpy()
    print(f"\nAttitude statistics (roll, pitch, yaw):")
    print(f"  Mean: {np.mean(attitudes, axis=(0, 1))[:3]}")
    print(f"  Std: {np.std(attitudes, axis=(0, 1))[:3]}")
    
    # Smoothness analysis
    print(f"\nTrajectory smoothness:")
    for i in range(min(3, num_trajectories)):  # Analyze first 3 trajectories
        traj = trajectories[i, :, 1:4].numpy()
        # Calculate jerk (third derivative approximation)
        acc = traj[2:] - 2*traj[1:-1] + traj[:-2]
        jerk = np.linalg.norm(acc, axis=1).mean()
        print(f"  Trajectory {i+1} average jerk: {jerk:.6f}")
    
    return {
        'start_std': start_std,
        'mean_pairwise_distance': np.mean(pairwise_distances),
        'end_radius': np.mean(np.linalg.norm(end_points[:, :2], axis=1)),
        'mean_speed': np.mean(speeds)
    }

# Test the generation function
if __name__ == "__main__":
    print("Generating trajectory cluster...")
    
    # Generate trajectories
    start_similar_length = 10
    trajectories = generate_circular_end_trajectories(
        num_trajectories=120,
        seq_len=60,
        start_point=[0, 0, 0],
        end_circle_radius=15.0,
        end_circle_height=25.0,
        start_similar_length=start_similar_length,
        smoothness_factor=0.85
    )
    
    print(f"Generated trajectory shape: {trajectories.shape}")
    
    # Analyze the cluster
    analysis_results = analyze_trajectory_cluster(trajectories, start_similar_length)
    
    # Visualize
    visualize_trajectory_cluster(trajectories, show_3d=True, show_2d=True)