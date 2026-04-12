
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_distributed_trajectories(
    num_trajectories=12,
    seq_len=60,
    start_point=None,
    end_point=None,
    max_deviation_radius=15.0,
    start_similar_length=10,
    end_similar_length=10,
    smoothness_factor=0.8
):
    """
    Generate a cluster of trajectories with:
    1. Same starting point
    2. Same ending point
    3. Distributed in the middle with different paths
    4. Smooth transitions between trajectories
    
    Args:
    - num_trajectories: Number of trajectories
    - seq_len: Number of time steps per trajectory
    - start_point: Starting point coordinates (x, y, z), None for default
    - end_point: Ending point coordinates (x, y, z), None for default
    - max_deviation_radius: Maximum deviation radius from direct path
    - start_similar_length: How many initial points should be similar
    - end_similar_length: How many final points should be similar
    - smoothness_factor: Smoothness of trajectories (0-1)
    
    Returns:
    - trajectories: Tensor of shape (num_trajectories, seq_len, 11)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set default starting and ending points with float type
    if start_point is None:
        start_point = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    else:
        start_point = torch.tensor(start_point, dtype=torch.float32)
    
    if end_point is None:
        end_point = torch.tensor([30.0, 0.0, 15.0], dtype=torch.float32)
    else:
        end_point = torch.tensor(end_point, dtype=torch.float32)
    
    # Move tensors to device
    start_point = start_point.to(device)
    end_point = end_point.to(device)
    
    # Initialize trajectory tensor
    trajectories = torch.zeros(num_trajectories, seq_len, 11, device=device, dtype=torch.float32)
    
    # Generate mid control points with different deviations
    angles = torch.linspace(0, 2 * torch.pi, num_trajectories + 1, device=device, dtype=torch.float32)[:num_trajectories]
    deviations = torch.linspace(0.3, 1.0, num_trajectories, device=device, dtype=torch.float32)  # Different deviation magnitudes
    
    # Store trajectories for debugging
    all_trajectories = []
    
    # Generate control points for each trajectory
    for i in range(num_trajectories):
        # Calculate midpoint of the direct path
        direct_midpoint = (start_point + end_point) * 0.5
        
        # Calculate perpendicular direction to the direct path
        direct_vector = end_point - start_point
        direct_length = torch.norm(direct_vector)
        
        if direct_length > 0.001:
            # Normalize the direct vector
            direct_normalized = direct_vector / direct_length
            
            # Find a perpendicular vector (cross product with up vector)
            up_vector = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
            perp_vector = torch.cross(direct_normalized, up_vector)
            
            # If the cross product is zero, use another perpendicular vector
            if torch.norm(perp_vector) < 0.001:
                perp_vector = torch.tensor([-direct_normalized[1], direct_normalized[0], 0.0], 
                                          device=device, dtype=torch.float32)
            
            # Normalize the perpendicular vector
            perp_normalized = perp_vector / torch.norm(perp_vector)
            
            # Create rotation matrix around the direct axis
            angle = angles[i]
            
            # Generate deviation in the perpendicular plane
            deviation_direction = torch.cos(angle) * perp_normalized
            
            # Add some vertical deviation as well
            vertical_scale = torch.sin(angle * 2)  # Different pattern for Z
            deviation = max_deviation_radius * deviations[i] * deviation_direction
            
            # Calculate the mid control point with deviation
            mid_control = direct_midpoint + deviation
            
            # Add vertical variation based on deviation pattern
            mid_control[2] += max_deviation_radius * 0.3 * vertical_scale
            
            # Generate control points for quintic Bezier curve
            # IMPORTANT: Control point 0 MUST be exactly start_point for all trajectories
            control_points = torch.zeros(5, 3, device=device, dtype=torch.float32)
            
            # Control point 1: Starting point (EXACTLY THE SAME FOR ALL)
            control_points[0] = start_point
            
            # Control point 2: Near starting point (25% of the way with small deviation)
            t1 = 0.25
            point1 = start_point * (1 - t1) + end_point * t1
            deviation1 = deviation * 0.3 * torch.sin(angle * 0.5)
            control_points[1] = point1 + deviation1
            
            # Control point 3: Main deviation point (50% of the way)
            control_points[2] = mid_control
            
            # Control point 4: Near ending point (75% of the way with small deviation)
            t2 = 0.75
            point2 = start_point * (1 - t2) + end_point * t2
            deviation2 = -deviation * 0.3 * torch.sin(angle * 0.5)  # Opposite phase
            control_points[3] = point2 + deviation2
            
            # Control point 5: Ending point (EXACTLY THE SAME FOR ALL)
            control_points[4] = end_point
            
        else:
            # If start and end are too close, use simpler control points
            control_points = torch.stack([
                start_point,
                start_point + torch.tensor([1.0, 0.0, 2.0], device=device, dtype=torch.float32),
                (start_point + end_point) * 0.5 + torch.tensor([0.0, 0.0, 5.0], device=device, dtype=torch.float32),
                end_point + torch.tensor([-1.0, 0.0, 2.0], device=device, dtype=torch.float32),
                end_point
            ])
        
        # Generate smooth trajectory using quintic Bezier curve
        t_vals = torch.linspace(0, 1, seq_len, device=device, dtype=torch.float32)
        trajectory = torch.zeros(seq_len, 3, device=device, dtype=torch.float32)
        
        # Quintic Bezier curve formula (for 5 control points)
        # At t=0: b0=1, others=0 -> exactly control_points[0] (start_point)
        # At t=1: b4=1, others=0 -> exactly control_points[4] (end_point)
        for j, t in enumerate(t_vals):
            # Basis functions for quintic Bezier
            omt = 1 - t
            b0 = omt**4
            b1 = 4 * t * omt**3
            b2 = 6 * t**2 * omt**2
            b3 = 4 * t**3 * omt
            b4 = t**4
            
            trajectory[j] = (b0 * control_points[0] +
                           b1 * control_points[1] +
                           b2 * control_points[2] +
                           b3 * control_points[3] +
                           b4 * control_points[4])
        
        # Add small noise to make trajectories more natural
        noise = torch.randn(seq_len, 3, device=device, dtype=torch.float32) * 0.05 * (1 - smoothness_factor)
        
        # CRITICAL FIX 1: Ensure first point has NO noise
        noise[0] = torch.zeros(3, device=device, dtype=torch.float32)
        
        # First start_similar_length points have less noise (more similar)
        for k in range(1, min(start_similar_length, seq_len)):
            noise[k] *= 0.1
        
        # Last end_similar_length points have less noise (more similar)
        for k in range(max(0, seq_len - end_similar_length), seq_len):
            noise[k] *= 0.1
        
        # CRITICAL FIX 2: Ensure last point has minimal noise
        noise[-1] *= 0.01
        
        trajectory += noise
        
        # CRITICAL FIX 3: Force exact start and end points
        trajectory[0] = start_point
        trajectory[-1] = end_point
        
        all_trajectories.append(trajectory.clone())
        
        # Calculate velocity (position differences)
        velocity = torch.zeros(seq_len, 3, device=device, dtype=torch.float32)
        velocity[1:] = trajectory[1:] - trajectory[:-1]
        
        # CRITICAL FIX 4: First point velocity should be calculated from the exact trajectory
        # Instead of setting it equal to the second point's velocity
        if seq_len > 1:
            velocity[0] = velocity[1]  # Keep this or set to zero?
        else:
            velocity[0] = torch.zeros(3, device=device, dtype=torch.float32)
        
        # Calculate speed magnitude
        speed = torch.norm(velocity, dim=1, keepdim=True)
        
        # Calculate attitude (roll, pitch, yaw)
        attitude = torch.zeros(seq_len, 6, device=device, dtype=torch.float32)
        
        for k in range(seq_len):
            if speed[k] > 0.01:  # Avoid division by zero
                v = velocity[k]
                spd = speed[k].item()
                
                # Yaw (ψ) - direction in XY plane
                yaw = torch.atan2(v[1], v[0])
                
                # Pitch (θ) - vertical angle
                pitch = torch.asin(torch.clamp(v[2] / spd, -0.99, 0.99))
                
                # Roll (φ) - simplified based on lateral acceleration
                if k > 0 and k < seq_len - 1:
                    # Estimate curvature from velocity change
                    acc = velocity[k] - velocity[k-1]
                    lateral_acc = torch.norm(acc[:2])
                    roll = torch.tanh(lateral_acc * 2.0) * 0.3  # Limited roll angle
                else:
                    roll = torch.tensor(0.0, device=device, dtype=torch.float32)
                
                # Store attitude: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
                attitude[k, 0] = roll
                attitude[k, 1] = pitch
                attitude[k, 2] = yaw
                
                # Angular rates (simplified as finite differences)
                if k > 0:
                    attitude[k, 3] = attitude[k, 0] - attitude[k-1, 0]  # roll rate
                    attitude[k, 4] = attitude[k, 1] - attitude[k-1, 1]  # pitch rate
                    attitude[k, 5] = attitude[k, 2] - attitude[k-1, 2]  # yaw rate
                else:
                    attitude[k, 3:] = torch.zeros(3, device=device, dtype=torch.float32)
            else:
                # Zero velocity: use neutral attitude
                attitude[k] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        
        # Fill trajectory data
        trajectories[i, :, 0] = speed.squeeze()  # Speed magnitude
        trajectories[i, :, 1:4] = trajectory  # Position (x, y, z)
        trajectories[i, :, 4:10] = attitude  # Attitude (6D)
        
        # Add style label (last dimension)
        trajectories[i, :, -1] = i % 14  # Cycle through 14 styles
    
    # DEBUG: Verify first points are identical
    first_points = trajectories[:, 0, 1:4]
    max_diff = torch.max(torch.abs(first_points[0] - first_points)).item()
    print(f"DEBUG: Maximum difference between first points: {max_diff:.10f}")
    if max_diff > 1e-6:
        print("WARNING: First points are not exactly identical!")
        print(f"First points sample:\n{first_points[:5]}")
    
    # DEBUG: Verify last points are identical
    last_points = trajectories[:, -1, 1:4]
    max_diff_last = torch.max(torch.abs(last_points[0] - last_points)).item()
    print(f"DEBUG: Maximum difference between last points: {max_diff_last:.10f}")
    
    return trajectories.cpu()

def visualize_distributed_trajectories(trajectories, show_3d=True, show_2d=True):
    """
    Visualize the distributed trajectories with same start and end
    
    Args:
    - trajectories: Trajectory data
    - show_3d: Whether to show 3D plot
    - show_2d: Whether to show 2D projections
    """
    num_trajectories = trajectories.shape[0]
    seq_len = trajectories.shape[1]
    
    # Create color mapping
    colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories))
    
    # Get start and end points
    start_point = trajectories[0, 0, 1:4].numpy()
    end_point = trajectories[0, -1, 1:4].numpy()
    
    # DEBUG: Check first points
    first_points = trajectories[:, 0, 1:4].numpy()
    print(f"\nVISUALIZATION DEBUG:")
    print(f"Start point mean: {np.mean(first_points, axis=0)}")
    print(f"Start point std: {np.std(first_points, axis=0)}")
    print(f"Max difference from mean: {np.max(np.abs(first_points - np.mean(first_points, axis=0)), axis=0)}")
    
    if show_3d:
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 3D View with enlarged first point markers
        ax1 = fig.add_subplot(331, projection='3d')
        for i in range(num_trajectories):
            traj = trajectories[i, :, 1:4].numpy()
            ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                    color=colors[i], alpha=0.7, linewidth=1.5)
        
        # Mark start and end points (make start point very visible)
        ax1.scatter(start_point[0], start_point[1], start_point[2], 
                   c='red', s=300, marker='o', label='Start', zorder=10, edgecolors='black', linewidth=2)
        ax1.scatter(end_point[0], end_point[1], end_point[2], 
                   c='green', s=300, marker='s', label='End', zorder=10, edgecolors='black', linewidth=2)
        
        # Plot direct path for reference
        ax1.plot([start_point[0], end_point[0]], 
                [start_point[1], end_point[1]], 
                [start_point[2], end_point[2]], 
                'k--', alpha=0.5, linewidth=1, label='Direct Path')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Distributed Trajectories\n(Same Start & End) - First Points Enlarged')
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
        
        # 2. XY Projection with zoomed-in start area
        for i in range(num_trajectories):
            traj = trajectories[i, :, 1:4].numpy()
            ax2.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.7, linewidth=1.5)
        
        ax2.scatter(start_point[0], start_point[1], c='red', s=200, marker='o', 
                   label='Start', zorder=10, edgecolors='black', linewidth=2)
        ax2.scatter(end_point[0], end_point[1], c='green', s=200, marker='s', 
                   label='End', zorder=10, edgecolors='black', linewidth=2)
        ax2.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                'k--', alpha=0.5, linewidth=1, label='Direct Path')
        
        # Zoom in on start area
        ax2ins = ax2.inset_axes([0.05, 0.6, 0.3, 0.3])
        for i in range(min(num_trajectories, 10)):  # Show only first 10 for clarity
            traj = trajectories[i, :, 1:4].numpy()
            ax2ins.plot(traj[:5, 0], traj[:5, 1], color=colors[i], alpha=0.7, linewidth=2)
        ax2ins.scatter(start_point[0], start_point[1], c='red', s=100, marker='o', 
                      zorder=10, edgecolors='black', linewidth=1)
        ax2ins.set_xlim(start_point[0] - 0.5, start_point[0] + 0.5)
        ax2ins.set_ylim(start_point[1] - 0.5, start_point[1] + 0.5)
        ax2ins.set_title('Zoom: Start Area')
        ax2ins.grid(True, alpha=0.3)
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('XY Projection with Start Zoom')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # 3. XZ Projection
        for i in range(num_trajectories):
            traj = trajectories[i, :, 1:4].numpy()
            ax3.plot(traj[:, 0], traj[:, 2], color=colors[i], alpha=0.7, linewidth=1.5)
        
        ax3.scatter(start_point[0], start_point[2], c='red', s=200, marker='o', 
                   label='Start', zorder=10, edgecolors='black', linewidth=2)
        ax3.scatter(end_point[0], end_point[2], c='green', s=200, marker='s', 
                   label='End', zorder=10, edgecolors='black', linewidth=2)
        ax3.plot([start_point[0], end_point[0]], [start_point[2], end_point[2]], 
                'k--', alpha=0.5, linewidth=1, label='Direct Path')
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('XZ Projection')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. YZ Projection
        for i in range(num_trajectories):
            traj = trajectories[i, :, 1:4].numpy()
            ax4.plot(traj[:, 1], traj[:, 2], color=colors[i], alpha=0.7, linewidth=1.5)
        
        ax4.scatter(start_point[1], start_point[2], c='red', s=200, marker='o', 
                   label='Start', zorder=10, edgecolors='black', linewidth=2)
        ax4.scatter(end_point[1], end_point[2], c='green', s=200, marker='s', 
                   label='End', zorder=10, edgecolors='black', linewidth=2)
        ax4.plot([start_point[1], end_point[1]], [start_point[2], end_point[2]], 
                'k--', alpha=0.5, linewidth=1, label='Direct Path')
        
        ax4.set_xlabel('Y')
        ax4.set_ylabel('Z')
        ax4.set_title('YZ Projection')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    if show_3d:
        # 5. First point analysis plot
        ax5 = plt.subplot(335)
        first_points = trajectories[:, 0, 1:4].numpy()
        
        # Plot distribution of first points
        for i in range(num_trajectories):
            ax5.scatter(first_points[i, 0], first_points[i, 1], 
                       color=colors[i], s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Calculate statistics
        mean_first = np.mean(first_points, axis=0)
        std_first = np.std(first_points, axis=0)
        
        # Draw mean point
        ax5.scatter(mean_first[0], mean_first[1], c='black', s=200, 
                   marker='X', label=f'Mean', zorder=10)
        
        # Draw std ellipse
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(xy=mean_first[:2], width=std_first[0]*4, 
                         height=std_first[1]*4, angle=0,
                         alpha=0.2, color='red', label=f'4×Std')
        ax5.add_patch(ellipse)
        
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_title(f'First Point Distribution (std_x={std_first[0]:.6f}, std_y={std_first[1]:.6f})')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(mean_first[0] - 0.1, mean_first[0] + 0.1)
        ax5.set_ylim(mean_first[1] - 0.1, mean_first[1] + 0.1)
        
        # 6. First 5 points comparison
        ax6 = plt.subplot(336)
        for i in range(min(num_trajectories, 10)):  # Show first 10 trajectories
            traj = trajectories[i, :5, 1:4].numpy()
            time_steps = np.arange(5)
            
            # Plot X coordinate
            ax6.plot(time_steps, traj[:, 0], color=colors[i], alpha=0.7, 
                    linewidth=2, label=f'Traj {i+1}' if i < 3 else "")
        
        ax6.set_xlabel('Time Step (0-4)')
        ax6.set_ylabel('X Coordinate')
        ax6.set_title('First 5 Points: X Coordinate Comparison')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Last point analysis plot
        ax7 = plt.subplot(337)
        last_points = trajectories[:, -1, 1:4].numpy()
        
        for i in range(num_trajectories):
            ax7.scatter(last_points[i, 0], last_points[i, 1], 
                       color=colors[i], s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        mean_last = np.mean(last_points, axis=0)
        std_last = np.std(last_points, axis=0)
        
        ax7.scatter(mean_last[0], mean_last[1], c='black', s=200, 
                   marker='X', label=f'Mean', zorder=10)
        
        ellipse_last = Ellipse(xy=mean_last[:2], width=std_last[0]*4, 
                              height=std_last[1]*4, angle=0,
                              alpha=0.2, color='green', label=f'4×Std')
        ax7.add_patch(ellipse_last)
        
        ax7.set_xlabel('X')
        ax7.set_ylabel('Y')
        ax7.set_title(f'Last Point Distribution (std_x={std_last[0]:.6f}, std_y={std_last[1]:.6f})')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Start and end similarity metrics
        ax8 = plt.subplot(338)
        
        # Calculate distances between trajectories at each time step
        start_distances = []
        end_distances = []
        
        for i in range(num_trajectories):
            for j in range(i+1, num_trajectories):
                # Start similarity (first 5 points)
                start_dist = np.mean(np.linalg.norm(
                    trajectories[i, :5, 1:4] - trajectories[j, :5, 1:4], axis=1
                ))
                start_distances.append(start_dist)
                
                # End similarity (last 5 points)
                end_dist = np.mean(np.linalg.norm(
                    trajectories[i, -5:, 1:4] - trajectories[j, -5:, 1:4], axis=1
                ))
                end_distances.append(end_dist)
        
        # Plot distribution of distances
        bins = np.linspace(0, max(max(start_distances), max(end_distances)), 20)
        ax8.hist(start_distances, bins=bins, alpha=0.5, color='red', label='Start (first 5 pts)')
        ax8.hist(end_distances, bins=bins, alpha=0.5, color='green', label='End (last 5 pts)')
        
        ax8.axvline(np.mean(start_distances), color='red', linestyle='--', linewidth=2)
        ax8.axvline(np.mean(end_distances), color='green', linestyle='--', linewidth=2)
        
        ax8.set_xlabel('Pairwise Distance')
        ax8.set_ylabel('Count')
        ax8.set_title(f'Similarity: Start mean={np.mean(start_distances):.6f}, End mean={np.mean(end_distances):.6f}')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Verification plot
        ax9 = plt.subplot(339)
        
        # Extract first points and calculate differences
        first_points = trajectories[:, 0, 1:4].numpy()
        reference = first_points[0]
        differences = np.linalg.norm(first_points - reference, axis=1)
        
        ax9.bar(range(num_trajectories), differences, color=colors, alpha=0.7)
        ax9.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax9.axhline(y=np.mean(differences), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(differences):.10f}')
        
        ax9.set_xlabel('Trajectory Index')
        ax9.set_ylabel('Distance from Trajectory 0')
        ax9.set_title('First Point Exactness Verification')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        ax9.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def analyze_distributed_trajectories(trajectories, start_similar_length=10, end_similar_length=10):
    """
    Analyze the distributed trajectories properties
    
    Args:
    - trajectories: Trajectory data
    - start_similar_length: Number of initial similar points
    - end_similar_length: Number of final similar points
    """
    num_trajectories = trajectories.shape[0]
    seq_len = trajectories.shape[1]
    
    print(f"\n{'='*80}")
    print("DISTRIBUTED TRAJECTORIES ANALYSIS - WITH EXACT START/END POINTS")
    print(f"{'='*80}")
    
    # Basic information
    print(f"Number of trajectories: {num_trajectories}")
    print(f"Sequence length: {seq_len}")
    print(f"State dimension: {trajectories.shape[2]}")
    
    # Starting point analysis - CRITICAL SECTION
    start_points = trajectories[:, 0, 1:4].numpy()
    start_std = np.std(start_points, axis=0)
    start_mean = np.mean(start_points, axis=0)
    
    print(f"\n{'='*40}")
    print("STARTING POINT ANALYSIS (CRITICAL)")
    print(f"{'='*40}")
    print(f"  Reference point (trajectory 0): {start_points[0]}")
    print(f"  Mean of all trajectories: {start_mean}")
    print(f"  Standard deviation: {start_std}")
    
    # Check exact equality
    all_equal = True
    max_difference = 0
    for i in range(1, num_trajectories):
        diff = np.max(np.abs(start_points[i] - start_points[0]))
        if diff > 1e-10:  # More strict tolerance
            all_equal = False
            max_difference = max(max_difference, diff)
    
    if all_equal:
        print(f"  ✓ ALL STARTING POINTS ARE IDENTICAL!")
        print(f"  Maximum difference: < 1e-10")
    else:
        print(f"  ✗ WARNING: Starting points are NOT identical!")
        print(f"  Maximum difference: {max_difference:.10f}")
    
    # Detailed comparison
    print(f"\n  Detailed comparison (first 5 trajectories):")
    for i in range(min(5, num_trajectories)):
        diff = start_points[i] - start_points[0]
        diff_norm = np.linalg.norm(diff)
        print(f"    Trajectory {i}: diff = {diff}, norm = {diff_norm:.10f}")
    
    # Ending point analysis
    end_points = trajectories[:, -1, 1:4].numpy()
    end_std = np.std(end_points, axis=0)
    end_mean = np.mean(end_points, axis=0)
    
    print(f"\n{'='*40}")
    print("ENDING POINT ANALYSIS")
    print(f"{'='*40}")
    print(f"  Reference point (trajectory 0): {end_points[0]}")
    print(f"  Mean of all trajectories: {end_mean}")
    print(f"  Standard deviation: {end_std}")
    
    # Check exact equality for end points
    all_equal_end = True
    max_difference_end = 0
    for i in range(1, num_trajectories):
        diff = np.max(np.abs(end_points[i] - end_points[0]))
        if diff > 1e-10:
            all_equal_end = False
            max_difference_end = max(max_difference_end, diff)
    
    if all_equal_end:
        print(f"  ✓ ALL ENDING POINTS ARE IDENTICAL!")
        print(f"  Maximum difference: < 1e-10")
    else:
        print(f"  ✗ WARNING: Ending points are NOT identical!")
        print(f"  Maximum difference: {max_difference_end:.10f}")
    
    # First N points similarity analysis
    print(f"\n{'='*40}")
    print(f"FIRST {start_similar_length} POINTS SIMILARITY")
    print(f"{'='*40}")
    first_n_points = trajectories[:, :start_similar_length, 1:4].numpy()
    
    # Calculate average pairwise distance
    pairwise_distances_start = []
    for i in range(num_trajectories):
        for j in range(i+1, num_trajectories):
            dist = np.mean(np.linalg.norm(
                first_n_points[i] - first_n_points[j], axis=1
            ))
            pairwise_distances_start.append(dist)
    
    print(f"  Average pairwise distance: {np.mean(pairwise_distances_start):.10f}")
    print(f"  Min/Max pairwise distance: {np.min(pairwise_distances_start):.10f}/{np.max(pairwise_distances_start):.10f}")
    
    # Last N points similarity analysis
    print(f"\n{'='*40}")
    print(f"LAST {end_similar_length} POINTS SIMILARITY")
    print(f"{'='*40}")
    last_n_points = trajectories[:, -end_similar_length:, 1:4].numpy()
    
    pairwise_distances_end = []
    for i in range(num_trajectories):
        for j in range(i+1, num_trajectories):
            dist = np.mean(np.linalg.norm(
                last_n_points[i] - last_n_points[j], axis=1
            ))
            pairwise_distances_end.append(dist)
    
    print(f"  Average pairwise distance: {np.mean(pairwise_distances_end):.10f}")
    print(f"  Min/Max pairwise distance: {np.min(pairwise_distances_end):.10f}/{np.max(pairwise_distances_end):.10f}")
    
    # Maximum deviation analysis (from direct line)
    print(f"\n{'='*40}")
    print("MAXIMUM DEVIATION ANALYSIS")
    print(f"{'='*40}")
    start_point_ref = start_points[0]
    end_point_ref = end_points[0]
    
    max_deviations = []
    for i in range(num_trajectories):
        traj = trajectories[i, :, 1:4].numpy()
        deviations = []
        for j in range(seq_len):
            t = j / (seq_len - 1) if seq_len > 1 else 0
            direct_point = start_point_ref * (1 - t) + end_point_ref * t
            dist = np.linalg.norm(traj[j] - direct_point)
            deviations.append(dist)
        max_deviations.append(np.max(deviations))
    
    print(f"  Mean max deviation: {np.mean(max_deviations):.6f}")
    print(f"  Min/Max deviation: {np.min(max_deviations):.6f}/{np.max(max_deviations):.6f}")
    
    # Path length analysis
    print(f"\n{'='*40}")
    print("PATH LENGTH ANALYSIS")
    print(f"{'='*40}")
    path_lengths = []
    for i in range(num_trajectories):
        traj = trajectories[i, :, 1:4].numpy()
        length = np.sum(np.linalg.norm(traj[1:] - traj[:-1], axis=1))
        path_lengths.append(length)
    
    direct_length = np.linalg.norm(end_point_ref - start_point_ref)
    print(f"  Direct path length: {direct_length:.6f}")
    print(f"  Mean trajectory length: {np.mean(path_lengths):.6f}")
    print(f"  Length increase ratio: {np.mean(path_lengths)/direct_length:.6f}")
    print(f"  Min/Max trajectory length: {np.min(path_lengths):.6f}/{np.max(path_lengths):.6f}")
    
    # Speed analysis
    speeds = trajectories[:, :, 0].numpy()
    print(f"\n{'='*40}")
    print("SPEED STATISTICS")
    print(f"{'='*40}")
    print(f"  Mean speed: {np.mean(speeds):.6f}")
    print(f"  Speed std: {np.std(speeds):.6f}")
    print(f"  Min/Max speed: {np.min(speeds):.6f}/{np.max(speeds):.6f}")
    
    # Diversity score (based on mid-point spread)
    print(f"\n{'='*40}")
    print("TRAJECTORY DIVERSITY")
    print(f"{'='*40}")
    mid_idx = seq_len // 2
    mid_points = trajectories[:, mid_idx, 1:4].numpy()
    
    # Calculate spread of mid-points
    mid_centroid = np.mean(mid_points, axis=0)
    distances_to_centroid = np.linalg.norm(mid_points - mid_centroid, axis=1)
    
    print(f"  Mid-point spread radius: {np.mean(distances_to_centroid):.6f}")
    print(f"  Mid-point max distance: {np.max(distances_to_centroid):.6f}")
    
    # Calculate coverage area
    x_range = np.max(mid_points[:, 0]) - np.min(mid_points[:, 0])
    y_range = np.max(mid_points[:, 1]) - np.min(mid_points[:, 1])
    z_range = np.max(mid_points[:, 2]) - np.min(mid_points[:, 2])
    
    print(f"  Mid-point coverage (X range): {x_range:.6f}")
    print(f"  Mid-point coverage (Y range): {y_range:.6f}")
    print(f"  Mid-point coverage (Z range): {z_range:.6f}")
    
    # Final verification
    print(f"\n{'='*80}")
    print("FINAL VERIFICATION")
    print(f"{'='*80}")
    
    # Calculate exact match percentage
    exact_start_matches = 0
    exact_end_matches = 0
    
    for i in range(num_trajectories):
        if np.allclose(trajectories[i, 0, 1:4], trajectories[0, 0, 1:4], atol=1e-10):
            exact_start_matches += 1
        if np.allclose(trajectories[i, -1, 1:4], trajectories[0, -1, 1:4], atol=1e-10):
            exact_end_matches += 1
    
    print(f"  Start points exactly matching: {exact_start_matches}/{num_trajectories} ({exact_start_matches/num_trajectories*100:.2f}%)")
    print(f"  End points exactly matching: {exact_end_matches}/{num_trajectories} ({exact_end_matches/num_trajectories*100:.2f}%)")
    
    if exact_start_matches == num_trajectories and exact_end_matches == num_trajectories:
        print(f"\n  ✓ SUCCESS: All trajectories have identical start and end points!")
    else:
        print(f"\n  ✗ WARNING: Not all trajectories have identical start and end points!")
    
    return {
        'start_std': start_std,
        'end_std': end_std,
        'start_exact_match': exact_start_matches == num_trajectories,
        'end_exact_match': exact_end_matches == num_trajectories,
        'mean_max_deviation': np.mean(max_deviations),
        'path_length_ratio': np.mean(path_lengths)/direct_length,
        'mid_point_spread': np.mean(distances_to_centroid)
    }

# Test the generation function
if __name__ == "__main__":
    print("Generating distributed trajectories with EXACT same start and end...")
    
    # Generate trajectories
    start_similar_length = 10
    end_similar_length = 10
    trajectories = generate_distributed_trajectories(
        num_trajectories=1200,
        seq_len=60,
        start_point=[0.0, 0.0, 0.0],  # Explicitly use float values
        end_point=[40.0, 0.0, 20.0],   # Explicitly use float values
        max_deviation_radius=12.0,
        start_similar_length=start_similar_length,
        end_similar_length=end_similar_length,
        smoothness_factor=0.85
    )
    
    print(f"\nGenerated trajectory shape: {trajectories.shape}")
    
    # Analyze the cluster
    analysis_results = analyze_distributed_trajectories(
        trajectories, 
        start_similar_length, 
        end_similar_length
    )
    
    # Visualize
    visualize_distributed_trajectories(trajectories, show_3d=True, show_2d=True)