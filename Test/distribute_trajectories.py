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
    Generate trajectories with EXACTLY same start (position, velocity, attitude)
    for initial segment and EXACTLY same end for final segment
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set default points
    if start_point is None:
        start_point = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    else:
        start_point = torch.tensor(start_point, dtype=torch.float32)
    
    if end_point is None:
        end_point = torch.tensor([30.0, 0.0, 15.0], dtype=torch.float32)
    else:
        end_point = torch.tensor(end_point, dtype=torch.float32)
    
    start_point = start_point.to(device)
    end_point = end_point.to(device)
    
    # Initialize trajectory tensor
    trajectories = torch.zeros(num_trajectories, seq_len, 11, device=device, dtype=torch.float32)
    
    # Generate control points
    angles = torch.linspace(0, 2 * torch.pi, num_trajectories + 1, device=device)[:num_trajectories]
    deviations = torch.linspace(0.3, 1.0, num_trajectories, device=device)
    
    # Generate reference trajectory
    reference_trajectory = None
    reference_velocity = None
    reference_attitude = None
    
    for traj_idx in range(num_trajectories):
        # Calculate control points
        direct_midpoint = (start_point + end_point) * 0.5
        direct_vector = end_point - start_point
        direct_length = torch.norm(direct_vector)
        
        if direct_length > 0.001:
            # Normalize direct vector
            direct_normalized = direct_vector / direct_length
            
            # Find perpendicular vector
            up_vector = torch.tensor([0.0, 0.0, 1.0], device=device)
            perp_vector = torch.cross(direct_normalized, up_vector)
            
            if torch.norm(perp_vector) < 0.001:
                perp_vector = torch.tensor([-direct_normalized[1], direct_normalized[0], 0.0], device=device)
            
            perp_normalized = perp_vector / torch.norm(perp_vector)
            
            # Generate deviation
            angle = angles[traj_idx]
            deviation_direction = torch.cos(angle) * perp_normalized
            vertical_scale = torch.sin(angle * 2)
            deviation = max_deviation_radius * deviations[traj_idx] * deviation_direction
            
            # Mid control point
            mid_control = direct_midpoint + deviation
            mid_control[2] += max_deviation_radius * 0.3 * vertical_scale
            
            # Quintic Bezier control points
            control_points = torch.zeros(5, 3, device=device)
            control_points[0] = start_point
            
            t1 = 0.25
            point1 = start_point * (1 - t1) + end_point * t1
            deviation1 = deviation * 0.3 * torch.sin(angle * 0.5)
            control_points[1] = point1 + deviation1
            
            control_points[2] = mid_control
            
            t2 = 0.75
            point2 = start_point * (1 - t2) + end_point * t2
            deviation2 = -deviation * 0.3 * torch.sin(angle * 0.5)
            control_points[3] = point2 + deviation2
            
            control_points[4] = end_point
            
        else:
            control_points = torch.stack([
                start_point,
                start_point + torch.tensor([1.0, 0.0, 2.0], device=device),
                (start_point + end_point) * 0.5 + torch.tensor([0.0, 0.0, 5.0], device=device),
                end_point + torch.tensor([-1.0, 0.0, 2.0], device=device),
                end_point
            ])
        
        # Generate trajectory using quintic Bezier
        t_vals = torch.linspace(0, 1, seq_len, device=device)
        trajectory = torch.zeros(seq_len, 3, device=device)
        
        for j, t in enumerate(t_vals):
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
        
        # Handle similarity zones
        noise = torch.randn(seq_len, 3, device=device) * 0.05 * (1 - smoothness_factor)
        
        # No noise in similarity zones
        noise[:start_similar_length] = 0
        noise[-end_similar_length:] = 0
        
        # Apply noise only to middle section
        middle_start = start_similar_length
        middle_end = seq_len - end_similar_length
        trajectory[middle_start:middle_end] += noise[middle_start:middle_end]
        
        # Force exact positions in similarity zones
        if traj_idx == 0:
            reference_trajectory = trajectory.clone()
            trajectory[0] = start_point
            trajectory[-1] = end_point
        else:
            trajectory[:start_similar_length] = reference_trajectory[:start_similar_length]
            trajectory[-end_similar_length:] = reference_trajectory[-end_similar_length:]
            trajectory[0] = start_point
            trajectory[-1] = end_point
        
        # Calculate velocity
        velocity = torch.zeros(seq_len, 3, device=device)
        velocity[1:] = trajectory[1:] - trajectory[:-1]
        
        if seq_len > 1:
            velocity[0] = velocity[1]
        else:
            velocity[0] = torch.zeros(3, device=device)
        
        # Use reference velocity in similarity zones
        if traj_idx == 0:
            reference_velocity = velocity.clone()
        else:
            velocity[:start_similar_length] = reference_velocity[:start_similar_length]
            velocity[-end_similar_length:] = reference_velocity[-end_similar_length:]
        
        # Calculate speed
        speed = torch.norm(velocity, dim=1, keepdim=True)
        
        # Calculate attitude
        attitude = torch.zeros(seq_len, 6, device=device)
        
        for k in range(seq_len):
            if speed[k] > 0.01:
                v = velocity[k]
                spd = speed[k].item()
                
                yaw = torch.atan2(v[1], v[0])
                pitch = torch.asin(torch.clamp(v[2] / spd, -0.99, 0.99))
                
                if k > 0 and k < seq_len - 1:
                    acc = velocity[k] - velocity[k-1]
                    lateral_acc = torch.norm(acc[:2])
                    roll = torch.tanh(lateral_acc * 2.0) * 0.3
                else:
                    roll = torch.tensor(0.0, device=device)
                
                attitude[k, 0] = roll
                attitude[k, 1] = pitch
                attitude[k, 2] = yaw
                
                if k > 0:
                    attitude[k, 3] = attitude[k, 0] - attitude[k-1, 0]
                    attitude[k, 4] = attitude[k, 1] - attitude[k-1, 1]
                    attitude[k, 5] = attitude[k, 2] - attitude[k-1, 2]
                else:
                    attitude[k, 3:] = torch.zeros(3, device=device)
            else:
                attitude[k] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
        
        # Use reference attitude in similarity zones
        if traj_idx == 0:
            reference_attitude = attitude.clone()
        else:
            attitude[:start_similar_length] = reference_attitude[:start_similar_length]
            attitude[-end_similar_length:] = reference_attitude[-end_similar_length:]
        
        # Fill trajectory data
        trajectories[traj_idx, :, 0] = speed.squeeze()
        trajectories[traj_idx, :, 1:4] = trajectory
        trajectories[traj_idx, :, 4:10] = attitude
        trajectories[traj_idx, :, -1] = traj_idx % 14
    
    return trajectories.cpu()

def visualize_with_exact_start(trajectories, start_similar_length=10, end_similar_length=10):
    """
    Visualize trajectories with emphasis on identical start/end segments
    """
    # Convert to numpy for plotting
    trajectories_np = trajectories.numpy() if torch.is_tensor(trajectories) else trajectories
    
    num_trajectories = trajectories_np.shape[0]
    seq_len = trajectories_np.shape[1]
    
    # Create color mapping
    colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories))
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 3D view with identical start/end emphasis
    ax1 = fig.add_subplot(231, projection='3d')
    
    for i in range(num_trajectories):
        traj = trajectories_np[i, :, 1:4]
        
        # Plot full trajectory
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=colors[i], alpha=0.3, linewidth=1)
        
        # Highlight identical start segment
        ax1.plot(traj[:start_similar_length, 0], 
                traj[:start_similar_length, 1], 
                traj[:start_similar_length, 2], 
                color='red', alpha=0.8, linewidth=3)
        
        # Highlight identical end segment
        ax1.plot(traj[-end_similar_length:, 0], 
                traj[-end_similar_length:, 1], 
                traj[-end_similar_length:, 2], 
                color='green', alpha=0.8, linewidth=3)
    
    # Mark start and end points
    start_point = trajectories_np[0, 0, 1:4]
    end_point = trajectories_np[0, -1, 1:4]
    
    ax1.scatter(start_point[0], start_point[1], start_point[2], 
               c='red', s=300, marker='o', label='Start', zorder=10)
    ax1.scatter(end_point[0], end_point[1], end_point[2], 
               c='green', s=300, marker='s', label='End', zorder=10)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'3D View: Identical Start/End Segments\n(Red: first {start_similar_length} points, Green: last {end_similar_length} points)')
    ax1.legend()
    
    # 2. Speed comparison in start zone
    ax2 = fig.add_subplot(232)
    
    time_steps = np.arange(start_similar_length)
    for i in range(min(num_trajectories, 20)):
        speed = trajectories_np[i, :start_similar_length, 0]
        ax2.plot(time_steps, speed, color=colors[i], alpha=0.5, linewidth=1)
    
    mean_speed = np.mean(trajectories_np[:, :start_similar_length, 0], axis=0)
    ax2.plot(time_steps, mean_speed, 'k-', linewidth=3, label=f'Mean speed')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Speed')
    ax2.set_title(f'Speed in Start Zone (first {start_similar_length} points)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Speed difference heatmap in start zone
    ax3 = fig.add_subplot(233)
    
    speeds_start = trajectories_np[:, :start_similar_length, 0]
    ref_speeds = speeds_start[0]
    speed_diffs = np.abs(speeds_start - ref_speeds)
    
    im = ax3.imshow(speed_diffs, aspect='auto', cmap='hot', 
                   extent=[0, start_similar_length-1, 0, num_trajectories-1])
    ax3.set_xlabel('Time Step in Start Zone')
    ax3.set_ylabel('Trajectory Index')
    ax3.set_title(f'Speed Difference from Trajectory 0\nin Start Zone (max diff: {np.max(speed_diffs):.6f})')
    plt.colorbar(im, ax=ax3, label='Speed Difference')
    
    # 4. Velocity vector visualization at start
    ax4 = fig.add_subplot(234)
    
    for i in range(min(num_trajectories, 15)):
        pos = trajectories_np[i, 0, 1:3]
        # Use pitch and yaw rates for velocity direction
        pitch_rate = trajectories_np[i, 0, 8]  # index 8 is pitch rate
        yaw_rate = trajectories_np[i, 0, 9]    # index 9 is yaw rate
        
        # Create velocity vector from rates
        vel = np.array([np.cos(yaw_rate), np.sin(yaw_rate)])
        vel = vel / (np.linalg.norm(vel) + 1e-10)
        
        ax4.quiver(pos[0], pos[1], vel[0], vel[1], 
                  color=colors[i], angles='xy', scale_units='xy', scale=0.5)
    
    ax4.scatter(start_point[0], start_point[1], c='red', s=200, marker='o', zorder=10)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('Velocity Vectors at Start Point (Time Step 0)')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    # 5. Attitude comparison in start zone
    ax5 = fig.add_subplot(235)
    
    for i in range(min(num_trajectories, 15)):
        yaw = trajectories_np[i, :start_similar_length, 6]  # yaw is at index 6
        ax5.plot(time_steps, yaw, color=colors[i], alpha=0.5, linewidth=1)
    
    mean_yaw = np.mean(trajectories_np[:, :start_similar_length, 6], axis=0)
    ax5.plot(time_steps, mean_yaw, 'k-', linewidth=3, label=f'Mean yaw')
    
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Yaw (radians)')
    ax5.set_title(f'Yaw Angle in Start Zone (first {start_similar_length} points)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Verification statistics (FIXED VERSION)
    ax6 = fig.add_subplot(236)
    
    max_diffs = []
    for i in range(1, num_trajectories):
        # FIX: Convert to numpy array first
        start_pos_i = trajectories_np[i, :start_similar_length, 1:4]
        start_pos_0 = trajectories_np[0, :start_similar_length, 1:4]
        
        start_speed_i = trajectories_np[i, :start_similar_length, 0]
        start_speed_0 = trajectories_np[0, :start_similar_length, 0]
        
        start_att_i = trajectories_np[i, :start_similar_length, 4:10]
        start_att_0 = trajectories_np[0, :start_similar_length, 4:10]
        
        diff_pos = np.max(np.abs(start_pos_i - start_pos_0))
        diff_speed = np.max(np.abs(start_speed_i - start_speed_0))
        diff_attitude = np.max(np.abs(start_att_i - start_att_0))
        
        max_diffs.append(max(diff_pos, diff_speed, diff_attitude))
    
    # Plot bar chart
    ax6.bar(range(1, num_trajectories), max_diffs, color=colors[1:num_trajectories])
    ax6.axhline(y=1e-10, color='red', linestyle='--', linewidth=2, label='Tolerance (1e-10)')
    
    ax6.set_xlabel('Trajectory Index')
    ax6.set_ylabel('Maximum Difference from Trajectory 0')
    ax6.set_title(f'Start Zone Identity Verification\n(Mean diff: {np.mean(max_diffs):.2e})')
    ax6.set_yscale('log')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_exact_start_trajectories(trajectories, start_similar_length=10, end_similar_length=10):
    """
    Detailed analysis of trajectories with identical start/end segments
    """
    # Convert to numpy for analysis
    trajectories_np = trajectories.numpy() if torch.is_tensor(trajectories) else trajectories
    
    num_trajectories = trajectories_np.shape[0]
    seq_len = trajectories_np.shape[1]
    
    print("\n" + "="*80)
    print("ANALYSIS: Trajectories with Identical Start/End Segments")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Number of trajectories: {num_trajectories}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Start similarity length: {start_similar_length} points")
    print(f"  End similarity length: {end_similar_length} points")
    print(f"  Middle diverse segment: {seq_len - start_similar_length - end_similar_length} points")
    
    # Extract start zone data
    start_positions = trajectories_np[:, :start_similar_length, 1:4]
    start_speeds = trajectories_np[:, :start_similar_length, 0]
    start_attitudes = trajectories_np[:, :start_similar_length, 4:10]
    
    # Calculate statistics
    pos_std = np.std(start_positions, axis=0)
    speed_std = np.std(start_speeds, axis=0)
    attitude_std = np.std(start_attitudes, axis=0)
    
    print(f"\nSTART ZONE ANALYSIS:")
    print(f"Position standard deviation (max):")
    print(f"  X: {np.max(pos_std[:, 0]):.10f}")
    print(f"  Y: {np.max(pos_std[:, 1]):.10f}")
    print(f"  Z: {np.max(pos_std[:, 2]):.10f}")
    print(f"Speed standard deviation (max): {np.max(speed_std):.10f}")
    print(f"Attitude standard deviation (max): {np.max(attitude_std):.10f}")
    
    # Extract end zone data
    end_positions = trajectories_np[:, -end_similar_length:, 1:4]
    end_speeds = trajectories_np[:, -end_similar_length:, 0]
    
    end_pos_std = np.std(end_positions, axis=0)
    end_speed_std = np.std(end_speeds, axis=0)
    
    print(f"\nEND ZONE ANALYSIS:")
    print(f"Position standard deviation (max):")
    print(f"  X: {np.max(end_pos_std[:, 0]):.10f}")
    print(f"  Y: {np.max(end_pos_std[:, 1]):.10f}")
    print(f"  Z: {np.max(end_pos_std[:, 2]):.10f}")
    print(f"Speed standard deviation (max): {np.max(end_speed_std):.10f}")
    
    # Middle zone analysis
    middle_start = start_similar_length
    middle_end = seq_len - end_similar_length
    
    if middle_end > middle_start:
        middle_length = middle_end - middle_start
        mid_point_idx = middle_start + middle_length // 2
        mid_points = trajectories_np[:, mid_point_idx, 1:4]
        
        mid_centroid = np.mean(mid_points, axis=0)
        distances = np.linalg.norm(mid_points - mid_centroid, axis=1)
        
        print(f"\nMIDDLE ZONE DIVERSITY:")
        print(f"  Middle zone length: {middle_length} points")
        print(f"  Mid-point spread radius: {np.mean(distances):.6f}")
        print(f"  Max distance from centroid: {np.max(distances):.6f}")
    
    # Final verification
    tolerance = 1e-10
    start_identical = True
    end_identical = True
    
    for i in range(1, num_trajectories):
        if (np.max(np.abs(start_positions[i] - start_positions[0])) > tolerance or
            np.max(np.abs(start_speeds[i] - start_speeds[0])) > tolerance or
            np.max(np.abs(start_attitudes[i] - start_attitudes[0])) > tolerance):
            start_identical = False
        
        if (np.max(np.abs(end_positions[i] - end_positions[0])) > tolerance or
            np.max(np.abs(end_speeds[i] - end_speeds[0])) > tolerance):
            end_identical = False
    
    print(f"\nFINAL VERIFICATION:")
    if start_identical and end_identical:
        print("✓ SUCCESS: All trajectories have IDENTICAL start and end segments")
        print("  (position, speed, attitude are exactly the same)")
    else:
        print("✗ WARNING: Start and/or end segments are not identical")
        if not start_identical:
            print("  - Start segments differ")
        if not end_identical:
            print("  - End segments differ")

# Main test function
if __name__ == "__main__":
    print("="*80)
    print("GENERATING TRAJECTORIES WITH IDENTICAL START/END SEGMENTS")
    print("="*80)
    
    # Configuration
    num_trajectories = 50
    seq_len = 60
    start_similar_length = 5
    end_similar_length = 5
    
    print(f"\nConfiguration:")
    print(f"  Number of trajectories: {num_trajectories}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Identical start points: first {start_similar_length}")
    print(f"  Identical end points: last {end_similar_length}")
    
    # Generate trajectories
    trajectories = generate_distributed_trajectories(
        num_trajectories=num_trajectories,
        seq_len=seq_len,
        start_point=[0.0, 0.0, 0.0],
        end_point=[40.0, 0.0, 20.0],
        max_deviation_radius=12.0,
        start_similar_length=start_similar_length,
        end_similar_length=end_similar_length,
        smoothness_factor=0.85
    )
    
    print(f"\nGenerated trajectory shape: {trajectories.shape}")
    
    # Detailed analysis
    analyze_exact_start_trajectories(
        trajectories, 
        start_similar_length, 
        end_similar_length
    )
    
    # Visualize
    visualize_with_exact_start(
        trajectories, 
        start_similar_length, 
        end_similar_length
    )
