import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

def smooth_connection_with_spline(history_segment, future_segment, num_blend=10):
    """
    Smoothly connect two trajectory segments using cubic spline interpolation.
    
    This method creates a smooth transition by fitting a cubic spline through the
    overlapping region between history and future segments, ensuring C2 continuity.
    
    Args:
        history_segment: Tensor of shape (history_len, state_dim) - the historical trajectory
        future_segment: Tensor of shape (seq_len, state_dim) - the future trajectory to connect
        num_blend: Number of frames to blend on each side of the connection
    
    Returns:
        adjusted_history: History segment with smoothed end portion
        adjusted_future: Future segment with smoothed beginning portion
    """
    from scipy import interpolate
    
    history_len = history_segment.shape[0]
    seq_len = future_segment.shape[0]
    
    # Extract the last 'num_blend' points from history and first 'num_blend' points from future
    # These form the overlapping region that will be interpolated
    hist_blend = history_segment[-num_blend:, 1:4].cpu().numpy()  # (num_blend, 3)
    future_blend = future_segment[:num_blend, 1:4].cpu().numpy()  # (num_blend, 3)
    
    # Concatenate all points in the blending region
    # Order: history end points first, then future start points
    blend_points = np.vstack([hist_blend, future_blend])  # (2*num_blend, 3)
    
    # Create a normalized parameter t from 0 to 1 for the blending region
    # This parameterizes the curve regardless of the number of points
    t = np.linspace(0, 1, len(blend_points))
    
    # Create cubic spline interpolators for each spatial dimension (X, Y, Z)
    # CubicSpline provides C2 continuity (continuous position, velocity, and acceleration)
    t_new = np.linspace(0, 1, len(blend_points))  # Same parameterization for evaluation
    spline_x = interpolate.CubicSpline(t, blend_points[:, 0])
    spline_y = interpolate.CubicSpline(t, blend_points[:, 1])
    spline_z = interpolate.CubicSpline(t, blend_points[:, 2])
    
    # Generate smoothly interpolated points along the spline
    smooth_points = np.column_stack([
        spline_x(t_new),
        spline_y(t_new),
        spline_z(t_new)
    ])  # (2*num_blend, 3)
    
    # Clone original tensors to avoid in-place modifications
    adjusted_history = history_segment.clone()
    adjusted_future = future_segment.clone()
    
    # Update the blending region in both segments
    # The first half of smooth_points corresponds to the history end
    # The second half corresponds to the future start
    for i in range(num_blend):
        # Update history segment end portion (working backwards from the connection point)
        adjusted_history[history_len - num_blend + i, 1:4] = torch.tensor(smooth_points[i])
        
        # Update future segment beginning portion (working forwards from the connection point)
        # Use symmetric indexing: smooth_points[-(num_blend - i)] maps to future positions
        adjusted_future[i, 1:4] = torch.tensor(smooth_points[-(num_blend - i)])
    
    # Update speed component based on new positions
    adjusted_history = update_speed_from_positions(adjusted_history)
    adjusted_future = update_speed_from_positions(adjusted_future)
    
    return adjusted_history, adjusted_future

def apply_smooth_connection(history_segment, future_segment, smooth_window=5):
    """
    Apply smooth blending at the connection boundary between history and future segments.
    
    Uses cosine-weighted blending to ensure C1 continuity (position and velocity continuous).
    
    Args:
        history_segment (torch.Tensor): History segment of shape (history_len, state_dim)
        future_segment (torch.Tensor): Future segment of shape (seq_len, state_dim)
        smooth_window (int): Number of frames to blend on each side of the connection
    
    Returns:
        tuple: (adjusted_history_segment, adjusted_future_segment)
    """
    history_len = history_segment.shape[0]
    seq_len = future_segment.shape[0]
    
    # Ensure window doesn't exceed available frames
    window = min(smooth_window, history_len, seq_len)
    
    if window < 2:
        return history_segment, future_segment
    
    # Get positions at the boundary
    history_pos = history_segment[:, 1:4]  # (history_len, 3)
    future_pos = future_segment[:, 1:4]    # (seq_len, 3)
    
    # Compute velocities at boundary (using finite differences)
    if history_len >= 2:
        history_vel = history_pos[-1] - history_pos[-2]
    else:
        history_vel = torch.zeros(3, device=history_pos.device)
    
    if seq_len >= 2:
        future_vel = future_pos[1] - future_pos[0]
    else:
        future_vel = torch.zeros(3, device=future_pos.device)
    
    # Compute desired connection velocity (average of both sides for smoothness)
    connection_vel = (history_vel + future_vel) / 2.0
    
    # Adjust history segment near the end
    adjusted_history = history_segment.clone()
    adjusted_future = future_segment.clone()
    
    for i in range(1, window + 1):
        # Cosine blending weight: smooth transition from 0 to 1
        alpha = 0.5 * (1 - torch.cos(torch.tensor(i / (window + 1) * 3.14159, device=history_pos.device)))
        
        # Adjust history positions near the end
        hist_idx = history_len - window + i - 1
        if hist_idx >= 0 and hist_idx < history_len:
            # Target position based on smooth velocity integration
            target_pos = history_pos[-1] - connection_vel * (window - i + 1)
            # Blend between original and target
            adjusted_history[hist_idx, 1:4] = (1 - alpha) * history_pos[hist_idx] + alpha * target_pos
        
        # Adjust future positions near the start
        fut_idx = i - 1
        if fut_idx < seq_len:
            # Target position based on smooth velocity integration
            target_pos = history_pos[-1] + connection_vel * i
            # Blend between original and target
            adjusted_future[fut_idx, 1:4] = (1 - alpha) * future_pos[fut_idx] + alpha * target_pos
    
    # Ensure exact continuity at the boundary
    adjusted_future[0, 1:4] = history_pos[-1]
    
    # Update velocities (speed component at index 0)
    adjusted_history = update_speed_from_positions(adjusted_history)
    adjusted_future = update_speed_from_positions(adjusted_future)
    
    return adjusted_history, adjusted_future

def update_speed_from_positions(trajectory):
    """
    Update speed component (index 0) based on position differences.
    
    Args:
        trajectory (torch.Tensor): Trajectory of shape (seq_len, state_dim)
    
    Returns:
        torch.Tensor: Trajectory with updated speed values
    """
    traj = trajectory.clone()
    positions = traj[:, 1:4]
    
    if positions.shape[0] >= 2:
        # Compute speeds as magnitude of velocity
        velocities = positions[1:] - positions[:-1]
        speeds = torch.norm(velocities, dim=1)
        
        # Assign speeds (first frame uses second frame's speed)
        traj[0, 0] = speeds[0]
        traj[1:, 0] = speeds
    
    return traj

def random_concatenate_trajectories_smooth(trajectories, history_len, seq_len, 
                                           num_concatenated=None, smooth_window=5):
    """
    Randomly concatenate different trajectories with smooth connection.
    
    Method: 
    1. Select history segment from trajectory 1: [0:history_len]
    2. Select future segment from trajectory 2: [history_len:history_len+seq_len]
    3. Translate future segment to align with history end point
    4. Apply smooth blending at the connection boundary
    
    Args:
        trajectories (torch.Tensor): Original trajectories of shape (N, total_len, state_dim)
        history_len (int): Length of history segment
        seq_len (int): Length of future prediction segment
        num_concatenated (int, optional): Number of concatenated trajectories to generate
        smooth_window (int): Number of frames to blend at the connection
    
    Returns:
        torch.Tensor: Smoothly concatenated trajectories
    """
    num_original = trajectories.shape[0]
    
    if num_concatenated is None:
        num_concatenated = num_original // 2
    
    device = trajectories.device
    concatenated_trajectories = []
    
    for i in range(num_concatenated):
        # Randomly select two different trajectories
        idx1 = torch.randint(0, num_original, (1,)).item()
        idx2 = torch.randint(0, num_original, (1,)).item()
        
        # Ensure they are different trajectories
        while idx1 == idx2 and num_original > 1:
            idx2 = torch.randint(0, num_original, (1,)).item()
        
        style1 = trajectories[idx1, 0, -1].item()
        style2 = trajectories[idx2, 0, -1].item()
        
        # if style1 != style2:
        #     print(f"Concatenating different styles: {style1} -> {style2}")
    
        # Extract segments
        # history_segment = trajectories[idx1, :history_len, :].clone()  # (history_len, state_dim)
        start_idx = trajectories.shape[1] - history_len
        history_segment = trajectories[idx1, start_idx:, :].clone()
        future_segment = trajectories[idx2, history_len:history_len+seq_len, :].clone()  # (seq_len, state_dim)
        
        # Get boundary positions
        history_end_pos = history_segment[-1, 1:4]  # (3,)
        future_start_pos = future_segment[0, 1:4]   # (3,)
        
        # Compute translation to align future start with history end
        translation = history_end_pos - future_start_pos
        
        # Apply translation to all positions in future segment
        future_segment[:, 1:4] = future_segment[:, 1:4] + translation.unsqueeze(0)
        
        # Apply smooth blending at the connection
        # history_segment, future_segment = apply_smooth_connection(
        #     history_segment, future_segment, smooth_window
        # )
        
        history_segment, future_segment = smooth_connection_with_spline(
            history_segment, 
            future_segment, 
            num_blend=8  # Blend over 8 frames (0.8 seconds at 10Hz)
        )

        # Concatenate along time dimension
        new_trajectory = torch.cat([history_segment, future_segment], dim=0)  # (total_len, state_dim)
        
        concatenated_trajectories.append(new_trajectory)
    
    # Stack all concatenated trajectories
    concatenated_trajectories = torch.stack(concatenated_trajectories, dim=0)
    
    return concatenated_trajectories

def augment_trajectories_with_smooth_concatenation(trajectories, history_len, seq_len,
                                                   concat_ratio=0.5, smooth_window=5,
                                                   validate_continuity=True):
    """
    Augment trajectory dataset by smoothly concatenating different trajectories.
    
    This creates new trajectories by combining history from one trajectory with 
    future from another, ensuring smooth position and velocity continuity at the boundary.
    
    Args:
        trajectories (torch.Tensor): Original trajectories of shape (N, total_len, state_dim)
        history_len (int): Length of history segment
        seq_len (int): Length of future prediction segment  
        concat_ratio (float): Ratio of concatenated trajectories to original (default: 0.5)
        smooth_window (int): Number of frames for smooth blending at connection
        validate_continuity (bool): If True, print continuity validation statistics
    
    Returns:
        torch.Tensor: Augmented trajectories (original + concatenated)
    """
    num_original = trajectories.shape[0]
    num_concatenated = int(num_original * concat_ratio)
    
    print(f"Augmenting trajectories with smooth concatenation:")
    print(f"  - Original trajectories: {num_original}")
    print(f"  - Concatenated trajectories: {num_concatenated}")
    print(f"  - Smooth window: {smooth_window}")
    
    # Generate smoothly concatenated trajectories
    concatenated = random_concatenate_trajectories_smooth(
        trajectories, history_len, seq_len, num_concatenated, smooth_window
    )
    
    if validate_continuity:
        validate_connection_continuity(concatenated, history_len)
    
    # Combine original and concatenated trajectories
    augmented_trajectories = torch.cat([trajectories, concatenated], dim=0)
    
    print(f"  - Total trajectories after augmentation: {augmented_trajectories.shape[0]}")
    
    return augmented_trajectories

def validate_connection_continuity(concatenated_trajectories, history_len):
    """
    Validate the continuity at the connection point of concatenated trajectories.
    
    Computes position jump and velocity discontinuity at the boundary.
    
    Args:
        concatened_trajectories (torch.Tensor): Concatenated trajectories
        history_len (int): Length of history segment (boundary index)
    """
    num_traj = concatenated_trajectories.shape[0]
    device = concatenated_trajectories.device
    
    position_jumps = []
    velocity_jumps = []
    
    with torch.no_grad():
        for i in range(num_traj):
            # Get positions around boundary
            pos_before = concatenated_trajectories[i, history_len-1, 1:4]
            pos_at = concatenated_trajectories[i, history_len, 1:4]
            pos_after = concatenated_trajectories[i, history_len+1, 1:4] if history_len+1 < concatenated_trajectories.shape[1] else pos_at
            
            # Position jump at boundary (should be near zero)
            pos_jump = torch.norm(pos_at - pos_before)
            position_jumps.append(pos_jump.item())
            
            # Velocity continuity
            vel_before = pos_at - pos_before
            vel_after = pos_after - pos_at
            vel_jump = torch.norm(vel_after - vel_before)
            velocity_jumps.append(vel_jump.item())
    
    # Compute statistics
    pos_jumps = torch.tensor(position_jumps)
    vel_jumps = torch.tensor(velocity_jumps)
    
    print(f"\nConnection Continuity Validation:")
    print(f"  - Position jump at boundary:")
    print(f"      Mean: {pos_jumps.mean():.6f}, Max: {pos_jumps.max():.6f}, Std: {pos_jumps.std():.6f}")
    print(f"  - Velocity discontinuity at boundary:")
    print(f"      Mean: {vel_jumps.mean():.6f}, Max: {vel_jumps.max():.6f}, Std: {vel_jumps.std():.6f}")
    
    # Check if continuity is good (position jump < 1e-4)
    if pos_jumps.mean() < 1e-4:
        print(f"  ✓ Position continuity is excellent")
    elif pos_jumps.mean() < 1e-2:
        print(f"  ✓ Position continuity is good")
    else:
        print(f"  ⚠ Position continuity may need improvement")

def generate_single_style_trajectory(style, seq_len=60, height=10.0, radius=5.0):
    """Generate trajectory data for a single style"""
    
    # Fixed parameters for consistency
    center_x, center_y, center_z = 0, 0, height
    current_radius = radius
    angular_velocity = 1.0
    
    # Normalized time steps
    norm_t = np.linspace(0, 1, seq_len)
    
    x = np.zeros(seq_len)
    y = np.zeros(seq_len)
    z = np.zeros(seq_len)
    
    # Default forward velocity component (helps alignment)
    fwd = 8.0 * norm_t  # slight forward translation so maneuvers don't perfectly close on themselves

    if style == 'power_loop':
        # Full vertical loop, upright entry/exit
        theta = 2 * np.pi * norm_t
        x = -radius * np.sin(theta) + fwd
        y = np.zeros_like(norm_t)
        z = height + radius * (1 - np.cos(theta))

    elif style == 'barrel_roll':
        # Classic aileron roll while flying a gentle helix
        rolls = 2.0
        pitch = 6.0
        theta = 2 * np.pi * rolls * norm_t
        x = radius * np.cos(theta) + fwd
        y = radius * np.sin(theta)
        z = height + pitch * norm_t

    elif style == 'split_s':
        # Real Split-S: half negative-g loop + 180° roll to upright
        loop_frac = 0.55
        roll_frac = 1.0 - loop_frac

        # Phase 1: half loop downward (negative g)
        theta_loop = np.pi * (norm_t / loop_frac)   # 0 → π
        mask1 = norm_t <= loop_frac
        x[mask1] = radius * np.sin(theta_loop[mask1])
        z[mask1] = height + radius * (1 - np.cos(theta_loop[mask1]))  # start high → go down

        # Phase 2: 180° roll while pulling level
        roll_t = (norm_t - loop_frac) / roll_frac
        mask2 = norm_t > loop_frac
        roll_angle = np.pi * roll_t
        lateral = radius * 0.4 * np.sin(roll_angle)  # slight lateral offset during roll
        x[mask2] = x[int(loop_frac*seq_len)] + lateral[mask2]
        z[mask2] = z[int(loop_frac*seq_len)-1] - 2.0 * (norm_t[mask2] - loop_frac) * height
        y[mask2] = radius * 0.6 * np.sin(roll_angle[mask2])  # visible roll in Y
        x += fwd

    elif style == 'immelmann':
        # Classic Immelmann: half positive-g loop + 180° roll on top
        loop_frac = 0.6
        roll_frac = 1.0 - loop_frac

        # Phase 1: half loop upward
        theta_loop = np.pi * (norm_t / loop_frac)
        mask1 = norm_t <= loop_frac
        x[mask1] = radius * np.sin(theta_loop[mask1])
        z[mask1] = height + radius * (1 - np.cos(theta_loop[mask1]))

        # Phase 2: half roll at the top
        roll_t = (norm_t - loop_frac) / roll_frac
        mask2 = norm_t > loop_frac
        roll_angle = np.pi * roll_t
        x[mask2] = x[int(loop_frac*seq_len)] + radius * 0.3 * np.sin(roll_angle[mask2])
        z[mask2] = z[int(loop_frac*seq_len)-1]
        y[mask2] = radius * 0.7 * np.sin(roll_angle[mask2])  # roll visible
        x += fwd

    elif style == 'wall_ride':
        # Vertical corkscrew climb with continuous roll
        turns = 2.5
        climb = 35.0
        theta = 2 * np.pi * turns * norm_t
        x = radius * np.cos(theta) + fwd * 0.3
        y = radius * np.sin(theta)
        z = height + climb * norm_t
        
    elif style == 'eight_figure':
        theta = 2 * np.pi * norm_t
        x = center_x + current_radius * np.sin(theta)
        y = center_y + 0.5 * current_radius * np.sin(2 * theta)
        z = np.full(seq_len, center_z)
        
    elif style == 'star':
        alpha, beta = 2.0, 3.0
        x = center_x + current_radius * np.sin(2 * np.pi * alpha * norm_t)
        y = center_y + current_radius * np.cos(2 * np.pi * beta * norm_t)
        z = center_z + current_radius * 0. * np.sin(2 * np.pi * (alpha + beta) * norm_t)
        
    elif style == 'half_moon':
        theta = np.pi * norm_t
        x = center_x + current_radius * np.cos(theta)
        y = center_y + current_radius * np.sin(theta)
        z = center_z + 0.1 * current_radius * np.sin(theta)
        
    elif style == 'sphinx':
        # Rising helix with sinusoidal pitch oscillation
        turns = 1.8
        climb = 28.0
        theta = 2 * np.pi * turns * norm_t
        x = radius * np.cos(theta) + fwd * 0.4
        y = radius * np.sin(theta)
        z = height + climb * norm_t + 6 * np.sin(4 * np.pi * norm_t)
        
    elif style == 'clover':
        alpha = 2.0
        theta = 2 * np.pi * norm_t
        x = center_x + current_radius * np.cos(theta) * np.cos(2 * np.pi * alpha * norm_t)
        y = center_y + current_radius * np.cos(theta) * np.sin(2 * np.pi * alpha * norm_t)
        z = np.full(seq_len, center_z)
        
    elif style == 'spiral_inward':
        turns = 2.0
        start_radius = current_radius * 2.0
        end_radius = current_radius * 0.2
        theta = 2 * np.pi * turns * norm_t * angular_velocity
        radius_t = start_radius + (end_radius - start_radius) * norm_t
        x = center_x + radius_t * np.cos(theta)
        y = center_y + radius_t * np.sin(theta)
        z = np.full(seq_len, center_z)
        
    elif style == 'spiral_outward':
        turns = 2.0
        start_radius = current_radius * 0.2
        end_radius = current_radius * 2.0
        theta = 2 * np.pi * turns * norm_t * angular_velocity
        radius_t = start_radius + (end_radius - start_radius) * norm_t
        x = center_x + radius_t * np.cos(theta)
        y = center_y + radius_t * np.sin(theta)
        z = np.full(seq_len, center_z)
        
    elif style == 'spiral_vertical_up':
        turns = 1.5
        climb_height = 25.0
        theta = 2 * np.pi * turns * norm_t * angular_velocity
        x = center_x + current_radius * np.cos(theta)
        y = np.full(seq_len, center_y)
        z = center_z + climb_height * norm_t
        
    elif style == 'spiral_vertical_down':
        turns = 1.5
        descent_height = 25.0
        theta = 2 * np.pi * turns * norm_t * angular_velocity
        x = center_x + current_radius * np.cos(theta)
        y = np.full(seq_len, center_y)
        z = center_z - descent_height * norm_t
        
    else:
        # Default straight line trajectory
        x = center_x + norm_t * 10
        y = np.full(seq_len, center_y)
        z = np.full(seq_len, center_z)
    
    # Create state array
    state = np.column_stack([x, y, z])
    return state

def generate_aerobatic_trajectories(num_trajectories, seq_len, height=10.0, radius=5.0):
    """Generates synthetic aerobatic trajectories with correct attitude angles."""
    trajectories = []

    maneuver_styles = ['power_loop', 'barrel_roll', 'split_s', 'immelmann', 'wall_ride', 'eight_figure', 'star', 'half_moon', 'sphinx', 'clover', 'spiral_inward', 'spiral_outward', 'spiral_vertical_up', 'spiral_vertical_down']
    
    for i in range(num_trajectories):
        # Randomly select a maneuver style
        style = np.random.choice(maneuver_styles)
        
        style_to_index = {
            'power_loop': 0,
            'barrel_roll': 1,
            'split_s': 2,
            'immelmann': 3,
            'wall_ride': 4,
            'eight_figure': 5,
            'star': 6,
            'half_moon': 7,
            'sphinx': 8,
            'clover': 9,
            'spiral_inward': 10,
            'spiral_outward': 11,
            'spiral_vertical_up': 12,
            'spiral_vertical_down': 13
        }
        
        style_idx = style_to_index.get(style, 0)

        # Random centers and scales
        center_x = np.random.uniform(-3, 3)
        center_y = np.random.uniform(-3, 3)
        center_z = height + np.random.uniform(0, 3)
        current_radius = radius * np.random.uniform(0.3, 1.2)
        
        # Generate the base trajectory using generate_single_style_trajectory
        base_trajectory = generate_single_style_trajectory(style, seq_len, center_z, current_radius)
        
        # Extract positions and apply random translation
        x = base_trajectory[:, 0] + center_x
        y = base_trajectory[:, 1] + center_y  
        z = base_trajectory[:, 2]
        
        # Calculate velocities using gradient
        dt = 0.10  # 0.1 second
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        vz = np.gradient(z, dt)
        
        # Compute speed
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # Calculate attitude angles from velocity direction
        # Yaw is fixed to 0 degrees (aircraft points along X-axis in horizontal plane)
        # Roll and pitch are derived from velocity vector
        
        # Pitch angle (θ): angle between velocity vector and horizontal plane
        # pitch = arcsin(vz / speed)  (positive = nose up, negative = nose down)
        horizontal_speed = np.sqrt(vx**2 + vy**2 + 1e-8)  # Avoid division by zero
        pitch = np.arcsin(np.clip(vz / (speed + 1e-8), -1.0, 1.0))
        
        # Roll angle (φ): rotation around the forward axis
        # With yaw fixed at 0, the aircraft's body X-axis aligns with world X-axis in horizontal projection
        # Roll is determined by the lateral acceleration or bank angle
        # For coordinated flight, roll relates to turn rate: tan(φ) = v * ψ̇ / g
        # Simplified: roll indicates how much the aircraft is banking into turns
        
        # Calculate heading angle (ψ) in horizontal plane
        heading = np.arctan2(vy, vx)  # ψ = atan2(vy, vx)
        
        # Calculate change in heading (turn rate)
        heading_rate = np.gradient(heading, dt)
        
        # Coordinated turn bank angle: tan(φ) = (v * ψ̇) / g
        # Use centripetal acceleration: a_centripetal = v * ψ̇
        v = speed + 1e-8
        centripetal_acc = v * heading_rate
        g = 9.81  # gravity
        
        # Roll angle from coordinated turn (clamp to reasonable range ±π/2)
        roll_turn = np.arctan2(centripetal_acc, g)
        
        # Additional roll component for maneuvers like barrel rolls
        # For barrel rolls, we can add a sinusoidal component
        roll_maneuver = np.zeros_like(roll_turn)
        
        # Add maneuver-specific roll effects
        if style in ['barrel_roll']:
            # Full roll rotation over the trajectory
            t = np.linspace(0, 2*np.pi, seq_len)
            roll_maneuver = 2 * np.pi * (t / (2*np.pi))  # One full rotation
        elif style in ['power_loop', 'split_s', 'immelmann']:
            # Loops involve pitch changes, minimal roll
            roll_maneuver = np.zeros_like(roll_turn)
        elif style in ['spiral_inward', 'spiral_outward']:
            # Spirals have continuous roll
            t = np.linspace(0, 2*np.pi, seq_len)
            roll_maneuver = np.pi * np.sin(t)  # Oscillating roll
        
        # Combine turn-based roll and maneuver-specific roll
        roll = roll_turn + roll_maneuver
        
        # Clamp roll to reasonable range [-π, π]
        roll = np.clip(roll, -np.pi, np.pi)
        
        # For straight-line flight (low turn rate), roll should be near 0
        # Smooth out small variations
        roll[np.abs(heading_rate) < 0.1] = roll[np.abs(heading_rate) < 0.1] * 0.5
        
        # Attitude representation: [roll, pitch, yaw]
        # Yaw is fixed to 0 as requested
        yaw = np.zeros(seq_len)
        
        # Combine into attitude array
        attitude = np.column_stack([roll, pitch, yaw])
        
        # Full state: [speed, x, y, z, roll, pitch, yaw, style_idx]
        attitude_3d = np.column_stack([roll, pitch, yaw])
        
        state = np.column_stack([speed, x, y, z, attitude_3d, np.full(seq_len, style_idx)])
        
        trajectories.append(state)
        
    return torch.tensor(np.stack(trajectories), dtype=torch.float32)

def generate_aerobatic_trajectories_pvR(num_trajectories, seq_len, height=10.0, radius=5.0, delta_T=0.10):
    """Generates synthetic aerobatic trajectories with correct attitude angles."""
    trajectories = []

    maneuver_styles = ['power_loop', 'barrel_roll', 'split_s', 'immelmann', 'wall_ride', 'eight_figure', 'star', 'half_moon', 'sphinx', 'clover', 'spiral_inward', 'spiral_outward', 'spiral_vertical_up', 'spiral_vertical_down']
    
    for i in range(num_trajectories):
        # Randomly select a maneuver style
        style = np.random.choice(maneuver_styles)
        
        style_to_index = {
            'power_loop': 0,
            'barrel_roll': 1,
            'split_s': 2,
            'immelmann': 3,
            'wall_ride': 4,
            'eight_figure': 5,
            'star': 6,
            'half_moon': 7,
            'sphinx': 8,
            'clover': 9,
            'spiral_inward': 10,
            'spiral_outward': 11,
            'spiral_vertical_up': 12,
            'spiral_vertical_down': 13
        }
        
        style_idx = style_to_index.get(style, 0)

        # Random centers and scales
        center_x = np.random.uniform(-3, 3)
        center_y = np.random.uniform(-3, 3)
        center_z = height + np.random.uniform(0, 3)
        current_radius = radius * np.random.uniform(0.3, 1.2)
        
        # Generate the base trajectory using generate_single_style_trajectory
        base_trajectory = generate_single_style_trajectory(style, seq_len, center_z, current_radius)
        
        # Extract positions and apply random translation
        x = base_trajectory[:, 0] + center_x
        y = base_trajectory[:, 1] + center_y  
        z = base_trajectory[:, 2]
        
        # Calculate velocities using gradient
        dt = 0.10  # 0.1 second
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        vz = np.gradient(z, dt)
        
        # Compute speed
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # Calculate attitude angles from velocity direction
        # Yaw is fixed to 0 degrees (aircraft points along X-axis in horizontal plane)
        # Roll and pitch are derived from velocity vector
        
        # Pitch angle (θ): angle between velocity vector and horizontal plane
        # pitch = arcsin(vz / speed)  (positive = nose up, negative = nose down)
        horizontal_speed = np.sqrt(vx**2 + vy**2 + 1e-8)  # Avoid division by zero
        pitch = np.arcsin(np.clip(vz / (speed + 1e-8), -1.0, 1.0))
        
        # Roll angle (φ): rotation around the forward axis
        # With yaw fixed at 0, the aircraft's body X-axis aligns with world X-axis in horizontal projection
        # Roll is determined by the lateral acceleration or bank angle
        # For coordinated flight, roll relates to turn rate: tan(φ) = v * ψ̇ / g
        # Simplified: roll indicates how much the aircraft is banking into turns
        
        # Calculate heading angle (ψ) in horizontal plane
        heading = np.arctan2(vy, vx)  # ψ = atan2(vy, vx)
        
        # Calculate change in heading (turn rate)
        heading_rate = np.gradient(heading, dt)
        
        # Coordinated turn bank angle: tan(φ) = (v * ψ̇) / g
        # Use centripetal acceleration: a_centripetal = v * ψ̇
        v = speed + 1e-8
        centripetal_acc = v * heading_rate
        g = 9.81  # gravity
        
        # Roll angle from coordinated turn (clamp to reasonable range ±π/2)
        roll_turn = np.arctan2(centripetal_acc, g)
        
        # Additional roll component for maneuvers like barrel rolls
        # For barrel rolls, we can add a sinusoidal component
        roll_maneuver = np.zeros_like(roll_turn)
        
        # Add maneuver-specific roll effects
        if style in ['barrel_roll']:
            # Full roll rotation over the trajectory
            t = np.linspace(0, 2*np.pi, seq_len)
            roll_maneuver = 2 * np.pi * (t / (2*np.pi))  # One full rotation
        elif style in ['power_loop', 'split_s', 'immelmann']:
            # Loops involve pitch changes, minimal roll
            roll_maneuver = np.zeros_like(roll_turn)
        elif style in ['spiral_inward', 'spiral_outward']:
            # Spirals have continuous roll
            t = np.linspace(0, 2*np.pi, seq_len)
            roll_maneuver = np.pi * np.sin(t)  # Oscillating roll
        
        # Combine turn-based roll and maneuver-specific roll
        roll = roll_turn + roll_maneuver
        
        # Clamp roll to reasonable range [-π, π]
        roll = np.clip(roll, -np.pi, np.pi)
        
        # For straight-line flight (low turn rate), roll should be near 0
        # Smooth out small variations
        roll[np.abs(heading_rate) < 0.1] = roll[np.abs(heading_rate) < 0.1] * 0.5
        
        # Attitude representation: [roll, pitch, yaw]
        # Yaw is fixed to 0 as requested
        yaw = np.zeros(seq_len)
        
        # Combine into attitude array
        attitude = np.column_stack([roll, pitch, yaw])
        
        # Full state: [vx, x, y, z, vy, vz, roll, pitch, yaw, style_idx]
        attitude_3d = np.column_stack([roll, pitch, yaw])
        
        state = np.column_stack([speed, x, y, z, vx, vy, vz, attitude_3d, np.full(seq_len, style_idx)])
        
        trajectories.append(state)
        
    return torch.tensor(np.stack(trajectories), dtype=torch.float32)

if __name__ == "__main__":
    # Run the random intensity visualization
    generate_aerobatic_trajectories_pvR(num_trajectories=5, seq_len=100)
