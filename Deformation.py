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
        history_segment, future_segment = apply_smooth_connection(
            history_segment, future_segment, smooth_window
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


# Trajectory deformation module
class TrajectoryDeformer:
    """
    Trajectory deformer for applying physically-inspired deformations and stretching
    to trajectory points while maintaining start and end constraints
    """
    
    def __init__(self, deformation_strength=0.3, modes=None):
        """
        Initialize trajectory deformer
        
        Args:
            deformation_strength: Deformation strength (0.0-1.0)
            modes: Deformation mode list, options: ['stretch', 'twist', 'wave', 'bend', 'random']
        """
        self.deformation_strength = deformation_strength
        if modes is None:
            self.modes = ['stretch', 'twist', 'wave', 'bend', 'random']
        else:
            self.modes = modes
    
    def apply_deformation(self, trajectory_points, fix_start=True, fix_end=True, intensity_scale=1.0):
        """
        Apply deformation to trajectory points with intensity control
        
        Args:
            trajectory_points: (N, 3) trajectory points array
            fix_start: Whether to fix start point
            fix_end: Whether to fix end point
            intensity_scale: Intensity scaling factor (0.0-2.0) to control deformation degree
            
        Returns:
            deformed_points: Deformed trajectory points
        """
        if len(trajectory_points) < 3:
            return trajectory_points.copy()
        
        points = trajectory_points.copy()
        N = len(points)
        
        # Apply randomly selected deformation mode
        mode = np.random.choice(self.modes)
        
        # Scale the deformation strength based on intensity parameter
        effective_strength = self.deformation_strength * intensity_scale
        
        if mode == 'stretch':
            points = self._apply_stretch(points, fix_start, fix_end, effective_strength)
        elif mode == 'twist':
            points = self._apply_twist(points, fix_start, fix_end, effective_strength)
        elif mode == 'wave':
            points = self._apply_wave(points, fix_start, fix_end, effective_strength)
        elif mode == 'bend':
            points = self._apply_bend(points, fix_start, fix_end, effective_strength)
        elif mode == 'random':
            points = self._apply_random(points, fix_start, fix_end, effective_strength)
        
        # Ensure start and end point constraints
        if fix_start:
            points[0] = trajectory_points[0]
        if fix_end:
            points[-1] = trajectory_points[-1]
        
        # Apply intensity scaling to the smoothing process
        smoothing_intensity = min(1.0, intensity_scale)
        points = self._smooth_deformation(points, trajectory_points, smoothing_intensity)
        
        return points
    
    def apply_deformation_with_random_intensity(self, trajectory_points, fix_start=True, fix_end=True, 
                                                intensity_range=(0.0, 2.0), distribution='uniform'):
        """
        Apply deformation with randomly generated intensity
        
        Args:
            trajectory_points: (N, 3) trajectory points array
            fix_start: Whether to fix start point
            fix_end: Whether to fix end point
            intensity_range: Tuple (min_intensity, max_intensity) for random generation
            distribution: Random distribution type - 'uniform', 'normal', 'beta', or 'mixed'
            
        Returns:
            deformed_points: Deformed trajectory points
            intensity: Generated intensity value
        """
        # Generate random intensity based on specified distribution
        min_intensity, max_intensity = intensity_range
        
        if distribution == 'uniform':
            # Uniform distribution between min and max
            intensity = np.random.uniform(min_intensity, max_intensity)
            
        elif distribution == 'normal':
            # Normal distribution centered around midpoint
            midpoint = (min_intensity + max_intensity) / 2
            scale = (max_intensity - min_intensity) / 4  # 4σ covers most of range
            intensity = np.random.normal(midpoint, scale)
            # Clip to range
            intensity = np.clip(intensity, min_intensity, max_intensity)
            
        elif distribution == 'beta':
            # Beta distribution for more control over shape
            alpha, beta = 2.0, 2.0  # Symmetric bell shape
            # Beta distribution is on [0,1], scale to our range
            beta_sample = np.random.beta(alpha, beta)
            intensity = min_intensity + beta_sample * (max_intensity - min_intensity)
            
        elif distribution == 'mixed':
            # Mixture of distributions for more diversity
            rand_choice = np.random.rand()
            if rand_choice < 0.4:  # 40% low intensity
                intensity = np.random.uniform(min_intensity, min_intensity + 0.3*(max_intensity - min_intensity))
            elif rand_choice < 0.8:  # 40% medium intensity
                intensity = np.random.uniform(min_intensity + 0.3*(max_intensity - min_intensity), 
                                             min_intensity + 0.7*(max_intensity - min_intensity))
            else:  # 20% high intensity
                intensity = np.random.uniform(min_intensity + 0.7*(max_intensity - min_intensity), max_intensity)
        
        else:
            # Default to uniform
            intensity = np.random.uniform(min_intensity, max_intensity)
        
        # Apply deformation with the generated intensity
        deformed_points = self.apply_deformation(
            trajectory_points, 
            fix_start=fix_start, 
            fix_end=fix_end, 
            intensity_scale=intensity
        )
        
        return deformed_points, intensity
    
    def _apply_stretch(self, points, fix_start, fix_end, strength):
        """Stretch deformation: Stretch or compress along trajectory direction"""
        N = len(points)
        stretch_strength = np.random.uniform(0.5, 1.5) * strength
        
        # Calculate main trajectory direction
        if fix_start and fix_end:
            start_to_end = points[-1] - points[0]
            # Add small epsilon to avoid division by zero
            norm_val = np.linalg.norm(start_to_end)
            if norm_val < 1e-8:
                return points
            main_direction = start_to_end / norm_val
            
            # Apply non-uniform stretching along main direction
            t = np.linspace(0, 1, N)
            stretch_factor = 1.0 + stretch_strength * np.sin(2 * np.pi * t)
            
            # Apply stretching
            for i in range(N):
                if i == 0 and fix_start:
                    continue
                if i == N-1 and fix_end:
                    continue
                # Offset relative to start point
                offset = points[i] - points[0]
                # Rescale offset
                offset_scaled = offset * stretch_factor[i]
                points[i] = points[0] + offset_scaled
        
        return points
    
    def _apply_twist(self, points, fix_start, fix_end, strength):
        """Twist deformation: Create spiral twisting of trajectory"""
        N = len(points)
        twist_strength = np.random.uniform(0.5, 2.0) * strength
        
        # Calculate center line (straight line from start to end)
        if fix_start and fix_end:
            center_line = np.linspace(points[0], points[-1], N)
            
            # Calculate offset of each point relative to center line
            offsets = points - center_line
            
            # Apply rotational twisting
            t = np.linspace(0, 1, N)
            twist_angle = twist_strength * 2 * np.pi * t
            
            for i in range(N):
                if i == 0 and fix_start:
                    continue
                if i == N-1 and fix_end:
                    continue
                
                # Rotate offset vector
                cos_a = np.cos(twist_angle[i])
                sin_a = np.sin(twist_angle[i])
                
                # 2D rotation matrix (in plane perpendicular to forward direction)
                R = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])
                
                offsets[i] = R @ offsets[i]
            
            # Reconstruct points
            points = center_line + offsets
        
        return points
    
    def _apply_wave(self, points, fix_start, fix_end, strength):
        """Wave deformation: Add sinusoidal perturbations"""
        N = len(points)
        wave_strength = strength
        
        # Superposition of waves with different frequencies and directions
        num_waves = np.random.randint(2, 5)
        
        for wave_idx in range(num_waves):
            # Random wave parameters
            amplitude = np.random.uniform(0.1, 0.5) * wave_strength
            frequency = np.random.uniform(1.0, 5.0)
            phase = np.random.uniform(0, 2 * np.pi)
            
            # Random wave direction
            wave_direction = np.random.randn(3)
            norm_val = np.linalg.norm(wave_direction)
            if norm_val < 1e-8:
                continue
            wave_direction = wave_direction / norm_val
            
            # Apply wave
            t = np.linspace(0, 1, N)
            wave_value = amplitude * np.sin(2 * np.pi * frequency * t + phase)
            
            for i in range(N):
                if (i == 0 and fix_start) or (i == N-1 and fix_end):
                    continue
                
                points[i] += wave_value[i] * wave_direction
        
        return points
    
    def _apply_bend(self, points, fix_start, fix_end, strength):
        """Bend deformation: Overall bending of trajectory"""
        N = len(points)
        bend_strength = np.random.uniform(0.5, 2.0) * strength
        
        if fix_start and fix_end:
            # Calculate center point of trajectory
            center = np.mean(points, axis=0)
            
            # Calculate direction from center to each point
            directions = points - center
            
            # Bend transformation: rotate on a plane
            bend_axis = np.random.randn(3)
            norm_val = np.linalg.norm(bend_axis)
            if norm_val < 1e-8:
                return points
            bend_axis = bend_axis / norm_val
            
            # Calculate rotation angle based on distance from start point
            distances_from_start = np.linalg.norm(points - points[0], axis=1)
            # Fix: Ensure total_distance is never zero
            total_distance = max(distances_from_start[-1], 1e-8) if N > 0 else 1.0
            normalized_distances = distances_from_start / total_distance
            
            for i in range(N):
                if i == 0 and fix_start:
                    continue
                if i == N-1 and fix_end:
                    continue
                
                # Rotation angle proportional to distance
                angle = bend_strength * normalized_distances[i]
                
                # Check for NaN or infinite values
                if np.isnan(angle) or np.isinf(angle):
                    continue
                
                # Rodrigues rotation formula
                k = bend_axis
                v = directions[i]
                
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                
                # Check for invalid trigonometric values
                if np.isnan(cos_a) or np.isnan(sin_a):
                    continue
                
                # Rotated vector
                v_rot = v * cos_a + np.cross(k, v) * sin_a + k * np.dot(k, v) * (1 - cos_a)
                
                points[i] = center + v_rot
        
        return points
    
    def _apply_random(self, points, fix_start, fix_end, strength):
        """Random deformation: Combine multiple small deformations"""
        N = len(points)
        
        # Randomly apply multiple small deformations
        num_deformations = np.random.randint(2, 4)
        
        for _ in range(num_deformations):
            # Randomly select small region for deformation
            if N > 10:
                start_idx = np.random.randint(0, N-5)
                end_idx = np.random.randint(start_idx+2, min(start_idx+10, N))
                
                # Apply local deformation to selected region
                sub_points = points[start_idx:end_idx].copy()
                
                # Small random offsets - scaled by strength
                sub_strength = np.random.uniform(0.1, 0.3) * strength
                random_offset = np.random.randn(len(sub_points), 3) * sub_strength
                
                # Keep start and end of local region unchanged
                if start_idx == 0 and fix_start:
                    random_offset[0] = 0
                if end_idx == N-1 and fix_end:
                    random_offset[-1] = 0
                
                points[start_idx:end_idx] += random_offset
        
        return points
    
    def _smooth_deformation(self, deformed_points, original_points, intensity_scale=1.0):
        """
        Smooth deformation to avoid abrupt changes
        
        Args:
            deformed_points: Deformed points
            original_points: Original points
            intensity_scale: Intensity scaling factor (0.0-1.0) for smoothing
            
        Returns:
            smoothed_points: Smoothed deformed points
        """
        N = len(deformed_points)
        if N < 3:
            return deformed_points
        
        # Use simple Gaussian filter for smoothing
        smoothed = deformed_points.copy()
        kernel_size = min(5, N)
        
        # Boundary handling
        for i in range(N):
            if i == 0 or i == N-1:
                continue
            
            # Get neighbor points
            start = max(0, i - kernel_size // 2)
            end = min(N, i + kernel_size // 2 + 1)
            
            # Weighted average, higher weight for center point
            weights = np.exp(-0.5 * ((np.arange(start, end) - i) ** 2))
            weights = weights / weights.sum()
            
            smoothed[i] = np.average(deformed_points[start:end], axis=0, weights=weights)
        
        # Adjust smoothing based on intensity
        # Higher intensity = less smoothing (more deformation preserved)
        # Lower intensity = more smoothing (closer to original)
        alpha = 0.7 * (1.0 - 0.3 * (1.0 - intensity_scale))  # Adaptive smoothing coefficient
        
        return alpha * smoothed + (1 - alpha) * original_points

# Enhanced trajectory generation function with random intensity
def generate_aerobatic_trajectories_deformation(num_trajectories, seq_len, height=10.0, radius=5.0, 
                                                           enable_deformation=True, deformation_strength=0.2,
                                                           intensity_range=(0.0, 2.0), distribution='uniform'):
    """
    Generate synthetic aerobatic trajectories with random intensity deformation
    
    Args:
        num_trajectories: Number of trajectories
        seq_len: Sequence length
        height: Base height
        radius: Base radius
        enable_deformation: Whether to enable deformation
        deformation_strength: Base deformation strength
        intensity_range: Tuple (min_intensity, max_intensity) for random generation
        distribution: Random distribution type - 'uniform', 'normal', 'beta', or 'mixed'
        
    Returns:
        trajectories: Trajectory tensor
        intensities: List of intensity values used for each trajectory
    """
    trajectories = []
    intensities = []
    
    maneuver_styles = ['power_loop', 'barrel_roll', 'split_s', 'immelmann', 'wall_ride', 
                      'eight_figure', 'star', 'half_moon', 'sphinx', 'clover', 
                      'spiral_inward', 'spiral_outward', 'spiral_vertical_up', 'spiral_vertical_down']
    
    # Initialize deformer
    deformer = TrajectoryDeformer(deformation_strength=deformation_strength) if enable_deformation else None
    
    for i in range(num_trajectories):
        # Randomly select maneuver style
        style = np.random.choice(maneuver_styles)
        
        style_to_index = {
            'power_loop': 0, 'barrel_roll': 1, 'split_s': 2, 'immelmann': 3,
            'wall_roll': 4, 'wall_ride': 4, 'eight_figure': 5, 'star': 6,
            'half_moon': 7, 'sphinx': 8, 'clover': 9, 'spiral_inward': 10,
            'spiral_outward': 11, 'spiral_vertical_up': 12, 'spiral_vertical_down': 13
        }
        
        style_idx = style_to_index.get(style, 0)
        
        # Random centers and scales
        center_x = np.random.uniform(-20, 20)
        center_y = np.random.uniform(-20, 20)
        center_z = height + np.random.uniform(-10, 10)
        current_radius = radius * np.random.uniform(0.8, 1.2)
        
        # Generate base trajectory
        base_trajectory = generate_single_style_trajectory(style, seq_len, center_z, current_radius)
        
        # Apply random translation
        x = base_trajectory[:, 0] + center_x
        y = base_trajectory[:, 1] + center_y
        z = base_trajectory[:, 2]
        
        # Apply deformation with random intensity (if needed)
        intensity = 0.0
        if enable_deformation and deformer is not None and seq_len > 10:
            # Combine points into (N, 3) format
            points = np.column_stack([x, y, z])
            
            # Apply deformation with random intensity
            deformed_points, intensity = deformer.apply_deformation_with_random_intensity(
                points, 
                fix_start=True, 
                fix_end=True,
                intensity_range=intensity_range,
                distribution=distribution
            )
            
            # Update coordinates
            x = deformed_points[:, 0]
            y = deformed_points[:, 1]
            z = deformed_points[:, 2]
        
        intensities.append(intensity)
        
        # Calculate velocity and direction
        dt = 0.10  # 0.1 second
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        vz = np.gradient(z, dt)
        
        # Calculate speed and direction
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        direction = np.stack([vx, vy, vz], axis=-1)
        norms = np.linalg.norm(direction, axis=-1, keepdims=True)
        direction = np.divide(direction, norms, where=norms>0, out=np.zeros_like(direction))
        
        # Attitude: direction + fixed components
        attitude = np.concatenate([direction, np.full((seq_len, 3), 0.1)], axis=-1)
        
        # Full state: [speed, x, y, z, attitude(6), style_index]
        state = np.column_stack([speed, x, y, z, attitude, np.full(seq_len, style_idx)])
        trajectories.append(state)
    
    return torch.tensor(np.stack(trajectories), dtype=torch.float32) #, intensities

# Visualization function for random intensity deformations
def visualize_random_intensity_deformations():
    """Visualize trajectory deformations with random intensity levels"""
    print("Generating random intensity deformation examples...")
    
    styles = ['power_loop', 'barrel_roll', 'split_s', 'immelmann']
    
    # Generate base trajectories for each style
    base_trajectories = {}
    for style in styles:
        base_traj = generate_single_style_trajectory(style, seq_len=60, height=10.0, radius=5.0)
        base_trajectories[style] = base_traj
    
    # Create deformer
    deformer = TrajectoryDeformer(deformation_strength=0.3)
    
    # Generate 4 random intensities for each style
    np.random.seed(42)  # For reproducibility
    
    # Visualization
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Trajectory Deformations with Random Intensity Levels', fontsize=16, fontweight='bold')
    
    # Create subplots: rows = styles, columns = random examples
    num_examples = 5  # Original + 4 random deformations
    
    for row_idx, style in enumerate(styles):
        base_traj = base_trajectories[style]
        
        for col_idx in range(num_examples):
            ax = fig.add_subplot(len(styles), num_examples, 
                                 row_idx * num_examples + col_idx + 1, 
                                 projection='3d')
            
            if col_idx == 0:
                # First column: original trajectory
                points = base_traj
                intensity = 0.0
                color = 'b'
                linestyle = '-'
                label = 'Original'
            else:
                # Random deformations
                # Generate random intensity using mixed distribution for diversity
                deformed_points, intensity = deformer.apply_deformation_with_random_intensity(
                    base_traj, 
                    fix_start=True, 
                    fix_end=True,
                    intensity_range=(0.0, 2.0),
                    distribution='mixed'
                )
                points = deformed_points
                color = 'r'
                linestyle = '-'
                label = 'Deformed'
            
            # Plot trajectory
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 
                   color=color, linewidth=2, alpha=0.8, label=label)
            
            # Plot original as reference for deformed trajectories
            if col_idx > 0:
                ax.plot(base_traj[:, 0], base_traj[:, 1], base_traj[:, 2], 
                       'b--', linewidth=1, alpha=0.3, label='Original')
            
            # Mark start and end points
            ax.scatter(points[0, 0], points[0, 1], points[0, 2], 
                      c='green', s=80, marker='o', label='Start' if col_idx == 0 else '')
            ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], 
                      c='red', s=80, marker='s', label='End' if col_idx == 0 else '')
            
            # Calculate deformation distance for statistics
            if col_idx > 0:
                deformation_dist = np.mean(np.linalg.norm(points - base_traj, axis=1))
            else:
                deformation_dist = 0.0
            
            # Set title with intensity and statistics
            if col_idx == 0:
                title = f'{style[:10]}...\nOriginal'
            else:
                title = f'{style[:10]}...\nIntensity: {intensity:.2f}\nAvg Δ: {deformation_dist:.3f}'
            ax.set_title(title, fontsize=9)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.grid(True, alpha=0.3)
            
            # Only show legend for first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)
    
    plt.tight_layout()
    plt.show()
    
    # Generate and display intensity statistics
    print("\n=== Random Intensity Statistics ===")
    all_intensities = []
    
    for style in styles:
        base_traj = base_trajectories[style]
        style_intensities = []
        
        # Generate 100 random intensities for statistics
        for _ in range(100):
            _, intensity = deformer.apply_deformation_with_random_intensity(
                base_traj, 
                fix_start=True, 
                fix_end=True,
                intensity_range=(0.0, 2.0),
                distribution='mixed'
            )
            style_intensities.append(intensity)
        
        all_intensities.extend(style_intensities)
        
        print(f"\n{style}:")
        print(f"  Min intensity: {np.min(style_intensities):.3f}")
        print(f"  Max intensity: {np.max(style_intensities):.3f}")
        print(f"  Mean intensity: {np.mean(style_intensities):.3f}")
        print(f"  Std intensity: {np.std(style_intensities):.3f}")
    
    # Overall statistics
    print(f"\nOverall statistics (all styles):")
    print(f"  Min intensity: {np.min(all_intensities):.3f}")
    print(f"  Max intensity: {np.max(all_intensities):.3f}")
    print(f"  Mean intensity: {np.mean(all_intensities):.3f}")
    print(f"  Std intensity: {np.std(all_intensities):.3f}")
    
    # Plot intensity distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(all_intensities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(all_intensities), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_intensities):.3f}')
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Randomly Generated Intensities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Test function for random intensity
def test_random_intensity_generation():
    """Test random intensity generation functionality"""
    print("\n=== Testing Random Intensity Generation ===")
    
    # Generate a simple trajectory
    base_traj = generate_single_style_trajectory('power_loop', seq_len=40, height=10.0, radius=5.0)
    
    # Create deformer
    deformer = TrajectoryDeformer(deformation_strength=0.25)
    
    # Test different distributions
    distributions = ['uniform', 'normal', 'beta', 'mixed']
    
    results = {}
    for dist in distributions:
        print(f"\nTesting '{dist}' distribution:")
        dist_intensities = []
        dist_deformations = []
        
        # Generate 50 samples
        for _ in range(50):
            deformed_traj, intensity = deformer.apply_deformation_with_random_intensity(
                base_traj, 
                fix_start=True, 
                fix_end=True,
                intensity_range=(0.0, 2.0),
                distribution=dist
            )
            
            # Calculate deformation
            deformation = np.mean(np.linalg.norm(deformed_traj - base_traj, axis=1))
            
            dist_intensities.append(intensity)
            dist_deformations.append(deformation)
        
        # Calculate statistics
        results[dist] = {
            'intensities': dist_intensities,
            'deformations': dist_deformations,
            'mean_intensity': np.mean(dist_intensities),
            'std_intensity': np.std(dist_intensities),
            'mean_deformation': np.mean(dist_deformations),
            'std_deformation': np.std(dist_deformations)
        }
        
        print(f"  Intensity - Mean: {results[dist]['mean_intensity']:.3f}, Std: {results[dist]['std_intensity']:.3f}")
        print(f"  Deformation - Mean: {results[dist]['mean_deformation']:.3f}, Std: {results[dist]['std_deformation']:.3f}")
    
    return results


if __name__ == "__main__":
    # Run the random intensity visualization
    visualize_random_intensity_deformations()
