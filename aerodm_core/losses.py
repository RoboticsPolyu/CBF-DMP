"""Training objectives for AeroDM."""

import torch
import torch.nn as nn

class AeroDMLoss(nn.Module):
    """
    Unified loss function for AeroDM training.
    Combines position, velocity, speed, attitude, and optional obstacle avoidance losses.
    Supports switching obstacle term via flag; always returns 4 values for consistency.
    Fixes: Proper safety margin for obstacles, Z-weighting, normalization by avg obstacles.
    """
    def __init__(self, config, enable_obstacle_term=False, safe_extra_factor=0.2, last_xyz_weight=1.5, xyz_weight=1.5, vel_weight=1.0, other_weight=1.0, obstacle_weight=10.0, continuity_weight=15.0, acc_weight=1.0, dynamics_consistency_weight=1.0):
        super().__init__()
        self.config = config
        # Flag to enable/disable obstacle distance penalty in total loss
        self.enable_obstacle_term = enable_obstacle_term
        # Safety buffer beyond obstacle surface (as fraction of radius, e.g., 0.2 = 20%)
        self.safe_extra_factor = safe_extra_factor
        # Extra weight for the last point's position loss (critical for trajectory endpoint accuracy)
        self.last_xyz_weight = last_xyz_weight
        # Extra weight for Z-axis losses (height is critical in aviation trajectories)
        self.xyz_weight = xyz_weight
        # Weight for velocity loss
        self.vel_weight = vel_weight
        # Weight for other losses
        self.other_weight = other_weight
        # Scaling factor for the entire obstacle loss term
        self.obstacle_weight = obstacle_weight
        # Weight for continuity loss
        self.continuity_weight = continuity_weight
        # Weight for acceleration loss
        self.acc_weight = acc_weight
        # Weight for position-velocity dynamics consistency
        self.dynamics_consistency_weight = dynamics_consistency_weight
        # Base MSE loss for all components
        self.mse_loss = nn.MSELoss()

    def compute_obstacle_distance_loss(self, pred_trajectory, obstacles_data, mean, std):
        """
        Computes obstacle avoidance loss.
        Penalizes trajectories that enter a safety buffer around obstacles.
        Formula: sum_over_obs_and_time [max(0, safe_extra - (dist_to_surface))^2] / (batch * seq * avg_num_obs)
        Where dist_to_surface = dist_to_center - radius.
        Returns 0 if disabled or no obstacles.
        """
        if not self.enable_obstacle_term or not obstacles_data or len(obstacles_data) == 0:
            # Return a scalar tensor with requires_grad for backprop compatibility
            return torch.tensor(0.0, device=pred_trajectory.device, requires_grad=True)

        batch_size, seq_len, _ = pred_trajectory.shape
        device = pred_trajectory.device
        # Initialize accumulators
        obstacle_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_num_obs = 0  # For normalization by average obstacles per batch

        # Denormalize positions for real-world distance computation
        # Assumes position indices: 1:4 (x,y,z)
        pos_std = std[0, 0, 1:4]
        pos_mean = mean[0, 0, 1:4]
        pred_pos_denorm = pred_trajectory[:, :, 1:4] * pos_std + pos_mean

        # Loop over batch samples
        for batch_idx in range(batch_size):
            # Get obstacles for this sample (empty list if none)
            batch_obs = obstacles_data[batch_idx] if batch_idx < len(obstacles_data) else []
            total_num_obs += len(batch_obs)  # Count total for avg

            # Loop over obstacles in this sample
            for obstacle in batch_obs:
                # Obstacle center (3D tensor) and radius (scalar)
                center = obstacle['center'].to(device)  # Shape: (3,)
                radius = obstacle['radius']  # Scalar float
                # Compute safety extra distance (proportional to radius)
                safe_extra = self.safe_extra_factor * radius

                # Euclidean distances from trajectory points to center
                # Shape: (seq_len,)
                distances = torch.norm(pred_pos_denorm[batch_idx] - center.unsqueeze(0), dim=1)

                # Distance to obstacle surface (positive outside, negative inside)
                surface_dist = distances - radius
                # Penalty only if inside safety buffer: clamp(safe_extra - surface_dist, 0)
                # E.g., if surface_dist < safe_extra, penalize the violation squared
                closeness_penalty = torch.clamp(safe_extra - surface_dist, min=0.0)
                # Accumulate squared penalties over time steps
                obstacle_loss = obstacle_loss + torch.sum(closeness_penalty ** 2)

        # Normalize: Average per batch, time step, and obstacle (prevents bias from varying obs count)
        avg_num_obs = total_num_obs / max(batch_size, 1.0)  # Avoid div-by-zero
        if avg_num_obs > 0:
            obstacle_loss = obstacle_loss / (batch_size * avg_num_obs) # obstacle_loss = obstacle_loss / (batch_size * seq_len * avg_num_obs)

        # Clamp to non-negative for stability (though clamp in penalty ensures this)
        return torch.clamp(obstacle_loss, min=0.0)

    def forward(self, pred_trajectory, gt_trajectory, obstacles_data=None, mean=None, std=None, history=None):
        """
        Computes total loss and components.
        Always returns (total_loss, position_loss, vel_loss, obstacle_loss, continuity_loss).
        - position_loss: Weighted MSE on positions (Z higher, last point x10).
        - vel_loss: MSE on velocity diffs.
        - obstacle_loss: 0 if disabled/no obs.
        - total: 2.0*position + 1.5*vel + other (speed + attitude) + obstacle_weight*obstacle.
        Handles seq_len <=1 for vel (returns 0).
        """
        batch_size, seq_len, state_dim = pred_trajectory.shape
        device = pred_trajectory.device

        # Extract components by indices: speed(0), pos(1:4), attitude(4:)
        pred_pos = pred_trajectory[:, :, 1:4]  # (B, T, 3) - positions x,y,z
        gt_pos = gt_trajectory[:, :, 1:4]
        pred_speed = pred_trajectory[:, :, 0:1]  # (B, T, 1) - speed
        gt_speed = gt_trajectory[:, :, 0:1]
        gt_vel = gt_trajectory[:, :, 4:7] # (B, T, 3) - velocity from GT (if available)
        pred_vel = pred_trajectory[:,:,4:7]
        pred_attitude = pred_trajectory[:, :, 7:10]  # (B, T, 3) - attitude (roll/pitch/yaw)
        gt_attitude = gt_trajectory[:, :, 7:10]

        # Position losses: Per-dimension MSE
        x_loss = self.mse_loss(pred_pos[:, :, 0], gt_pos[:, :, 0])
        y_loss = self.mse_loss(pred_pos[:, :, 1], gt_pos[:, :, 1])
        # Z loss with extra weight for height accuracy
        z_loss = self.mse_loss(pred_pos[:, :, 2], gt_pos[:, :, 2])

        # Last time-step losses (higher weight for endpoint accuracy)
        last_x_loss = self.mse_loss(pred_pos[:, -1, 0], gt_pos[:, -1, 0])
        last_y_loss = self.mse_loss(pred_pos[:, -1, 1], gt_pos[:, -1, 1])
        last_z_loss = self.mse_loss(pred_pos[:, -1, 2], gt_pos[:, -1, 2])

        last_xyz_loss = last_x_loss + last_y_loss + last_z_loss
        position_loss = x_loss + y_loss + z_loss

        if seq_len > 1:
            # Per-dimension MSE on velocities
            vel_x_loss = self.mse_loss(pred_vel[:, :, 0], gt_vel[:, :, 0])
            vel_y_loss = self.mse_loss(pred_vel[:, :, 1], gt_vel[:, :, 1])
            vel_z_loss = self.mse_loss(pred_vel[:, :, 2], gt_vel[:, :, 2])
            vel_loss = vel_x_loss + vel_y_loss + vel_z_loss
        else:
            # No velocity if single timestep
            vel_loss = torch.tensor(0.0, device=device)

        dynamics_consistency_loss = torch.tensor(0.0, device=device)
        if seq_len > 1 and mean is not None and std is not None:
            pos_std = std[0, 0, 1:4].to(device=device, dtype=pred_trajectory.dtype)
            pos_mean = mean[0, 0, 1:4].to(device=device, dtype=pred_trajectory.dtype)
            vel_std = std[0, 0, 4:7].to(device=device, dtype=pred_trajectory.dtype)
            vel_mean = mean[0, 0, 4:7].to(device=device, dtype=pred_trajectory.dtype)
            pred_pos_denorm = pred_pos * pos_std + pos_mean
            pred_vel_denorm = pred_vel * vel_std + vel_mean
            dynamics_residual = (
                pred_pos_denorm[:, 1:]
                - pred_pos_denorm[:, :-1]
                - self.config.delta_T * pred_vel_denorm[:, :-1]
            )
            dynamics_consistency_loss = 0.5 * dynamics_residual.square().mean()

        if seq_len >= 3:
            # vel = (B, T-1, 3)
            vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
            # acc = (B, T-2, 3) - acceleration
            acc = vel[:, 1:, :] - vel[:, :-1, :]
            # Smoothness loss = mean squared acceleration
            acc_smoothness = acc.pow(2).mean()

        # Other losses: Explicit speed and attitude (no overlap with position)
        speed_loss = self.mse_loss(pred_speed, gt_speed)
        attitude_loss = self.mse_loss(pred_attitude, gt_attitude)
        other_loss = speed_loss + attitude_loss

        # Obstacle loss: Computed only if enabled and params provided
        obstacle_loss = torch.tensor(0.0, device=device)
        if self.enable_obstacle_term and obstacles_data is not None and mean is not None and std is not None:
            obstacle_loss = self.compute_obstacle_distance_loss(pred_trajectory, obstacles_data, mean, std)

        # New: Continuity loss (MSE between last history and first pred timestep)
        continuity_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if history is not None and pred_trajectory.size(1) > 0:
            # Focus on position components (indices 1:4) for smoothness
            last_history_pos = history[:, -1, 1:4]
            first_pred_pos = pred_trajectory[:, 0, 1:4]
            continuity_loss = self.mse_loss(first_pred_pos, last_history_pos)

        # Total weighted loss
        total_loss = self.last_xyz_weight * last_xyz_loss + self.xyz_weight * position_loss + self.vel_weight * vel_loss + self.other_weight * other_loss + self.obstacle_weight * obstacle_loss + self.continuity_weight * continuity_loss + self.acc_weight * acc_smoothness + self.dynamics_consistency_weight * dynamics_consistency_loss

        return total_loss, position_loss, vel_loss, obstacle_loss, continuity_loss
