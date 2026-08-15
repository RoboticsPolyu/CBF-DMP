"""Barrier objectives and analytic guidance gradients."""

import torch
import torch.nn.functional as F

def compute_barrier_and_grad(x, mean, std, obstacles_data=None, safety_margin = 0.20):
    """
    Compute barrier V and its gradient ∇V for the trajectory x.
    Fixed version with proper gradient computation.
    """
    # Denormalize positions for barrier computation
    pos_denorm = x[:, :, 1:4] * std[0, 0, 1:4] + mean[0, 0, 1:4]
    batch_size, seq_len, _ = pos_denorm.shape

    # Initialize gradient tensor for denormalized positions
    grad_pos_denorm = torch.zeros_like(pos_denorm)

    pos_denorm = pos_denorm.clone().detach()

    # Initialize barrier value
    V_total = torch.zeros(batch_size, device=x.device)

    # Process each obstacle
    if obstacles_data is not None:
        for batch_idx in range(batch_size):
            # Get obstacles for this sample
            batch_obs = obstacles_data[batch_idx] if batch_idx < len(obstacles_data) else []

            for obstacle in batch_obs:
                center = obstacle['center'].to(x.device)
                radius = obstacle['radius'] + safety_margin  # Add safety margin

                # Euclidean distances
                distances = torch.norm(pos_denorm[batch_idx] - center.unsqueeze(0), dim=1)

                # Closeness function: r - distance
                h = radius * radius - distances * distances
                # h = (p-o)^T*(p-o) - r^2
                # P_free(x) = Φ(z) = Φ( h(x) / σ_h(x) )
                # log(P_free) = log(Φ(z)) ; ∇log(P_free) = ( φ(z) / Φ(z) ) * ∇h / σ_h
                # φ(z) = exp(-0.5*z^2) / sqrt(2π)

                # Barrier violation term: max(0, h)
                violation = torch.clamp(h, min=0.0)
                V_obstacle = torch.sum(violation) # sum( min(0, r^2 - d^2  ) r^2 - d^2 = r^2 - ||p - c||^2
                V_total[batch_idx] += V_obstacle

                # Compute gradient for this obstacle
                violation_mask = (h > 0).float().unsqueeze(1)
                direction_vec = pos_denorm[batch_idx] - center.unsqueeze(0)

                epsilon = 1e-6
                grad_dist = direction_vec / (distances.unsqueeze(1) + epsilon)
                grad_V_obs = -2 * violation.unsqueeze(1) * grad_dist

                grad_pos_denorm[batch_idx] = grad_pos_denorm[batch_idx] + grad_V_obs

    # Map gradient back to normalized space
    grad_x = torch.zeros_like(x, device=x.device)
    std_scaled = std[0, 0, 1:4].to(x.device)
    grad_x[:, :, 1:4] = grad_pos_denorm / std_scaled

    # V_total is sum of barrier violations
    V_avg = V_total.mean()

    return V_avg, grad_x

def compute_barrier_and_grad_logistic(
        x, mean, std, obstacles_data=None, safety_margin=0.20, sigma=0.5,
        delta_t=0.1, cbf_decay_rate=1.0,
        dynamics_consistency_weight=1.0):
    """
    Compute the velocity-aware DCBF barrier and its gradient.

    The SafeDiff discrete-time CBF for a static spherical obstacle is:
        h_v = gamma * (||p-c||^2 - R_s^2)
              + 2 * dt * (p-c)^T v + dt^2 * ||v||^2
        P_safe = sigmoid(h_v / sigma), V = -log(P_safe)

    Here R_s = radius * (1 + safety_margin), because ``safety_margin`` is
    configured as a fraction of obstacle radius. The returned gradient is with
    respect to the normalized state, including both position and velocity.

    Args:
        x: Normalized trajectory tensor (batch, seq_len, state_dim)
        mean: Normalization mean tensor
        std: Normalization std tensor
        obstacles_data: List of obstacle dictionaries for each batch sample
        safety_margin: Safety buffer as a fraction of obstacle radius
        sigma: Smoothness parameter for logistic function (controls transition sharpness)
        delta_t: Physical trajectory sampling period
        cbf_decay_rate: Discrete-time CBF gamma in (0, 1]

    Returns:
        V_avg: Average barrier violation value (scalar tensor)
        grad_x: Gradient of barrier w.r.t normalized trajectory (same shape as x)
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if delta_t <= 0:
        raise ValueError("delta_t must be positive")
    if not 0 < cbf_decay_rate <= 1:
        raise ValueError("cbf_decay_rate must be in (0, 1]")
    if dynamics_consistency_weight < 0:
        raise ValueError("dynamics_consistency_weight must be non-negative")

    # State layout: speed(0), position(1:4), velocity(4:7), attitude(7:10).
    pos_std = std[0, 0, 1:4].to(device=x.device, dtype=x.dtype)
    pos_mean = mean[0, 0, 1:4].to(device=x.device, dtype=x.dtype)
    vel_std = std[0, 0, 4:7].to(device=x.device, dtype=x.dtype)
    vel_mean = mean[0, 0, 4:7].to(device=x.device, dtype=x.dtype)
    pos_denorm = x[:, :, 1:4] * pos_std + pos_mean
    vel_denorm = x[:, :, 4:7] * vel_std + vel_mean
    batch_size = pos_denorm.shape[0]

    grad_pos_denorm = torch.zeros_like(pos_denorm)
    grad_vel_denorm = torch.zeros_like(vel_denorm)
    grad_vel_dynamics_denorm = torch.zeros_like(vel_denorm)
    V_total = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

    if obstacles_data is not None:
        for batch_idx in range(batch_size):
            batch_obs = obstacles_data[batch_idx] if batch_idx < len(obstacles_data) else []
            for obstacle in batch_obs:
                center = obstacle['center'].to(device=x.device, dtype=x.dtype)
                radius = torch.as_tensor(
                    obstacle['radius'], device=x.device, dtype=x.dtype)
                safety_radius = radius * (1.0 + safety_margin)

                delta = pos_denorm[batch_idx] - center.unsqueeze(0)
                velocity = vel_denorm[batch_idx]
                h_v = (
                    cbf_decay_rate
                    * (torch.sum(delta.square(), dim=1) - safety_radius.square())
                    + 2.0 * delta_t * torch.sum(delta * velocity, dim=1)
                    + delta_t ** 2 * torch.sum(velocity.square(), dim=1)
                )

                # softplus(-z) is a stable implementation of -log(sigmoid(z)).
                logit_safe = h_v / sigma
                V_total[batch_idx] += F.softplus(-logit_safe).sum()

                dV_dh = -(1.0 - torch.sigmoid(logit_safe)) / sigma
                dh_dp = 2.0 * cbf_decay_rate * delta + 2.0 * delta_t * velocity
                dh_dv = 2.0 * delta_t * delta + 2.0 * delta_t ** 2 * velocity
                grad_pos_denorm[batch_idx] += dV_dh.unsqueeze(1) * dh_dp
                grad_vel_denorm[batch_idx] += dV_dh.unsqueeze(1) * dh_dv

    # Position-velocity dynamics consistency:
    # p[k+1] = p[k] + delta_t * v[k]. This term is averaged over all
    # transitions and coordinates, so its scale is independent of seq_len.
    if pos_denorm.shape[1] > 1 and dynamics_consistency_weight > 0:
        for batch_idx in range(batch_size):
            dynamics_residual = (
                pos_denorm[batch_idx, 1:]
                - pos_denorm[batch_idx, :-1]
                - delta_t * vel_denorm[batch_idx, :-1]
            )
            dynamics_scale = (
                dynamics_consistency_weight / dynamics_residual.numel()
            )
            V_total[batch_idx] += (
                0.5 * dynamics_consistency_weight
                * dynamics_residual.square().mean()
            )
            grad_pos_denorm[batch_idx, :-1] -= (
                dynamics_scale * dynamics_residual
            )
            grad_pos_denorm[batch_idx, 1:] += (
                dynamics_scale * dynamics_residual
            )
            grad_vel_dynamics_denorm[batch_idx, :-1] -= (
                dynamics_scale * delta_t * dynamics_residual
            )

    # If y = x_norm * std + mean, then dV/dx_norm = dV/dy * std.
    grad_x = torch.zeros_like(x)
    grad_x[:, :, 1:4] = grad_pos_denorm * pos_std
    # Keep the existing CBF velocity-guidance setting unchanged; only the new
    # dynamics-consistency gradient is applied to the velocity channels.
    grad_x[:, :, 4:7] = grad_vel_dynamics_denorm * vel_std
    return V_total.mean(), grad_x
