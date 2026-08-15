"""Trajectory normalization, targets, and obstacle generation utilities."""

import numpy as np
import torch

def normalize_trajectories(trajectories, mean=None, std=None):
    """
    Normalize all dimensions to zero mean and unit variance,
    but then restore the original style dimension values.

    Args:
        trajectories: Shape (batch, seq_len, state_dim) where state_dim includes style

    Returns:
        normalized_trajectories: State dimensions normalized, style restored to original
        mean: Mean for ALL dimensions (including style)
        std: Std for ALL dimensions (including style)
    """
    # Save original style values (last dimension)
    original_style = trajectories[:, :, -1:]  # Shape: (batch, seq_len, 1)

    if mean is None:
            # Normalize everything (including style)
            mean = trajectories.mean(dim=(0, 1), keepdim=True)
    if std is None:
            std = trajectories.std(dim=(0, 1), keepdim=True)
            std = torch.where(std < 1e-8, torch.ones_like(std), std)

    normalized = (trajectories - mean) / std

    # Restore original style values (overwrite the normalized style dimension)
    normalized[:, :, -1:] = original_style

    return normalized, mean, std

def denormalize_trajectories(trajectories_norm, mean, std):
    return trajectories_norm * std + mean

def normalize_obstacle(obstacle_center, mean, std):
    """Normalize obstacle center using the same normalization parameters"""
    # Extract position normalization parameters (indices 1:4 for x,y,z)
    pos_mean = mean[0, 0, 1:4].cpu().numpy()
    pos_std = std[0, 0, 1:4].cpu().numpy()
    # Normalize obstacle center
    obstacle_norm = (obstacle_center - pos_mean) / pos_std
    return obstacle_norm

def denormalize_obstacle(obstacle_center_norm, mean, std):
    """Denormalize obstacle center using the same normalization parameters"""
    # Extract position normalization parameters (indices 1:4 for x,y,z)
    pos_mean = mean[0, 0, 1:4].cpu().numpy()
    pos_std = std[0, 0, 1:4].cpu().numpy()
    # Denormalize obstacle center
    obstacle_denorm = obstacle_center_norm * pos_std + pos_mean
    return obstacle_denorm

def generate_target_waypoints(trajectory):
    # Target is the final position (indices 1:4 for x, y, z)
    target_pos = trajectory[:, -1, 1:4]  # (batch, 3)

    # Add validity flag (1 = valid target)
    valid_flag = torch.ones(target_pos.shape[0], 1, device=target_pos.device)

    # Concatenate: [x, y, z, valid]
    return torch.cat([target_pos, valid_flag], dim=-1)

def add_target_noise(target, bound=1.0):
    """
    Add uniform noise to the target waypoint's position components (x, y, z).

    Args:
        target: Tensor of shape (batch_size, 4) where last dimension = [x, y, z, valid_flag]
        bound: Float, noise range is (-bound, +bound) for each coordinate

    Returns:
        target_noisy: Tensor with same shape, where x, y, z have added noise
    """
    # Generate uniform noise in range [-bound, bound] for x, y, z
    noise = (torch.rand_like(target[:, :3]) * 2 - 1) * bound  # Shape: (batch_size, 3)

    # Add noise to position components
    target_noisy = target.clone()
    target_noisy[:, :3] = target_noisy[:, :3] + noise

    return target_noisy

def project_target_outside_obstacles(target, obstacles, safety_margin=0.2,
                                     clearance=1e-4):
    """Project target positions onto obstacle safety surfaces when necessary.

    The safety surface of obstacle ``i`` has radius
    ``radius_i * (1 + safety_margin)``. The target validity flag and any other
    non-position components are preserved.
    """
    projected_target = target.clone()
    if not obstacles:
        return projected_target

    flat_targets = projected_target.reshape(-1, projected_target.shape[-1])

    for target_row in flat_targets:
        position = target_row[:3]
        original_position = position.clone()
        centers = [
            obstacle['center'].to(
                device=position.device, dtype=position.dtype)
            for obstacle in obstacles
        ]
        safe_radii = [
            float(obstacle['radius']) * (1.0 + safety_margin)
            for obstacle in obstacles
        ]

        violations = [
            safe_radius - torch.linalg.norm(original_position - center).item()
            for center, safe_radius in zip(centers, safe_radii)
        ]
        if max(violations) <= 0.0:
            continue

        # Use the outward normal of the most deeply violated safety sphere.
        deepest_idx = int(np.argmax(violations))
        direction = original_position - centers[deepest_idx]
        direction_norm = torch.linalg.norm(direction)
        if direction_norm <= 1e-12:
            # At a sphere center, point away from the obstacle cluster. If the
            # cluster is symmetric, use a deterministic axis as a fallback.
            mean_center = torch.stack(centers).mean(dim=0)
            direction = centers[deepest_idx] - mean_center
            direction_norm = torch.linalg.norm(direction)
            if direction_norm <= 1e-12:
                direction = torch.zeros_like(original_position)
                direction[0] = 1.0
                direction_norm = torch.tensor(
                    1.0, device=position.device, dtype=position.dtype)
        direction = direction / direction_norm

        # Find the farthest positive sphere exit along this ray. Moving just
        # beyond it puts the target outside every safety sphere intersected by
        # the ray, including overlapping expanded safety regions.
        farthest_exit = torch.zeros(
            (), device=position.device, dtype=position.dtype)
        for center, safe_radius in zip(centers, safe_radii):
            relative = original_position - center
            ray_projection = torch.dot(direction, relative)
            discriminant = (
                ray_projection.square()
                - (torch.dot(relative, relative) - safe_radius ** 2)
            )
            if discriminant >= 0.0:
                exit_distance = -ray_projection + torch.sqrt(discriminant)
                if exit_distance > farthest_exit:
                    farthest_exit = exit_distance

        position.copy_(
            original_position + direction * (farthest_exit + clearance))

    return projected_target

def generate_random_obstacles(trajectory, num_obstacles_range=(1, 5), radius_range=(0.5, 2.0), check_collision=True, device='cpu'):
    """
    Generate spherical obstacles that intersect the reference trajectory.

    Every accepted obstacle contains at least one sample from the second half
    of the reference trajectory.
    When ``check_collision`` is True, accepted obstacles also satisfy
    distance(center_i, center_j) >= radius_i + radius_j.
    Returns list of obstacle dictionaries with center and radius.
    """
    # Randomly determine the number of obstacles to generate
    num_obstacles = np.random.randint(num_obstacles_range[0], num_obstacles_range[1] + 1)
    obstacles = []

    # Extract trajectory positions (x, y, z) and move to CPU for numpy operations
    traj_pos = trajectory[:, 1:4].cpu().numpy()
    if len(traj_pos) == 0:
        raise ValueError("Cannot generate obstacles for an empty trajectory.")

    start_idx = 30
    end_idx = len(traj_pos) - 15

    # print(f"Generating {num_obstacles} random non-colliding obstacles around trajectory")
    for i in range(num_obstacles):
        attempts = 0
        max_attempts = 100 # Prevent infinite loop in crowded spaces
        valid_placement = False

        while not valid_placement and attempts < max_attempts:
            # Choose a point from the latter half of the reference trajectory,
            # then place the center strictly less than one radius away so the
            # sphere is guaranteed to intersect that part of the trajectory.
            reference_idx = np.random.randint(start_idx, end_idx)
            reference_point = traj_pos[reference_idx]
            radius = np.random.uniform(radius_range[0], radius_range[1])
            direction = np.random.normal(size=3)
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-12:
                attempts += 1
                continue
            direction /= direction_norm
            offset_distance = np.random.uniform(0.0, 0.8 * radius)
            center = reference_point + direction * offset_distance

            valid_placement = True
            if check_collision:
                for prev_obstacle in obstacles:
                    prev_center = prev_obstacle['center'].cpu().numpy()
                    prev_radius = prev_obstacle['radius']
                    dist = np.linalg.norm(center - prev_center)
                    if dist < radius + prev_radius:
                        valid_placement = False
                        break
            attempts += 1

        if not valid_placement:
            print(
                f"Warning: Could not place obstacle {i} intersecting the "
                f"reference trajectory without obstacle overlap after "
                f"{max_attempts} attempts. Skipping."
            )
            continue

        # Create obstacle dictionary with proper tensor
        obstacle = {
            'center': torch.tensor(center, dtype=torch.float32, device=device),
            'radius': float(radius),  # Ensure radius is a float, not tensor
            'id': i
        }
        obstacles.append(obstacle)
        # print(f"Placed Obstacle {i}: center={center}, radius={radius:.3f}")

    return obstacles

def denormalize_target(target_norm, mean, std):
    # Extract position normalization parameters (indices 1:4 for x,y,z)
    pos_mean = mean[0, 0, 1:4]  # Shape: (3,)
    pos_std = std[0, 0, 1:4]    # Shape: (3,)

    # Extract x,y,z from target
    target_pos = target_norm[..., :3]  # Shape: (batch, 3)
    target_valid = target_norm[..., 3:]  # Shape: (batch, 1)

    # Denormalize only the position part
    target_pos_denorm = target_pos * pos_std + pos_mean

    # Concatenate with original valid flag
    return torch.cat([target_pos_denorm, target_valid], dim=-1)

def normalize_target(target_denorm, mean, std):
    """
    Normalize target waypoint (only first 3 dimensions: x, y, z).
    Preserves the 4th dimension (valid_flag) unchanged.

    Args:
        target_denorm: Denormalized target tensor of shape (batch, 4)
                       where last dim is [x, y, z, valid_flag]
        mean: Mean tensor of shape (1, 1, state_dim)
        std: Std tensor of shape (1, 1, state_dim)

    Returns:
        target_norm: Normalized target tensor of shape (batch, 4)
                     with x,y,z normalized and valid_flag preserved
    """
    # Extract position normalization parameters (indices 1:4 for x,y,z)
    pos_mean = mean[0, 0, 1:4]  # Shape: (3,)
    pos_std = std[0, 0, 1:4]    # Shape: (3,)

    # Extract x,y,z from target
    target_pos = target_denorm[..., :3]  # Shape: (batch, 3)
    target_valid = target_denorm[..., 3:]  # Shape: (batch, 1)

    # Normalize only the position part
    target_pos_norm = (target_pos - pos_mean) / pos_std

    # Concatenate with original valid flag
    return torch.cat([target_pos_norm, target_valid], dim=-1)
