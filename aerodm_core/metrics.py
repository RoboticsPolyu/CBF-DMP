"""Geometry and quantitative evaluation metrics."""

import numpy as np

def get_collision_mask(trajectory_pos, obstacles, safety_margin=0.0):
    """Return a Boolean mask for samples lying inside any spherical obstacle."""
    trajectory_pos = np.asarray(trajectory_pos)
    collision_mask = np.zeros(len(trajectory_pos), dtype=bool)
    if not obstacles:
        return collision_mask

    for obstacle in obstacles:
        center = obstacle['center']
        if hasattr(center, 'detach'):
            center = center.detach()
        if hasattr(center, 'cpu'):
            center = center.cpu()
        center = np.asarray(center, dtype=float)

        radius = obstacle['radius']
        if hasattr(radius, 'detach'):
            radius = radius.detach()
        if hasattr(radius, 'cpu'):
            radius = radius.cpu()
        radius = float(np.asarray(radius)) + safety_margin

        collision_mask |= np.linalg.norm(trajectory_pos - center, axis=1) < radius

    return collision_mask

def compute_collision_rate(trajectories_pos, obstacles_data, safety_margin=0.0):
    """Compute collision rate for a batch of trajectories."""
    batch_size = trajectories_pos.shape[0]

    if hasattr(trajectories_pos, 'cpu'):
        trajectories_pos = trajectories_pos.cpu().numpy()

    collision_counts = np.zeros(batch_size)
    trajectories_with_collision_count = 0

    for batch_idx in range(batch_size):
        batch_obs = obstacles_data[batch_idx] if batch_idx < len(obstacles_data) else []
        traj_pos = trajectories_pos[batch_idx]
        has_collision = False

        for t in range(len(traj_pos)):
            point = traj_pos[t]
            for obstacle in batch_obs:
                center = obstacle['center']
                if hasattr(center, 'cpu'):
                    center = center.cpu().numpy()
                radius = obstacle['radius'] + safety_margin

                dist = np.linalg.norm(point - center)
                if dist < radius:
                    collision_counts[batch_idx] += 1
                    has_collision = True
                    break

        if has_collision:
            trajectories_with_collision_count += 1

        collision_counts[batch_idx] = collision_counts[batch_idx] / len(traj_pos)

    collision_rate = trajectories_with_collision_count / batch_size
    avg_collision_percentage = np.mean(collision_counts) * 100

    return {
        'collision_rate': collision_rate,
        'avg_collision_percentage': avg_collision_percentage,
        'trajectories_with_collision': trajectories_with_collision_count,
        'total_trajectories': batch_size
    }

def compute_success_rates(trajectories_pos, ground_truth_positions, target_positions, obstacles_data):
    """
    Compute comprehensive success rates based on multiple criteria.

    Success definitions:
    1. Collision-free: Trajectory has no collisions with obstacles
    2. Trajectory accuracy: Position error < 2.0 (normalized units)
    3. Final point accuracy: Final position error < 1.0
    4. Target reach: Distance to target < 2.0 at final step
    5. Overall success: Meets all criteria
    """
    batch_size = trajectories_pos.shape[0]

    if hasattr(trajectories_pos, 'cpu'):
        trajectories_pos = trajectories_pos.cpu().numpy()
    if hasattr(ground_truth_positions, 'cpu'):
        ground_truth_positions = ground_truth_positions.cpu().numpy()
    if hasattr(target_positions, 'cpu'):
        target_positions = target_positions.cpu().numpy()

    # Success criteria thresholds (can be adjusted)
    criteria = {
        'max_position_error': 2.0,      # Max allowed position error
        'max_final_error': 1.0,          # Max allowed final position error
        'target_reach_radius': 2.0,      # Distance to target considered "reached"
        'min_safety_distance': 0.3       # Min distance to obstacle surface
    }

    # Initialize success flags
    collision_free = np.ones(batch_size, dtype=bool)
    trajectory_accurate = np.ones(batch_size, dtype=bool)
    final_point_accurate = np.ones(batch_size, dtype=bool)
    target_reached = np.ones(batch_size, dtype=bool)
    safe_distance_maintained = np.ones(batch_size, dtype=bool)

    # Detailed metrics storage
    max_errors = []
    final_errors = []
    target_distances = []
    min_surface_distances = []
    collision_percentages = []

    for batch_idx in range(batch_size):
        traj_pos = trajectories_pos[batch_idx]
        gt_pos = ground_truth_positions[batch_idx]
        target_pos = target_positions[batch_idx]

        # 1. Collision check and safety distance
        batch_obs = obstacles_data[batch_idx] if batch_idx < len(obstacles_data) else []
        has_collision = False
        min_surface_dist = float('inf')
        collision_steps = 0

        for t in range(len(traj_pos)):
            point = traj_pos[t]
            for obstacle in batch_obs:
                center = obstacle['center']
                if hasattr(center, 'cpu'):
                    center = center.cpu().numpy()
                radius = obstacle['radius']

                dist_to_center = np.linalg.norm(point - center)
                surface_dist = dist_to_center - radius
                min_surface_dist = min(min_surface_dist, surface_dist)

                if dist_to_center < radius:
                    has_collision = True
                    collision_steps += 1
                    break

        collision_free[batch_idx] = not has_collision
        safe_distance_maintained[batch_idx] = min_surface_dist >= criteria['min_safety_distance']
        min_surface_distances.append(min_surface_dist)
        collision_percentages.append(collision_steps / len(traj_pos) * 100)

        # 2. Trajectory accuracy (max position error)
        timestep_errors = np.linalg.norm(traj_pos - gt_pos, axis=1)
        max_error = np.max(timestep_errors)
        max_errors.append(max_error)
        trajectory_accurate[batch_idx] = max_error < criteria['max_position_error']

        # 3. Final point accuracy
        final_error = np.linalg.norm(traj_pos[-1] - gt_pos[-1])
        final_errors.append(final_error)
        final_point_accurate[batch_idx] = final_error < criteria['max_final_error']

        # 4. Target reach
        dist_to_target = np.linalg.norm(traj_pos[-1] - target_pos)
        target_distances.append(dist_to_target)
        target_reached[batch_idx] = dist_to_target < criteria['target_reach_radius']

    # Compute overall success (meets all criteria)
    overall_success = (
        collision_free &
        trajectory_accurate &
        final_point_accurate &
        target_reached
    )

    # Calculate success rates
    results = {
        'collision_free_rate': np.mean(collision_free) * 100,
        'trajectory_accuracy_rate': np.mean(trajectory_accurate) * 100,
        'final_point_accuracy_rate': np.mean(final_point_accurate) * 100,
        'target_reach_rate': np.mean(target_reached) * 100,
        'safety_distance_rate': np.mean(safe_distance_maintained) * 100,
        'overall_success_rate': np.mean(overall_success) * 100,
        'detailed_metrics': {
            'mean_max_error': np.mean(max_errors),
            'std_max_error': np.std(max_errors),
            'mean_final_error': np.mean(final_errors),
            'std_final_error': np.std(final_errors),
            'mean_target_distance': np.mean(target_distances),
            'std_target_distance': np.std(target_distances),
            'mean_min_surface_dist': np.mean(min_surface_distances),
            'std_min_surface_dist': np.std(min_surface_distances),
            'mean_collision_percentage': np.mean(collision_percentages),
            'collision_free_count': np.sum(collision_free),
            'total_trajectories': batch_size
        }
    }

    return results

def compute_trajectory_errors(pred_trajectory, gt_trajectory):
    """Compute trajectory prediction errors."""
    if hasattr(pred_trajectory, 'cpu'):
        pred_trajectory = pred_trajectory.cpu().numpy()
    if hasattr(gt_trajectory, 'cpu'):
        gt_trajectory = gt_trajectory.cpu().numpy()

    batch_size, seq_len, _ = pred_trajectory.shape
    timestep_errors = np.linalg.norm(pred_trajectory - gt_trajectory, axis=2)

    mean_ae = np.mean(timestep_errors)
    std_ae = np.std(timestep_errors)
    rmse = np.sqrt(np.mean(timestep_errors ** 2))

    final_point_error = np.linalg.norm(pred_trajectory[:, -1, :] - gt_trajectory[:, -1, :], axis=1)
    mean_final_error = np.mean(final_point_error)
    std_final_error = np.std(final_point_error)

    axis_errors = {
        'x': np.mean(np.abs(pred_trajectory[:, :, 0] - gt_trajectory[:, :, 0])),
        'y': np.mean(np.abs(pred_trajectory[:, :, 1] - gt_trajectory[:, :, 1])),
        'z': np.mean(np.abs(pred_trajectory[:, :, 2] - gt_trajectory[:, :, 2]))
    }

    return {
        'mean_ae': mean_ae,
        'std_ae': std_ae,
        'rmse': rmse,
        'mean_final_error': mean_final_error,
        'std_final_error': std_final_error,
        'axis_errors': axis_errors,
        'max_error': np.max(timestep_errors),
        'percentile_95': np.percentile(timestep_errors, 95)
    }
