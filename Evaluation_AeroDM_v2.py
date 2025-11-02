# VerifyAeroDM_CBF_OBSMLP_v2.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict

# Import your model and utility functions
from AeroDM_SafeTrj_v2 import (
    Config, AeroDM,
    generate_aerobatic_trajectories,
    normalize_trajectories, denormalize_trajectories,
    generate_target_waypoints, generate_action_styles, 
    generate_history_segments, generate_random_obstacles
)

def load_model(checkpoint_path, device):
    config = Config()
    model = AeroDM(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # Move diffusion process tensors to device
    model.diffusion_process.betas = model.diffusion_process.betas.to(device)
    model.diffusion_process.alphas = model.diffusion_process.alphas.to(device)
    model.diffusion_process.alpha_bars = model.diffusion_process.alpha_bars.to(device)
    return model

def plot_obstacle_aware_trajectory_comparison(
    original_denorm, unguided_denorm, guided_denorm,
    history=None, target=None,
    obstacles: List[Dict] = None,
    title="Trajectory Comparison with Multiple Obstacles",
    num_subplots=1
):
    """
    Plot 3D trajectory comparison with multiple spherical obstacles.
    Imitates plot_trajectory_comparison but supports multiple obstacles.
    """
    fig = plt.figure(figsize=(15, 5 * num_subplots))
    fig.suptitle(title, fontsize=16)

    # Extract positions
    original_pos = original_denorm[0, :, 1:4].cpu().numpy()
    unguided_pos = unguided_denorm[0, :, 1:4].cpu().numpy()
    guided_pos = guided_denorm[0, :, 1:4].cpu().numpy()

    if history is not None:
        history_pos = history[0, :, 1:4].cpu().numpy()
    else:
        history_pos = None

    # Fixed bounds for visualization
    fixed_min = np.array([-20.0, -20.0, -20.0])
    fixed_max = np.array([20.0, 20.0, 20.0])

    # 3D Plot
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], 'b-', label='Original', linewidth=2)
    ax1.plot(unguided_pos[:, 0], unguided_pos[:, 1], unguided_pos[:, 2], 'orange', label='Unguided Sample', linewidth=2)
    ax1.plot(guided_pos[:, 0], guided_pos[:, 1], guided_pos[:, 2], 'g-', label='Guided Sample', linewidth=2)

    if history_pos is not None:
        ax1.plot(history_pos[:, 0], history_pos[:, 1], history_pos[:, 2], 'gray', linestyle='--', label='History', alpha=0.5)

    # Plot multiple obstacles as spheres
    if obstacles:
        for obstacle in obstacles:
            center = obstacle['center'].cpu().numpy()
            radius = obstacle['radius']
            # Simple sphere approximation with wireframe
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, color='red', alpha=0.3)

    ax1.set_xlim(fixed_min[0], fixed_max[0])
    ax1.set_ylim(fixed_min[1], fixed_max[1])
    ax1.set_zlim(fixed_min[2], fixed_max[2])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('3D Trajectories with Obstacles')
    ax1.grid(True)

    # X-Y Projection
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(original_pos[:, 0], original_pos[:, 1], 'b-', label='Original')
    ax2.plot(unguided_pos[:, 0], unguided_pos[:, 1], 'orange', label='Unguided')
    ax2.plot(guided_pos[:, 0], guided_pos[:, 1], 'g-', label='Guided')
    if history_pos is not None:
        ax2.plot(history_pos[:, 0], history_pos[:, 1], 'gray', linestyle='--', label='History', alpha=0.5)
    if obstacles:
        for obstacle in obstacles:
            center = obstacle['center'].cpu().numpy()
            circle = plt.Circle((center[0], center[1]), obstacle['radius'], color='red', fill=False, alpha=0.5)
            ax2.add_patch(circle)
    if target is not None:
        target_x = target[0, 0].cpu().item()
        target_y = target[0, 1].cpu().item()
        ax2.scatter(target_x, target_y, color='purple', s=100, marker='*', label='Target')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.set_title('X-Y Projection')
    ax2.grid(True)
    ax2.set_xlim(fixed_min[0], fixed_max[0])
    ax2.set_ylim(fixed_min[1], fixed_max[1])

    # Distance to nearest obstacle over time
    ax3 = fig.add_subplot(1, 3, 3)
    time_steps = np.arange(len(original_pos))
    if obstacles:
        # Compute min distance for each trajectory
        min_dist_original = [min([np.linalg.norm(original_pos[t] - obs['center'].cpu().numpy()) - obs['radius'] 
                                  for obs in obstacles]) for t in time_steps]
        min_dist_unguided = [min([np.linalg.norm(unguided_pos[t] - obs['center'].cpu().numpy()) - obs['radius'] 
                                  for obs in obstacles]) for t in time_steps]
        min_dist_guided = [min([np.linalg.norm(guided_pos[t] - obs['center'].cpu().numpy()) - obs['radius'] 
                                for obs in obstacles]) for t in time_steps]

        ax3.plot(time_steps, min_dist_original, 'b-', label='Original Min Dist')
        ax3.plot(time_steps, min_dist_unguided, 'orange', label='Unguided Min Dist')
        ax3.plot(time_steps, min_dist_guided, 'g-', label='Guided Min Dist')
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Collision Threshold')
    else:
        ax3.text(0.5, 0.5, 'No Obstacles', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Min Distance to Obstacle')
    ax3.legend()
    ax3.set_title('Obstacle Avoidance Performance')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

def visualize_samples(model, mean, std, num_samples=3):
    device = next(model.parameters()).device
    config = model.config

    # Generate test trajectories
    test_trajectories = generate_aerobatic_trajectories(
        num_trajectories=num_samples,
        seq_len=config.seq_len + config.history_len
    ).to(device)

    # Normalize
    test_trajectories_norm, _, _ = normalize_trajectories(test_trajectories)
    test_trajectories_norm = test_trajectories_norm.to(device)

    for i in range(num_samples):
        full_traj = test_trajectories_norm[i:i+1]
        target = generate_target_waypoints(full_traj)
        action = generate_action_styles(1, config.action_dim, device=device)
        history = generate_history_segments(full_traj, config.history_len, device=device)
        x_0 = full_traj[:, config.history_len:, :]

        # Generate random obstacles for this sample
        traj_denorm = denormalize_trajectories(full_traj, mean, std)
        obstacles = generate_random_obstacles(
            traj_denorm[0],
            num_obstacles_range=(1, 5),  # 1-5 obstacles for visualization
            radius_range=(0.5, 1.5),
            device=device
        )

        model.set_normalization_params(mean, std)
        # Set obstacles data for model input
        model.set_obstacles_data(obstacles)

        # Denormalize target
        target_denorm = target * std[0, 0, 1:4] + mean[0, 0, 1:4]
        history_denorm = denormalize_trajectories(history, mean, std) if history is not None else None

        # Sample unguided (no guidance, but obstacles for model input)
        model.config.enable_cbf_guidance = False
        sampled_unguided = model.sample(target, action, history, batch_size=1, enable_guidance=False)
        sampled_unguided_denorm = denormalize_trajectories(sampled_unguided, mean, std)

        # Sample guided (with CBF guidance and obstacles)
        model.config.enable_cbf_guidance = True
        sampled_guided = model.sample(target, action, history, batch_size=1, enable_guidance=True, guidance_gamma=config.guidance_gamma)
        sampled_guided_denorm = denormalize_trajectories(sampled_guided, mean, std)

        # Denormalize original
        x_0_denorm = denormalize_trajectories(x_0, mean, std)

        plot_obstacle_aware_trajectory_comparison(
            x_0_denorm, sampled_unguided_denorm, sampled_guided_denorm,
            history=history_denorm, target=target_denorm,
            obstacles=obstacles,
            title=f"Visualization Sample {i+1} (Guided: Green avoids Red Obstacles)"
        )

if __name__ == "__main__":
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Path to your trained model checkpoint
    checkpoint_path = "model/enhanced_obstacle_aware_aerodm.pth"

    # Load model
    print("Loading trained Enhanced Obstacle-Aware AeroDM model...")
    model = load_model(checkpoint_path, device)

    # Load mean and std from checkpoint or recompute from training data
    checkpoint = torch.load(checkpoint_path, map_location=device)
    mean = checkpoint.get('mean', None)
    std = checkpoint.get('std', None)
    if mean is None or std is None:
        # If not saved, generate training data and compute mean/std
        print("Mean/Std not found in checkpoint, generating training data for normalization...")
        trajectories = generate_aerobatic_trajectories(
            num_trajectories=500,
            seq_len=model.config.seq_len + model.config.history_len
        ).to(device)
        _, mean, std = normalize_trajectories(trajectories)
        mean = mean.to(device)
        std = std.to(device)
    else:
        mean = mean.to(device)
        std = std.to(device)

    # Visualize samples
    visualize_samples(model, mean, std, num_samples=3)