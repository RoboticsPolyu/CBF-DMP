# Evaluation_AeroDM_CBF.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import matplotlib.pyplot as plt
import numpy as np

# Import your model and utility functions
from AeroDM_Barrier_Guid import (
    Config, AeroDM,
    generate_aerobatic_trajectories,
    normalize_trajectories, denormalize_trajectories,
    generate_target_waypoints, generate_action_styles, generate_history_segments,
    plot_trajectory_comparison
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

        # Sample unguided
        sampled_unguided = model.sample(target, action, history, batch_size=1, enable_guidance=False)
        sampled_unguided_denorm = denormalize_trajectories(sampled_unguided, mean, std)

        # Sample guided
        sampled_guided = model.sample(target, action, history, batch_size=1, enable_guidance=True, guidance_gamma=config.guidance_gamma)
        sampled_guided_denorm = denormalize_trajectories(sampled_guided, mean, std)

        # Denormalize others
        x_0_denorm = denormalize_trajectories(x_0, mean, std)
        history_denorm = denormalize_trajectories(history, mean, std)
        target_denorm = target * std[0, 0, 1:4] + mean[0, 0, 1:4]

        obstacle_center = config.get_obstacle_center('cpu').numpy()

        plot_trajectory_comparison(
            x_0_denorm, sampled_unguided_denorm, sampled_guided_denorm,
            history=history_denorm, target=target_denorm,
            title=f"Visualization Sample {i+1} (Guided: Green avoids Red Obstacle)",
            obstacle_center=obstacle_center,
            obstacle_radius=config.obstacle_radius
        )

if __name__ == "__main__":
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Path to your trained model checkpoint
    checkpoint_path = "model/improved_aerodm_with_cbf.pth"

    # Load model
    print("Loading trained AeroDM model...")
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
    model.set_normalization_params(mean, std)
    model.config.guidance_gamma = 1000.0
    # model.config.diffusion_steps = 50
    # Visualize samples
    visualize_samples(model, mean, std, num_samples=100)
