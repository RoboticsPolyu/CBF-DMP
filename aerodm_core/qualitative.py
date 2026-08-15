"""Qualitative trajectory and maneuver-style experiments."""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from Trajectory_Gen import generate_aerobatic_trajectories

from .data import (denormalize_target, denormalize_trajectories,
                   generate_random_obstacles, generate_target_waypoints,
                   normalize_target, project_target_outside_obstacles)
from .visualization.styles import plot_action_comparison, plot_style_statistics

def generate_trj_demos():
    # Generate example enhanced circular trajectories for demonstration
    print("Generating example enhanced circular trajectories...")
    demo_trajectories = generate_aerobatic_trajectories(num_trajectories=18, seq_len=60)

    # Extract style indices from the trajectories (last dimension)
    # Style index is stored as the last element in the state vector
    style_indices = demo_trajectories[:, 0, -1].long().numpy()

    # Define style names mapping (same as in generate_aerobatic_trajectories)
    style_names = {
        0: 'power_loop',
        1: 'barrel_roll',
        2: 'split_s',
        3: 'immelmann',
        4: 'wall_ride',
        5: 'eight_figure',
        6: 'star',
        7: 'half_moon',
        8: 'sphinx',
        9: 'clover',
        10: 'spiral_inward',
        11: 'spiral_outward',
        12: 'spiral_vertical_up',
        13: 'spiral_vertical_down'
    }

    # Get style names for each trajectory
    trajectory_styles = [style_names.get(idx, 'unknown') for idx in style_indices]

    # Visualize some training data with z-axis focus
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Enhanced Circular Trajectories with Style Information', fontsize=16, fontweight='bold')

    for i in range(6):
        # First row: Trajectories 1-6
        ax = fig.add_subplot(3, 6, i+1, projection='3d')
        trajectory = demo_trajectories[i, :, 1:4].numpy()
        style = trajectory_styles[i]

        # Color coding based on style
        if 'loop' in style:
            color = 'blue'
        elif 'roll' in style:
            color = 'red'
        elif 'spiral' in style:
            color = 'green'
        elif 'figure' in style:
            color = 'purple'
        else:
            color = 'orange'

        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color=color, linewidth=2, alpha=0.8)
        ax.set_title(f'Traj {i+1}: {style}', fontsize=9, pad=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True, alpha=0.3)

        # Second row: Trajectories 7-12
        ax = fig.add_subplot(3, 6, i+7, projection='3d')
        trajectory = demo_trajectories[i+6, :, 1:4].numpy()
        style = trajectory_styles[i+6]

        if 'loop' in style:
            color = 'blue'
        elif 'roll' in style:
            color = 'red'
        elif 'spiral' in style:
            color = 'green'
        elif 'figure' in style:
            color = 'purple'
        else:
            color = 'orange'

        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color=color, linewidth=2, alpha=0.8)
        ax.set_title(f'Traj {i+7}: {style}', fontsize=9, pad=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True, alpha=0.3)

        # Third row: Trajectories 13-18
        ax = fig.add_subplot(3, 6, i+13, projection='3d')
        trajectory = demo_trajectories[i+12, :, 1:4].numpy()
        style = trajectory_styles[i+12]

        if 'loop' in style:
            color = 'blue'
        elif 'roll' in style:
            color = 'red'
        elif 'spiral' in style:
            color = 'green'
        elif 'figure' in style:
            color = 'purple'
        else:
            color = 'orange'

        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color=color, linewidth=2, alpha=0.8)
        ax.set_title(f'Traj {i+13}: {style}', fontsize=9, pad=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True, alpha=0.3)

    # Add legend for style-color mapping
    plt.figtext(0.5, 0.01,
                'Color Legend: Blue=Loops, Red=Rolls, Green=Spirals, Purple=Figures, Orange=Others',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to accommodate title and legend
    plt.show()

    # Print style distribution
    print("\n=== Style Distribution in Generated Trajectories ===")
    style_counts = {}
    for style in trajectory_styles:
        style_counts[style] = style_counts.get(style, 0) + 1

    for style, count in style_counts.items():
        print(f"{style}: {count} trajectories")

    print(f"Total: {len(trajectory_styles)} trajectories")

def test_action_effect(model, trajectories_norm, mean, std, num_test_samples=5, show_flag=True):
    """Test the effect of different actions (maneuver styles) on trajectory generation"""
    print("\nTesting action (maneuver style) effects on trajectory generation...")
    config = model.config
    device = next(model.parameters()).device

    mean_state = mean[..., :-1]
    std_state = std[..., :-1]

    # Set normalization parameters
    model.set_normalization_params(mean, std)

    # Define style names for display
    style_names = {
        0: 'power_loop',
        1: 'barrel_roll',
        2: 'split_s',
        3: 'immelmann',
        4: 'wall_ride',
        5: 'eight_figure',
        6: 'star',
        7: 'half_moon',
        8: 'sphinx',
        9: 'clover',
        10: 'spiral_inward',
        11: 'spiral_outward',
        12: 'spiral_vertical_up',
        13: 'spiral_vertical_down'
    }

    # Select a subset of styles to test
    test_styles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # All styles

    model.eval()
    with torch.no_grad():
        # Use a fixed test sample from the dataset
        for i in range(min(num_test_samples, trajectories_norm.shape[0])):
            full_traj = trajectories_norm[i:i+1]

            # Extract history from the test sample
            state_without_style = full_traj[:, :, :-1]
            history = state_without_style[:, :config.history_len, :]

            # Get the target from the ground truth trajectory
            x_0 = state_without_style[:, config.history_len:config.history_len+config.seq_len, :]
            target_norm = generate_target_waypoints(x_0)
            target_denorm = denormalize_target(target_norm, mean_state, std_state)

            # Generate fixed obstacles for consistent comparison
            x_0_denorm = denormalize_trajectories(x_0, mean_state, std_state)
            obstacles = generate_random_obstacles(
                x_0_denorm[0],
                num_obstacles_range=(3, 5),
                radius_range=(0.5, 1.0),
                check_collision=False,
                device=device
            )
            target_denorm = project_target_outside_obstacles(
                target_denorm, obstacles,
                safety_margin=config.safe_extra_factor)
            target_norm = normalize_target(target_denorm, mean_state, std_state)
            model.set_obstacles_data([obstacles])

            print(f"\n{'='*80}")
            print(f"TEST SAMPLE {i+1} - Comparing All Maneuver Styles")
            print(f"Target Waypoint: {target_denorm[0].cpu().numpy()}")
            print(f"Number of obstacles: {len(obstacles)}")
            print(f"{'='*80}")

            # Store results for all styles
            all_trajectories = {}
            all_trajectories_norm = {}

            # Generate trajectories for each test style
            for style_idx in test_styles:
                # Create one-hot action for this style
                action = F.one_hot(torch.tensor([style_idx]), num_classes=config.action_dim).float().to(device)
                style_name = style_names.get(style_idx, f'style_{style_idx}')

                # Sample with CBF guidance
                sampled_norm = model.sample(
                    target_norm,
                    action,
                    history,
                    batch_size=1,
                    enable_guidance=True,
                    guidance_gamma=config.guidance_gamma,
                    plot_all_steps=False
                )

                sampled_denorm = denormalize_trajectories(sampled_norm, mean_state, std_state)
                all_trajectories[style_name] = sampled_denorm[0].cpu().numpy()
                all_trajectories_norm[style_name] = sampled_norm[0].cpu().numpy()

            # Plot comparison of all styles
            plot_action_comparison(
                all_trajectories,
                target_denorm[0].cpu().numpy(),
                obstacles,
                history[0].cpu().numpy() if history is not None else None,
                mean_state, std_state,
                show_flag,
                sample_idx=i+1
            )

            # Plot style statistics
            plot_style_statistics(all_trajectories, show_flag, sample_idx=i+1)
