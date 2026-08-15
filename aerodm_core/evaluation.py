"""Quantitative model evaluation workflow."""

import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import (denormalize_target, denormalize_trajectories,
                   generate_random_obstacles, generate_target_waypoints,
                   normalize_target, project_target_outside_obstacles)
from .reporting import (print_inference_time_statistics,
                        print_metrics_with_success_rates)
from .visualization.results import plot_combined_trajectories, plot_test_results

def test_model_performance_cb_eva(model, trajectories_norm, mean, std, num_test_samples=50,
                                  show_flag=True, plot_combined=True,
                                  show_collision_points=True):
    """Testing with obstacle-aware transformer and compute collision rates, trajectory errors & success rates"""
    print("\nTesting obstacle-aware model performance...")
    config = model.config
    device = next(model.parameters()).device

    # Define style names mapping
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

    mean_state = mean[..., :-1]  # Shape: (1, 1, 10)
    std_state = std[..., :-1]    # Shape: (1, 1, 10)

    # Set normalization parameters
    model.set_normalization_params(mean, std)

    # Store data for first 9 cases for combined visualization
    combined_cases = []

    # Storage for metrics computation
    all_unguided_positions = []
    all_guided_positions = []
    all_ground_truth_positions = []
    all_target_positions = []
    all_obstacles_data = []

    # ============ NEW: Timing storage ============
    inference_times_unguided = []
    inference_times_guided = []

    model.eval()
    with torch.no_grad():
        for i in range(min(num_test_samples, trajectories_norm.shape[0])):
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Processing test sample {i+1}/{min(num_test_samples, trajectories_norm.shape[0])}")

            # Prepare test sample
            full_traj = trajectories_norm[i:i+1]

            style_info = full_traj[:, :, -1:]  # Shape: (B, T_full, 1)
            state_without_style = full_traj[:, :, :-1]  # Shape: (B, T_full, state_dim-1)

            # Split into history and sequence-to-predict
            history = state_without_style[:, :config.history_len, :]
            x_0 = state_without_style[:, config.history_len:config.history_len+config.seq_len, :]
            target_norm = generate_target_waypoints(x_0)

            # Denormalize for obstacle generation and plotting
            x_0_denorm = denormalize_trajectories(x_0, mean_state, std_state)
            target_denorm = denormalize_target(target_norm, mean_state, std_state)
            # target_denorm = add_target_noise(target_denorm, bound=1.0)

            # Extract style information
            history_style = style_info[:, 0, 0]
            pred_style = style_info[:, -1, 0]

            style_indices = pred_style.long()
            action = F.one_hot(style_indices, num_classes=config.action_dim).float()

            # Generate random obstacles (consistent for both unguided and guided)
            obstacles = generate_random_obstacles(x_0_denorm[0],
                                                  num_obstacles_range=(3, 5),
                                                  radius_range=(0.5, 1.0),
                                                  check_collision=True,
                                                  device=device)

            # Keep the conditioning target outside every expanded obstacle.
            target_denorm = project_target_outside_obstacles(
                target_denorm, obstacles,
                safety_margin=config.safe_extra_factor)
            target_norm = normalize_target(target_denorm, mean_state, std_state)

            # Set obstacles data for model input
            model.set_obstacles_data([obstacles])

            # Get style names for display
            hist_idx = history_style.item()
            pred_idx = pred_style.item()

            print(f"\n{'='*60}")
            print(f"TEST SAMPLE {i+1}")
            print(f"History Style: {style_names.get(hist_idx, f'Style_{hist_idx}')} (idx={hist_idx})")
            print(f"Prediction Style: {style_names.get(pred_idx, f'Style_{pred_idx}')} (idx={pred_idx})")
            print(f"Generated {len(obstacles)} random obstacles")
            print(f"{'='*60}")

            # ============ NEW: Time unguided sampling ============
            start_time = time.time()

            sampled_unguided_norm = model.sample(
                target_norm,
                action,
                history,
                batch_size=1,
                enable_guidance=False,
                plot_all_steps=False
            )

            end_time = time.time()
            inference_time_unguided = end_time - start_time
            inference_times_unguided.append(inference_time_unguided)

            sampled_unguided_denorm = denormalize_trajectories(sampled_unguided_norm, mean_state, std_state)

            # ============ NEW: Time guided sampling ============
            start_time = time.time()

            sampled_guided_norm = model.sample(
                target_norm,
                action,
                history,
                batch_size=1,
                enable_guidance=True,
                guidance_gamma=config.guidance_gamma,
                plot_all_steps=False
            )

            end_time = time.time()
            inference_time_guided = end_time - start_time
            inference_times_guided.append(inference_time_guided)

            sampled_guided_denorm = denormalize_trajectories(sampled_guided_norm, mean_state, std_state)

            # Print timing information
            print(f"  ⏱️ Unguided inference time: {inference_time_unguided:.4f}s")
            print(f"  ⏱️ Guided inference time:   {inference_time_guided:.4f}s")
            print(f"  📊 Speedup (guided vs unguided): {inference_time_unguided/inference_time_guided:.2f}x")

            # Plot individual test results
            plot_test_results(
                x_0_denorm,
                sampled_unguided_denorm,
                sampled_guided_denorm,
                denormalize_trajectories(history, mean_state, std_state) if history is not None else None,
                target_denorm,
                obstacles,
                show_flag,
                step_idx=i+1,
                history_style=history_style,
                pred_style=pred_style,
                style_names=style_names,
                show_collision_points=show_collision_points
            )
            # Store positions for metrics computation (only x,y,z)
            all_unguided_positions.append(sampled_unguided_denorm[0, :, 1:4].cpu().numpy())
            all_guided_positions.append(sampled_guided_denorm[0, :, 1:4].cpu().numpy())
            all_ground_truth_positions.append(x_0_denorm[0, :, 1:4].cpu().numpy())
            all_target_positions.append(target_denorm[0, :3].cpu().numpy())
            all_obstacles_data.append(obstacles)

            # Store first 9 cases for combined visualization
            if i < 9 and plot_combined:
                combined_cases.append({
                    'idx': i + 1,
                    'original': x_0_denorm[0, :, 1:4].detach().cpu().numpy(),
                    'guided': sampled_guided_denorm[0, :, 1:4].detach().cpu().numpy(),
                    'unguided': sampled_unguided_denorm[0, :, 1:4].detach().cpu().numpy(),
                    'history': denormalize_trajectories(history, mean_state, std_state)[0, :, 1:4].detach().cpu().numpy() if history is not None else None,
                    'target': target_denorm[0, :3].detach().cpu().numpy() if target_denorm is not None else None,
                    'obstacles': obstacles,
                    'hist_style': style_names.get(hist_idx, f'Style_{hist_idx}'),
                    'pred_style': style_names.get(pred_idx, f'Style_{pred_idx}')
                })

            # Plot individual test results (only for first 9 to avoid excessive plots)
            if i < 9:
                plot_test_results(
                    x_0_denorm,
                    sampled_unguided_denorm,
                    sampled_guided_denorm,
                    denormalize_trajectories(history, mean_state, std_state) if history is not None else None,
                    target_denorm,
                    obstacles,
                    show_flag,
                    step_idx=i+1,
                    history_style=history_style,
                    pred_style=pred_style,
                    style_names=style_names,
                    show_collision_points=show_collision_points
                )

    # ============ NEW: Print inference time statistics ============
    print_inference_time_statistics(inference_times_unguided, inference_times_guided, config)

    # Convert to numpy arrays for metrics computation
    all_unguided_positions = np.stack(all_unguided_positions)  # (N, seq_len, 3)
    all_guided_positions = np.stack(all_guided_positions)
    all_ground_truth_positions = np.stack(all_ground_truth_positions)
    all_target_positions = np.stack(all_target_positions)

    # Compute and print all metrics including success rates
    print_metrics_with_success_rates(
        all_unguided_positions,
        all_guided_positions,
        all_ground_truth_positions,
        all_target_positions,
        all_obstacles_data,
        config
    )

    # Plot combined trajectories for first 9 cases
    if plot_combined and len(combined_cases) > 0:
        plot_combined_trajectories(
            combined_cases, show_flag,
            show_collision_points=show_collision_points)
