"""Progress and experiment-result reporting."""

import os
import time
from datetime import datetime

import numpy as np

from .config import Config
from .metrics import compute_collision_rate, compute_success_rates, compute_trajectory_errors

def print_inference_time_statistics(times_unguided, times_guided, config=None):
    """
    Print statistics about inference times.

    Args:
        times_unguided: List of inference times for unguided sampling
        times_guided: List of inference times for guided sampling
    """
    config = config or Config()
    print("\n" + "="*60)
    print("INFERENCE TIME STATISTICS")
    print("="*60)

    if len(times_unguided) == 0:
        print("No inference times recorded.")
        return

    # Convert to numpy arrays for statistical computation
    t_unguided = np.array(times_unguided)
    t_guided = np.array(times_guided)

    # Compute statistics
    def compute_stats(times):
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'q25': np.percentile(times, 25),
            'q75': np.percentile(times, 75),
            'total': np.sum(times)
        }

    stats_unguided = compute_stats(t_unguided)
    stats_guided = compute_stats(t_guided)

    # Print unguided statistics
    print("\n📊 UNGUIDED Sampling (without CBF):")
    print(f"  Total samples: {len(t_unguided)}")
    print(f"  Mean time:     {stats_unguided['mean']:.4f} s")
    print(f"  Std dev:       {stats_unguided['std']:.4f} s")
    print(f"  Min time:      {stats_unguided['min']:.4f} s")
    print(f"  Max time:      {stats_unguided['max']:.4f} s")
    print(f"  Median:        {stats_unguided['median']:.4f} s")
    print(f"  Q25-Q75:       [{stats_unguided['q25']:.4f}, {stats_unguided['q75']:.4f}] s")
    print(f"  Total time:    {stats_unguided['total']:.2f} s")

    # Print guided statistics
    print("\n📊 GUIDED Sampling (with CBF):")
    print(f"  Total samples: {len(t_guided)}")
    print(f"  Mean time:     {stats_guided['mean']:.4f} s")
    print(f"  Std dev:       {stats_guided['std']:.4f} s")
    print(f"  Min time:      {stats_guided['min']:.4f} s")
    print(f"  Max time:      {stats_guided['max']:.4f} s")
    print(f"  Median:        {stats_guided['median']:.4f} s")
    print(f"  Q25-Q75:       [{stats_guided['q25']:.4f}, {stats_guided['q75']:.4f}] s")
    print(f"  Total time:    {stats_guided['total']:.2f} s")

    # Compute and print comparison
    print("\n📈 COMPARISON (Guided vs Unguided):")

    # Mean time difference
    mean_diff = stats_guided['mean'] - stats_unguided['mean']
    mean_pct = (mean_diff / stats_unguided['mean']) * 100

    print(f"  Mean time difference: {mean_diff:+.4f} s ({mean_pct:+.2f}%)")

    # Speed of guided relative to unguided (slower = speedup < 1)
    speedup = stats_unguided['mean'] / stats_guided['mean']
    if speedup > 1:
        print(f"  ⚡ Guided is {speedup:.2f}x FASTER than unguided")
    elif speedup < 1:
        print(f"  ⚡ Guided is {1/speedup:.2f}x SLOWER than unguided (overhead)")
    else:
        print(f"  ⚡ Guided and unguided have similar speed")

    diffusion_steps = config.inference_diffusion_steps  # From config
    time_per_step_unguided = stats_unguided['mean'] / diffusion_steps
    time_per_step_guided = stats_guided['mean'] / diffusion_steps

    print(f"\n  Time per diffusion step:")
    print(f"    Unguided: {time_per_step_unguided*1000:.2f} ms/step")
    print(f"    Guided:   {time_per_step_guided*1000:.2f} ms/step")

    # Performance grade
    print("\n" + "-"*60)
    if stats_guided['mean'] < 0.5:
        grade = "A (Excellent) - <0.5s per trajectory"
    elif stats_guided['mean'] < 1.0:
        grade = "B (Good) - <1.0s per trajectory"
    elif stats_guided['mean'] < 2.0:
        grade = "C (Fair) - <2.0s per trajectory"
    else:
        grade = "D (Slow) - >2.0s per trajectory"

    print(f"  Performance Grade: {grade}")
    print("="*60)

    # Also append to results file if available
    try:
        # Find the most recent results file
        import glob
        result_files = glob.glob("Results/metrics_summary_*.txt")
        if result_files:
            latest_file = max(result_files, key=os.path.getctime)
            with open(latest_file, 'a', encoding='utf-8') as f:
                f.write("\n\n" + "="*60 + "\n")
                f.write("INFERENCE TIME STATISTICS\n")
                f.write("="*60 + "\n")
                f.write(f"\nUNGUIDED Sampling (without CBF):\n")
                f.write(f"  Mean: {stats_unguided['mean']:.4f} s ± {stats_unguided['std']:.4f}\n")
                f.write(f"  Min: {stats_unguided['min']:.4f} s, Max: {stats_unguided['max']:.4f} s\n")
                f.write(f"\nGUIDED Sampling (with CBF):\n")
                f.write(f"  Mean: {stats_guided['mean']:.4f} s ± {stats_guided['std']:.4f}\n")
                f.write(f"  Min: {stats_guided['min']:.4f} s, Max: {stats_guided['max']:.4f} s\n")
                f.write(f"\nGuided vs Unguided: {mean_diff:+.4f} s ({mean_pct:+.2f}%)\n")
                f.write("="*60 + "\n")
            print(f"✅ Inference time statistics appended to: {latest_file}")
    except Exception as e:
        print(f"⚠️ Could not append to results file: {e}")

    return {
        'unguided': stats_unguided,
        'guided': stats_guided,
        'mean_diff': mean_diff,
        'mean_pct': mean_pct,
        'speedup': speedup
    }

def config_to_string(config):
    """Convert Config parameters to a formatted string."""
    lines = []
    lines.append("="*80)
    lines.append("CONFIGURATION PARAMETERS")
    lines.append("="*80)
    lines.append(f"Training Parameters:")
    lines.append(f"  num_epochs: {config.num_epochs}")
    lines.append(f"  batch_size: {config.batch_size}")
    lines.append(f"  base_num_trajectories: {config._base_num_trajectories}")
    lines.append(f"  num_test_samples: {config.num_test_samples}")
    lines.append("")
    lines.append(f"Model Dimensions:")
    lines.append(f"  latent_dim: {config.latent_dim}")
    lines.append(f"  obs_latent_dim: {config.obs_latent_dim}")
    lines.append(f"  num_layers: {config.num_layers}")
    lines.append(f"  num_heads: {config.num_heads}")
    lines.append(f"  dropout: {config.dropout}")
    lines.append("")
    lines.append(f"Diffusion Parameters:")
    lines.append(f"  diffusion_steps: {config.diffusion_steps}")
    lines.append(f"  beta_start: {config.beta_start}")
    lines.append(f"  beta_end: {config.beta_end}")
    lines.append("")
    lines.append(f"Sequence Parameters:")
    lines.append(f"  seq_len: {config.seq_len}")
    lines.append(f"  state_dim: {config.state_dim}")
    lines.append(f"  history_len: {config.history_len}")
    lines.append("")
    lines.append(f"Condition Dimensions:")
    lines.append(f"  target_dim: {config.target_dim}")
    lines.append(f"  action_dim: {config.action_dim}")
    lines.append("")
    lines.append(f"Obstacle Parameters:")
    lines.append(f"  max_obstacles: {config.max_obstacles}")
    lines.append(f"  obstacle_feat_dim: {config.obstacle_feat_dim}")
    lines.append(f"  enable_obstacle_encoding: {config.enable_obstacle_encoding}")
    lines.append(f"  use_obstacle_loss: {config.use_obstacle_loss}")
    lines.append("")
    lines.append(f"CBF Guidance Parameters:")
    lines.append(f"  guidance_gamma: {config.guidance_gamma}")
    lines.append(f"  safe_extra_factor: {config.safe_extra_factor}")
    lines.append(f"  barrier_sigma: {config.barrier_sigma}")
    lines.append(f"  cbf_decay_rate: {config.cbf_decay_rate}")
    lines.append("")
    lines.append(f"Loss Weights:")
    lines.append(f"  last_xyz_weight: {config.last_xyz_weight}")
    lines.append(f"  xyz_weight: {config.xyz_weight}")
    lines.append(f"  vel_weight: {config.vel_weight}")
    lines.append(f"  other_weight: {config.other_weight}")
    lines.append(f"  obstacle_weight: {config.obstacle_weight}")
    lines.append(f"  continuity_weight: {config.continuity_weight}")
    lines.append(f"  acc_weight: {config.acc_weight}")
    lines.append(f"  dynamics_consistency_weight: {config.dynamics_consistency_weight}")
    lines.append("")
    lines.append(f"Other Parameters:")
    lines.append(f"  delta_T: {config.delta_T}")
    lines.append(f"  drop_style_prob: {config.drop_style_prob}")
    lines.append(f"  drop_target_prob: {config.drop_target_prob}")
    lines.append(f"  guidance_scale: {config.guidance_scale}")
    lines.append(f"  show_flag: {config.show_flag}")
    lines.append("="*80)

    return '\n'.join(lines)

def print_metrics_with_success_rates(unguided_positions, guided_positions,
                                      ground_truth_positions, target_positions,
                                      obstacles_data, config=None):
    """Print comprehensive metrics including success rates and save to timestamped file."""

    config = config or Config()

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Results/metrics_summary_{timestamp}.txt"

    # Capture all output
    output_lines = []

    def write_and_print(text):
        """Write text to both console and file buffer."""
        print(text)
        output_lines.append(text)

    # Write config parameters first
    write_and_print(config_to_string(config))
    write_and_print("\n")

    write_and_print("\n" + "="*80)
    write_and_print("COMPREHENSIVE METRICS SUMMARY WITH SUCCESS RATES")
    write_and_print("="*80)

    # Compute collision rates
    write_and_print("\n" + "-"*60)
    write_and_print("COLLISION RATE ANALYSIS")
    write_and_print("-"*60)

    unguided_collision = compute_collision_rate(unguided_positions, obstacles_data, safety_margin=0.0)
    guided_collision = compute_collision_rate(guided_positions, obstacles_data, safety_margin=0.0)

    write_and_print(f"\nUNGuided Trajectories (without CBF):")
    write_and_print(f"  - Collision Rate: {unguided_collision['collision_rate']*100:.2f}%")
    write_and_print(f"  - Avg Collision % per trajectory: {unguided_collision['avg_collision_percentage']:.2f}%")
    write_and_print(f"  - Trajectories with collision: {unguided_collision['trajectories_with_collision']}/{unguided_collision['total_trajectories']}")

    write_and_print(f"\nGuided Trajectories (with CBF):")
    write_and_print(f"  - Collision Rate: {guided_collision['collision_rate']*100:.2f}%")
    write_and_print(f"  - Avg Collision % per trajectory: {guided_collision['avg_collision_percentage']:.2f}%")
    write_and_print(f"  - Trajectories with collision: {guided_collision['trajectories_with_collision']}/{guided_collision['total_trajectories']}")

    # Compute trajectory errors
    write_and_print("\n" + "-"*60)
    write_and_print("TRAJECTORY PREDICTION ERROR ANALYSIS")
    write_and_print("-"*60)

    unguided_errors = compute_trajectory_errors(unguided_positions, ground_truth_positions)
    guided_errors = compute_trajectory_errors(guided_positions, ground_truth_positions)

    write_and_print(f"\nUNGuided Trajectory Errors:")
    write_and_print(f"  - Mean AE: {unguided_errors['mean_ae']:.4f}")
    write_and_print(f"  - RMSE: {unguided_errors['rmse']:.4f}")
    write_and_print(f"  - Mean Final Error: {unguided_errors['mean_final_error']:.4f}")
    write_and_print(f"  - 95th Percentile: {unguided_errors['percentile_95']:.4f}")

    write_and_print(f"\nGuided Trajectory Errors:")
    write_and_print(f"  - Mean AE: {guided_errors['mean_ae']:.4f}")
    write_and_print(f"  - RMSE: {guided_errors['rmse']:.4f}")
    write_and_print(f"  - Mean Final Error: {guided_errors['mean_final_error']:.4f}")
    write_and_print(f"  - 95th Percentile: {guided_errors['percentile_95']:.4f}")

    # Compute success rates
    write_and_print("\n" + "-"*60)
    write_and_print("SUCCESS RATE ANALYSIS")
    write_and_print("-"*60)

    write_and_print("\nSuccess Criteria:")
    write_and_print("  ✓ Collision-free: No intersection with obstacles")
    write_and_print("  ✓ Trajectory accuracy: Max position error < 2.0")
    write_and_print("  ✓ Final point accuracy: Final position error < 1.0")
    write_and_print("  ✓ Target reach: Distance to target < 2.0 at final step")
    write_and_print("  ✓ Overall success: Meets ALL above criteria")

    # Unguided success rates
    write_and_print("\n" + "="*60)
    write_and_print("UNGuided Trajectories (without CBF)")
    write_and_print("="*60)
    unguided_success = compute_success_rates(
        unguided_positions, ground_truth_positions, target_positions, obstacles_data
    )

    write_and_print(f"\n  📊 Collision-Free Rate:     {unguided_success['collision_free_rate']:6.2f}%")
    write_and_print(f"  📊 Trajectory Accuracy Rate: {unguided_success['trajectory_accuracy_rate']:6.2f}%")
    write_and_print(f"  📊 Final Point Accuracy Rate:{unguided_success['final_point_accuracy_rate']:6.2f}%")
    write_and_print(f"  📊 Target Reach Rate:        {unguided_success['target_reach_rate']:6.2f}%")
    write_and_print(f"  📊 Safety Distance Rate:     {unguided_success['safety_distance_rate']:6.2f}%")
    write_and_print(f"  ⭐ OVERALL SUCCESS RATE:     {unguided_success['overall_success_rate']:6.2f}%")

    # Guided success rates
    write_and_print("\n" + "="*60)
    write_and_print("Guided Trajectories (with CBF)")
    write_and_print("="*60)
    guided_success = compute_success_rates(
        guided_positions, ground_truth_positions, target_positions, obstacles_data
    )

    write_and_print(f"\n  📊 Collision-Free Rate:     {guided_success['collision_free_rate']:6.2f}%")
    write_and_print(f"  📊 Trajectory Accuracy Rate: {guided_success['trajectory_accuracy_rate']:6.2f}%")
    write_and_print(f"  📊 Final Point Accuracy Rate:{guided_success['final_point_accuracy_rate']:6.2f}%")
    write_and_print(f"  📊 Target Reach Rate:        {guided_success['target_reach_rate']:6.2f}%")
    write_and_print(f"  📊 Safety Distance Rate:     {guided_success['safety_distance_rate']:6.2f}%")
    write_and_print(f"  ⭐ OVERALL SUCCESS RATE:     {guided_success['overall_success_rate']:6.2f}%")

    # Improvement analysis
    write_and_print("\n" + "-"*60)
    write_and_print("IMPROVEMENT ANALYSIS (Guided vs Unguided)")
    write_and_print("-"*60)

    collision_improvement = (unguided_collision['collision_rate'] - guided_collision['collision_rate']) / max(unguided_collision['collision_rate'], 1e-6) * 100
    error_improvement = (unguided_errors['mean_ae'] - guided_errors['mean_ae']) / max(unguided_errors['mean_ae'], 1e-6) * 100
    success_improvement = guided_success['overall_success_rate'] - unguided_success['overall_success_rate']

    write_and_print(f"\n  🚀 Collision Rate Reduction:  {collision_improvement:+.2f}%")
    write_and_print(f"  🚀 Mean AE Reduction:         {error_improvement:+.2f}%")
    write_and_print(f"  🚀 Overall Success Increase:  {success_improvement:+.2f} percentage points")

    # Safety summary
    write_and_print("\n" + "-"*60)
    write_and_print("SAFETY & PERFORMANCE SUMMARY")
    write_and_print("-"*60)

    if guided_success['overall_success_rate'] >= 80:
        grade = "A (Excellent)"
    elif guided_success['overall_success_rate'] >= 60:
        grade = "B (Good)"
    elif guided_success['overall_success_rate'] >= 40:
        grade = "C (Fair)"
    else:
        grade = "D (Poor)"

    write_and_print(f"\n  Performance Grade: {grade}")
    write_and_print(f"  CBF Guidance prevents {collision_improvement:.1f}% of collisions")
    write_and_print(f"  Trade-off: {abs(error_improvement):.1f}% change in trajectory error for {collision_improvement:.1f}% safety improvement")

    # Print detailed metrics
    write_and_print("\n" + "-"*60)
    write_and_print("DETAILED METRICS")
    write_and_print("-"*60)

    write_and_print(f"\n  Unguided - Mean Max Error: {unguided_success['detailed_metrics']['mean_max_error']:.4f} ± {unguided_success['detailed_metrics']['std_max_error']:.4f}")
    write_and_print(f"  Guided   - Mean Max Error: {guided_success['detailed_metrics']['mean_max_error']:.4f} ± {guided_success['detailed_metrics']['std_max_error']:.4f}")

    write_and_print(f"\n  Unguided - Mean Final Error: {unguided_success['detailed_metrics']['mean_final_error']:.4f} ± {unguided_success['detailed_metrics']['std_final_error']:.4f}")
    write_and_print(f"  Guided   - Mean Final Error: {guided_success['detailed_metrics']['mean_final_error']:.4f} ± {guided_success['detailed_metrics']['std_final_error']:.4f}")

    write_and_print(f"\n  Unguided - Mean Target Distance: {unguided_success['detailed_metrics']['mean_target_distance']:.4f}")
    write_and_print(f"  Guided   - Mean Target Distance: {guided_success['detailed_metrics']['mean_target_distance']:.4f}")

    write_and_print(f"\n  Unguided - Mean Min Surface Dist: {unguided_success['detailed_metrics']['mean_min_surface_dist']:.4f}")
    write_and_print(f"  Guided   - Mean Min Surface Dist: {guided_success['detailed_metrics']['mean_min_surface_dist']:.4f}")

    write_and_print("\n" + "="*80)

    # Save to file
    try:
        # Create Results directory if it doesn't exist
        import os
        os.makedirs("Results", exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"\n✅ Metrics saved to: {filename}")
    except Exception as e:
        print(f"\n❌ Error saving file: {e}")

    return {
        'unguided_success': unguided_success,
        'guided_success': guided_success,
        'unguided_errors': unguided_errors,
        'guided_errors': guided_errors,
        'filename': filename
    }

def format_progress(epoch, num_epochs, start_time, avg_total, avg_position, avg_vel, avg_obstacle, avg_continuity, use_obstacle_loss):
    progress = (epoch + 1) / num_epochs
    bar_length = 30
    filled_length = int(bar_length * progress)
    bar = '=' * filled_length + ' ' * (bar_length - filled_length)

    if use_obstacle_loss:
        loss_info = (f"Total: {avg_total:.4f} | Pos: {avg_position:.4f} | "
                    f"Vel: {avg_vel:.4f} | Obs: {avg_obstacle:.4f} | Cont: {avg_continuity:.4f}")
    else:
        loss_info = (f"Total: {avg_total:.4f} | Pos: {avg_position:.4f} | "
                    f"Vel: {avg_vel:.4f} | Cont: {avg_continuity:.4f}")

    return f"\rEpoch {(epoch+1):4d}/{num_epochs} [{bar}] {progress*100:5.1f}% | Time: {time.time()-start_time:6.2f}s | {loss_info}"
