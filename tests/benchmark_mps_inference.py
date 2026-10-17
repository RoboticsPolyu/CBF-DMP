"""Local MPS throughput benchmark of the production 60-step sampler.

From the repository root:
    python tests/benchmark_mps_inference.py --num-trajectories 200 --batch-size 32
Use --batch-size 1 for serial per-request latency instead of amortized throughput.
"""
import argparse
import contextlib
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from AeroDM_ObstacleAware_Training_Test import (
    Config, AeroDM, load_model_checkpoint, normalize_trajectories,
    generate_target_waypoints, generate_random_obstacles,
)
from Trajectory_Gen import generate_aerobatic_trajectories_pvR


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be >= 1')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-trajectories', '-n', type=positive_int, default=200)
    parser.add_argument('--batch-size', type=positive_int, default=Config.test_batch_size)
    parser.add_argument('--repeats', type=positive_int, default=3)
    parser.add_argument('--warmup', type=positive_int, default=2, help='warm-up batches per mode and batch shape')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/Config.model_save_dir/Config.model_filename)
    parser.add_argument('--output', type=Path, help='JSON output path')
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error('MPS is unavailable. Run with an Apple Silicon PyTorch environment; no CPU fallback is used.')
    device = torch.device('mps')
    config = Config()
    config.device = 'mps'
    config.diffusion_steps = 60
    config.enable_test_plots = False
    config.plot_guidance_steps = False
    config.show_flag = False
    config.train_model = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = load_model_checkpoint(args.checkpoint, 'cpu')
    model = AeroDM(config).to(device).eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    mean_cpu, std_cpu = checkpoint['mean'].cpu(), checkpoint['std'].cpu()
    model.set_normalization_params(mean_cpu.to(device), std_cpu.to(device))

    # Prepare identical physical trajectories/obstacles for both modes outside timing.
    raw = generate_aerobatic_trajectories_pvR(
        args.num_trajectories, config.history_len + config.seq_len, delta_T=config.delta_T)
    normalized, _, _ = normalize_trajectories(raw, mean=mean_cpu, std=std_cpu)
    batches = []
    for start in range(0, args.num_trajectories, args.batch_size):
        stop = min(start + args.batch_size, args.num_trajectories)
        full = normalized[start:stop].to(device)
        history = full[:, :config.history_len, :-1]
        future = full[:, config.history_len:, :-1]
        target = generate_target_waypoints(future)
        action = F.one_hot(full[:, -1, -1].long(), config.action_dim).float()
        obstacles = [generate_random_obstacles(
            trajectory[config.history_len:, :-1], num_obstacles_range=(3, 5),
            radius_range=(0.5, 1.0), check_collision=False, device=device)
            for trajectory in raw[start:stop]]
        batches.append((history, target, action, obstacles))
    torch.mps.synchronize()

    def run_batch(batch, guided, seed):
        history, target, action, obstacles = batch
        torch.manual_seed(seed)  # paired initial and diffusion noise
        model.obstacles_data = obstacles
        torch.mps.synchronize()
        started = time.perf_counter()
        result = model.sample(target, action, history, batch_size=history.shape[0],
                              enable_guidance=guided,
                              guidance_gamma=config.guidance_gamma if guided else None,
                              plot_all_steps=False)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError('Non-finite sample detected')
        return elapsed

    print(f'MPS | N={args.num_trajectories} | batch={args.batch_size} | steps=60 | repeats={args.repeats}', flush=True)
    print('Guided includes CFG and barrier guidance; setup, data generation and plotting are excluded.', flush=True)
    totals = {'unguided': [], 'guided': []}
    all_batch_times = {'unguided': [], 'guided': []}
    with open(os.devnull, 'w') as quiet:
        # Warm each batch shape, including a partial final batch.
        shapes = {b[0].shape[0]: b for b in batches}
        with contextlib.redirect_stdout(quiet), torch.no_grad():
            for batch in shapes.values():
                for guided in (False, True):
                    for i in range(args.warmup):
                        run_batch(batch, guided, args.seed+i)
        for repeat in range(args.repeats):
            times = {False: [], True: []}
            for i, batch in enumerate(batches):
                # Alternate mode order to reduce systematic ordering bias.
                order = (False, True) if (repeat+i)%2 == 0 else (True, False)
                for guided in order:
                    with contextlib.redirect_stdout(quiet), torch.no_grad():
                        elapsed = run_batch(batch, guided, args.seed+1000+repeat*len(batches)+i)
                    times[guided].append(elapsed)
            for guided, name in ((False, 'unguided'), (True, 'guided')):
                totals[name].append(sum(times[guided]))
                all_batch_times[name].append(times[guided])
            print(f'Repeat {repeat+1}: unguided={totals["unguided"][-1]:.4f}s, guided={totals["guided"][-1]:.4f}s', flush=True)

    results = {}
    print('\nMode        Total mean (s)   Mean (ms/trajectory)   Trajectories/s')
    for name, values in totals.items():
        total = statistics.mean(values)
        results[name] = {'mean_total_seconds': total,
                         'std_total_seconds': statistics.stdev(values) if len(values)>1 else 0.0,
                         'mean_seconds_per_trajectory': total/args.num_trajectories,
                         'trajectories_per_second': args.num_trajectories/total,
                         'repeat_total_seconds': values, 'batch_seconds': all_batch_times[name]}
        print(f'{name:10s}  {total:14.4f}   {1000*total/args.num_trajectories:20.3f}   {args.num_trajectories/total:14.2f}')
    ratio = results['guided']['mean_total_seconds']/results['unguided']['mean_total_seconds']
    print(f'Guided / unguided: {ratio:.3f}x')
    print('Per-trajectory time is amortized batch time. Use --batch-size 1 for serial latency.')
    report = {'device': 'mps', 'torch_version': torch.__version__, 'num_trajectories': args.num_trajectories,
              'batch_size': args.batch_size, 'diffusion_steps': 60, 'repeats': args.repeats,
              'warmup_per_shape_and_mode': args.warmup, 'seed': args.seed,
              'checkpoint': str(args.checkpoint.resolve()), 'guidance_gamma': config.guidance_gamma,
              'barrier_type': config.barrier_type, 'barrier_sigma': config.barrier_sigma,
              'guidance_scale': config.guidance_scale, 'enable_obstacle_encoding': config.enable_obstacle_encoding,
              'guided_to_unguided_ratio': ratio, 'results': results,
              'timing_scope': 'Synchronized model.sample wall time, including obstacle packing and sampler diagnostics; excludes loading, data preparation, warmup, finite checks and plotting.'}
    output = args.output or ROOT/'Results'/f'mps_inference_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
