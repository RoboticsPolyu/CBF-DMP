"""One fixed case/style, ten target perturbations within a 2 m ball, no obstacles.

Run: python experiment_target_neighborhood.py --device mps
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from AeroDM_ObstacleAware_Training_Test import (
    AeroDM, Config, load_model_checkpoint, normalize_target,
)
from Trajectory_Gen import MANEUVER_STYLES, generate_aerobatic_trajectories_pvR


def sample_targets(center, rng):
    # Uniform in volume, not a cube or only the sphere surface.
    directions = rng.normal(size=(10, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    offsets = directions * (2.0 * rng.random((10, 1)) ** (1.0 / 3.0))
    return (center + offsets).astype(np.float32)


def select_cases(config):
    cases = {}
    for _ in range(100):
        trajectories = generate_aerobatic_trajectories_pvR(
            64, config.history_len + config.seq_len, delta_T=config.delta_T)
        for trajectory in trajectories:
            style = int(trajectory[-1, -1])
            cases.setdefault(style, trajectory)
        if len(cases) == len(MANEUVER_STYLES):
            return cases
    raise RuntimeError('Could not find a case for every style')


def plot_case(path, name, history, original, targets, predictions):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*history.T, color='black', linewidth=3, label='Fixed history')
    ax.plot(*np.vstack([history[-1], original]).T, color='black',
            linestyle='--', linewidth=2.5, label='Original trajectory')
    ax.scatter(*original[-1], color='black', marker='*', s=150, label='Original target')
    colors = plt.get_cmap('tab10').colors
    for i, (target, prediction) in enumerate(zip(targets, predictions)):
        ax.plot(*np.vstack([history[-1], prediction]).T, color=colors[i],
                linewidth=1.4, alpha=0.85, label=f'Target {i+1}')
        ax.scatter(*target, color=colors[i], marker='x', s=65)
        ax.scatter(*prediction[-1], color=colors[i], marker='o', s=15)
    points = np.concatenate([history, original, targets, predictions.reshape(-1, 3)])
    midpoint = (points.min(0) + points.max(0)) / 2
    radius = max(float(np.ptp(points, axis=0).max()) * 0.55, 1)
    ax.set_xlim(midpoint[0]-radius, midpoint[0]+radius)
    ax.set_ylim(midpoint[1]-radius, midpoint[1]+radius)
    ax.set_zlim(midpoint[2]-radius, midpoint[2]+radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.savefig(path, format='svg', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(Config.model_save_dir)/Config.model_filename)
    parser.add_argument('--device', default=Config.device)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('Figs/target_neighborhood'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; use --device cpu in this environment')
    config = Config()
    checkpoint = load_model_checkpoint(args.checkpoint, 'cpu')
    weights = checkpoint['model_state_dict']
    action_dim = weights['diffusion_model.cond_embed.action_embed.0.weight'].shape[1]
    if action_dim != len(MANEUVER_STYLES):
        raise ValueError(f'Checkpoint has {action_dim} styles; this experiment requires '
                         f'{len(MANEUVER_STYLES)}. Select a matching checkpoint with --checkpoint.')
    model = AeroDM(config).to(device)
    model.load_state_dict(weights)
    model.eval()
    mean = checkpoint['mean'].to(device=device, dtype=torch.float32)
    std = checkpoint['std'].to(device=device, dtype=torch.float32)
    state_mean, state_std = mean[..., :config.state_dim], std[..., :config.state_dim]
    model.set_normalization_params(mean, std)
    model.set_obstacles_data(None)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cases = select_cases(config)
    rng = np.random.default_rng(args.seed)
    output = args.output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(checkpoint=str(args.checkpoint.resolve()), seed=args.seed,
                    device=str(device), radius_m=2.0, targets_per_style=10,
                    shared_diffusion_noise=True, obstacles=False, styles=list(MANEUVER_STYLES),
                    diffusion_steps=config.diffusion_steps)
    (output/'experiment.json').write_text(json.dumps(metadata, indent=2))
    with torch.no_grad():
        for style, name in enumerate(MANEUVER_STYLES):
            case = cases[style]
            states = case[:, :config.state_dim].to(device)
            history = ((states-state_mean[0])/state_std[0])[:config.history_len].unsqueeze(0)
            original = case[config.history_len:, 1:4].numpy()
            history_xyz = case[:config.history_len, 1:4].numpy()
            targets = sample_targets(original[-1], rng)
            target_tensor = torch.cat([torch.from_numpy(targets), torch.ones(10, 1)], dim=1).to(device)
            normalized_targets = normalize_target(target_tensor, state_mean, state_std)
            action = F.one_hot(torch.tensor([style], device=device), config.action_dim).float()
            predictions = []
            for i in range(10):
                # Same initial and reverse-process noise for all ten targets of this case.
                torch.manual_seed(args.seed + 1000 + style)
                with redirect_stdout(io.StringIO()):
                    prediction = model.sample(normalized_targets[i:i+1], action, history,
                                              batch_size=1, enable_guidance=False,
                                              plot_all_steps=False)
                prediction = prediction * state_std + state_mean
                predictions.append(prediction[0, :, 1:4].cpu().numpy())
            predictions = np.stack(predictions)
            if not np.isfinite(predictions).all():
                raise RuntimeError(f'Nonfinite predictions for {name}')
            errors = np.linalg.norm(predictions[:, -1]-targets, axis=1)
            np.savez_compressed(output/f'{style:02d}_{name}.npz', case=case.numpy(),
                                history=history_xyz, original=original, targets=targets,
                                predictions=predictions, endpoint_errors_m=errors)
            plot_case(output/f'{style:02d}_{name}.svg', name, history_xyz, original, targets, predictions)
            print(f'[{style+1}/{len(MANEUVER_STYLES)}] {name}: mean endpoint error {errors.mean():.3f} m', flush=True)
    print(f'Results: {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
