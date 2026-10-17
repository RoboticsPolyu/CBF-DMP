"""Multi-history AeroDM demo: python demo_multi_agent.py --agents 4 --last-m 15."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from AeroDM_ObstacleAware_Training_Test import (
    AeroDM, Config, load_model_checkpoint, generate_target_waypoints,
    generate_aerobatic_trajectories_pvR, normalize_trajectories,
)


def pairwise_distances(positions):
    """Return [N*(N-1)/2, T]; compare matching times only, no self pairs."""
    i, j = torch.triu_indices(positions.shape[0], positions.shape[0], 1,
                             device=positions.device)
    return (positions[i] - positions[j]).norm(dim=-1)


class MultiAgentGuidance:
    """Descent on sum_{i<j,t} .5 * relu(distance - ||p_i-p_j||)^2.

    Each iteration updates all agents simultaneously in physical units. Exactly
    coincident points use deterministic antisymmetric directions to break ties.
    Only position channels 1:4 are changed; this is a geometric demo, not a
    dynamically constrained planner. The input batch must be ONE shared scene.
    """
    def __init__(self, mean, std, collision_distance=1.0, strength=0.4,
                 iterations=20, safety_margin=0.05):
        if not math.isfinite(collision_distance) or collision_distance <= 0:
            raise ValueError('collision_distance must be finite and positive')
        if not math.isfinite(strength) or not 0 < strength <= 0.5:
            raise ValueError('strength must be in (0, 0.5]')
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if not math.isfinite(safety_margin) or safety_margin < 0:
            raise ValueError('safety_margin must be finite and nonnegative')
        self.mean, self.std = mean[..., 1:4], std[..., 1:4]
        if not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('position std must be finite and positive')
        self.distance, self.strength, self.iterations = collision_distance + safety_margin, strength, iterations
        self.calls = 0

    @torch.no_grad()
    def __call__(self, prediction):
        self.calls += 1
        n = prediction.shape[0]
        if n < 2:
            return prediction
        pos = prediction[..., 1:4] * self.std + self.mean
        # Unique directions for coincident agents; antisymmetry preserves centroid.
        angles = torch.arange(n, device=pos.device, dtype=pos.dtype) * (2 * math.pi / n)
        anchors = torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), -1)
        fallback = anchors[:, None] - anchors[None, :]
        fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        mask = ~torch.eye(n, dtype=torch.bool, device=pos.device)
        for _ in range(self.iterations):
            delta = pos[:, None] - pos[None, :]  # [agent, other agent, time, xyz]
            dist = delta.norm(dim=-1)
            direction = delta / dist[..., None].clamp_min(1e-8)
            direction = torch.where((dist < 1e-8)[..., None], fallback[:, :, None], direction)
            overlap = (self.distance - dist).clamp_min(0) * mask[..., None]
            # Degree normalization bounds the collective step for large fleets.
            correction = (overlap[..., None] * direction).sum(dim=1) / (n - 1)
            pos = pos + self.strength * correction
        result = prediction.clone()
        result[..., 1:4] = (pos - self.mean) / self.std
        return result



@torch.no_grad()
def sample_multi_agent(model, history, target, action, guidance=None, last_m=15):
    """Joint DDPM sampling. M counts reverse diffusion steps, not future frames."""
    config = model.config
    if not isinstance(last_m, int) or not 0 <= last_m <= config.diffusion_steps:
        raise ValueError('last_m must be an integer in [0, diffusion_steps]')
    n = history.shape[0]
    x = torch.randn(n, config.seq_len, config.state_dim, device=history.device)
    last = history[:, -1, 1:4]
    velocity = last - history[:, -2, 1:4] if history.shape[1] > 1 else torch.zeros_like(last)
    x[:, :1, 1:4] = 0.5 * x[:, :1, 1:4] + 0.5 * (last + velocity)[:, None]
    alphas, betas, bars = model.diffusion_process._schedule_on(history.device)
    for step in reversed(range(config.diffusion_steps)):
        t = torch.full((n,), step, dtype=torch.long, device=x.device)
        if config.guidance_scale != 1.0:
            conditional, unconditional = model(
                torch.cat((x, x)), torch.cat((t, t)), torch.cat((target, target)),
                torch.cat((action, torch.zeros_like(action))), torch.cat((history, history)),
                None).chunk(2)
            clean = unconditional + config.guidance_scale * (conditional - unconditional)
        else:
            clean = model(x, t, target, action, history, None)
        if guidance is not None and step < last_m:
            clean = guidance(clean)
        if step == 0:
            return clean
        noise = (x - bars[step].sqrt() * clean) / (1 - bars[step]).sqrt()
        mu = (x - betas[step] / (1 - bars[step]).sqrt() * noise) / alphas[step].sqrt()
        variance = betas[step] * (1 - bars[step - 1]) / (1 - bars[step])
        x = mu + variance.sqrt() * torch.randn_like(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agents', type=int, default=4)
    parser.add_argument('--last-m', type=int, default=60, help='Final M reverse diffusion steps')
    parser.add_argument('--collision-distance', type=float, default=1.0, help='Physical center-to-center distance')
    parser.add_argument('--safety-margin', type=float, default=0.05, help='Extra physical clearance used by guidance')
    parser.add_argument('--strength', type=float, default=0.4)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', default=str(Path(Config.model_save_dir) / Config.model_filename))
    parser.add_argument('--input', type=Path, help='Optional NPZ: history [N,H,10], target [N,3], style [N], physical units')
    parser.add_argument('--output', type=Path, default=Path('output/multi_agent'))
    args = parser.parse_args()
    if args.agents < 2:
        parser.error('--agents must be >= 2')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    checkpoint = load_model_checkpoint(args.checkpoint, device)
    config = Config()
    saved = checkpoint.get('config_values', {})
    if not saved and 'config' in checkpoint:
        saved = {k: getattr(checkpoint['config'], k) for k in dir(checkpoint['config']) if not k.startswith('_')}
    for key, value in saved.items():
        if hasattr(config, key) and not callable(value):
            setattr(config, key, value)
    config.device = args.device
    if not 0 <= args.last_m <= config.diffusion_steps:
        parser.error('--last-m must be between 0 and checkpoint diffusion_steps')
    model = AeroDM(config).to(device).eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    mean, std = checkpoint['mean'].to(device), checkpoint['std'].to(device)
    model.set_normalization_params(mean, std)
    if args.input:
        with np.load(args.input) as data:
            history_real = torch.as_tensor(data['history'], dtype=torch.float32, device=device)
            target_real = torch.as_tensor(data['target'], dtype=torch.float32, device=device)
            styles = torch.as_tensor(data['style'], dtype=torch.long, device=device)
        n = history_real.shape[0]
        if (n < 2 or history_real.shape != (n, config.history_len, config.state_dim)
                or target_real.shape != (n, 3) or styles.shape != (n,)):
            raise ValueError('Input shapes must be history [N,H,10], target [N,3], style [N], N>=2')
        if not torch.isfinite(history_real).all() or not torch.isfinite(target_real).all():
            raise ValueError('Input states and targets must be finite')
        history = (history_real - mean[..., :config.state_dim]) / std[..., :config.state_dim]
        target = torch.cat(((target_real - mean[0, 0, 1:4]) / std[0, 0, 1:4], torch.ones(n, 1, device=device)), -1)
    else:
        # Translate distinct generated maneuvers to meet at a shared future time.
        raw = generate_aerobatic_trajectories_pvR(args.agents, config.history_len + config.seq_len,
                                                delta_T=config.delta_T).to(device)
        meeting = raw.new_tensor([0., 0., 10.])
        raw[..., 1:4] += (meeting - raw[:, config.history_len + config.seq_len // 2, 1:4])[:, None]
        normalized, _, _ = normalize_trajectories(raw, mean, std)
        history = normalized[:, :config.history_len, :config.state_dim]
        history_real = raw[:, :config.history_len, :config.state_dim]
        target = generate_target_waypoints(normalized[:, config.history_len:, :config.state_dim])
        styles = raw[:, -1, -1].long()
        n = args.agents
    action = torch.nn.functional.one_hot(styles, config.action_dim).float()
    guidance = MultiAgentGuidance(mean, std, args.collision_distance, args.strength, args.iterations, args.safety_margin)
    outputs = {}
    for name, enabled in [('unguided', False), ('guided', True)]:
        torch.manual_seed(args.seed)  # Identical initial and reverse-process noise.
        result = sample_multi_agent(model, history, target, action,
                                    guidance if enabled else None, args.last_m)
        outputs[name] = result * std[..., :config.state_dim] + mean[..., :config.state_dim]
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    # Two-column paper figure (183 mm wide); all sizes are in points.
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6,
                         'ytick.labelsize': 6, 'legend.fontsize': 6,
                         'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
                         'mathtext.fontset': 'dejavuserif', 'text.color': 'black',
                         'axes.labelcolor': 'black', 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(7.2, 3.0))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1],
                           left=0.015, right=0.98, bottom=0.34, top=0.97, wspace=0.30)
    # Avoid using the collision-marker red for an aircraft trajectory.
    palette = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#332288']
    def colors(index):
        return palette[index % len(palette)]
    all_points = torch.cat([history_real[..., 1:4].reshape(-1, 3)] +
                           [v[..., 1:4].reshape(-1, 3) for v in outputs.values()]).cpu().numpy()
    lower, upper = all_points.min(0), all_points.max(0)
    span = np.maximum(upper - lower, 0.5)
    lower, upper = lower - 0.07 * span, upper + 0.07 * span
    time = np.arange(1, config.seq_len + 1) * config.delta_T
    for panel, (name, states) in enumerate(outputs.items(), 1):
        positions = states[..., 1:4]
        distances = pairwise_distances(positions)
        metrics[name] = dict(min_distance=float(distances.min()),
                             colliding_pair_timesteps=int((distances < args.collision_distance).sum()))
        ax = fig.add_subplot(grid[0, panel - 1], projection='3d', computed_zorder=False)
        ax.view_init(elev=25, azim=-58)
        ax.set_box_aspect(upper - lower)
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
               zlim=(lower[2], upper[2]))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(4))
            axis.pane.fill = False
            axis._axinfo['grid'].update(color='#e2e8f0', linewidth=0.4)
        ax.tick_params(labelsize=6, pad=-2)
        for i in range(n):
            h = history_real[i, :, 1:4].cpu().numpy()
            p = positions[i].cpu().numpy()
            ax.plot(*h.T, '-', color=colors(i % 10), alpha=0.65, linewidth=0.8)
            ax.plot(*p.T, '--', color=colors(i % 10), label=f'UAV {i+1}', linewidth=1.0)
            ax.scatter(*p[-1], color=colors(i % 10), marker='x', s=10, linewidths=0.7)
        # Mark both UAV positions at each colliding synchronous timestep.
        # Connecting segments identify the offending pair, not a flight path.
        pair_i, pair_j = torch.triu_indices(n, n, 1, device=positions.device)
        pair_ids, frames = (distances < args.collision_distance).nonzero(as_tuple=True)
        collision_points = torch.zeros(positions.shape[:2], dtype=torch.bool,
                                       device=positions.device)
        collision_points[pair_i[pair_ids], frames] = True
        collision_points[pair_j[pair_ids], frames] = True
        points = positions[collision_points].cpu().numpy()
        if len(points):
            ax.scatter(*points.T, s=10, facecolors='none', edgecolors='crimson',
                       linewidths=0.6, depthshade=False, label='Collision points', zorder=5)
            segments = torch.stack((positions[pair_i[pair_ids], frames],
                                    positions[pair_j[pair_ids], frames]), dim=1).cpu().numpy()
            for segment in segments:
                ax.plot(*segment.T, color='crimson', linestyle=':', linewidth=0.4, alpha=0.35)
        ax.set(xlabel=r'$x$ (m)', ylabel=r'$y$ (m)', zlabel=r'$z$ (m)')
        ax.xaxis.labelpad = -5
        ax.yaxis.labelpad = -5
        ax.zaxis.labelpad = -4
    ax = fig.add_subplot(grid[0, 2])
    ax.axhspan(0, args.collision_distance, color='#fff1f2', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e2e8f0', linewidth=0.4)
    for name, states in outputs.items():
        minimum = pairwise_distances(states[..., 1:4]).min(0).values.cpu().numpy()
        line, = ax.plot(time, minimum, label='Without guidance' if name == 'unguided' else 'With guidance',
                        linewidth=1.0, linestyle='--' if name == 'unguided' else '-',
                        color='#555555' if name == 'unguided' else '#0072B2')
        colliding = minimum < args.collision_distance
        if colliding.any():
            ax.scatter(time[colliding], minimum[colliding], s=10, marker='x',
                       color='crimson', label='_nolegend_', linewidths=0.7, zorder=5)
    ax.axhline(args.collision_distance, color='crimson', linestyle=':', linewidth=0.8, label=r'$d_{\mathrm{safe}}$')
    ax.set(xlabel='Future time (s)', ylabel='Minimum separation (m)',
           xlim=(time[0], time[-1]), ylim=(0, None))
    ax.legend(fontsize=5.7, frameon=False, loc='upper left', handlelength=2.2)
    ax.tick_params(width=0.6, length=2)
    ax.yaxis.labelpad = 2
    handles = [Line2D([0], [0], color=colors(i % 10), lw=1, label=f'UAV {i+1}')
               for i in range(n)]
    handles += [Line2D([0], [0], color='#64748b', ls='-', label='History'),
                Line2D([0], [0], color='#64748b', ls='--', label='Prediction'),
                Line2D([0], [0], color='#64748b', marker='x', ls='none', label='Final point'),
                Line2D([0], [0], color='crimson', marker='o', markerfacecolor='none',
                       ls='none', label='Collision point'),
                Line2D([0], [0], color='crimson', ls=':', label='Colliding pair')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.015),
               ncol=min(len(handles), 5), frameon=False, fontsize=6,
               handlelength=2, columnspacing=1.5, markerscale=0.7)
    fig.savefig(args.output / 'multi_uav_collision.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(args.output / 'multi_uav_collision.svg', bbox_inches='tight', pad_inches=0.03)
    fig.savefig(args.output / 'multi_uav_collision.pdf', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    np.savez(args.output / 'trajectories.npz', history=history_real.cpu().numpy(),
             target=(target[:, :3] * std[0, 0, 1:4] + mean[0, 0, 1:4]).cpu().numpy(),
             style=styles.cpu().numpy(), **{k: v.cpu().numpy() for k, v in outputs.items()})
    report = dict(metrics=metrics, guidance_calls=guidance.calls,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
