"""Reusable training, checkpoint, and evaluation orchestration."""

from dataclasses import dataclass
from datetime import datetime
import __main__
import os
import sys
import time

import torch
import torch.nn.functional as F

from Trajectory_Gen import generate_aerobatic_trajectories_pvR

from .config import Config
from .data import (
    denormalize_target,
    denormalize_trajectories,
    generate_random_obstacles,
    generate_target_waypoints,
    normalize_target,
    normalize_trajectories,
    project_target_outside_obstacles,
)
from .evaluation import test_model_performance_cb_eva
from .losses import AeroDMLoss
from .model import AeroDM
from .reporting import format_progress


@dataclass
class DatasetBundle:
    """Raw and normalized train/test trajectories plus normalization state."""

    train_trajectories: torch.Tensor
    test_trajectories: torch.Tensor
    train_norm: torch.Tensor
    test_norm: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def renormalize_test(self, mean, std, device):
        """Apply checkpoint statistics to raw test data."""
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.test_norm, _, _ = normalize_trajectories(
            self.test_trajectories.to(device), mean=self.mean, std=self.std)


def select_device():
    """Select CUDA, Apple MPS, or CPU in priority order."""
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def build_criterion(config):
    """Create the configured training objective."""
    return AeroDMLoss(
        config,
        enable_obstacle_term=config.use_obstacle_loss,
        safe_extra_factor=config.safe_extra_factor,
        last_xyz_weight=config.last_xyz_weight,
        xyz_weight=config.xyz_weight,
        vel_weight=config.vel_weight,
        other_weight=config.other_weight,
        obstacle_weight=config.obstacle_weight,
        continuity_weight=config.continuity_weight,
        acc_weight=config.acc_weight,
        dynamics_consistency_weight=config.dynamics_consistency_weight,
    )


def prepare_datasets(config, device, seed=42):
    """Generate, split, normalize, and place trajectory data on the device."""
    print("Generating training data with obstacle-aware transformer...")
    trajectories = generate_aerobatic_trajectories_pvR(
        num_trajectories=config._base_num_trajectories,
        seq_len=config.seq_len + config.history_len,
        delta_T=config.delta_T,
    )

    torch.manual_seed(seed)
    indices = torch.randperm(trajectories.shape[0])
    train_size = int(0.9 * trajectories.shape[0])
    train_trajectories = trajectories[indices[:train_size]]
    test_trajectories = trajectories[indices[train_size:]]

    train_norm, mean, std = normalize_trajectories(train_trajectories)
    test_norm, _, _ = normalize_trajectories(
        test_trajectories, mean=mean, std=std)

    return DatasetBundle(
        train_trajectories=train_trajectories,
        test_trajectories=test_trajectories,
        train_norm=train_norm.to(device),
        test_norm=test_norm.to(device),
        mean=mean.to(device),
        std=std.to(device),
    )


def _build_training_obstacles(full_traj, target, config, mean, std, device):
    """Generate batch obstacles and project targets outside their buffers."""
    batch_size = full_traj.shape[0]
    obstacles_for_batch = []
    for batch_idx in range(batch_size):
        trajectory = denormalize_trajectories(
            full_traj[batch_idx:batch_idx + 1, config.history_len:],
            mean,
            std,
        )
        obstacles_for_batch.append(generate_random_obstacles(
            trajectory[0],
            num_obstacles_range=(3, 5),
            radius_range=(0.5, 1.0),
            check_collision=False,
            device=device,
        ))

    target_denorm = denormalize_target(target, mean, std)
    projected_targets = [
        project_target_outside_obstacles(
            target_denorm[index:index + 1],
            obstacles_for_batch[index],
            safety_margin=config.safe_extra_factor,
        )
        for index in range(batch_size)
    ]
    target = normalize_target(torch.cat(projected_targets, dim=0), mean, std)
    return obstacles_for_batch, target


def train(model, criterion, optimizer, datasets, config, device):
    """Train the diffusion model and return epoch loss histories."""
    print("=" * 60)
    print("TRAINING MODE: Training new model from scratch")
    print("=" * 60)

    train_size = datasets.train_trajectories.shape[0]
    losses = {
        "total": [], "position": [], "vel": [],
        "obstacle": [], "continuity": [],
    }
    mode = "with obstacle-aware loss" if config.use_obstacle_loss else "with basic loss"
    print(f"Starting training {mode}...")
    start_time = time.time()

    for epoch in range(config.num_epochs):
        model.train()
        totals = {
            "total": 0.0, "position": 0.0, "vel": 0.0,
            "obstacle": 0.0, "continuity": 0.0,
        }
        num_batches = 0
        shuffled_indices = torch.randperm(train_size)

        for start in range(0, train_size, config.batch_size):
            batch_indices = shuffled_indices[start:start + config.batch_size]
            full_traj = datasets.train_norm[batch_indices]
            actual_batch_size = full_traj.shape[0]

            style_info = full_traj[:, :, -1:]
            state = full_traj[:, :, :-1]
            history = state[:, :config.history_len]
            x_0 = state[:, config.history_len:config.history_len + config.seq_len]
            target = generate_target_waypoints(x_0)
            action = F.one_hot(
                style_info[:, -1, 0].long(),
                num_classes=config.action_dim,
            ).float()

            timestep = torch.randint(
                0, config.diffusion_steps, (actual_batch_size,), device=device)
            x_t, _ = model.diffusion_process.q_sample(
                x_0, timestep, torch.randn_like(x_0))

            obstacles = None
            if config.use_obstacle_loss or config.enable_obstacle_encoding:
                obstacles, target = _build_training_obstacles(
                    full_traj, target, config,
                    datasets.mean, datasets.std, device)

            pred_x0 = model(
                x_t, timestep, target, action, history,
                obstacles_data=obstacles)
            batch_losses = criterion(
                pred_x0,
                x_0,
                obstacles if config.use_obstacle_loss else None,
                datasets.mean,
                datasets.std,
                history,
            )
            total_loss, position_loss, vel_loss, obstacle_loss, continuity_loss = batch_losses

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            totals["total"] += total_loss.item()
            totals["position"] += position_loss.item()
            totals["vel"] += vel_loss.item()
            totals["obstacle"] += obstacle_loss.item()
            totals["continuity"] += continuity_loss.item()
            num_batches += 1

        averages = {key: value / max(num_batches, 1) for key, value in totals.items()}
        for key in ("total", "position", "vel", "continuity"):
            losses[key].append(averages[key])
        if config.use_obstacle_loss:
            losses["obstacle"].append(averages["obstacle"])

        sys.stdout.write(format_progress(
            epoch,
            config.num_epochs,
            start_time,
            averages["total"],
            averages["position"],
            averages["vel"],
            averages["obstacle"],
            averages["continuity"],
            config.use_obstacle_loss,
        ))
        sys.stdout.flush()

    print(f"\nTraining completed after {config.num_epochs} epochs.")
    print(f"Training finished in {time.time() - start_time:.2f} seconds.")
    return losses


def save_checkpoint(model, optimizer, datasets, losses, config):
    """Save timestamped and latest checkpoints, returning the latest path."""
    os.makedirs(config.model_save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = config.model_filename.split("_")[0]
    timestamped_path = os.path.join(
        config.model_save_dir, f"{base_name}_{timestamp}.pth")
    latest_path = os.path.join(config.model_save_dir, config.model_filename)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": config.num_epochs - 1,
        "loss": losses,
        "mean": datasets.mean,
        "std": datasets.std,
        "use_obstacle_loss": config.use_obstacle_loss,
        "config": config,
    }
    torch.save(checkpoint, timestamped_path)
    torch.save(checkpoint, latest_path)
    print(f"Model saved to: {timestamped_path}")
    print(f"Latest model saved to: {latest_path}")
    return latest_path


def load_checkpoint(model, datasets, config, device):
    """Load model and normalization state from the configured checkpoint."""
    path = os.path.join(config.model_save_dir, config.model_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. Set train_model=True to train it.")

    print("=" * 60)
    print("LOADING MODE: Loading pre-trained model")
    print("=" * 60)
    print(f"Loading model from: {path}")
    # Historical checkpoints were written while the monolithic script ran as
    # ``__main__`` and therefore reference ``__main__.Config`` in the pickle.
    if not hasattr(__main__, "Config"):
        __main__.Config = Config
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    datasets.renormalize_test(
        checkpoint["mean"], checkpoint["std"], device)
    model.set_normalization_params(datasets.mean, datasets.std)
    config.use_obstacle_loss = checkpoint.get(
        "use_obstacle_loss", config.use_obstacle_loss)

    print("Model loaded successfully!")
    print(f"  - Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  - Use obstacle loss: {config.use_obstacle_loss}")
    if checkpoint.get("loss", {}).get("total"):
        print(f"  - Final loss: {checkpoint['loss']['total'][-1]:.4f}")
    return checkpoint


def run_experiment(config=None):
    """Run the configured train/load workflow followed by evaluation."""
    config = config or Config()
    device = select_device()
    print("Training Obstacle-Aware AeroDM with Transformer Integration and Obstacle-Aware Loss...")
    print(f"Using device: {device}")

    datasets = prepare_datasets(config, device)
    criterion = build_criterion(config)
    model = AeroDM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.set_normalization_params(datasets.mean, datasets.std)

    if config.train_model:
        losses = train(
            model, criterion, optimizer, datasets, config, device)
        save_checkpoint(model, optimizer, datasets, losses, config)
    else:
        load_checkpoint(model, datasets, config, device)

    print("\n" + "=" * 60)
    print("TESTING PHASE: Running model evaluation")
    print("=" * 60)
    test_model_performance_cb_eva(
        model,
        datasets.test_norm,
        datasets.mean,
        datasets.std,
        num_test_samples=config.num_test_samples,
        show_flag=config.show_flag,
        show_collision_points=config.show_collision_points,
    )
    return model, datasets
