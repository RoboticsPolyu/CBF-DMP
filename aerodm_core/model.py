"""High-level AeroDM model and sampling interface."""

import torch
import torch.nn as nn

from .diffusion import ObstacleAwareDiffusionProcess
from .models import ObstacleAwareDiffusionTransformer

class AeroDM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.diffusion_model = ObstacleAwareDiffusionTransformer(config)
        self.diffusion_process = ObstacleAwareDiffusionProcess(config)
        self.mean = None
        self.std = None
        self.obstacles_data = None

    def forward(self, x_t, t, target=None, action=None, history=None, obstacles_data=None):
        return self.diffusion_model(x_t, t, target, action, history, obstacles_data)

    def set_normalization_params(self, mean, std):
        """Set normalization parameters for CBF guidance"""
        self.mean = mean
        self.std = std

    def set_obstacles_data(self, obstacles_data):
        """Set obstacles data: must be list of list of dicts"""
        if obstacles_data is None:
            self.obstacles_data = None
        else:
            self.obstacles_data = obstacles_data
        print(f"Set obstacles_data: {len(self.obstacles_data) if self.obstacles_data else 0} batches")

    def sample(self, target=None, action=None, history=None, batch_size=1, enable_guidance=True,
               guidance_gamma=None, plot_all_steps=False):
        device = next(self.parameters()).device

        if target is None:
            target = torch.ones(batch_size, self.config.target_dim).to(device)* 1e-6
        if action is None:
            action = torch.zeros(batch_size, self.config.action_dim).to(device)

        # Initialize with noise
        x_t = torch.randn(batch_size, self.config.seq_len, self.config.state_dim).to(device)

        # Optional: Soft init from history for better continuity
        if history is not None:
            last_pos = history[:, -1, 1:4]
            last_vel = history[:, -1, 1:4] - history[:, -2, 1:4] if history.size(1) > 1 else torch.zeros_like(last_pos)
            init_first_pos = last_pos.unsqueeze(1) + last_vel.unsqueeze(1)
            x_t[:, :1, 1:4] = 0.5 * x_t[:, :1, 1:4] + 0.5 * init_first_pos

        print(f"\n{'='*50}")
        print("STARTING OBSTACLE-AWARE REVERSE DIFFUSION PROCESS")
        print(f"Initial noise stats - Mean: {x_t.mean().item():.4f}, Std: {x_t.std().item():.4f}")
        print(f"Total steps: {self.config.inference_diffusion_steps}")
        print(f"CBF Guidance: {enable_guidance}")
        if self.obstacles_data:
            print(f"Number of obstacles: {len(self.obstacles_data)}")
            print(f"Obstacle information integrated into transformer")
        print(f"{'='*50}")

        # Reverse diffusion process
        step_counter = 0
        for t_step in reversed(range(self.config.inference_diffusion_steps)):
            t_batch = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
            gamma = guidance_gamma if enable_guidance else None

            # Plot every step if requested, or key steps for overview
            # plot_step = plot_all_steps or (t_step % max(1, self.config.inference_diffusion_steps // 5) == 0) or t_step == 0

            # debug:
            plot_step = False
            x_t = self.diffusion_process.p_sample(
                self.diffusion_model, x_t, t_batch, target, action, history, enable_guidance,
                gamma, self.mean, self.std, plot_step=plot_step, step_idx=step_counter,
                obstacles_data=self.obstacles_data
            )
            step_counter += 1

        print(f"\n{'='*50}")
        print("OBSTACLE-AWARE REVERSE DIFFUSION PROCESS COMPLETED")
        print(f"Final trajectory stats - Mean: {x_t.mean().item():.4f}, Std: {x_t.std().item():.4f}")
        print(f"{'='*50}")

        return x_t
