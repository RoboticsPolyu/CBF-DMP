"""DDPM forward and reverse processes with obstacle guidance."""

import numpy as np
import matplotlib.pyplot as plt
import torch

from .guidance import compute_barrier_and_grad, compute_barrier_and_grad_logistic

class ObstacleAwareDiffusionProcess:
    def __init__(self, config):
        self.config = config
        self.num_timesteps = config.diffusion_steps

        # Linear noise schedule - initialize on CPU, will move to device when needed
        self.betas = torch.linspace(config.beta_start, config.beta_end, config.diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)

        if t.dim() == 1:
            t = t.view(-1, 1, 1)

        # Move alpha_bars to the same device as x_0
        alpha_bars = self.alpha_bars.to(x_0.device)
        alpha_bar_t = alpha_bars[t]
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

        return x_t, noise

    def p_sample(self, model, x_t, t, target=None, action=None, history=None, enable_guidance=True, guidance_gamma=None,
                mean=None, std=None, plot_step=False, step_idx=0, obstacles_data=None):
        """
        Reverse diffusion process with obstacle-aware sampling.
        """
        batch_size = x_t.size(0)
        device = x_t.device
        # print("p_sample - enable_cbf_guidance: ", enable_guidance, "mean:", mean, "std: ", std, "gamma: ", guidance_gamma, "obstacles_data size: ", len(obstacles_data))
        with torch.no_grad():
            # Model prediction with obstacle information
            pred_x0 = model(x_t, t, target, action, history, obstacles_data)

            # Conditional prediction (with action) for guidance
            pred_x0_cond = model(x_t, t, target, action, history, obstacles_data)

            #  non-conditional prediction (zero out action) for guidance
            zero_action = torch.zeros_like(action)
            pred_x0_uncond = model(x_t, t, target, zero_action, history, obstacles_data)

            # Classifier-free guidance: pred_x0 = pred_x0_uncond + guidance_scale * (pred_x0_cond - pred_x0_uncond)
            if enable_guidance and self.config.guidance_scale != 1.0:
                pred_x0 = pred_x0_uncond + self.config.guidance_scale * (pred_x0_cond - pred_x0_uncond)
            else:
                pred_x0 = pred_x0_cond

            # Expand t for broadcasting
            t_exp = t.view(batch_size, 1, 1) if t.dim() == 1 else t.view(-1, 1, 1)

            # Move diffusion parameters to the same device as x_t
            alphas = self.alphas.to(x_t.device)
            betas = self.betas.to(x_t.device)
            alpha_bars = self.alpha_bars.to(x_t.device)

            alpha_bar_t = alpha_bars[t_exp.squeeze(1)].view(batch_size, 1, 1)
            alpha_t = alphas[t_exp.squeeze(1)].view(batch_size, 1, 1)
            beta_t = betas[t_exp.squeeze(1)].view(batch_size, 1, 1)
            one_minus_alpha_bar_t = 1 - alpha_bar_t

            # Compute predicted noise from pred_x0
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(one_minus_alpha_bar_t)
            # Predict noise ε_pred = (x_t - sqrt(α_bar_t) * pred_x0) / sqrt(1 - α_bar_t)
            ε_pred = (x_t - sqrt_alpha_bar_t * pred_x0) / sqrt_one_minus_alpha_bar_t

            barrier_info = None
            if enable_guidance and guidance_gamma is not None and mean is not None and std is not None:
                # Compute γ_t (scheduled: strongest at t=0 for final safety enforcement)
                gamma_t = guidance_gamma * (1.0 - t_exp.squeeze(1).float() / self.config.diffusion_steps)

                # Compute barrier gradient ∇V with multiple obstacles
                V, grad_V = compute_barrier_and_grad_logistic(
                    pred_x0, mean, std, obstacles_data,
                    safety_margin=self.config.safe_extra_factor,
                    sigma=self.config.barrier_sigma,
                    delta_t=self.config.delta_T,
                    cbf_decay_rate=self.config.cbf_decay_rate,
                    dynamics_consistency_weight=(
                        self.config.dynamics_consistency_weight))

                # V, grad_V = compute_barrier_and_grad(pred_x0, mean, std, obstacles_data, safety_margin=config.safe_extra_factor);

                barrier_info = {'V': V, 'grad_V': grad_V, 'gamma_t': gamma_t}
                # print(barrier_info)
                # Guided score: s_guided = s_theta - γ_t ∇V
                sigma_t = sqrt_one_minus_alpha_bar_t
                # ε_guided = mu_pred - gamma_t.view(batch_size, 1, 1) * grad_V
                # norm(grad_V) << norm(mu_pred), sigma_t: 1->0, gamma_t: 0->guidance_gamma
                ε_guided = ε_pred + gamma_t.view(batch_size, 1, 1) * grad_V * sqrt_one_minus_alpha_bar_t
                # ε_pred = score_function * - sqrt_one_minus_alpha_bar_t
                # norm(score_function) = norm(ε_pred) / sigma_t; norm(ε_pred)^2 ~ χ2(D)

                s_norm = torch.norm(ε_pred, dim=(1,2), keepdim=True)
                grad_norm = torch.norm(gamma_t.view(batch_size, 1, 1) * grad_V, dim=(1,2), keepdim=True)
                # print("s_norm: ", s_norm, "grad_norm: ", grad_norm,  "gamma_t: ", gamma_t, "sigma_t: ", sigma_t)

            else:
                ε_guided = ε_pred

            # Compute mean μ using guided noise (standard DDPM formula)
            coeff = (1 - alpha_t) / sqrt_one_minus_alpha_bar_t
            # x_{t-1} mean = 1/sqrt(α_t) * (x_t - (1 - α_t) / sqrt(1 - α_bar_t) * ε_guided)
            mu = (1 / torch.sqrt(alpha_t)) * (x_t - coeff * ε_guided)

            # For t=0, return pred_x0 (or guided equivalent)
            is_t_zero = (t_exp.squeeze(1) == 0).all()
            if is_t_zero:
                # Compute guided pred_x0 for consistency
                pred_x0_guided = (x_t - sqrt_one_minus_alpha_bar_t * ε_guided) / sqrt_alpha_bar_t

                if plot_step:
                    self._plot_diffusion_step(x_t, pred_x0_guided, t, step_idx, barrier_info, is_final=True, mean=mean, std=std, obstacles_data=obstacles_data)
                return pred_x0_guided

            # Variance (DDPM posterior variance)
            alpha_bar_prev = alpha_bars[t_exp.squeeze(1) - 1].view(batch_size, 1, 1) if t.min() > 0 else torch.ones_like(alpha_bar_t)
            # var = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
            var = beta_t * (1 - alpha_bar_prev) / one_minus_alpha_bar_t
            sigma = torch.sqrt(var)

            # Sample noise
            z = torch.randn_like(x_t)

            # x_{t-1} = μ + σ * z
            x_prev = mu + sigma * z

            if plot_step:
                self._plot_diffusion_step(x_t, x_prev, t, step_idx, barrier_info, is_final=False, mean=mean, std=std, obstacles_data=obstacles_data)

            return x_prev

    def _plot_diffusion_step(self, x_t, x_prev, t, step_idx, barrier_info=None, is_final=False, mean=None, std=None, obstacles_data=None):
        """
        Plot the current diffusion step with obstacles.
        Denormalizes positions for visualization if mean/std provided; otherwise plots raw normalized values.
        Supports 3D trajectories, projections, stats, CBF info, and step details.
        """
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f'Reverse Diffusion Process - Step {step_idx} (t={t[0].item()})', fontsize=16)

        # Extract position coordinates with conditional denormalization
        if mean is not None and std is not None:
            # Denormalize for real-world scale (assumes position indices 1:4 for x,y,z)
            pos_mean = mean[0, 0, 1:4].cpu().numpy()
            pos_std = std[0, 0, 1:4].cpu().numpy()
            x_t_pos = x_t[0, :, 1:4].cpu().numpy() * pos_std + pos_mean
            x_prev_pos = x_prev[0, :, 1:4].cpu().numpy() * pos_std + pos_mean
        else:
            # Fallback: Plot raw normalized positions (mean~0, std~1)
            x_t_pos = x_t[0, :, 1:4].cpu().numpy()
            x_prev_pos = x_prev[0, :, 1:4].cpu().numpy()

        # Fixed bounds based on trajectory generation ranges (centers -20~20, radius~10, climb~40; safe cover -50 to 50)
        # fixed_min = np.array([-20.0, -20.0, -20.0])
        # fixed_max = np.array([20.0, 20.0, 20.0])

        # 1. 3D trajectory evolution with obstacles
        ax1 = fig.add_subplot(241, projection='3d')
        ax1.plot(x_t_pos[:, 0], x_t_pos[:, 1], x_t_pos[:, 2], 'r-', label='x_t (current)', linewidth=2, alpha=0.7)
        ax1.plot(x_prev_pos[:, 0], x_prev_pos[:, 1], x_prev_pos[:, 2], 'b-', label='x_prev (denoised)', linewidth=2, alpha=0.7)

        # Plot obstacles if available (assumes centers are already denormalized)
        # if obstacles_data:
        #     for obstacle in obstacles_data:
        #         center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
        #         radius = obstacle['radius']

        #         # Create sphere surface for 3D visualization
        #         u = np.linspace(0, 2 * np.pi, 10)
        #         v = np.linspace(0, np.pi, 10)
        #         x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        #         y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        #         z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        #         ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='red')

        # Set fixed equal-range limits for X/Y/Z to prevent distortion (spheres look spherical)
        # ax1.set_xlim(fixed_min[0], fixed_max[0])
        # ax1.set_ylim(fixed_min[1], fixed_max[1])
        # ax1.set_zlim(fixed_min[2], fixed_max[2])

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        ax1.set_title('3D Trajectory Evolution with Obstacles')
        ax1.grid(True)

        # 2. Position components over time
        time_steps = np.arange(len(x_t_pos))
        ax2 = fig.add_subplot(242)
        ax2.plot(time_steps, x_t_pos[:, 0], 'r-', label='x_t X', linewidth=2, alpha=0.7)
        ax2.plot(time_steps, x_prev_pos[:, 0], 'b-', label='x_prev X', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('X Position')
        ax2.legend()
        ax2.set_title('X Position Over Time')
        ax2.grid(True)

        ax3 = fig.add_subplot(243)
        ax3.plot(time_steps, x_t_pos[:, 1], 'r-', label='x_t Y', linewidth=2, alpha=0.7)
        ax3.plot(time_steps, x_prev_pos[:, 1], 'b-', label='x_prev Y', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Y Position')
        ax3.legend()
        ax3.set_title('Y Position Over Time')
        ax3.grid(True)

        ax4 = fig.add_subplot(244)
        ax4.plot(time_steps, x_t_pos[:, 2], 'r-', label='x_t Z', linewidth=2, alpha=0.7)
        ax4.plot(time_steps, x_prev_pos[:, 2], 'b-', label='x_prev Z', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Z Position')
        ax4.legend()
        ax4.set_title('Z Position Over Time')
        ax4.grid(True)

        # 3. Noise and prediction statistics (on full normalized states)
        ax5 = fig.add_subplot(245)
        stats_labels = ['x_t Mean', 'x_t Std', 'x_prev Mean', 'x_prev Std']
        stats_values = [
            x_t.mean().item(), x_t.std().item(),
            x_prev.mean().item(), x_prev.std().item()
        ]
        bars = ax5.bar(stats_labels, stats_values, color=['red', 'red', 'blue', 'blue'])
        ax5.set_ylabel('Value')
        ax5.set_title('Statistical Properties')
        for bar, value in zip(bars, stats_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.3f}',
                    ha='center', va='bottom')

        # 4. Position differences (using denormalized positions)
        ax6 = fig.add_subplot(246)
        pos_diff = np.linalg.norm(x_prev_pos - x_t_pos, axis=1)
        ax6.plot(time_steps, pos_diff, 'g-', linewidth=2)
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Position Difference')
        ax6.set_title('Position Change Magnitude')
        ax6.grid(True)

        # 5. CBF Barrier information (if available)
        if barrier_info is not None:
            ax7 = fig.add_subplot(247)
            V = barrier_info['V'].item()
            gamma_t = barrier_info['gamma_t'][0].item() if barrier_info['gamma_t'].numel() == 1 else barrier_info['gamma_t'].mean().item()
            grad_norm = barrier_info['grad_V'].norm().item()

            cbf_data = [V, gamma_t, grad_norm]
            cbf_labels = ['Barrier V', 'Gamma_t', 'Grad Norm']
            bars = ax7.bar(cbf_labels, cbf_data, color=['purple', 'orange', 'green'])
            ax7.set_ylabel('Value')
            ax7.set_title('CBF Guidance Information')
            for bar, value in zip(bars, cbf_data):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.3f}',
                        ha='center', va='bottom')

        # 6. Step information
        ax8 = fig.add_subplot(248)
        step_info = {
            'Step': step_idx,
            'Timestep': t[0].item(),
            'Is Final': is_final,
            'Batch Size': x_t.size(0),
            'Seq Len': x_t.size(1)
        }
        ax8.axis('off')
        info_text = '\n'.join([f'{k}: {v}' for k, v in step_info.items()])
        ax8.text(0.1, 0.9, info_text, transform=ax8.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Modified display/save logic
        if self.config.show_flag:
            plt.show()
        else:
            # Save as SVG with descriptive filename
            filename = f"Figs/diffusion_step_{step_idx:03d}_t_{t[0].item():03d}.svg"
            plt.savefig(filename, format='svg', bbox_inches='tight')
            plt.close()  # Close the figure to free memory

        # Print step information (normalized stats for debugging)
        print(f"\n=== Diffusion Step {step_idx} (t={t[0].item()}) ===")
        print(f"x_t shape: {x_t.shape}")
        print(f"x_t stats - Mean: {x_t.mean().item():.4f}, Std: {x_t.std().item():.4f}")
        print(f"x_prev stats - Mean: {x_prev.mean().item():.4f}, Std: {x_prev.std().item():.4f}")
        if barrier_info is not None:
            print(f"CBF - Barrier V: {barrier_info['V'].item():.4f}, Gamma_t: {barrier_info['gamma_t'][0].item():.4f}")
