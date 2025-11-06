# AeroDM + Barrier Function (CBF) implementation
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt

# Configuration parameters based on the paper
class Config:
    # Model dimensions
    latent_dim = 256
    num_layers = 4
    num_heads = 4
    dropout = 0.1
    
    # Diffusion parameters
    diffusion_steps = 30
    beta_start = 0.0001
    beta_end = 0.02
    
    # Sequence parameters
    seq_len = 60  # N_a = 60 time steps
    state_dim = 10  # x_i ∈ R^10: s(1) + p(3) + r(6)
    history_len = 5  # 5-frame historical observations
    
    # Condition dimensions
    target_dim = 3  # p_t ∈ R^3
    action_dim = 5   # 5 maneuver styles

    # CBF Guidance parameters (from CoDiG paper)
    enable_cbf_guidance = False  # Disabled by default; toggle for inference
    guidance_gamma = 100.0  # Base gamma for barrier guidance
    obstacle_radius = 5.0  # Safe distance radius

    @staticmethod
    def get_obstacle_center(device='cpu'):
        return torch.tensor([5.0, 5.0, 10.0], device=device)  # Example obstacle center
    
# Transformer positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# MLP embedding for conditional inputs
class ConditionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Diffusion timestep embedding
        self.t_embed = nn.Sequential(
            nn.Linear(1, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )
        
        # Target waypoint embedding
        self.target_embed = nn.Sequential(
            nn.Linear(config.target_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )
        
        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(config.action_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )

    def forward(self, t, target, action):
        t_emb = self.t_embed(t.unsqueeze(-1).float())
        target_emb = self.target_embed(target)
        action_emb = self.action_embed(action)
        
        # Combine conditions (simple addition as in the paper)
        cond_emb = t_emb + target_emb + action_emb
        
        return cond_emb

# Diffusion Transformer (decoder only)
class DiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.state_dim, config.latent_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.latent_dim)
        
        # Transformer layers
        transformer_layer = nn.TransformerDecoderLayer(
            d_model=config.latent_dim,
            nhead=config.num_heads,
            dim_feedforward=config.latent_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(
            transformer_layer, num_layers=config.num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.latent_dim, config.state_dim)
        
        # Condition embedding
        self.cond_embed = ConditionEmbedding(config)

    def forward(self, x, t, target, action, history=None):
        batch_size, seq_len, _ = x.shape
        
        # Project input to latent space
        x_emb = self.input_proj(x)
        
        # Add positional encoding
        x_emb = self.pos_encoding(x_emb.transpose(0, 1)).transpose(0, 1)
        
        # Prepare transformer input
        if history is not None:
            if history.size(0) != batch_size:
                if history.size(0) == 1:
                    history = history.repeat(batch_size, 1, 1)
                else:
                    raise ValueError(f"History data batch size mismatch")
            
            history_emb = self.input_proj(history)
            history_emb = self.pos_encoding(history_emb.transpose(0, 1)).transpose(0, 1)
            
            # Concatenate history data with current sequence along sequence dimension
            transformer_input = torch.cat([history_emb, x_emb], dim=1)
            total_seq_len = history_emb.size(1) + seq_len
        else:
            transformer_input = x_emb
            total_seq_len = seq_len
        
        # Generate causal mask
        memory_mask = self._generate_square_subsequent_mask(total_seq_len).to(x.device)
        
        # Get condition embedding and expand to sequence length
        cond_emb = self.cond_embed(t, target, action)
        cond_seq = cond_emb.unsqueeze(1).expand(-1, total_seq_len, -1)
        
        # Add condition to transformer input
        transformer_input = transformer_input + cond_seq
        
        # Self-attention with causal mask
        transformer_output = self.transformer(
            tgt=transformer_input,
            memory=transformer_input,
            tgt_mask=memory_mask,
            memory_mask=memory_mask
        )
        
        # Extract current sequence part (exclude history data)
        if history is not None:
            current_output = transformer_output[:, -seq_len:, :]
        else:
            current_output = transformer_output
        
        # Final projection
        output = self.output_proj(current_output)
        
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def compute_barrier_and_grad(x, config, mean, std):
    """
    Compute barrier V and its gradient ∇V for the trajectory x.
    Example: Quadratic barrier for spherical obstacle avoidance.
    V = sum_τ max(0, r - ||pos_τ - center||)^2
    ∇V affects only position components (indices 1:4).
    """
    # Ensure mean and std are on the same device as x
    device = x.device
    mean = mean.to(device)
    std = std.to(device)
    
    # Handle different possible shapes of mean/std
    if mean.dim() == 3:
        pos_mean = mean[0, 0, 1:4]  # Shape (1, 1, state_dim)
        pos_std = std[0, 0, 1:4]
    elif mean.dim() == 2:
        pos_mean = mean[0, 1:4]     # Shape (1, state_dim)
        pos_std = std[0, 1:4]
    else:
        # Fallback: assume last dimension is state_dim
        pos_mean = mean[..., 1:4].reshape(-1, 3)[0]  # Take first element
        pos_std = std[..., 1:4].reshape(-1, 3)[0]
    
    # Denormalize positions for barrier computation
    pos_denorm = x[:, :, 1:4] * pos_std + pos_mean
    
    # Get obstacle center on the correct device
    center = config.get_obstacle_center(device).unsqueeze(0).unsqueeze(0)  # (1,1,3)
    r = config.obstacle_radius
    
    # Compute distances and barrier values
    dist = torch.norm(pos_denorm - center, dim=-1, keepdim=False)  # (batch, seq_len)
    excess = torch.clamp(r - dist, min=0.0)  # (batch, seq_len)
    V = (excess ** 2).sum(dim=-1)  # (batch,)
    
    # Gradient computation with robust division
    grad_pos_denorm = torch.zeros_like(pos_denorm)
    unsafe_mask = dist < r
    
    if unsafe_mask.any():
        # Safe division with epsilon to avoid NaN
        diff = pos_denorm - center
        dist_safe = dist.unsqueeze(-1) + 1e-12  # Increased epsilon for stability
        direction = diff / dist_safe
        
        # Compute gradient: ∇_pos V_τ = -2 * excess_τ * (pos_τ - center) / dist_τ
        grad_update = -2.0 * excess.unsqueeze(-1) * direction
        
        # Only apply to unsafe points
        grad_pos_denorm[unsafe_mask] = grad_update[unsafe_mask]
    
    # Normalize gradient back to normalized space using chain rule
    # Since x_norm = (x_denorm - mean) / std, then dx_norm/dx_denorm = 1/std
    # So grad_x_norm = grad_x_denorm / std
    grad_pos = grad_pos_denorm * pos_std
    
    # Embed into full state gradient (only positions affected)
    grad_x = torch.zeros_like(x)
    grad_x[:, :, 1:4] = grad_pos
    
    # Debug information
    print(f"=== CBF Barrier Computation ===")
    print(f"Input x shape: {x.shape}")
    print(f"Position range - Min: {pos_denorm.min().item():.3f}, Max: {pos_denorm.max().item():.3f}")
    print(f"Distance range - Min: {dist.min().item():.3f}, Max: {dist.max().item():.3f}")
    print(f"Barrier V: {V.mean().item():.6f} (per sample)")
    print(f"grad_V shape: {grad_x.shape}")
    print(f"grad_V norm: {torch.norm(grad_x):.6f}")
    print(f"grad_V min/max: {grad_x.min():.6f} / {grad_x.max():.6f}")
    print(f"Number of unsafe points: {unsafe_mask.sum().item()}")
    print(f"Unsafe percentage: {unsafe_mask.float().mean().item()*100:.2f}%")
    print("=" * 40)
    
    return V, grad_x

# Diffusion process with linear noise schedule and CoDiG CBF guidance
class DiffusionProcess:
    def __init__(self, config):
        self.config = config
        self.num_timesteps = config.diffusion_steps
        
        # Linear noise schedule
        self.betas = torch.linspace(config.beta_start, config.beta_end, config.diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        if t.dim() == 1:
            t = t.view(-1, 1, 1)
        
        alpha_bar_t = self.alpha_bars[t].to(x_0.device)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        return x_t, noise

    def p_sample(self, model, x_t, t, target, action, history=None, guidance_gamma=None, mean=None, std=None, plot_step=False, step_idx=0):
        """
        Reverse diffusion process: p(x_{t-1} | x_t) with optional CoDiG CBF guidance.
        Vectorized over batch for efficiency.
        """
        batch_size = x_t.size(0)
        device = x_t.device
        
        with torch.no_grad():
            # 1. 预测 x_0
            pred_x0 = model(x_t, t, target, action, history)  # (batch, seq, state_dim)
            
            # 2. 正确提取扩散参数
            # 确保时间步是整数索引
            t_indices = t.long()
            
            # 正确广播参数
            alpha_bar_t = self.alpha_bars[t_indices].view(batch_size, 1, 1).to(device)
            alpha_t = self.alphas[t_indices].view(batch_size, 1, 1).to(device)
            beta_t = self.betas[t_indices].view(batch_size, 1, 1).to(device)
            one_minus_alpha_bar_t = 1.0 - alpha_bar_t
            
            # 3. 计算预测的噪声 (标准DDPM公式)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(one_minus_alpha_bar_t)
            
            # ε_θ = (x_t - √ᾱ_t * x_0) / √(1-ᾱ_t)
            ε_pred = (x_t - sqrt_alpha_bar_t * pred_x0) / (sqrt_one_minus_alpha_bar_t + 1e-8)
            
            # 4. CBF引导 (如果启用)
            ε_guided = ε_pred.clone()
            barrier_info = None
            
            if self.config.enable_cbf_guidance and guidance_gamma is not None and mean is not None and std is not None:
                # 计算时间相关的gamma
                gamma_t = guidance_gamma * (t_indices.float() / self.num_timesteps).view(batch_size, 1, 1)
                
                # 计算障碍函数梯度
                V, grad_V = compute_barrier_and_grad(pred_x0, self.config, mean, std)
                barrier_info = {'V': V, 'grad_V': grad_V, 'gamma_t': gamma_t}

                # 引导的分数: s_guided = s_theta - γ_t ∇V
                # 注意: s_theta = -ε_θ / σ_t, 其中 σ_t = √(1-ᾱ_t)
                s_theta = -ε_pred / (sqrt_one_minus_alpha_bar_t + 1e-8)
                s_guided = s_theta - gamma_t * grad_V
                
                # 引导的噪声: ε_guided = -s_guided * σ_t
                ε_guided = -s_guided * sqrt_one_minus_alpha_bar_t
            
            # 5. 计算后验均值 (DDPM公式)
            # μ_θ = (1/√α_t) * [x_t - (β_t/√(1-ᾱ_t)) * ε_guided]
            coeff = beta_t / (sqrt_one_minus_alpha_bar_t + 1e-8)
            mu = (1.0 / torch.sqrt(alpha_t)) * (x_t - coeff * ε_guided)
            
            # 6. 处理最后一步 (t=0)
            if t_indices.min() == 0:
                # 对于t=0，直接返回预测的x_0
                is_last_step = (t_indices == 0)
                if is_last_step.any():
                    # 对于最后一步，使用引导的x_0预测
                    pred_x0_guided = (x_t - sqrt_one_minus_alpha_bar_t * ε_guided) / (sqrt_alpha_bar_t + 1e-8)
                    
                    # 只替换最后一步的预测
                    result = x_t.clone()
                    result[is_last_step] = pred_x0_guided[is_last_step]
                    
                    # if plot_step:
                    #     self._plot_diffusion_step(x_t, result, t, step_idx, barrier_info, is_final=True)
                    return result
            
            # 7. 计算后验方差并采样
            # 前一个时间步的ᾱ
            t_prev = torch.clamp(t_indices - 1, min=0)
            alpha_bar_prev = self.alpha_bars[t_prev].view(batch_size, 1, 1).to(device)
            
            # 后验方差: σ_t^2 = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)
            var = beta_t * (1.0 - alpha_bar_prev) / (one_minus_alpha_bar_t + 1e-8)
            sigma = torch.sqrt(var)
            
            # 8. 采样 x_{t-1}
            z = torch.randn_like(x_t)
            x_prev = mu + sigma * z
            
            # if plot_step:
            #     self._plot_diffusion_step(x_t, x_prev, t, step_idx, barrier_info, is_final=False)
            
            return x_prev
    
    # Plotting function for diffusion steps for debugging
    def _plot_diffusion_step(self, x_t, x_prev, t, step_idx, barrier_info=None, is_final=False):
        """Plot the current diffusion step"""
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f'Reverse Diffusion Process - Step {step_idx} (t={t[0].item()})', fontsize=16)
        
        # Extract position coordinates (assuming x_t has shape [batch, seq_len, state_dim])
        x_t_pos = x_t[0, :, 1:4].cpu().numpy()  # positions at indices 1-3
        x_prev_pos = x_prev[0, :, 1:4].cpu().numpy()
        
        # 1. 3D trajectory evolution
        ax1 = fig.add_subplot(241, projection='3d')
        ax1.plot(x_t_pos[:, 0], x_t_pos[:, 1], x_t_pos[:, 2], 'r-', label='x_t (current)', linewidth=2, alpha=0.7)
        ax1.plot(x_prev_pos[:, 0], x_prev_pos[:, 1], x_prev_pos[:, 2], 'b-', label='x_prev (denoised)', linewidth=2, alpha=0.7)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        ax1.set_title('3D Trajectory Evolution')
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
        
        # 3. Noise and prediction statistics
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
        
        # 4. Position differences
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
            V = barrier_info['V'][0].item() if barrier_info['V'].numel() == 1 else barrier_info['V'].mean().item()
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
        plt.show()
        
        # Print step information
        print(f"\n=== Diffusion Step {step_idx} (t={t[0].item()}) ===")
        print(f"x_t shape: {x_t.shape}")
        print(f"x_t stats - Mean: {x_t.mean().item():.4f}, Std: {x_t.std().item():.4f}")
        print(f"x_prev stats - Mean: {x_prev.mean().item():.4f}, Std: {x_prev.std().item():.4f}")
        if barrier_info is not None:
            print(f"CBF - Barrier V: {barrier_info['V'][0].item():.4f}, Gamma_t: {barrier_info['gamma_t'][0].item():.4f}")

# Complete Aerobatic Diffusion Model (AeroDM)
class AeroDM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.diffusion_model = DiffusionTransformer(config)
        self.diffusion_process = DiffusionProcess(config)
        self.mean = None
        self.std = None
        
    def forward(self, x_t, t, target, action, history=None):
        return self.diffusion_model(x_t, t, target, action, history)
    
    def set_normalization_params(self, mean, std):
        """Set normalization parameters for CBF guidance"""
        self.mean = mean
        self.std = std
    
    def sample(self, target, action, history=None, batch_size=1, enable_guidance=True, guidance_gamma=None, plot_all_steps=False):
        device = next(self.parameters()).device
        
        if target.size(0) != batch_size:
            if target.size(0) == 1:
                target = target.repeat(batch_size, 1)
        
        if action.size(0) != batch_size:
            if action.size(0) == 1:
                action = action.repeat(batch_size, 1)
        
        if history is not None and history.size(0) != batch_size:
            if history.size(0) == 1:
                history = history.repeat(batch_size, 1, 1)
        
        # Initialize with noise
        x_t = torch.randn(batch_size, self.config.seq_len, self.config.state_dim).to(device)
        
        print(f"\n{'='*50}")
        print("STARTING REVERSE DIFFUSION PROCESS")
        print(f"Initial noise stats - Mean: {x_t.mean().item():.4f}, Std: {x_t.std().item():.4f}")
        print(f"Total steps: {self.config.diffusion_steps}")
        print(f"CBF Guidance: {enable_guidance}")
        print(f"{'='*50}")
        
        # Store original config setting
        original_guidance_setting = self.config.enable_cbf_guidance
        
        # Override with parameter
        self.config.enable_cbf_guidance = enable_guidance

        # Reverse diffusion process
        step_counter = 0
        for t_step in reversed(range(self.config.diffusion_steps)):
            t_batch = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
            gamma = guidance_gamma if enable_guidance else None
            
            # Plot every step if requested, or key steps for overview
            plot_step = plot_all_steps or (t_step % max(1, self.config.diffusion_steps // 5) == 0) or t_step == 0
            
            x_t = self.diffusion_process.p_sample(
                self.diffusion_model, x_t, t_batch, target, action, history, 
                gamma, self.mean, self.std, plot_step=plot_step, step_idx=step_counter
            )
            step_counter += 1
        
        print(f"\n{'='*50}")
        print("REVERSE DIFFUSION PROCESS COMPLETED")
        print(f"Final trajectory stats - Mean: {x_t.mean().item():.4f}, Std: {x_t.std().item():.4f}")
        print(f"{'='*50}")
        
        return x_t

# Improved Loss Function with Balanced Z-Axis Learning
class AeroDMLoss(nn.Module):
    def __init__(self, config, continuity_weight=10.0,  position_weight=2.0, vel_weight=1.0, last_pos_weight=10.0):
        super().__init__()
        self.config = config
        # loss weights
        self.continuity_weight = continuity_weight
        self.position_weight = position_weight
        self.vel_weight = vel_weight
        self.last_pos_weight = last_pos_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred_trajectory, gt_trajectory, history=None):
        # Separate position dimensions for balanced learning
        pred_pos = pred_trajectory[:, :, 1:4]
        gt_pos = gt_trajectory[:, :, 1:4]
        device = pred_trajectory.device

        # Basic losses
        x_loss = self.mse_loss(pred_pos[:, :, 0], gt_pos[:, :, 0])
        y_loss = self.mse_loss(pred_pos[:, :, 1], gt_pos[:, :, 1])
        z_loss = self.mse_loss(pred_pos[:, :, 2], gt_pos[:, :, 2])

        # All dimensional losses in the last time step
        last_x_loss = self.mse_loss(pred_pos[:, -1, 0], gt_pos[:, -1, 0])
        last_y_loss = self.mse_loss(pred_pos[:, -1, 1], gt_pos[:, -1, 1])
        last_z_loss = self.mse_loss(pred_pos[:, -1, 2], gt_pos[:, -1, 2])

        # Combination loss, with the last point having a higher weight
        position_loss = (x_loss + y_loss + z_loss + 
                         self.last_pos_weight * (last_x_loss + last_y_loss + last_z_loss))  
        
        # Velocity regularization with dimension balancing
        pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        gt_vel = gt_pos[:, 1:, :] - gt_pos[:, :-1, :]
        
        vel_x_loss = self.mse_loss(pred_vel[:, :, 0], gt_vel[:, :, 0])
        vel_y_loss = self.mse_loss(pred_vel[:, :, 1], gt_vel[:, :, 1])
        vel_z_loss = self.mse_loss(pred_vel[:, :, 2], gt_vel[:, :, 2])
        vel_loss = vel_x_loss + vel_y_loss + vel_z_loss
        
        # Other state components (velocity, attitude)
        other_components_loss = self.mse_loss(
            torch.cat([pred_trajectory[:, :, :1], pred_trajectory[:, :, 4:]], dim=2),
            torch.cat([gt_trajectory[:, :, :1], gt_trajectory[:, :, 4:]], dim=2)
        )
        
        # New: Continuity loss (MSE between last history and first pred timestep)
        continuity_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if history is not None and pred_trajectory.size(1) > 0:
            # Focus on position components (indices 1:4) for smoothness
            last_history_pos = history[:, -1, 1:4]
            first_pred_pos = pred_trajectory[:, 0, 1:4]
            continuity_loss = self.position_weight * self.mse_loss(first_pred_pos, last_history_pos)
            # Optional: Add velocity continuity (delta from last history to first pred)
            if history.size(1) > 1:
                last_history_vel = history[:, -1, 1:4] - history[:, -2, 1:4]
                first_pred_vel = pred_trajectory[:, 0, 1:4] - last_history_pos  # Approx
                continuity_loss += self.vel_weight* self.mse_loss(first_pred_vel, last_history_vel)

        total_loss = self.position_weight * position_loss + self.vel_weight * vel_loss + self.continuity_weight * continuity_loss + other_components_loss
        
        return total_loss, position_loss, vel_loss

def normalize_trajectories(trajectories):
    """Normalize each dimension to zero mean and unit variance"""
    mean = trajectories.mean(dim=(0, 1), keepdim=True)
    std = trajectories.std(dim=(0, 1), keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)  # avoid division by zero
    return (trajectories - mean) / std, mean, std

def denormalize_trajectories(trajectories_norm, mean, std):
    return trajectories_norm * std + mean

def normalize_obstacle(obstacle_center, mean, std):
    """Normalize obstacle center using the same normalization parameters"""
    # Extract position normalization parameters (indices 1:4 for x,y,z)
    pos_mean = mean[0, 0, 1:4].cpu().numpy()
    pos_std = std[0, 0, 1:4].cpu().numpy()
    
    # Normalize obstacle center
    obstacle_norm = (obstacle_center - pos_mean) / pos_std
    return obstacle_norm

def denormalize_obstacle(obstacle_norm, mean, std):
    """Denormalize obstacle center for visualization"""
    pos_mean = mean[0, 0, 1:4].cpu().numpy()
    pos_std = std[0, 0, 1:4].cpu().numpy()
    
    obstacle_denorm = obstacle_norm * pos_std + pos_mean
    return obstacle_denorm

def plot_training_losses(losses, show_flag=False):
    """Plot training losses over epochs"""
    plt.figure(figsize=(10, 6))
    epochs = range(len(losses['total']))
    
    plt.plot(epochs, losses['total'], 'b-', label='Total Loss', linewidth=2)
    plt.plot(epochs, losses['position'], 'r--', label='Position Loss', linewidth=2)
    plt.plot(epochs, losses['vel'], 'g-.', label='Velocity Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    if show_flag:
        plt.show()
    else:
        filename = "Figs/training_losses_bg.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight')
        plt.close()

# Aerobatic Trajectory Generation
def generate_aerobatic_trajectories(num_trajectories=100, seq_len=60, radius=10.0, height=0.0):
    """Generate diverse aerobatic trajectories based on eleven maneuver styles:
    (a) Power Loop, (b) Barrel Roll, (c) Split-S, (d) Immelmann Turn, (e) Wall Ride,
    (f) Eight Figure, (g) Patrick, (h) Star, (i) Half Moon, (j) Sphinx, (k) Clover."""
    trajectories = []
    maneuver_styles = ['power_loop', 'barrel_roll', 'split_s', 'immelmann', 'wall_ride', 'eight_figure', 'star', 'half_moon', 'sphinx', 'clover']
    
    # maneuver_styles = ['barrel_roll']
    
    # def smooth_trajectory(positions, smoothing_factor=0.1):
    #     """Apply smoothing to trajectory positions using a simple moving average"""
    #     smoothed = np.zeros_like(positions)
    #     for i in range(len(positions)):
    #         start_idx = max(0, i - 1)
    #         end_idx = min(len(positions), i + 2)
    #         smoothed[i] = np.mean(positions[start_idx:end_idx], axis=0)
    #     return smoothing_factor * smoothed + (1 - smoothing_factor) * positions
    
    def smooth_trajectory(positions, smoothing_factor=0.1):
        return positions
    
    for i in range(num_trajectories):
        # Randomly select a maneuver style
        style = np.random.choice(maneuver_styles)
        
        # Random centers and scales
        center_x = np.random.uniform(-20, 20)
        center_y = np.random.uniform(-20, 20)
        center_z = height + np.random.uniform(-10, 10)
        
        current_radius = radius * np.random.uniform(0.8, 1.2)
        angular_velocity = np.random.uniform(0.5, 2.0)
        
        # Normalize time steps to [0, 1] - exactly one period
        norm_t = np.linspace(0, 1, seq_len)
        
        # Compute positions and velocities based on style
        if style == 'power_loop':
            # Full vertical loop in xz plane, starting at bottom with forward velocity
            theta = 2 * np.pi * norm_t * angular_velocity
            x = center_x + current_radius * np.sin(theta)
            y = np.full(seq_len, center_y)
            z = center_z - current_radius * np.cos(theta)
            vx = current_radius * angular_velocity * 2 * np.pi * np.cos(theta)
            vy = np.zeros(seq_len)
            vz = current_radius * angular_velocity * 2 * np.pi * np.sin(theta)
        
        elif style == 'barrel_roll':
            # Helical path (corkscrew) along x
            theta = 2 * np.pi * norm_t * angular_velocity
            forward_speed = np.random.uniform(5.0, 15.0)  # Forward component
            x = center_x + forward_speed * norm_t
            y = center_y + current_radius * np.cos(theta)
            z = center_z + current_radius * np.sin(theta)
            vx = np.full(seq_len, forward_speed)
            vy = -current_radius * angular_velocity * 2 * np.pi * np.sin(theta)
            vz = current_radius * angular_velocity * 2 * np.pi * np.cos(theta)
        
        elif style == 'split_s':
            # Descending half-loop (semicircle down) in xz plane
            theta = np.pi * norm_t * angular_velocity
            x = center_x - current_radius * (1 - np.cos(theta))
            y = np.full(seq_len, center_y)
            z = center_z - current_radius * np.sin(theta)
            vx = -current_radius * angular_velocity * np.pi * np.sin(theta)
            vy = np.zeros(seq_len)
            vz = -current_radius * angular_velocity * np.pi * np.cos(theta)
        
        elif style == 'immelmann':
            # Ascending half-loop (semicircle up) in xz plane
            theta = np.pi * norm_t * angular_velocity
            x = center_x - current_radius * (1 - np.cos(theta))
            y = np.full(seq_len, center_y)
            z = center_z + current_radius * np.sin(theta)
            vx = -current_radius * angular_velocity * np.pi * np.sin(theta)
            vy = np.zeros(seq_len)
            vz = current_radius * angular_velocity * np.pi * np.cos(theta)
        
        elif style == 'wall_ride':
            # Vertical helix climb (spiral up)
            turns = np.random.uniform(0.5, 1.5)  # Number of turns
            climb_height = np.random.uniform(20.0, 40.0)
            theta = 2 * np.pi * turns * norm_t * angular_velocity
            x = center_x + current_radius * np.cos(theta)
            y = center_y + current_radius * np.sin(theta)
            z = center_z + climb_height * norm_t
            vx = -current_radius * (2 * np.pi * turns * angular_velocity) * np.sin(theta)
            vy = current_radius * (2 * np.pi * turns * angular_velocity) * np.cos(theta)
            vz = np.full(seq_len, climb_height)
        
        elif style == 'eight_figure':
            # Lemniscate (infinity symbol) in xy plane - exactly one period
            theta = 2 * np.pi * norm_t
            a = current_radius
            denom = 1 + np.sin(theta)**2
            x = center_x + a * np.cos(theta) / denom
            y = center_y + a * np.sin(theta) * np.cos(theta) / denom
            z = center_z + current_radius * 0.1 * np.sin(2 * theta)  # Smooth vertical variation
            
            # Numerical derivatives for smooth velocity
            dt = 1.0 / seq_len
            x_smooth = smooth_trajectory(x)
            y_smooth = smooth_trajectory(y)
            z_smooth = smooth_trajectory(z)
            vx = np.gradient(x_smooth, dt)
            vy = np.gradient(y_smooth, dt)
            vz = np.gradient(z_smooth, dt)
            x, y, z = x_smooth, y_smooth, z_smooth
        
        elif style == 'patrick':
            # Complex 3D star-like pattern - exactly one period
            freq1, freq2, freq3 = 2.0, 3.0, 1.5  # Fixed frequencies for period control
            theta = 2 * np.pi * norm_t
            x = center_x + current_radius * np.sin(freq1 * theta) * np.cos(freq2 * theta)
            y = center_y + current_radius * np.sin(freq2 * theta) * np.cos(freq3 * theta)
            z = center_z + current_radius * np.sin(freq3 * theta) * np.cos(freq1 * theta)
            
            # Apply smoothing and compute derivatives
            x_smooth = smooth_trajectory(x, 0.2)
            y_smooth = smooth_trajectory(y, 0.2)
            z_smooth = smooth_trajectory(z, 0.2)
            dt = 1.0 / seq_len
            vx = np.gradient(x_smooth, dt)
            vy = np.gradient(y_smooth, dt)
            vz = np.gradient(z_smooth, dt)
            x, y, z = x_smooth, y_smooth, z_smooth
        
        elif style == 'star':
            # 5-point star pattern in xy plane with vertical oscillation - exactly one period
            points = 5
            theta = 2 * np.pi * norm_t
            r = current_radius * (1 + 0.3 * np.sin(points * theta))  # Smoother star shape
            x = center_x + r * np.cos(theta)
            y = center_y + r * np.sin(theta)
            z = center_z + current_radius * 0. * np.sin(3 * theta)
            
            # Apply smoothing
            positions = np.column_stack([x, y, z])
            smoothed_positions = smooth_trajectory(positions)
            x, y, z = smoothed_positions[:, 0], smoothed_positions[:, 1], smoothed_positions[:, 2]
            dt = 1.0 / seq_len
            vx = np.gradient(x, dt)
            vy = np.gradient(y, dt)
            vz = np.gradient(z, dt)
        
        elif style == 'half_moon':
            # Crescent moon shape in xy plane - exactly one half period
            theta = np.pi * norm_t
            r = current_radius * (1 + 0.3 * np.cos(2 * theta))  # Smoother crescent
            x = center_x + r * np.cos(theta)
            y = center_y + r * np.sin(theta)
            z = center_z + current_radius * 0.1 * np.sin(2 * theta)  # Smooth vertical variation
            
            # Apply smoothing
            x_smooth = smooth_trajectory(x)
            y_smooth = smooth_trajectory(y)
            z_smooth = smooth_trajectory(z)
            dt = 1.0 / seq_len
            vx = np.gradient(x_smooth, dt)
            vy = np.gradient(y_smooth, dt)
            vz = np.gradient(z_smooth, dt)
            x, y, z = x_smooth, y_smooth, z_smooth
        
        elif style == 'sphinx':
            # Pyramid-like triangular pattern - exactly one period
            theta = 2 * np.pi * norm_t
            # Smoother triangular wave using sine approximation
            triangle_wave = 0.5 * np.sin(2 * theta) + 0.3 * np.sin(4 * theta)
            x = center_x + current_radius * np.cos(theta)
            y = center_y + current_radius * np.sin(theta)
            z = center_z + current_radius * 0.3 * triangle_wave
            
            # Apply smoothing
            positions = np.column_stack([x, y, z])
            smoothed_positions = smooth_trajectory(positions, 0.15)
            x, y, z = smoothed_positions[:, 0], smoothed_positions[:, 1], smoothed_positions[:, 2]
            dt = 1.0 / seq_len
            vx = np.gradient(x, dt)
            vy = np.gradient(y, dt)
            vz = np.gradient(z, dt)
        
        elif style == 'clover':
            # 4-leaf clover pattern - exactly one period
            leaves = 4
            theta = 2 * np.pi * norm_t
            r = current_radius * (1 + 0.2 * np.sin(leaves * theta))  # Smoother clover
            x = center_x + r * np.cos(theta)
            y = center_y + r * np.sin(theta)
            z = center_z + current_radius * 0.0 * np.cos(2 * theta)
            
            # Apply smoothing
            x_smooth = smooth_trajectory(x)
            y_smooth = smooth_trajectory(y)
            z_smooth = smooth_trajectory(z)
            dt = 1.0 / seq_len
            vx = np.gradient(x_smooth, dt)
            vy = np.gradient(y_smooth, dt)
            vz = np.gradient(z_smooth, dt)
            x, y, z = x_smooth, y_smooth, z_smooth
        
        # Compute speed and direction
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        direction = np.stack([vx, vy, vz], axis=-1)
        norms = np.linalg.norm(direction, axis=-1, keepdims=True)
        direction = np.divide(direction, norms, where=norms>0, out=np.zeros_like(direction))
        
        # Attitude: direction + fixed components (e.g., for roll/pitch/yaw approximation)
        attitude = np.concatenate([direction, np.full((seq_len, 3), 0.1)], axis=-1)
        
        # Full state: [speed, x, y, z, attitude(6)]
        state = np.column_stack([speed, x, y, z, attitude])
        trajectories.append(state)
    
    return torch.tensor(np.stack(trajectories), dtype=torch.float32)

def generate_target_waypoints(trajectories):
    """Generate target waypoints from trajectories (usually the trajectory endpoint)"""
    batch_size = trajectories.shape[0]
    target_waypoints = trajectories[:, -1, 1:4]  # Take final position as target
    return target_waypoints

def generate_action_styles(batch_size, action_dim=5, device='cpu'):
    """Generate action style vectors"""
    styles = []
    for i in range(batch_size):
        style = np.zeros(action_dim)
        num_features = np.random.randint(1, 3)
        features = np.random.choice(action_dim, num_features, replace=False)
        style[features] = np.random.uniform(0.5, 1.0, num_features)
        styles.append(style)
    
    return torch.tensor(styles, dtype=torch.float32, device=device)

def generate_history_segments(trajectories, history_len=5, device=None):
    """Generate history segments from trajectories"""
    if device is None:
        device = trajectories.device
    batch_size, seq_len, state_dim = trajectories.shape
    histories = []
    
    for i in range(batch_size):
        # start_idx = np.random.randint(0, max(1, seq_len - history_len - 10))
        start_idx = 0
        history_segment = trajectories[i, start_idx:start_idx+history_len]
        
        if len(history_segment) < history_len:
            padding = torch.zeros(history_len - len(history_segment), state_dim, device=device)
            history_segment = torch.cat([history_segment, padding], dim=0)
        
        histories.append(history_segment)
    
    return torch.stack(histories)

def plot_trajectory_comparison(original, reconstructed, sampled, history=None, target=None, title="Figs/Barrier Guidance Trajectory Comparison", obstacle_center=None, obstacle_radius=None, show_flag=False):
    """Enhanced visualization with history, target, and obstacle (for CBF demo)"""
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(title, fontsize=16)
    
    # Extract position coordinates
    original_pos = original[0, :, 1:4].detach().cpu().numpy()
    reconstructed_pos = reconstructed[0, :, 1:4].detach().cpu().numpy()
    sampled_pos = sampled[0, :, 1:4].detach().cpu().numpy()
    
    # Extract history and target if provided
    history_pos = None
    if history is not None:
        history_pos = history[0, :, 1:4].detach().cpu().numpy()
    
    target_pos = None
    if target is not None:
        target_pos = target[0, :].detach().cpu().numpy()  # target is [x, y, z]
    
    # 1. 3D trajectory plot with history, target, and obstacle
    ax1 = fig.add_subplot(331, projection='3d')
    
    # Plot history (if available)
    if history_pos is not None:
        ax1.plot(history_pos[:, 0], history_pos[:, 1], history_pos[:, 2], 
                'm-', label='History', linewidth=4, alpha=0.8, marker='o', markersize=4)
    
    # Plot main trajectories
    ax1.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], 
             'b-', label='Original Trajectory', linewidth=3, alpha=0.8)
    
    ax1.plot(reconstructed_pos[:, 0], reconstructed_pos[:, 1], reconstructed_pos[:, 2], 
             'r.-', label='Reconstructed Trajectory', linewidth=2, alpha=0.8)
    ax1.plot(sampled_pos[:, 0], sampled_pos[:, 1], sampled_pos[:, 2], 
             'g.-', label='Sampled Trajectory', linewidth=2, alpha=0.8)
    
    # Plot target (if available)
    if target_pos is not None:
        ax1.scatter(target_pos[0], target_pos[1], target_pos[2], 
                   c='yellow', s=200, marker='*', label='Target Waypoint', edgecolors='black', linewidth=2)
    
    # Plot obstacle (if provided) - FIXED: Use scatter for obstacle to avoid legend issues
    if obstacle_center is not None and obstacle_radius is not None:
        # Create a proxy artist for the legend
        from matplotlib.patches import Circle
        from mpl_toolkits.mplot3d.art3d import Patch3DCollection
        
        # Plot obstacle as a transparent sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        obs_x = obstacle_center[0] + obstacle_radius * np.outer(np.cos(u), np.sin(v))
        obs_y = obstacle_center[1] + obstacle_radius * np.outer(np.sin(u), np.sin(v))
        obs_z = obstacle_center[2] + obstacle_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Create surface plot but don't add to legend directly
        obstacle_surface = ax1.plot_surface(obs_x, obs_y, obs_z, alpha=0.3, color='red')
        
        # Create a proxy artist for the legend
        proxy_artist = plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.3)
    
    # Create legend with custom handling for obstacle
    legend_handles = [
        plt.Line2D([0], [0], color='m', linewidth=4, marker='o', markersize=4, label='History'),
        plt.Line2D([0], [0], color='b', linewidth=3, label='Original Trajectory'),
        plt.Line2D([0], [0], color='r', linewidth=2, marker='.', label='Reconstructed Trajectory'),
        plt.Line2D([0], [0], color='g', linewidth=2, marker='.', label='Sampled Trajectory'),
        plt.Line2D([0], [0], color='yellow', marker='*', markersize=10, linestyle='None', 
                  markeredgecolor='black', markeredgewidth=2, label='Target Waypoint')
    ]
    
    if obstacle_center is not None and obstacle_radius is not None:
        legend_handles.append(proxy_artist)
    
    ax1.legend(handles=legend_handles)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory Comparison with History, Target & Obstacle')
    ax1.grid(True)
    
    # 2. X-Y projection
    ax2 = fig.add_subplot(332)
    if history_pos is not None:
        ax2.plot(history_pos[:, 0], history_pos[:, 1], 'm-', label='History', linewidth=4, alpha=0.8, marker='o', markersize=4)
    ax2.plot(original_pos[:, 0], original_pos[:, 1], 'b-', label='Original', linewidth=3, alpha=0.8)
    ax2.plot(reconstructed_pos[:, 0], reconstructed_pos[:, 1], 'r.-', label='Reconstructed', linewidth=2, alpha=0.8)
    ax2.plot(sampled_pos[:, 0], sampled_pos[:, 1], 'g.-', label='Sampled', linewidth=2, alpha=0.8)
    if target_pos is not None:
        ax2.scatter(target_pos[0], target_pos[1], c='yellow', s=200, marker='*', label='Target', edgecolors='black', linewidth=2)
    if obstacle_center is not None and obstacle_radius is not None:
        circle = plt.Circle((obstacle_center[0], obstacle_center[1]), obstacle_radius, color='red', alpha=0.3, label='Obstacle')
        ax2.add_patch(circle)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.set_title('X-Y Projection')
    ax2.grid(True)
    ax2.axis('equal')
    
    # 3. X-Z projection
    ax3 = fig.add_subplot(333)
    if history_pos is not None:
        ax3.plot(history_pos[:, 0], history_pos[:, 2], 'm-', label='History', linewidth=4, alpha=0.8, marker='o', markersize=4)
    ax3.plot(original_pos[:, 0], original_pos[:, 2], 'b-', label='Original', linewidth=3, alpha=0.8)
    ax3.plot(reconstructed_pos[:, 0], reconstructed_pos[:, 2], 'r.-', label='Reconstructed', linewidth=2, alpha=0.8)
    ax3.plot(sampled_pos[:, 0], sampled_pos[:, 2], 'g.-', label='Sampled', linewidth=2, alpha=0.8)
    if target_pos is not None:
        ax3.scatter(target_pos[0], target_pos[2], c='yellow', s=200, marker='*', label='Target', edgecolors='black', linewidth=2)
    if obstacle_center is not None and obstacle_radius is not None:
        circle = plt.Circle((obstacle_center[0], obstacle_center[2]), obstacle_radius, color='red', alpha=0.3, label='Obstacle')
        ax3.add_patch(circle)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.legend()
    ax3.set_title('X-Z Projection')
    ax3.grid(True)
    ax3.axis('equal')
    
    # 4. Y-Z projection
    ax4 = fig.add_subplot(334)
    if history_pos is not None:
        ax4.plot(history_pos[:, 1], history_pos[:, 2], 'm-', label='History', linewidth=4, alpha=0.8, marker='o', markersize=4)
    ax4.plot(original_pos[:, 1], original_pos[:, 2], 'b-', label='Original', linewidth=3, alpha=0.8)
    ax4.plot(reconstructed_pos[:, 1], reconstructed_pos[:, 2], 'r.-', label='Reconstructed', linewidth=2, alpha=0.8)
    ax4.plot(sampled_pos[:, 1], sampled_pos[:, 2], 'g.-', label='Sampled', linewidth=2, alpha=0.8)
    if target_pos is not None:
        ax4.scatter(target_pos[1], target_pos[2], c='yellow', s=200, marker='*', label='Target', edgecolors='black', linewidth=2)
    if obstacle_center is not None and obstacle_radius is not None:
        circle = plt.Circle((obstacle_center[1], obstacle_center[2]), obstacle_radius, color='red', alpha=0.3, label='Obstacle')
        ax4.add_patch(circle)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.legend()
    ax4.set_title('Y-Z Projection')
    ax4.grid(True)
    ax4.axis('equal')
    
    # 5. Position components over time
    time_steps = np.arange(original_pos.shape[0])
    ax5 = fig.add_subplot(335)
    ax5.plot(time_steps, original_pos[:, 0], 'b-', label='Original X', linewidth=2)
    ax5.plot(time_steps, reconstructed_pos[:, 0], 'r--', label='Reconstructed X', linewidth=2)
    ax5.plot(time_steps, sampled_pos[:, 0], 'g-.', label='Sampled X', linewidth=2)
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('X Position')
    ax5.legend()
    ax5.set_title('X Position Over Time')
    ax5.grid(True)
    
    ax6 = fig.add_subplot(336)
    ax6.plot(time_steps, original_pos[:, 1], 'b-', label='Original Y', linewidth=2)
    ax6.plot(time_steps, reconstructed_pos[:, 1], 'r--', label='Reconstructed Y', linewidth=2)
    ax6.plot(time_steps, sampled_pos[:, 1], 'g-.', label='Sampled Y', linewidth=2)
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Y Position')
    ax6.legend()
    ax6.set_title('Y Position Over Time')
    ax6.grid(True)
    
    ax7 = fig.add_subplot(337)
    ax7.plot(time_steps, original_pos[:, 2], 'b-', label='Original Z', linewidth=2)
    ax7.plot(time_steps, reconstructed_pos[:, 2], 'r--', label='Reconstructed Z', linewidth=2)
    ax7.plot(time_steps, sampled_pos[:, 2], 'g-.', label='Sampled Z', linewidth=2)
    ax7.set_xlabel('Time Step')
    ax7.set_ylabel('Z Position')
    ax7.legend()
    ax7.set_title('Z Position Over Time')
    ax7.grid(True)
    
    # 8. Speed comparison
    original_speed = original[0, :, 0].detach().cpu().numpy()
    reconstructed_speed = reconstructed[0, :, 0].detach().cpu().numpy()
    sampled_speed = sampled[0, :, 0].detach().cpu().numpy()
    
    ax8 = fig.add_subplot(338)
    ax8.plot(time_steps, original_speed, 'b-', label='Original Speed', linewidth=2)
    ax8.plot(time_steps, reconstructed_speed, 'r--', label='Reconstructed Speed', linewidth=2)
    ax8.plot(time_steps, sampled_speed, 'g-.', label='Sampled Speed', linewidth=2)
    ax8.set_xlabel('Time Step')
    ax8.set_ylabel('Speed')
    ax8.legend()
    ax8.set_title('Speed Over Time')
    ax8.grid(True)
    
    # 9. Error analysis
    recon_error = np.linalg.norm(reconstructed_pos - original_pos, axis=1)
    sampled_error = np.linalg.norm(sampled_pos - original_pos, axis=1)
    
    ax9 = fig.add_subplot(339)
    ax9.plot(time_steps, recon_error, 'r-', label='Reconstruction Error', linewidth=2)
    ax9.plot(time_steps, sampled_error, 'g-', label='Sampling Error', linewidth=2)
    ax9.set_xlabel('Time Step')
    ax9.set_ylabel('Position Error')
    ax9.legend()
    ax9.set_title('Position Error Over Time')
    ax9.grid(True)
    
    plt.tight_layout()
    
    if show_flag:
        plt.show()
    else:
        # Create filename from title
        filename = title.replace(' ', '_').replace('\n', '_').replace(':', '') + ".svg"
        plt.savefig(filename, format='svg', bbox_inches='tight')
        plt.close()

# Update the test function to demonstrate CBF guidance with diffusion visualization
def test_model_performance(model, trajectories_norm, mean, std, num_test_samples=3, show_flag=False):
    """Enhanced testing with history, target, and CBF guidance visualization"""
    print("\nTesting model performance with CBF guidance...")
    config = model.config  # Access config from model
    device = next(model.parameters()).device
    
    # Set normalization parameters for CBF guidance
    model.set_normalization_params(mean, std)
    
    model.eval()
    with torch.no_grad():
        for i in range(min(num_test_samples, trajectories_norm.shape[0])):
            # Prepare test sample
            full_traj = trajectories_norm[i:i+1]  # (1, 65, 10) full trajectory
            target_norm = generate_target_waypoints(full_traj)  # From full trajectory end
            action = generate_action_styles(1, config.action_dim, device=device)
            history = generate_history_segments(full_traj, config.history_len, device=device)  # First 5 steps
            x_0 = full_traj[:, config.history_len:, :]  # Last 60 steps (prediction sequence)
            
            # Denormalize target for plotting only (model uses normalized)
            target_denorm = target_norm * std[0, 0, 1:4] + mean[0, 0, 1:4]  # Position dims only
            
            print(f"\n{'='*60}")
            print(f"TEST SAMPLE {i+1}")
            print(f"{'='*60}")
            
            # Sample with CBF guidance and plot all diffusion steps
            sampled_guided = model.sample(target_norm, action, history, batch_size=1, 
                                        enable_guidance=True, guidance_gamma=config.guidance_gamma,
                                        plot_all_steps=True)  # Set to True to plot every step
            
            # For comparison, also sample without guidance (no plots)
            sampled_unguided = model.sample(target_norm, action, history, batch_size=1, 
                                          enable_guidance=False, plot_all_steps=False)
            
            # Denormalize others
            sampled_guided_denorm = denormalize_trajectories(sampled_guided, mean, std)
            sampled_unguided_denorm = denormalize_trajectories(sampled_unguided, mean, std)
            x_0_denorm = denormalize_trajectories(x_0, mean, std)
            history_denorm = denormalize_trajectories(history, mean, std)
            
            # Get obstacle center and normalize it for CBF computation
            obstacle_center_original = config.get_obstacle_center('cpu').numpy()
            obstacle_center_norm = normalize_obstacle(obstacle_center_original, mean, std)
            
            print(f"Original obstacle center: {obstacle_center_original}")
            print(f"Normalized obstacle center: {obstacle_center_norm}")
            
            # For visualization, use denormalized obstacle
            obstacle_center_viz = obstacle_center_original
            
            # Visualize: unguided vs guided, with obstacle
            plot_trajectory_comparison(
                x_0_denorm, sampled_unguided_denorm, sampled_guided_denorm, 
                history=history_denorm, target=target_denorm,
                title=f"Figs/CBF Guidance Test Sample {i+1}\n(Guided: Green avoids Red Obstacle)",
                obstacle_center=obstacle_center_viz,
                obstacle_radius=config.obstacle_radius,
                show_flag=show_flag
            )
            
    model.train()

def train_aerodm_cbf(show_flag=False):
    """Enhanced training function with z-axis improvements"""
    config = Config()
    model = AeroDM(config)
    
    # Set device to MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.config = config  # Attach config to model for access
    
    # Explicitly disable CBF guidance during training
    model.config.enable_cbf_guidance = False
    
    # Move diffusion process tensors to device
    model.diffusion_process.betas = model.diffusion_process.betas.to(device)
    model.diffusion_process.alphas = model.diffusion_process.alphas.to(device)
    model.diffusion_process.alpha_bars = model.diffusion_process.alpha_bars.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = AeroDMLoss(config)  # Use improved loss
    
    # Training parameters
    num_epochs = 50
    batch_size = 8
    num_trajectories = 10000
    
    print("Generating enhanced circular trajectory data...")
    # Generate enhanced training data
    trajectories = generate_aerobatic_trajectories(
        num_trajectories=num_trajectories,
        seq_len=config.seq_len + config.history_len
    )
    
    # Normalize trajectories
    trajectories_norm, mean, std = normalize_trajectories(trajectories)
    trajectories_norm = trajectories_norm.to(device)
    mean = mean.to(device)
    std = std.to(device)
    
    # Store losses for plotting
    losses = {'total': [], 'position': [], 'vel': []}
    
    print("Starting improved training with z-axis focus...")
    for epoch in range(num_epochs):
        epoch_total_loss = 0
        epoch_position_loss = 0
        epoch_vel_loss = 0
        num_batches = 0
        
        # Random batch training
        indices = torch.randperm(num_trajectories)
        for i in range(0, num_trajectories, batch_size):
            if i + batch_size > num_trajectories:
                actual_batch_size = num_trajectories - i
            else:
                actual_batch_size = batch_size
                
            batch_indices = indices[i:i+actual_batch_size]
            full_traj = trajectories_norm[batch_indices]  # (bs, 65, 10) full trajectory
            target = generate_target_waypoints(full_traj)  # From full trajectory end
            action = generate_action_styles(actual_batch_size, config.action_dim, device=device)
            history = generate_history_segments(full_traj, config.history_len, device=device)  # First 5 steps
            x_0 = full_traj[:, config.history_len:, :]  # FIXED: Last 60 steps (prediction sequence)
            
            # Sample diffusion time step
            t = torch.randint(0, config.diffusion_steps, (actual_batch_size,), device=device)
            
            # Forward diffusion
            noisy_x, noise = model.diffusion_process.q_sample(x_0, t)
            
            # Model prediction
            pred_x0 = model(noisy_x, t, target, action, history)
            
            # Calculate loss
            total_loss, position_loss, vel_loss = criterion(pred_x0, x_0)
            
            # Backward propagation
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_total_loss += total_loss.item()
            epoch_position_loss += position_loss.item()
            epoch_vel_loss += vel_loss.item()
            num_batches += 1
        
        # Calculate average loss
        if num_batches > 0:
            avg_total = epoch_total_loss / num_batches
            avg_position = epoch_position_loss / num_batches
            avg_vel = epoch_vel_loss / num_batches
            
            losses['total'].append(avg_total)
            losses['position'].append(avg_position)
            losses['vel'].append(avg_vel)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Total Loss: {avg_total:.4f}, "
                  f"Position: {avg_position:.4f}, Vel: {avg_vel:.4f}")
        if avg_position < 0.02:
            print("Early stopping as position loss is below threshold.")
            break

    # Plot training losses
    plot_training_losses(losses, show_flag=show_flag)
    
    # Enable CBF guidance for inference/testing
    model.config.enable_cbf_guidance = True
    
    # Test model performance with CBF and diffusion visualization
    test_model_performance(model, trajectories_norm, mean, std, num_test_samples=30, show_flag=show_flag)  # Reduced for demonstration
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': losses,
        'mean': mean,
        'std': std
    }, "model/improved_aerodm_with_cbf.pth")

    return model, losses, trajectories, mean, std

# Main execution
if __name__ == "__main__":
    print("Training Improved AeroDM with CBF Guidance Integration...")
    
    show_flag = False 

    # Train with enhanced method and CBF
    trained_model, losses, trajectories, mean, std = train_aerodm_cbf(show_flag=show_flag)
    
    print("Training completed!")