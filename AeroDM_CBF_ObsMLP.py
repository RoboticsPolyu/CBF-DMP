import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    # Obstacle parameters
    max_obstacles = 10  # Maximum number of obstacles to process
    obstacle_feat_dim = 4  # [x, y, z, radius]

    # CBF Guidance parameters (from CoDiG paper)
    enable_cbf_guidance = True  # Disabled by default; toggle for inference
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

# Obstacle Encoder MLP Module
class ObstacleEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # MLP for encoding individual obstacles
        self.obstacle_mlp = nn.Sequential(
            nn.Linear(config.obstacle_feat_dim, config.latent_dim // 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim // 2, config.latent_dim),
            nn.ReLU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )
        
        # Global obstacle context encoder
        self.global_obstacle_encoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.ReLU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )
        
        # Learnable obstacle query for aggregation
        self.obstacle_query = nn.Parameter(torch.randn(1, config.latent_dim))
        
    def forward(self, obstacles_data):
        """
        Encode obstacle information into a latent representation.
        """
        if obstacles_data is None or len(obstacles_data) == 0:
            # No obstacles, return zero embedding
            batch_size = 1
            return torch.zeros(batch_size, self.config.latent_dim, device=next(self.parameters()).device)
        
        device = next(self.parameters()).device
        batch_size = 1
        
        if isinstance(obstacles_data, list):
            # Process list of obstacle dictionaries
            obstacle_tensors = []
            for obstacle in obstacles_data:
                center = obstacle['center'].to(device)
                radius = obstacle['radius']
                if isinstance(radius, torch.Tensor):
                    radius_tensor = radius.to(device)
                else:
                    radius_tensor = torch.tensor([radius], device=device)
                obstacle_feat = torch.cat([center, radius_tensor])
                obstacle_tensors.append(obstacle_feat)
            
            if len(obstacle_tensors) == 0:
                return torch.zeros(batch_size, self.config.latent_dim, device=device)
            
            # Stack and pad to max_obstacles
            obstacle_tensor = torch.stack(obstacle_tensors)
            if len(obstacle_tensors) < self.config.max_obstacles:
                padding = torch.zeros(self.config.max_obstacles - len(obstacle_tensors), 
                                    self.config.obstacle_feat_dim, device=device)  # 在正确设备上创建padding
                obstacle_tensor = torch.cat([obstacle_tensor, padding], dim=0)
            elif len(obstacle_tensors) > self.config.max_obstacles:
                obstacle_tensor = obstacle_tensor[:self.config.max_obstacles]
            
            obstacle_tensor = obstacle_tensor.unsqueeze(0)  # Add batch dimension
            
        else:
            # Assume obstacles_data is already a tensor
            obstacle_tensor = obstacles_data.to(device)  # 移动到模型设备
            if obstacle_tensor.dim() == 2:
                obstacle_tensor = obstacle_tensor.unsqueeze(0)
            batch_size = obstacle_tensor.size(0)
        
        # Encode each obstacle individually
        encoded_obstacles = self.obstacle_mlp(obstacle_tensor)
        
        # Global aggregation using attention mechanism
        obstacle_query = self.obstacle_query.expand(batch_size, -1, -1)
        
        # Simple attention-based aggregation
        attention_weights = F.softmax(torch.bmm(obstacle_query, encoded_obstacles.transpose(1, 2)), dim=-1)
        aggregated_obstacle = torch.bmm(attention_weights, encoded_obstacles)
        
        # Final encoding
        obstacle_emb = self.global_obstacle_encoder(aggregated_obstacle.squeeze(1))
        
        return obstacle_emb
    
# Enhanced Condition Embedding with Obstacle Information
class EnhancedConditionEmbedding(nn.Module):
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
        
        # Obstacle embedding
        self.obstacle_encoder = ObstacleEncoder(config)
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.latent_dim * 4, config.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim * 2, config.latent_dim)
        )

    def forward(self, t, target, action, obstacles_data=None):
        # Individual embeddings
        t_emb = self.t_embed(t.unsqueeze(-1).float())
        target_emb = self.target_embed(target)
        action_emb = self.action_embed(action)
        
        # Obstacle embedding
        if obstacles_data is not None:
            obstacle_emb = self.obstacle_encoder(obstacles_data)
        else:
            obstacle_emb = torch.zeros_like(t_emb)
        
        # Combine all conditions with feature fusion
        combined_emb = torch.cat([t_emb, target_emb, action_emb, obstacle_emb], dim=-1)
        cond_emb = self.fusion_layer(combined_emb)
        
        return cond_emb

# Enhanced Diffusion Transformer with Obstacle Awareness
class ObstacleAwareDiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.state_dim, config.latent_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.latent_dim)
        
        # Enhanced condition embedding with obstacle information
        self.cond_embed = EnhancedConditionEmbedding(config)
        
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

    def forward(self, x, t, target, action, history=None, obstacles_data=None):
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
        
        # Get enhanced condition embedding with obstacle information
        cond_emb = self.cond_embed(t, target, action, obstacles_data)
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

# Enhanced CBF Barrier Function with Multiple Obstacles
def compute_barrier_and_grad(x, config, mean, std, obstacles_data=None):
    """
    Compute barrier V and its gradient ∇V for the trajectory x.
    Extended to handle multiple spherical obstacles.
    V = sum_τ sum_obs max(0, r_obs - ||pos_τ - center_obs||)^2
    ∇V affects only position components (indices 1:4).
    """
    # Denormalize positions for barrier computation
    pos_denorm = x[:, :, 1:4] * std[0, 0, 1:4] + mean[0, 0, 1:4]
    
    batch_size, seq_len, _ = pos_denorm.shape
    
    # Initialize barrier value and gradient
    V_total = torch.zeros(batch_size, device=x.device)
    grad_pos_denorm = torch.zeros_like(pos_denorm)
    
    # Process each obstacle
    if obstacles_data is not None:
        for obstacle in obstacles_data:
            center = obstacle['center'].unsqueeze(0).unsqueeze(0).to(x.device)  # (1,1,3)
            r = obstacle['radius']
            
            # Compute distance to obstacle center
            dist = torch.norm(pos_denorm - center, dim=-1, keepdim=False)  # (batch, seq_len)
            excess = torch.clamp(r - dist, min=0.0)  # (batch, seq_len)
            
            # Accumulate barrier value
            V_total += (excess ** 2).sum(dim=-1)  # (batch,)
            
            # Compute gradient for this obstacle
            unsafe_mask = dist < r
            if unsafe_mask.any():
                direction = (pos_denorm - center) / (dist.unsqueeze(-1) + 1e-8)  # Unit vector away from center
                grad_pos_denorm[unsafe_mask] += -2.0 * excess[unsafe_mask].unsqueeze(-1) * direction[unsafe_mask]
    
    # Normalize gradient back to normalized space
    grad_pos = grad_pos_denorm / std[0, 0, 1:4]
    
    # Embed into full state gradient (only positions affected)
    grad_x = torch.zeros_like(x)
    grad_x[:, :, 1:4] = grad_pos
    
    # Print grad_V information
    print(f"grad_V shape: {grad_x.shape}")
    print(f"grad_V norm: {torch.norm(grad_x):.6f}")
    print(f"grad_V min/max: {grad_x.min():.6f} / {grad_x.max():.6f}")
    print(f"Total barrier value V: {V_total.mean().item():.6f}")
    
    return V_total, grad_x

def generate_random_obstacles(trajectory, num_obstacles_range=(0, 10), radius_range=(0.01, 0.30), device='cpu'):
    """
    Generate random spherical obstacles around the trajectory.
    Returns list of obstacle dictionaries with center and radius.
    """
    num_obstacles = np.random.randint(num_obstacles_range[0], num_obstacles_range[1] + 1)
    obstacles = []
    
    # Get trajectory bounds for obstacle placement
    traj_pos = trajectory[:, 1:4].cpu().numpy()
    min_bounds = traj_pos.min(axis=0)
    max_bounds = traj_pos.max(axis=0)
    
    bounds_range = max_bounds - min_bounds
    expanded_min = min_bounds - 0.5 * bounds_range
    expanded_max = max_bounds + 0.5 * bounds_range
    
    # print(f"Generating {num_obstacles} random obstacles around trajectory")
    
    for i in range(num_obstacles):
        center = np.random.uniform(expanded_min, expanded_max)
        radius = np.random.uniform(radius_range[0], radius_range[1])
        
        obstacle = {
            'center': torch.tensor(center, dtype=torch.float32, device=device),  
            'radius': radius,
            'id': i
        }
        obstacles.append(obstacle)
        
        # print(f"Obstacle {i}: center={center}, radius={radius:.3f}")
    
    return obstacles

# Enhanced Diffusion Process with Obstacle-Aware Sampling
class ObstacleAwareDiffusionProcess:
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
    
    def p_sample(self, model, x_t, t, target, action, history=None, guidance_gamma=None, 
                mean=None, std=None, plot_step=False, step_idx=0, obstacles_data=None):
        """
        Reverse diffusion process with obstacle-aware sampling.
        """
        batch_size = x_t.size(0)
        device = x_t.device
        
        with torch.no_grad():
            # Model prediction with obstacle information
            pred_x0 = model(x_t, t, target, action, history, obstacles_data)
            
            # Expand t for broadcasting
            t_exp = t.view(batch_size, 1, 1) if t.dim() == 1 else t.view(-1, 1, 1)
            
            alpha_bar_t = self.alpha_bars[t_exp.squeeze(1)].view(batch_size, 1, 1)
            alpha_t = self.alphas[t_exp.squeeze(1)].view(batch_size, 1, 1)
            beta_t = self.betas[t_exp.squeeze(1)].view(batch_size, 1, 1)
            one_minus_alpha_bar_t = 1 - alpha_bar_t
            
            # Compute predicted noise from pred_x0
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(one_minus_alpha_bar_t)
            ε_pred = (x_t - sqrt_alpha_bar_t * pred_x0) / sqrt_one_minus_alpha_bar_t
            
            # Compute score s_theta ≈ -ε_pred / sqrt(1 - α_bar_t)
            sigma_t = sqrt_one_minus_alpha_bar_t
            s_theta = - ε_pred / sigma_t
            
            # Apply CBF guidance if enabled
            ε_guided = ε_pred.clone()
            barrier_info = None
            if self.config.enable_cbf_guidance and guidance_gamma is not None and mean is not None and std is not None:
                # Compute γ_t (scheduled: increases with t for stronger late guidance)
                gamma_t = guidance_gamma * (t_exp.squeeze(1).float() / self.num_timesteps)
                
                # Compute barrier gradient ∇V with multiple obstacles
                V, grad_V = compute_barrier_and_grad(x_t, self.config, mean, std, obstacles_data)
                barrier_info = {'V': V, 'grad_V': grad_V, 'gamma_t': gamma_t}

                # Guided score: s_guided = s_theta - γ_t ∇V
                s_guided = s_theta - gamma_t.view(batch_size, 1, 1) * grad_V
                
                # Guided noise
                ε_guided = - s_guided * sigma_t
            
            # Compute mean μ using guided noise (standard DDPM formula)
            coeff = (1 - alpha_t) / sqrt_one_minus_alpha_bar_t
            mu = (1 / torch.sqrt(alpha_t)) * (x_t - coeff * ε_guided)
            
            # For t=0, return pred_x0 (or guided equivalent)
            is_t_zero = (t_exp.squeeze(1) == 0).all()
            if is_t_zero:
                # Compute guided pred_x0 for consistency
                pred_x0_guided = (x_t - sqrt_one_minus_alpha_bar_t * ε_guided) / sqrt_alpha_bar_t
                
                if plot_step:
                    self._plot_diffusion_step(x_t, pred_x0_guided, t, step_idx, barrier_info, is_final=True, obstacles_data=obstacles_data)
                return pred_x0_guided
            
            # Variance (DDPM posterior variance)
            alpha_bar_prev = self.alpha_bars[t_exp.squeeze(1) - 1].view(batch_size, 1, 1) if t.min() > 0 else torch.ones_like(alpha_bar_t)
            var = beta_t * (1 - alpha_bar_prev) / one_minus_alpha_bar_t
            sigma = torch.sqrt(var)
            
            # Sample noise
            z = torch.randn_like(x_t)
            
            # x_{t-1}
            x_prev = mu + sigma * z
            
            if plot_step:
                self._plot_diffusion_step(x_t, x_prev, t, step_idx, barrier_info, is_final=False, obstacles_data=obstacles_data)
            
            return x_prev

    def _plot_diffusion_step(self, x_t, x_prev, t, step_idx, barrier_info=None, is_final=False, obstacles_data=None):
        """Plot the current diffusion step with obstacles"""
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f'Reverse Diffusion Process - Step {step_idx} (t={t[0].item()})', fontsize=16)
        
        # Extract position coordinates
        x_t_pos = x_t[0, :, 1:4].cpu().numpy()
        x_prev_pos = x_prev[0, :, 1:4].cpu().numpy()
        
        # 1. 3D trajectory evolution with obstacles
        ax1 = fig.add_subplot(241, projection='3d')
        ax1.plot(x_t_pos[:, 0], x_t_pos[:, 1], x_t_pos[:, 2], 'r-', label='x_t (current)', linewidth=2, alpha=0.7)
        ax1.plot(x_prev_pos[:, 0], x_prev_pos[:, 1], x_prev_pos[:, 2], 'b-', label='x_prev (denoised)', linewidth=2, alpha=0.7)
        
        # Plot obstacles if available
        if obstacles_data:
            for obstacle in obstacles_data:
                center = obstacle['center'].numpy()
                radius = obstacle['radius']
                
                # Create sphere surface
                u = np.linspace(0, 2 * np.pi, 10)
                v = np.linspace(0, np.pi, 10)
                x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                
                ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='red')
        
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

# Enhanced AeroDM with Obstacle Awareness
class EnhancedAeroDM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.diffusion_model = ObstacleAwareDiffusionTransformer(config)
        self.diffusion_process = ObstacleAwareDiffusionProcess(config)
        self.mean = None
        self.std = None
        self.obstacles_data = None
        
    def forward(self, x_t, t, target, action, history=None, obstacles_data=None):
        return self.diffusion_model(x_t, t, target, action, history, obstacles_data)
    
    def set_normalization_params(self, mean, std):
        """Set normalization parameters for CBF guidance"""
        self.mean = mean
        self.std = std
    
    def set_obstacles_data(self, obstacles_data):
        """Set obstacles data for CBF guidance and model input"""
        self.obstacles_data = obstacles_data
        print(f"Set {len(obstacles_data)} obstacles for model input and CBF guidance")
    
    def sample(self, target, action, history=None, batch_size=1, enable_guidance=True, 
               guidance_gamma=None, plot_all_steps=False):
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
        print("STARTING OBSTACLE-AWARE REVERSE DIFFUSION PROCESS")
        print(f"Initial noise stats - Mean: {x_t.mean().item():.4f}, Std: {x_t.std().item():.4f}")
        print(f"Total steps: {self.config.diffusion_steps}")
        print(f"CBF Guidance: {enable_guidance}")
        if self.obstacles_data:
            print(f"Number of obstacles: {len(self.obstacles_data)}")
            print(f"Obstacle information integrated into transformer")
        print(f"{'='*50}")
        
        # Reverse diffusion process
        step_counter = 0
        for t_step in reversed(range(self.config.diffusion_steps)):
            t_batch = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
            gamma = guidance_gamma if enable_guidance else None
            
            # Plot every step if requested, or key steps for overview
            plot_step = plot_all_steps or (t_step % max(1, self.config.diffusion_steps // 5) == 0) or t_step == 0
            
            x_t = self.diffusion_process.p_sample(
                self.diffusion_model, x_t, t_batch, target, action, history, 
                gamma, self.mean, self.std, plot_step=plot_step, step_idx=step_counter,
                obstacles_data=self.obstacles_data
            )
            step_counter += 1
        
        print(f"\n{'='*50}")
        print("OBSTACLE-AWARE REVERSE DIFFUSION PROCESS COMPLETED")
        print(f"Final trajectory stats - Mean: {x_t.mean().item():.4f}, Std: {x_t.std().item():.4f}")
        print(f"{'='*50}")
        
        return x_t

# Improved Loss Function with Balanced Z-Axis Learning
class ImprovedAeroDMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred_trajectory, gt_trajectory):
        # Separate position dimensions for balanced learning
        pred_pos = pred_trajectory[:, :, 1:4]
        gt_pos = gt_trajectory[:, :, 1:4]

        # Basic loss
        x_loss = self.mse_loss(pred_pos[:, :, 0], gt_pos[:, :, 0])
        y_loss = self.mse_loss(pred_pos[:, :, 1], gt_pos[:, :, 1])
        z_loss = self.mse_loss(pred_pos[:, :, 2], gt_pos[:, :, 2])

        # All dimensional losses in the last time step
        last_x_loss = self.mse_loss(pred_pos[:, -1, 0], gt_pos[:, -1, 0])
        last_y_loss = self.mse_loss(pred_pos[:, -1, 1], gt_pos[:, -1, 1])
        last_z_loss = self.mse_loss(pred_pos[:, -1, 2], gt_pos[:, -1, 2])

        # Combination loss, with the last point having a higher weight
        position_loss = (x_loss + y_loss + z_loss + 
                         10.0 * (last_x_loss + last_y_loss + last_z_loss))  # 最后一个点权重加倍
        
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
        
        total_loss = 2.0 * position_loss + 1.5 * vel_loss + other_components_loss
        
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

def plot_training_losses(losses):
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
    plt.show()

def generate_enhanced_circular_trajectories(num_trajectories=100, seq_len=60, radius=10.0, height=0.0):
    """Generate diverse aerobatic trajectories based on five maneuver styles:
    (a) Power Loop, (b) Barrel Roll, (c) Split-S, (d) Immelmann Turn, (e) Wall Ride."""
    trajectories = []
    maneuver_styles = ['power_loop', 'barrel_roll', 'split_s', 'immelmann', 'wall_ride']
    
    for i in range(num_trajectories):
        # Randomly select a maneuver style
        style = np.random.choice(maneuver_styles)
        
        # Random centers and scales
        center_x = np.random.uniform(-20, 20)
        center_y = np.random.uniform(-20, 20)
        center_z = height + np.random.uniform(-10, 10)
        
        current_radius = radius * np.random.uniform(0.8, 1.2)
        angular_velocity = np.random.uniform(0.5, 2.0)
        
        # Normalize time steps to [0, 1]
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
    
    return torch.tensor(trajectories, dtype=torch.float32)

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

# Update the test function to demonstrate CBF guidance with diffusion visualization
def test_enhanced_model_performance(model, trajectories_norm, mean, std, num_test_samples=3):
    """Enhanced testing with obstacle-aware transformer"""
    print("\nTesting enhanced obstacle-aware model performance...")
    config = model.config
    device = next(model.parameters()).device
    
    # Set normalization parameters
    model.set_normalization_params(mean, std)
    
    model.eval()
    with torch.no_grad():
        for i in range(min(num_test_samples, trajectories_norm.shape[0])):
            # Prepare test sample
            full_traj = trajectories_norm[i:i+1]
            target_norm = generate_target_waypoints(full_traj)
            action = generate_action_styles(1, config.action_dim, device=device)
            history = generate_history_segments(full_traj, config.history_len, device=device)
            x_0 = full_traj[:, config.history_len:, :]
            
            # Denormalize for obstacle generation and plotting
            x_0_denorm = denormalize_trajectories(x_0, mean, std)
            target_denorm = target_norm * std[0, 0, 1:4] + mean[0, 0, 1:4]
            
            # Generate random obstacles
            obstacles = generate_random_obstacles(x_0_denorm[0], num_obstacles_range=(0, 10), radius_range=(0.05, 0.30))

            # Set obstacles data for model input
            model.set_obstacles_data(obstacles)
            
            print(f"\n{'='*60}")
            print(f"ENHANCED TEST SAMPLE {i+1}")
            print(f"Generated {len(obstacles)} random obstacles")
            print(f"Obstacle information integrated into transformer via MLP encoder")
            print(f"{'='*60}")
            
            # Sample with obstacle-aware transformer and CBF guidance
            sampled_guided = model.sample(target_norm, action, history, batch_size=1, 
                                        enable_guidance=True, guidance_gamma=config.guidance_gamma,
                                        plot_all_steps=False)
            
            # For comparison, sample without guidance but with obstacle awareness
            sampled_unguided = model.sample(target_norm, action, history, batch_size=1, 
                                          enable_guidance=False, plot_all_steps=False)
            
            # Denormalize results
            sampled_guided_denorm = denormalize_trajectories(sampled_guided, mean, std)
            sampled_unguided_denorm = denormalize_trajectories(sampled_unguided, mean, std)
            history_denorm = denormalize_trajectories(history, mean, std)
            
            # Enhanced visualization with obstacles
            plot_enhanced_trajectory_comparison(
                x_0_denorm, sampled_unguided_denorm, sampled_guided_denorm, 
                history=history_denorm, target=target_denorm, obstacles=obstacles,
                title=f"Enhanced Obstacle-Aware Test Sample {i+1}\n(Transformer + CBF Guidance)"
            )

    model.train()

def plot_enhanced_trajectory_comparison(original, reconstructed, sampled, history=None, 
                                      target=None, obstacles=None, title="Enhanced Trajectory Comparison"):
    """Enhanced visualization with multiple obstacles and trajectory comparison"""
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(title, fontsize=16)
    
    # Extract position coordinates from tensors
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
    
    # 1. 3D trajectory plot with history, target, and multiple obstacles
    ax1 = fig.add_subplot(341, projection='3d')
    
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
    
    # Plot multiple obstacles (if provided)
    obstacle_proxies = []
    if obstacles is not None and len(obstacles) > 0:
        for i, obstacle in enumerate(obstacles):
            center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
            radius = obstacle['radius']
            
            # Create sphere surface for 3D obstacle visualization
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 10)
            obs_x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            obs_y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            obs_z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Use different colors for multiple obstacles
            colors = ['red', 'orange', 'purple', 'brown', 'pink']
            color = colors[i % len(colors)]
            # alpha = 0.3 + (i * 0.1)  # Vary transparency
            alpha = 0.6
            
            ax1.plot_surface(obs_x, obs_y, obs_z, alpha=alpha, color=color)
            
            # Create proxy artist for legend (only once)
            if i == 0:
                obstacle_proxies.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha, label=f'Obstacles'))
    
    # Create legend with custom handling
    legend_handles = [
        plt.Line2D([0], [0], color='m', linewidth=4, marker='o', markersize=4, label='History'),
        plt.Line2D([0], [0], color='b', linewidth=3, label='Original Trajectory'),
        plt.Line2D([0], [0], color='r', linewidth=2, marker='.', label='Reconstructed Trajectory'),
        plt.Line2D([0], [0], color='g', linewidth=2, marker='.', label='Sampled Trajectory'),
        plt.Line2D([0], [0], color='yellow', marker='*', markersize=10, linestyle='None', 
                  markeredgecolor='black', markeredgewidth=2, label='Target Waypoint')
    ]
    
    legend_handles.extend(obstacle_proxies)
    ax1.legend(handles=legend_handles)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory with Multiple Obstacles')
    ax1.grid(True)
    
    # 2. X-Y projection with obstacles
    ax2 = fig.add_subplot(342)
    if history_pos is not None:
        ax2.plot(history_pos[:, 0], history_pos[:, 1], 'm-', label='History', linewidth=4, alpha=0.8, marker='o', markersize=4)
    ax2.plot(original_pos[:, 0], original_pos[:, 1], 'b-', label='Original', linewidth=3, alpha=0.8)
    ax2.plot(reconstructed_pos[:, 0], reconstructed_pos[:, 1], 'r.-', label='Reconstructed', linewidth=2, alpha=0.8)
    ax2.plot(sampled_pos[:, 0], sampled_pos[:, 1], 'g.-', label='Sampled', linewidth=2, alpha=0.8)
    
    if target_pos is not None:
        ax2.scatter(target_pos[0], target_pos[1], c='yellow', s=200, marker='*', label='Target', edgecolors='black', linewidth=2)
    
    # Plot obstacles in 2D projection
    if obstacles is not None:
        colors = ['red', 'orange', 'purple', 'brown', 'pink']
        for i, obstacle in enumerate(obstacles):
            center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
            radius = obstacle['radius']
            color = colors[i % len(colors)]
            circle = plt.Circle((center[0], center[1]), radius, color=color, alpha=0.3, 
                              label=f'Obstacle {i+1}' if i < 3 else "")
            ax2.add_patch(circle)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.set_title('X-Y Projection with Obstacles')
    ax2.grid(True)
    ax2.axis('equal')
    
    # 3. X-Z projection with obstacles
    ax3 = fig.add_subplot(343)
    if history_pos is not None:
        ax3.plot(history_pos[:, 0], history_pos[:, 2], 'm-', label='History', linewidth=4, alpha=0.8, marker='o', markersize=4)
    ax3.plot(original_pos[:, 0], original_pos[:, 2], 'b-', label='Original', linewidth=3, alpha=0.8)
    ax3.plot(reconstructed_pos[:, 0], reconstructed_pos[:, 2], 'r.-', label='Reconstructed', linewidth=2, alpha=0.8)
    ax3.plot(sampled_pos[:, 0], sampled_pos[:, 2], 'g.-', label='Sampled', linewidth=2, alpha=0.8)
    
    if target_pos is not None:
        ax3.scatter(target_pos[0], target_pos[2], c='yellow', s=200, marker='*', label='Target', edgecolors='black', linewidth=2)
    
    # Plot obstacles in X-Z projection
    if obstacles is not None:
        for i, obstacle in enumerate(obstacles):
            center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
            radius = obstacle['radius']
            color = colors[i % len(colors)]
            circle = plt.Circle((center[0], center[2]), radius, color=color, alpha=0.3)
            ax3.add_patch(circle)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.legend()
    ax3.set_title('X-Z Projection with Obstacles')
    ax3.grid(True)
    ax3.axis('equal')
    
    # 4. Y-Z projection with obstacles
    ax4 = fig.add_subplot(344)
    if history_pos is not None:
        ax4.plot(history_pos[:, 1], history_pos[:, 2], 'm-', label='History', linewidth=4, alpha=0.8, marker='o', markersize=4)
    ax4.plot(original_pos[:, 1], original_pos[:, 2], 'b-', label='Original', linewidth=3, alpha=0.8)
    ax4.plot(reconstructed_pos[:, 1], reconstructed_pos[:, 2], 'r.-', label='Reconstructed', linewidth=2, alpha=0.8)
    ax4.plot(sampled_pos[:, 1], sampled_pos[:, 2], 'g.-', label='Sampled', linewidth=2, alpha=0.8)
    
    if target_pos is not None:
        ax4.scatter(target_pos[1], target_pos[2], c='yellow', s=200, marker='*', label='Target', edgecolors='black', linewidth=2)
    
    # Plot obstacles in Y-Z projection
    if obstacles is not None:
        for i, obstacle in enumerate(obstacles):
            center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
            radius = obstacle['radius']
            color = colors[i % len(colors)]
            circle = plt.Circle((center[1], center[2]), radius, color=color, alpha=0.3)
            ax4.add_patch(circle)
    
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.legend()
    ax4.set_title('Y-Z Projection with Obstacles')
    ax4.grid(True)
    ax4.axis('equal')
    
    # 5. Position components over time
    time_steps = np.arange(original_pos.shape[0])
    ax5 = fig.add_subplot(345)
    ax5.plot(time_steps, original_pos[:, 0], 'b-', label='Original X', linewidth=2)
    ax5.plot(time_steps, reconstructed_pos[:, 0], 'r--', label='Reconstructed X', linewidth=2)
    ax5.plot(time_steps, sampled_pos[:, 0], 'g-.', label='Sampled X', linewidth=2)
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('X Position')
    ax5.legend()
    ax5.set_title('X Position Over Time')
    ax5.grid(True)
    
    ax6 = fig.add_subplot(346)
    ax6.plot(time_steps, original_pos[:, 1], 'b-', label='Original Y', linewidth=2)
    ax6.plot(time_steps, reconstructed_pos[:, 1], 'r--', label='Reconstructed Y', linewidth=2)
    ax6.plot(time_steps, sampled_pos[:, 1], 'g-.', label='Sampled Y', linewidth=2)
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Y Position')
    ax6.legend()
    ax6.set_title('Y Position Over Time')
    ax6.grid(True)
    
    ax7 = fig.add_subplot(347)
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
    
    ax8 = fig.add_subplot(348)
    ax8.plot(time_steps, original_speed, 'b-', label='Original Speed', linewidth=2)
    ax8.plot(time_steps, reconstructed_speed, 'r--', label='Reconstructed Speed', linewidth=2)
    ax8.plot(time_steps, sampled_speed, 'g-.', label='Sampled Speed', linewidth=2)
    ax8.set_xlabel('Time Step')
    ax8.set_ylabel('Speed')
    ax8.legend()
    ax8.set_title('Speed Over Time')
    ax8.grid(True)
    
    # 9. Obstacle distance analysis
    ax9 = fig.add_subplot(349)
    if obstacles is not None and len(obstacles) > 0:
        # Calculate minimum distance to any obstacle for each trajectory
        min_dist_original = []
        min_dist_sampled = []
        
        for t in range(original_pos.shape[0]):
            # For original trajectory
            dists_original = [np.linalg.norm(original_pos[t] - obstacle['center'].cpu().numpy()) - obstacle['radius'] 
                            for obstacle in obstacles]
            min_dist_original.append(min(dists_original))
            
            # For sampled trajectory
            dists_sampled = [np.linalg.norm(sampled_pos[t] - obstacle['center'].cpu().numpy()) - obstacle['radius'] 
                           for obstacle in obstacles]
            min_dist_sampled.append(min(dists_sampled))
        
        ax9.plot(time_steps, min_dist_original, 'b-', label='Original Min Distance', linewidth=2)
        ax9.plot(time_steps, min_dist_sampled, 'g-', label='Sampled Min Distance', linewidth=2)
        ax9.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Collision Threshold')
        ax9.set_xlabel('Time Step')
        ax9.set_ylabel('Distance to Nearest Obstacle')
        ax9.legend()
        ax9.set_title('Obstacle Avoidance Performance')
        ax9.grid(True)
    else:
        ax9.text(0.5, 0.5, 'No Obstacles\nAvailable', ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Obstacle Distance Analysis')
    
    # 10. Error analysis
    recon_error = np.linalg.norm(reconstructed_pos - original_pos, axis=1)
    sampled_error = np.linalg.norm(sampled_pos - original_pos, axis=1)
    
    ax10 = fig.add_subplot(3,4,10)
    ax10.plot(time_steps, recon_error, 'r-', label='Reconstruction Error', linewidth=2)
    ax10.plot(time_steps, sampled_error, 'g-', label='Sampling Error', linewidth=2)
    ax10.set_xlabel('Time Step')
    ax10.set_ylabel('Position Error')
    ax10.legend()
    ax10.set_title('Position Error Over Time')
    ax10.grid(True)
    
    # 11. Obstacle information summary
    ax11 = fig.add_subplot(3,4,11)
    if obstacles is not None:
        obstacle_info = []
        for i, obstacle in enumerate(obstacles):
            center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
            radius = obstacle['radius']
            obstacle_info.append(f'Obs {i+1}: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})\nR: {radius:.1f}')
        
        ax11.axis('off')
        obstacle_text = f"Obstacles: {len(obstacles)}\n\n" + "\n".join(obstacle_info[:5])  # Show first 5
        if len(obstacles) > 5:
            obstacle_text += f"\n... and {len(obstacles) - 5} more"
        ax11.text(0.1, 0.9, obstacle_text, transform=ax11.transAxes, fontsize=10, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    else:
        ax11.axis('off')
        ax11.text(0.5, 0.5, 'No Obstacles', ha='center', va='center', transform=ax11.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    ax11.set_title('Obstacle Information')
    
    # 12. Performance statistics
    ax12 = fig.add_subplot(3,4,12)
    stats_data = [
        np.mean(recon_error), np.std(recon_error), np.max(recon_error),
        np.mean(sampled_error), np.std(sampled_error), np.max(sampled_error)
    ]
    stats_labels = ['Recon Mean', 'Recon Std', 'Recon Max', 'Sample Mean', 'Sample Std', 'Sample Max']
    bars = ax12.bar(range(len(stats_data)), stats_data, color=['red', 'red', 'red', 'green', 'green', 'green'])
    ax12.set_xticks(range(len(stats_data)))
    ax12.set_xticklabels(stats_labels, rotation=45)
    ax12.set_ylabel('Error Value')
    ax12.set_title('Performance Statistics')
    
    # Add value labels on bars
    for bar, value in zip(bars, stats_data):
        ax12.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.3f}', 
                 ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("\n=== Enhanced Trajectory Analysis ===")
    print(f"Reconstruction - Total Error: {np.mean(recon_error):.4f} ± {np.std(recon_error):.4f}")
    print(f"Sampling - Total Error: {np.mean(sampled_error):.4f} ± {np.std(sampled_error):.4f}")
    
    if obstacles is not None and len(obstacles) > 0:
        # Calculate collision statistics
        collisions_original = sum(1 for dist in min_dist_original if dist < 0)
        collisions_sampled = sum(1 for dist in min_dist_sampled if dist < 0)
        print(f"Original Trajectory - Collisions: {collisions_original}/{len(min_dist_original)} time steps")
        print(f"Sampled Trajectory - Collisions: {collisions_sampled}/{len(min_dist_sampled)} time steps")
        print(f"Number of obstacles: {len(obstacles)}")

# Enhanced training function
# def train_enhanced_aerodm():
#     """Enhanced training with obstacle-aware transformer"""
#     config = Config()
#     model = EnhancedAeroDM(config)
    
#     # Set device
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#     model = model.to(device)
#     model.config = config
    
#     # Disable CBF guidance during training (obstacle info still goes to transformer)
#     model.config.enable_cbf_guidance = False
    
#     # Move diffusion process tensors to device
#     model.diffusion_process.betas = model.diffusion_process.betas.to(device)
#     model.diffusion_process.alphas = model.diffusion_process.alphas.to(device)
#     model.diffusion_process.alpha_bars = model.diffusion_process.alpha_bars.to(device)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#     criterion = ImprovedAeroDMLoss(config)
    
#     # Training parameters
#     num_epochs = 50
#     batch_size = 8
#     num_trajectories = 20000
    
#     print("Generating training data with obstacle-aware transformer...")
#     trajectories = generate_enhanced_circular_trajectories(
#         num_trajectories=num_trajectories,
#         seq_len=config.seq_len + config.history_len
#     )
    
#     # Normalize trajectories
#     trajectories_norm, mean, std = normalize_trajectories(trajectories)
#     trajectories_norm = trajectories_norm.to(device)
#     mean = mean.to(device)
#     std = std.to(device)
    
#     # Store losses
#     losses = {'total': [], 'position': [], 'vel': []}
    
#     print("Starting enhanced training with obstacle-aware transformer...")
#     for epoch in range(num_epochs):
#         epoch_total_loss = 0
#         epoch_position_loss = 0
#         epoch_vel_loss = 0
#         num_batches = 0
        
#         indices = torch.randperm(num_trajectories)
#         for i in range(0, num_trajectories, batch_size):
#             if i + batch_size > num_trajectories:
#                 actual_batch_size = num_trajectories - i
#             else:
#                 actual_batch_size = batch_size
                
#             batch_indices = indices[i:i+actual_batch_size]
#             full_traj = trajectories_norm[batch_indices]
#             target = generate_target_waypoints(full_traj)
#             action = generate_action_styles(actual_batch_size, config.action_dim, device=device)
#             history = generate_history_segments(full_traj, config.history_len, device=device)
#             x_0 = full_traj[:, config.history_len:, :]
            
#             # Generate random obstacles for this batch (for transformer input)
#             obstacles_for_batch = []
#             for j in range(actual_batch_size):
#                 traj_denorm = denormalize_trajectories(full_traj[j:j+1], mean, std)
#                 obstacles = generate_random_obstacles(traj_denorm[0], num_obstacles_range=(0, 15), radius_range=(0.5, 1.30), device=device)            
#                 obstacles_for_batch.append(obstacles)
            
#             # Sample diffusion time step
#             t = torch.randint(0, config.diffusion_steps, (actual_batch_size,), device=device)
            
#             # Forward diffusion
#             noisy_x, noise = model.diffusion_process.q_sample(x_0, t)
            
#             # Model prediction with obstacle information - 修改这里：逐个处理batch中的每个样本
#             pred_x0_list = []
#             for j in range(actual_batch_size):
#                 # 对batch中的每个样本单独调用模型
#                 pred_x0_j = model(
#                     noisy_x[j:j+1], 
#                     t[j:j+1], 
#                     target[j:j+1], 
#                     action[j:j+1], 
#                     history[j:j+1] if history is not None else None,
#                     obstacles_for_batch[j]  # 传递单个样本的障碍物列表
#                 )
#                 pred_x0_list.append(pred_x0_j)
            
#             # 合并预测结果
#             pred_x0 = torch.cat(pred_x0_list, dim=0)
            
#             # Calculate loss
#             total_loss, position_loss, vel_loss = criterion(pred_x0, x_0)
            
#             # Backward propagation
#             optimizer.zero_grad()
#             total_loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             epoch_total_loss += total_loss.item()
#             epoch_position_loss += position_loss.item()
#             epoch_vel_loss += vel_loss.item()
#             num_batches += 1
        
#         # Calculate average loss
#         if num_batches > 0:
#             avg_total = epoch_total_loss / num_batches
#             avg_position = epoch_position_loss / num_batches
#             avg_vel = epoch_vel_loss / num_batches
            
#             losses['total'].append(avg_total)
#             losses['position'].append(avg_position)
#             losses['vel'].append(avg_vel)
        
#         if epoch % 5 == 0:
#             print(f"Epoch {epoch}: Total Loss: {avg_total:.4f}, "
#                   f"Position: {avg_position:.4f}, Vel: {avg_vel:.4f}")
#         if avg_position < 0.02:
#             print("Early stopping as position loss is below threshold.")
#             break

#     # Plot training losses
#     plot_training_losses(losses)
    
#     # Enable CBF guidance for inference
#     model.config.enable_cbf_guidance = True
    
#     # Test enhanced model
#     test_enhanced_model_performance(model, trajectories_norm, mean, std, num_test_samples=30)
    
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'epoch': epoch,
#         'loss': losses,
#         'mean': mean,
#         'std': std
#     }, "model/enhanced_obstacle_aware_aerodm.pth")

#     return model, losses, trajectories, mean, std

# Main execution
# if __name__ == "__main__":
#     print("Training Enhanced Obstacle-Aware AeroDM with Transformer Integration...")
    
#     # Train with enhanced obstacle-aware model
#     trained_model, losses, trajectories, mean, std = train_enhanced_aerodm()
    
#     print("Training completed! Obstacle information integrated into transformer via MLP encoder.")


# Enhanced Loss Function with Obstacle Distance Term
class ObstacleAwareAeroDMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.obstacle_weight = 10.0  # Weight for obstacle distance term
        
    def compute_obstacle_distance_loss(self, pred_trajectory, obstacles_data, mean, std):
        """
        Compute obstacle distance loss to encourage obstacle avoidance.
        Loss = sum over obstacles and time steps of max(0, safety_margin - distance)^2
        """
        if not obstacles_data or len(obstacles_data) == 0:
            return torch.tensor(0.0, device=pred_trajectory.device)
        
        batch_size, seq_len, _ = pred_trajectory.shape
        obstacle_loss = torch.tensor(0.0, device=pred_trajectory.device)
        
        # Denormalize positions for distance computation
        pred_pos_denorm = pred_trajectory[:, :, 1:4] * std[0, 0, 1:4] + mean[0, 0, 1:4]
        
        for batch_idx in range(batch_size):
            # Get obstacles for this batch sample
            if isinstance(obstacles_data, list):
                obstacles = obstacles_data[batch_idx] if batch_idx < len(obstacles_data) else []
            else:
                obstacles = obstacles_data
                
            for obstacle in obstacles:
                center = obstacle['center'].to(pred_trajectory.device)
                radius = obstacle['radius']
                safety_margin = radius * 1.2  # Add 20% safety margin
                
                # Compute distances from trajectory to obstacle center
                distances = torch.norm(pred_pos_denorm[batch_idx] - center, dim=1)
                
                # Effective distance considering obstacle radius
                effective_distances = distances - radius
                
                # Penalize trajectories that get too close to obstacles
                closeness_penalty = torch.clamp(safety_margin - effective_distances, min=0.0)
                obstacle_loss += torch.sum(closeness_penalty ** 2)
        
        # Normalize by batch size and sequence length
        if batch_size > 0 and len(obstacles_data) > 0:
            obstacle_loss = obstacle_loss / (batch_size * seq_len)
            
        return obstacle_loss
    
    def forward(self, pred_trajectory, gt_trajectory, obstacles_data=None, mean=None, std=None):
        # Separate position dimensions for balanced learning
        pred_pos = pred_trajectory[:, :, 1:4]
        gt_pos = gt_trajectory[:, :, 1:4]

        # Basic loss components
        x_loss = self.mse_loss(pred_pos[:, :, 0], gt_pos[:, :, 0])
        y_loss = self.mse_loss(pred_pos[:, :, 1], gt_pos[:, :, 1])
        z_loss = self.mse_loss(pred_pos[:, :, 2], gt_pos[:, :, 2])

        # Last time step losses with higher weight
        last_x_loss = self.mse_loss(pred_pos[:, -1, 0], gt_pos[:, -1, 0])
        last_y_loss = self.mse_loss(pred_pos[:, -1, 1], gt_pos[:, -1, 1])
        last_z_loss = self.mse_loss(pred_pos[:, -1, 2], gt_pos[:, -1, 2])

        # Combined position loss
        position_loss = (x_loss + y_loss + z_loss + 
                         10.0 * (last_x_loss + last_y_loss + last_z_loss))
        
        # Velocity regularization
        pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        gt_vel = gt_pos[:, 1:, :] - gt_pos[:, :-1, :]
        
        vel_x_loss = self.mse_loss(pred_vel[:, :, 0], gt_vel[:, :, 0])
        vel_y_loss = self.mse_loss(pred_vel[:, :, 1], gt_vel[:, :, 1])
        vel_z_loss = self.mse_loss(pred_vel[:, :, 2], gt_vel[:, :, 2])
        vel_loss = vel_x_loss + vel_y_loss + vel_z_loss
        
        # Other state components
        other_components_loss = self.mse_loss(
            torch.cat([pred_trajectory[:, :, :1], pred_trajectory[:, :, 4:]], dim=2),
            torch.cat([gt_trajectory[:, :, :1], gt_trajectory[:, :, 4:]], dim=2)
        )
        
        # Obstacle distance loss (if obstacles and normalization params are provided)
        obstacle_loss = torch.tensor(0.0, device=pred_trajectory.device)
        if obstacles_data is not None and mean is not None and std is not None:
            obstacle_loss = self.compute_obstacle_distance_loss(
                pred_trajectory, obstacles_data, mean, std
            )
        
        # Total loss with obstacle term
        total_loss = (2.0 * position_loss + 
                     1.5 * vel_loss + 
                     other_components_loss + 
                     self.obstacle_weight * obstacle_loss)
        
        return total_loss, position_loss, vel_loss, obstacle_loss

# Update the training function to use the new loss
def train_enhanced_aerodm():
    """Enhanced training with obstacle-aware transformer and obstacle-aware loss"""
    config = Config()
    model = EnhancedAeroDM(config)
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.config = config
    
    # Disable CBF guidance during training (obstacle info still goes to transformer)
    model.config.enable_cbf_guidance = False
    
    # Move diffusion process tensors to device
    model.diffusion_process.betas = model.diffusion_process.betas.to(device)
    model.diffusion_process.alphas = model.diffusion_process.alphas.to(device)
    model.diffusion_process.alpha_bars = model.diffusion_process.alpha_bars.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = ObstacleAwareAeroDMLoss(config)  # Use the new loss function
    
    # Training parameters
    num_epochs = 50
    batch_size = 8
    num_trajectories = 20000
    
    print("Generating training data with obstacle-aware transformer...")
    trajectories = generate_enhanced_circular_trajectories(
        num_trajectories=num_trajectories,
        seq_len=config.seq_len + config.history_len
    )
    
    # Normalize trajectories
    trajectories_norm, mean, std = normalize_trajectories(trajectories)
    trajectories_norm = trajectories_norm.to(device)
    mean = mean.to(device)
    std = std.to(device)
    
    # Store losses
    losses = {'total': [], 'position': [], 'vel': [], 'obstacle': []}
    
    print("Starting enhanced training with obstacle-aware transformer and loss...")
    for epoch in range(num_epochs):
        epoch_total_loss = 0
        epoch_position_loss = 0
        epoch_vel_loss = 0
        epoch_obstacle_loss = 0
        num_batches = 0
        
        indices = torch.randperm(num_trajectories)
        for i in range(0, num_trajectories, batch_size):
            if i + batch_size > num_trajectories:
                actual_batch_size = num_trajectories - i
            else:
                actual_batch_size = batch_size
                
            batch_indices = indices[i:i+actual_batch_size]
            full_traj = trajectories_norm[batch_indices]
            target = generate_target_waypoints(full_traj)
            action = generate_action_styles(actual_batch_size, config.action_dim, device=device)
            history = generate_history_segments(full_traj, config.history_len, device=device)
            x_0 = full_traj[:, config.history_len:, :]
            
            # Generate random obstacles for this batch (for transformer input and loss)
            obstacles_for_batch = []
            for j in range(actual_batch_size):
                traj_denorm = denormalize_trajectories(full_traj[j:j+1], mean, std)
                obstacles = generate_random_obstacles(traj_denorm[0], num_obstacles_range=(0, 15), radius_range=(0.5, 1.30), device=device)            
                obstacles_for_batch.append(obstacles)
            
            # Sample diffusion time step
            t = torch.randint(0, config.diffusion_steps, (actual_batch_size,), device=device)
            
            # Forward diffusion
            noisy_x, noise = model.diffusion_process.q_sample(x_0, t)
            
            # Model prediction with obstacle information
            pred_x0_list = []
            for j in range(actual_batch_size):
                pred_x0_j = model(
                    noisy_x[j:j+1], 
                    t[j:j+1], 
                    target[j:j+1], 
                    action[j:j+1], 
                    history[j:j+1] if history is not None else None,
                    obstacles_for_batch[j]
                )
                pred_x0_list.append(pred_x0_j)
            
            pred_x0 = torch.cat(pred_x0_list, dim=0)
            
            # Calculate loss with obstacle awareness
            total_loss, position_loss, vel_loss, obstacle_loss = criterion(
                pred_x0, x_0, obstacles_for_batch, mean, std
            )
            
            # Backward propagation
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_total_loss += total_loss.item()
            epoch_position_loss += position_loss.item()
            epoch_vel_loss += vel_loss.item()
            epoch_obstacle_loss += obstacle_loss.item()
            num_batches += 1
        
        # Calculate average loss
        if num_batches > 0:
            avg_total = epoch_total_loss / num_batches
            avg_position = epoch_position_loss / num_batches
            avg_vel = epoch_vel_loss / num_batches
            avg_obstacle = epoch_obstacle_loss / num_batches
            
            losses['total'].append(avg_total)
            losses['position'].append(avg_position)
            losses['vel'].append(avg_vel)
            losses['obstacle'].append(avg_obstacle)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Total Loss: {avg_total:.4f}, "
                  f"Position: {avg_position:.4f}, Vel: {avg_vel:.4f}, Obstacle: {avg_obstacle:.4f}")
        if avg_position < 0.02:
            print("Early stopping as position loss is below threshold.")
            break

    # Enhanced plotting with obstacle loss
    def plot_enhanced_training_losses(losses):
        """Plot training losses including obstacle loss"""
        plt.figure(figsize=(12, 8))
        epochs = range(len(losses['total']))
        
        plt.subplot(2, 1, 1)
        plt.plot(epochs, losses['total'], 'b-', label='Total Loss', linewidth=2)
        plt.plot(epochs, losses['position'], 'r--', label='Position Loss', linewidth=2)
        plt.plot(epochs, losses['vel'], 'g-.', label='Velocity Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses - Main Components')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(epochs, losses['obstacle'], 'm-', label='Obstacle Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Obstacle Loss')
        plt.title('Obstacle Distance Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    plot_enhanced_training_losses(losses)
    
    # Enable CBF guidance for inference
    model.config.enable_cbf_guidance = True
    
    # Test enhanced model
    test_enhanced_model_performance(model, trajectories_norm, mean, std, num_test_samples=30)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': losses,
        'mean': mean,
        'std': std
    }, "model/enhanced_obstacle_aware_aerodm.pth")

    return model, losses, trajectories, mean, std

# Update the main execution
if __name__ == "__main__":
    print("Training Enhanced Obstacle-Aware AeroDM with Transformer Integration and Obstacle-Aware Loss...")
    
    # Train with enhanced obstacle-aware model and loss
    trained_model, losses, trajectories, mean, std = train_enhanced_aerodm()
    
    print("Training completed! Obstacle information integrated into both transformer and loss function.")