#AeroDM + Barrier Function Guidance + Obstacle Encoding

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

    # Obstacle parameters
    max_obstacles = 10  # Maximum number of obstacles to process
    obstacle_feat_dim = 4  # [x, y, z, radius]
    enable_obstacle_encoding = False  # Toggle obstacle encoding in the model

    # CBF Guidance parameters (from CoDiG paper)
    enable_cbf_guidance = False  # Disabled by default; toggle for inference
    guidance_gamma = 100.0  # Base gamma for barrier guidance
    obstacle_radius = 5.0  # Safe distance radius
    
    # Plotting control
    show_flag = False  # Set to False to save plots as SVG instead of displaying
    
    @staticmethod
    def get_obstacle_center(device='cpu'):
        return torch.tensor([5.0, 5.0, 10.0], device=device)  # Example obstacle center
    
    @staticmethod
    def set_show_flag(flag):
        Config.show_flag = flag

# Transformer positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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
        if obstacles_data is None:
            # No obstacles, return zero embedding
            batch_size = 1
            return torch.zeros(batch_size, self.config.latent_dim, device=next(self.parameters()).device)
        if obstacles_data is not None and len(obstacles_data) == 0:
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
    
# Condition Embedding with Obstacle Information
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
        if obstacles_data is not None and self.config.enable_obstacle_encoding:
            obstacle_emb = self.obstacle_encoder(obstacles_data)
        else:
            obstacle_emb = torch.zeros_like(t_emb)
        
        # Combine all conditions with feature fusion
        combined_emb = torch.cat([t_emb, target_emb, action_emb, obstacle_emb], dim=-1)
        cond_emb = self.fusion_layer(combined_emb)
        
        return cond_emb

# Diffusion Transformer with Obstacle Awareness
class ObstacleAwareDiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.state_dim, config.latent_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.latent_dim)
        
        # condition embedding with obstacle information
        self.cond_embed = ConditionEmbedding(config)
        
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
        
        # Project input to latent space (no PE yet)
        x_proj = self.input_proj(x)
        
        # Prepare transformer input
        if history is not None:
            if history.size(0) != batch_size:
                if history.size(0) == 1:
                    history = history.repeat(batch_size, 1, 1)
                else:
                    raise ValueError(f"History data batch size mismatch")
            
            history_proj = self.input_proj(history)
            # Concatenate projected history and current *before* adding PE
            transformer_input = torch.cat([history_proj, x_proj], dim=1)
            total_seq_len = history_proj.size(1) + seq_len
        else:
            transformer_input = x_proj
            total_seq_len = seq_len
        
        # Now add positional encoding to the *combined* input (correct absolute positions)
        transformer_input = self.pos_encoding(transformer_input.transpose(0, 1)).transpose(0, 1)
        
        # Generate causal mask for the total sequence
        memory_mask = self._generate_square_subsequent_mask(total_seq_len).to(x.device)
        
        # Get condition embedding with obstacle information
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

# CBF Barrier Function with Multiple Obstacles
def compute_barrier_and_grad(x, config, mean, std, obstacles_data=None):
    """
    Compute barrier V and its gradient ∇V for the trajectory x.
    Extended to handle multiple spherical obstacles.
    V = sum_τ sum_obs max(0, r_obs - ||pos_τ - center_obs||)^2
    ∇V affects only position components (indices 1:4).
    """
    # Denormalize positions for barrier computation
    # Uses std and mean to denormalize only position components (1:4)
    pos_denorm = x[:, :, 1:4] * std[0, 0, 1:4] + mean[0, 0, 1:4]
    batch_size, seq_len, _ = pos_denorm.shape
    
    # Initialize barrier value and gradient
    V_total = torch.zeros(batch_size, device=x.device)
    # Gradient will be computed on the denormalized positions
    grad_pos_denorm = torch.zeros_like(pos_denorm, requires_grad=True)
    # Ensure pos_denorm requires grad for backprop
    pos_denorm = pos_denorm.clone().detach().requires_grad_(True)
    
    # Process each obstacle
    if obstacles_data is not None:
        # Handle obstacles_data: list-of-lists (per-batch) or single list (broadcast)
        if not isinstance(obstacles_data, list):
            obstacles_data = [obstacles_data] * batch_size  # Broadcast to batch

        for batch_idx in range(batch_size):
            # Get obstacles for this sample
            batch_obs = obstacles_data[batch_idx] if batch_idx < len(obstacles_data) else []
            
            for obstacle in batch_obs:
                center = obstacle['center'].to(x.device)  # Shape: (3,)
                radius = obstacle['radius']  # Scalar float
                
                # Euclidean distances from trajectory points to center
                # Shape: (seq_len, 3) -> (seq_len,)
                distances = torch.norm(pos_denorm[batch_idx] - center.unsqueeze(0), dim=1)
                
                # Closeness function: r - distance (positive means inside/near obstacle)
                h = radius - distances
                
                # Barrier violation term: max(0, h)^2
                violation = torch.clamp(h, min=0.0)
                V_obstacle = torch.sum(violation ** 2)
                V_total[batch_idx] += V_obstacle
                
                # Compute gradient for this obstacle
                # The gradient of max(0, h)^2 with respect to pos_denorm is:
                # 2 * max(0, h) * grad(h)
                # grad(h) = grad(r - distances) = -grad(distances)
                # grad(distances) = (pos_denorm - center) / distances (if distances > 0)
                
                # Compute mask for violated points (where h > 0, i.e., distance < radius)
                violation_mask = (h > 0).float().unsqueeze(1) # (seq_len, 1)
                
                # Compute direction vector from center to point (pos_denorm - center)
                direction_vec = pos_denorm[batch_idx] - center.unsqueeze(0) # (seq_len, 3)
                
                # Compute gradient of distance w.r.t pos_denorm
                # Add small epsilon to distances to avoid div by zero
                epsilon = 1e-6
                grad_dist = direction_vec / (distances.unsqueeze(1) + epsilon) # (seq_len, 3)
                
                # Grad(V_obs) w.r.t. pos_denorm: 2 * violation * (-grad_dist)
                grad_V_obs = -2 * violation.unsqueeze(1) * grad_dist
                
                # Apply violation mask (grad is 0 if not violated)
                grad_V_obs = grad_V_obs * violation_mask
                
                # Accumulate gradients (note: pos_denorm is still a leaf node)
                # We update the manually tracked gradient
                grad_pos_denorm[batch_idx] += grad_V_obs
    
    # Map the gradient back to the normalized space (x)
    # grad_x = grad_pos_denorm * (d(pos_denorm)/d(x))
    # pos_denorm = x[:, :, 1:4] * std + mean
    # d(pos_denorm)/d(x) = std
    grad_x = torch.zeros_like(x, device=x.device)
    grad_x[:, :, 1:4] = grad_pos_denorm / std[0, 0, 1:4].to(x.device)
    
    # V_total is sum of barrier violations over all obstacles and timesteps in the batch
    V_avg = V_total.mean() 
    
    return V_avg, grad_x

# Diffusion Process with Obstacle-Aware Sampling
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
                # Compute γ_t (scheduled: strongest at t=0 for final safety enforcement)
                gamma_t = guidance_gamma * (1.0 - t_exp.squeeze(1).float() / self.num_timesteps)
                
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
                    self._plot_diffusion_step(x_t, pred_x0_guided, t, step_idx, barrier_info, is_final=True, mean=mean, std=std, obstacles_data=obstacles_data)
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
        fixed_min = np.array([-20.0, -20.0, -20.0])
        fixed_max = np.array([20.0, 20.0, 20.0])

        # 1. 3D trajectory evolution with obstacles
        ax1 = fig.add_subplot(241, projection='3d')
        ax1.plot(x_t_pos[:, 0], x_t_pos[:, 1], x_t_pos[:, 2], 'r-', label='x_t (current)', linewidth=2, alpha=0.7)
        ax1.plot(x_prev_pos[:, 0], x_prev_pos[:, 1], x_prev_pos[:, 2], 'b-', label='x_prev (denoised)', linewidth=2, alpha=0.7)
        
        # Plot obstacles if available (assumes centers are already denormalized)
        if obstacles_data:
            for obstacle in obstacles_data:
                center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
                radius = obstacle['radius']
                
                # Create sphere surface for 3D visualization
                u = np.linspace(0, 2 * np.pi, 10)
                v = np.linspace(0, np.pi, 10)
                x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                
                ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='red')
        
        # Set fixed equal-range limits for X/Y/Z to prevent distortion (spheres look spherical)
        ax1.set_xlim(fixed_min[0], fixed_max[0])
        ax1.set_ylim(fixed_min[1], fixed_max[1])
        ax1.set_zlim(fixed_min[2], fixed_max[2])

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
            
# AeroDM with Obstacle Awareness
class AeroDM(nn.Module):
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
        
        # Optional: Soft init from history for better continuity
        if history is not None:
            last_pos = history[:, -1, 1:4]
            last_vel = history[:, -1, 1:4] - history[:, -2, 1:4] if history.size(1) > 1 else torch.zeros_like(last_pos)
            init_first_pos = last_pos.unsqueeze(1) + last_vel.unsqueeze(1)
            x_t[:, :1, 1:4] = 0.5 * x_t[:, :1, 1:4] + 0.5 * init_first_pos
        
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

# Pos Error, Velocity Error, Obstacle Distance Loss
class AeroDMLoss(nn.Module):
    """
    Unified loss function for AeroDM training.
    Combines position, velocity, speed, attitude, and optional obstacle avoidance losses.
    Supports switching obstacle term via flag; always returns 4 values for consistency.
    Fixes: Proper safety margin for obstacles, Z-weighting, normalization by avg obstacles.
    """
    def __init__(self, config, enable_obstacle_term=False, safe_extra_factor=0.2, z_weight=1.5, obstacle_weight=10.0, continuity_weight=15.0):
        super().__init__()
        self.config = config
        # Flag to enable/disable obstacle distance penalty in total loss
        self.enable_obstacle_term = enable_obstacle_term
        # Safety buffer beyond obstacle surface (as fraction of radius, e.g., 0.2 = 20%)
        self.safe_extra_factor = safe_extra_factor
        # Extra weight for Z-axis losses (height is critical in aviation trajectories)
        self.z_weight = z_weight
        # Scaling factor for the entire obstacle loss term
        self.obstacle_weight = obstacle_weight
        # Weight for continuity loss
        self.continuity_weight = continuity_weight
        # Base MSE loss for all components
        self.mse_loss = nn.MSELoss()
    
    def compute_obstacle_distance_loss(self, pred_trajectory, obstacles_data, mean, std):
        """
        Computes obstacle avoidance loss.
        Penalizes trajectories that enter a safety buffer around obstacles.
        Formula: sum_over_obs_and_time [max(0, safe_extra - (dist_to_surface))^2] / (batch * seq * avg_num_obs)
        Where dist_to_surface = dist_to_center - radius.
        Returns 0 if disabled or no obstacles.
        """
        if not self.enable_obstacle_term or not obstacles_data or len(obstacles_data) == 0:
            # Return a scalar tensor with requires_grad for backprop compatibility
            return torch.tensor(0.0, device=pred_trajectory.device, requires_grad=True)
        
        batch_size, seq_len, _ = pred_trajectory.shape
        device = pred_trajectory.device
        # Initialize accumulators
        obstacle_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_num_obs = 0  # For normalization by average obstacles per batch
        
        # Denormalize positions for real-world distance computation
        # Assumes position indices: 1:4 (x,y,z)
        pos_std = std[0, 0, 1:4]
        pos_mean = mean[0, 0, 1:4]
        pred_pos_denorm = pred_trajectory[:, :, 1:4] * pos_std + pos_mean
        
        # Handle obstacles_data: list-of-lists (per-batch) or single list (broadcast)
        if not isinstance(obstacles_data, list):
            obstacles_data = [obstacles_data] * batch_size  # Broadcast to batch
        
        # Loop over batch samples
        for batch_idx in range(batch_size):
            # Get obstacles for this sample (empty list if none)
            batch_obs = obstacles_data[batch_idx] if batch_idx < len(obstacles_data) else []
            total_num_obs += len(batch_obs)  # Count total for avg
            
            # Loop over obstacles in this sample
            for obstacle in batch_obs:
                # Obstacle center (3D tensor) and radius (scalar)
                center = obstacle['center'].to(device)  # Shape: (3,)
                radius = obstacle['radius']  # Scalar float
                # Compute safety extra distance (proportional to radius)
                safe_extra = self.safe_extra_factor * radius
                
                # Euclidean distances from trajectory points to center
                # Shape: (seq_len,)
                distances = torch.norm(pred_pos_denorm[batch_idx] - center.unsqueeze(0), dim=1)
                
                # Distance to obstacle surface (positive outside, negative inside)
                surface_dist = distances - radius
                # Penalty only if inside safety buffer: clamp(safe_extra - surface_dist, 0)
                # E.g., if surface_dist < safe_extra, penalize the violation squared
                closeness_penalty = torch.clamp(safe_extra - surface_dist, min=0.0)
                # Accumulate squared penalties over time steps
                obstacle_loss = obstacle_loss + torch.sum(closeness_penalty ** 2)
        
        # Normalize: Average per batch, time step, and obstacle (prevents bias from varying obs count)
        avg_num_obs = total_num_obs / max(batch_size, 1.0)  # Avoid div-by-zero
        if avg_num_obs > 0:
            obstacle_loss = obstacle_loss / (batch_size * seq_len * avg_num_obs)
        
        # Clamp to non-negative for stability (though clamp in penalty ensures this)
        return torch.clamp(obstacle_loss, min=0.0)
    
    def forward(self, pred_trajectory, gt_trajectory, obstacles_data=None, mean=None, std=None, history=None):
        """
        Computes total loss and components.
        Always returns (total_loss, position_loss, vel_loss, obstacle_loss, continuity_loss).
        - position_loss: Weighted MSE on positions (Z higher, last point x10).
        - vel_loss: MSE on velocity diffs (from position deltas, Z weighted).
        - obstacle_loss: 0 if disabled/no obs.
        - total: 2.0*position + 1.5*vel + other (speed + attitude) + obstacle_weight*obstacle.
        Handles seq_len <=1 for vel (returns 0).
        """
        batch_size, seq_len, state_dim = pred_trajectory.shape
        device = pred_trajectory.device
        
        # Extract components by indices: speed(0), pos(1:4), attitude(4:)
        pred_pos = pred_trajectory[:, :, 1:4]  # (B, T, 3) - positions x,y,z
        gt_pos = gt_trajectory[:, :, 1:4]
        pred_speed = pred_trajectory[:, :, 0:1]  # (B, T, 1) - speed
        gt_speed = gt_trajectory[:, :, 0:1]
        pred_attitude = pred_trajectory[:, :, 4:]  # (B, T, 6) - attitude (roll/pitch/yaw approx)
        gt_attitude = gt_trajectory[:, :, 4:]
        
        # Position losses: Per-dimension MSE
        x_loss = self.mse_loss(pred_pos[:, :, 0], gt_pos[:, :, 0])
        y_loss = self.mse_loss(pred_pos[:, :, 1], gt_pos[:, :, 1])
        # Z loss with extra weight for height accuracy
        z_loss = self.z_weight * self.mse_loss(pred_pos[:, :, 2], gt_pos[:, :, 2])
        
        # Last time-step losses (higher weight for endpoint accuracy)
        last_x_loss = self.mse_loss(pred_pos[:, -1, 0], gt_pos[:, -1, 0])
        last_y_loss = self.mse_loss(pred_pos[:, -1, 1], gt_pos[:, -1, 1])
        last_z_loss = self.z_weight * self.mse_loss(pred_pos[:, -1, 2], gt_pos[:, -1, 2])
        
        # Combine: Base + 10x last point
        position_loss = (x_loss + y_loss + z_loss + 
                         10.0 * (last_x_loss + last_y_loss + last_z_loss))
        
        # Velocity loss: From position differences (assumes uniform dt=1)
        if seq_len > 1:
            # Compute deltas: (B, T-1, 3)
            pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
            gt_vel = gt_pos[:, 1:, :] - gt_pos[:, :-1, :]
            # Per-dimension MSE on velocities
            vel_x_loss = self.mse_loss(pred_vel[:, :, 0], gt_vel[:, :, 0])
            vel_y_loss = self.mse_loss(pred_vel[:, :, 1], gt_vel[:, :, 1])
            vel_z_loss = self.z_weight * self.mse_loss(pred_vel[:, :, 2], gt_vel[:, :, 2])
            vel_loss = vel_x_loss + vel_y_loss + vel_z_loss
        else:
            # No velocity if single timestep
            vel_loss = torch.tensor(0.0, device=device)
        
        # Other losses: Explicit speed and attitude (no overlap with position)
        speed_loss = self.mse_loss(pred_speed, gt_speed)
        attitude_loss = self.mse_loss(pred_attitude, gt_attitude)
        other_loss = speed_loss + attitude_loss
        
        # Obstacle loss: Computed only if enabled and params provided
        obstacle_loss = torch.tensor(0.0, device=device)
        if self.enable_obstacle_term and obstacles_data is not None and mean is not None and std is not None:
            obstacle_loss = self.compute_obstacle_distance_loss(pred_trajectory, obstacles_data, mean, std)
        
        # New: Continuity loss (MSE between last history and first pred timestep)
        continuity_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if history is not None and pred_trajectory.size(1) > 0:
            # Focus on position components (indices 1:4) for smoothness
            last_history_pos = history[:, -1, 1:4]
            first_pred_pos = pred_trajectory[:, 0, 1:4]
            continuity_loss = self.mse_loss(first_pred_pos, last_history_pos)
            
            # Optional: Add velocity continuity (delta from last history to first pred)
            if history.size(1) > 1:
                last_history_vel = history[:, -1, 1:4] - history[:, -2, 1:4]
                first_pred_vel = pred_trajectory[:, 0, 1:4] - last_history_pos # Approx
                continuity_loss += self.mse_loss(first_pred_vel, last_history_vel)
        
        # Total weighted loss
        total_loss = (2.0 * position_loss + 1.5 * vel_loss + other_loss + 
                      self.obstacle_weight * obstacle_loss + self.continuity_weight * continuity_loss)
        
        return total_loss, position_loss, vel_loss, obstacle_loss, continuity_loss

def normalize_trajectories(trajectories):
    """Normalize each dimension to zero mean and unit variance"""
    mean = trajectories.mean(dim=(0, 1), keepdim=True)
    std = trajectories.std(dim=(0, 1), keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std) # avoid division by zero
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

def denormalize_obstacle(obstacle_center_norm, mean, std):
    """Denormalize obstacle center using the same normalization parameters"""
    # Extract position normalization parameters (indices 1:4 for x,y,z)
    pos_mean = mean[0, 0, 1:4].cpu().numpy()
    pos_std = std[0, 0, 1:4].cpu().numpy()
    # Denormalize obstacle center
    obstacle_denorm = obstacle_center_norm * pos_std + pos_mean
    return obstacle_denorm

# Trajectory Generation Utilities

def generate_target_waypoints(trajectory):
    """Extract a target waypoint (e.g., the final position) from a trajectory."""
    # Target is the final position (indices 1:4 for x, y, z)
    return trajectory[:, -1, 1:4]

def generate_action_styles(batch_size, action_dim, device='cpu'):
    """Generate one-hot or soft style embedding for maneuver type."""
    # Simple one-hot vector indicating a style
    style_idx = torch.randint(0, action_dim, (batch_size,))
    action = F.one_hot(style_idx, num_classes=action_dim).float().to(device)
    return action

def generate_history_segments(trajectories, history_len, device=None):
    """Extract history segments from trajectories"""
    if device is None:
        device = trajectories.device
    batch_size, seq_len, state_dim = trajectories.shape
    histories = []
    for i in range(batch_size):
        # start_idx is 0 for simplicity (use the first history_len steps)
        start_idx = 0
        history_segment = trajectories[i, start_idx:start_idx+history_len]
        if len(history_segment) < history_len:
            padding = torch.zeros(history_len - len(history_segment), state_dim, device=device)
            history_segment = torch.cat([history_segment, padding], dim=0)
        histories.append(history_segment)
    return torch.stack(histories)

def generate_aerobatic_trajectories(num_trajectories, seq_len, height=10.0, radius=5.0):
    """Generates synthetic aerobatic trajectories (Power Loop, Barrel Roll, Split S, Immelmann, Wall Ride, Figure Eight, Star, Half Moon, Sphinx, Clover)."""
    trajectories = []
    maneuver_styles = ['power_loop', 'barrel_roll', 'split_s', 'immelmann', 'wall_ride', 'eight_figure', 'star', 'half_moon', 'sphinx', 'clover']
    
    def smooth_trajectory(positions, smoothing_factor=0.1):
        # """Apply smoothing to trajectory positions using a simple moving average"""
        # smoothed = np.zeros_like(positions)
        # for i in range(len(positions)):
        #     start_idx = max(0, i - 1)
        #     end_idx = min(len(positions), i + 2)
        #     smoothed[i] = np.mean(positions[start_idx:end_idx], axis=0)
        # return smoothing_factor * smoothed + (1 - smoothing_factor) * positions
        return positions # Keeping the original simple implementation
        
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
            theta = np.pi * norm_t * angular_velocity
            x = center_x - current_radius * (1 - np.cos(theta))
            y = np.full(seq_len, center_y)
            z = center_z + current_radius * np.sin(theta)
            vx = -current_radius * angular_velocity * np.pi * np.sin(theta)
            vy = np.zeros(seq_len)
            vz = current_radius * angular_velocity * np.pi * np.cos(theta)

        elif style == 'barrel_roll':
            # Helical motion
            pitch = np.random.uniform(5.0, 10.0)
            theta = 2 * np.pi * norm_t * angular_velocity
            x = center_x + current_radius * np.cos(theta)
            y = center_y + current_radius * np.sin(theta)
            z = center_z + pitch * norm_t
            vx = -current_radius * angular_velocity * 2 * np.pi * np.sin(theta)
            vy = current_radius * angular_velocity * 2 * np.pi * np.cos(theta)
            vz = np.full(seq_len, pitch)

        elif style == 'split_s':
            # Half loop down, then inverted flight and recovery
            theta = np.pi * norm_t
            x = center_x + current_radius * np.sin(theta)
            y = np.full(seq_len, center_y)
            z = center_z - current_radius * (1 - np.cos(theta))
            vx = current_radius * np.pi * np.cos(theta)
            vy = np.zeros(seq_len)
            vz = -current_radius * np.pi * np.sin(theta)

        elif style == 'immelmann':
            # Half loop up, then half roll to recover
            theta = np.pi * norm_t
            x = center_x + current_radius * np.sin(theta)
            y = np.full(seq_len, center_y)
            z = center_z + current_radius * (1 - np.cos(theta))
            vx = current_radius * np.pi * np.cos(theta)
            vy = np.zeros(seq_len)
            vz = current_radius * np.pi * np.sin(theta)

        elif style == 'wall_ride':
            # Vertical helix climb (spiral up)
            turns = np.random.uniform(0.5, 1.5) # Number of turns
            climb_height = np.random.uniform(20.0, 40.0)
            theta = 2 * np.pi * turns * norm_t * angular_velocity
            x = center_x + current_radius * np.cos(theta)
            y = center_y + current_radius * np.sin(theta)
            z = center_z + climb_height * norm_t
            vx = -current_radius * angular_velocity * 2 * np.pi * turns * np.sin(theta)
            vy = current_radius * angular_velocity * 2 * np.pi * turns * np.cos(theta)
            vz = np.full(seq_len, climb_height)
        
        elif style == 'eight_figure':
            # Figure eight in the xy plane
            theta = 2 * np.pi * norm_t
            x = center_x + current_radius * np.sin(theta)
            y = center_y + 0.5 * current_radius * np.sin(2 * theta)
            z = np.full(seq_len, center_z)
            vx = current_radius * 2 * np.pi * np.cos(theta)
            vy = 0.5 * current_radius * 4 * np.pi * np.cos(2 * theta)
            vz = np.zeros(seq_len)

        elif style == 'star':
            # 3D Star/Lissajous-like curve
            alpha = 2.0
            beta = 3.0
            x = center_x + current_radius * np.sin(2 * np.pi * alpha * norm_t)
            y = center_y + current_radius * np.cos(2 * np.pi * beta * norm_t)
            z = center_z + current_radius * 0.5 * np.sin(2 * np.pi * (alpha + beta) * norm_t)
            vx = current_radius * 2 * np.pi * alpha * np.cos(2 * np.pi * alpha * norm_t)
            vy = -current_radius * 2 * np.pi * beta * np.sin(2 * np.pi * beta * norm_t)
            vz = current_radius * 0.5 * 2 * np.pi * (alpha + beta) * np.cos(2 * np.pi * (alpha + beta) * norm_t)

        elif style == 'half_moon':
            # Semicircle arc, primarily in xy plane, with some small z variation
            theta = np.pi * norm_t
            x = center_x + current_radius * np.cos(theta)
            y = center_y + current_radius * np.sin(theta)
            z = center_z + 0.1 * current_radius * np.sin(theta)
            vx = -current_radius * np.pi * np.sin(theta)
            vy = current_radius * np.pi * np.cos(theta)
            vz = 0.1 * current_radius * np.pi * np.cos(theta)

        elif style == 'sphinx':
            # Similar to wall ride, but with pitch variation for 'nose-up' maneuver
            turns = np.random.uniform(0.5, 1.5)
            climb_height = np.random.uniform(10.0, 30.0)
            theta = 2 * np.pi * turns * norm_t * angular_velocity
            x = center_x + current_radius * np.cos(theta)
            y = center_y + current_radius * np.sin(theta)
            z = center_z + climb_height * norm_t + 5 * np.sin(np.pi * norm_t) # Pitch variation
            vx = -current_radius * angular_velocity * 2 * np.pi * turns * np.sin(theta)
            vy = current_radius * angular_velocity * 2 * np.pi * turns * np.cos(theta)
            vz = np.full(seq_len, climb_height) + 5 * np.pi * np.cos(np.pi * norm_t)

        elif style == 'clover':
            # Four leaf clover shape (like two overlapping figure eights)
            alpha = 2.0
            x = center_x + current_radius * np.cos(2 * np.pi * norm_t) * np.cos(2 * np.pi * alpha * norm_t)
            y = center_y + current_radius * np.cos(2 * np.pi * norm_t) * np.sin(2 * np.pi * alpha * norm_t)
            z = np.full(seq_len, center_z)
            # Use gradient for velocity approximation (too complex to derive analytically)
            x_smooth = smooth_trajectory(x)
            y_smooth = smooth_trajectory(y)
            z_smooth = smooth_trajectory(z)
            dt = 1.0 / seq_len
            vx = np.gradient(x_smooth, dt)
            vy = np.gradient(y_smooth, dt)
            vz = np.gradient(z_smooth, dt)
            x, y, z = x_smooth, y_smooth, z_smooth
        
        else:
            # Simple straight line (fallback)
            x = center_x + norm_t * 10
            y = np.full(seq_len, center_y)
            z = np.full(seq_len, center_z)
            vx = np.full(seq_len, 10.0)
            vy = np.zeros(seq_len)
            vz = np.zeros(seq_len)

        # Smooth and re-calculate velocity if not done above
        if style not in ['clover']:
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
        
    return torch.tensor(trajectories, dtype=torch.float32)

def generate_random_obstacles(trajectory, num_obstacles_range=(1, 5), radius_range=(0.5, 2.0), check_collision=True, device='cpu'):
    """
    Generates a set of non-colliding spherical obstacles whose centers are placed near the trajectory.
    Ensures no two obstacles overlap by checking distance >= sum of radii during placement.
    Returns list of obstacle dictionaries with center and radius.
    """
    # Randomly determine the number of obstacles to generate
    num_obstacles = np.random.randint(num_obstacles_range[0], num_obstacles_range[1] + 1)
    obstacles = []
    
    # Extract trajectory positions (x, y, z) and move to CPU for numpy operations
    traj_pos = trajectory[:, 1:4].cpu().numpy()
    
    # Compute bounding box of the trajectory
    min_bounds = traj_pos.min(axis=0)
    max_bounds = traj_pos.max(axis=0)
    
    # Calculate the range of the bounds
    bounds_range = max_bounds - min_bounds
    
    # Expand the placement area by 50% of the trajectory range to allow space around it
    expanded_min = min_bounds - 0.5 * bounds_range
    expanded_max = max_bounds + 0.5 * bounds_range
    
    # print(f"Generating {num_obstacles} random non-colliding obstacles around trajectory")
    for i in range(num_obstacles):
        attempts = 0
        max_attempts = 100 # Prevent infinite loop in crowded spaces
        collision = True
        
        if check_collision:
            while collision and attempts < max_attempts:
                # Sample a random center within the expanded bounds
                center = np.random.uniform(expanded_min, expanded_max)
                # Sample a random radius within the given range
                radius = np.random.uniform(radius_range[0], radius_range[1])
                
                # Check for collisions with existing obstacles
                collision = False
                for prev_obstacle in obstacles:
                    prev_center = prev_obstacle['center'].cpu().numpy()
                    prev_radius = prev_obstacle['radius']
                    # Compute Euclidean distance between centers
                    dist = np.linalg.norm(center - prev_center)
                    # Check if distance < sum of radii (collision)
                    if dist < radius + prev_radius:
                        collision = True
                        break
                attempts += 1
            if collision:
                print(f"Warning: Could not place obstacle {i} without collision after {max_attempts} attempts. Skipping.")
                continue
        else:
            center = np.random.uniform(expanded_min, expanded_max)
            # Sample a random radius within the given range
            radius = np.random.uniform(radius_range[0], radius_range[1])
            
        # Create obstacle dictionary
        obstacle = {
            'center': torch.tensor(center, dtype=torch.float32, device=device),
            'radius': radius,
            'id': i
        }
        obstacles.append(obstacle)
        # print(f"Placed Obstacle {i}: center={center}, radius={radius:.3f}")

    return obstacles

# Plotting Utilities (for visual testing)

def plot_trajectories_demo(demo_trajectories, rows=3, cols=6):
    """Utility to plot a grid of 3D trajectories for visualization."""
    num_trajectories = demo_trajectories.shape[0]
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(f'Sample of {num_trajectories} Generated Aerobatic Trajectories (3D View)', fontsize=20, fontweight='bold')

    for i in range(min(rows * cols, num_trajectories)):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        
        # Trajectory is [speed, x, y, z, attitude(6)]. Position is [1:4]
        trajectory = demo_trajectories[i, :, 1:4].numpy()

        # Custom coloring based on index for variation
        if i % 3 == 0:
            color = 'blue'
            marker_color = 'red'
        elif i % 3 == 1:
            color = 'green'
            marker_color = 'orange'
        else:
            color = 'purple'
            marker_color = 'cyan'
            
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color, linewidth=2.5, alpha=0.8)
        # Mark every 10th point
        ax.scatter(trajectory[::10, 0], trajectory[::10, 1], trajectory[::10, 2], 
                color=marker_color, s=20, alpha=0.6, marker='o')
        
        ax.set_title(f'Trajectory {i+1}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('X', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Remove fill for better 3D visualization
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to accommodate title and text
    plt.show()

def plot_test_results(original, sampled_unguided_denorm, sampled_guided_denorm, history, target, obstacles=None, show_flag=True, step_idx=0):
    """
    Plot test results including original, reconstructed (unguided), and guided samples.
    Supports 3D, 2D projections, and time-series plots.
    """
    original_pos = original[0, :, 1:4].detach().cpu().numpy()
    reconstructed_pos = sampled_unguided_denorm[0, :, 1:4].detach().cpu().numpy()
    sampled_pos = sampled_guided_denorm[0, :, 1:4].detach().cpu().numpy()
    
    time_steps = np.arange(len(original_pos))

    # History positions (denormalized)
    history_pos = history[0, :, 1:4].detach().cpu().numpy() if history is not None else None
    
    # Target position (denormalized)
    target_pos = target[0, :].detach().cpu().numpy() # target is [x, y, z]

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f'AeroDM Trajectory Generation Results (Test Sample {step_idx})', fontsize=20, fontweight='bold')
    
    # Fixed bounds based on trajectory generation ranges (centers -20~20, radius~10, climb~40; safe cover -50 to 50)
    fixed_min = np.array([-20.0, -20.0, -20.0])
    fixed_max = np.array([20.0, 20.0, 20.0])

    # 1. 3D trajectory plot with history, target, and multiple obstacles
    ax1 = fig.add_subplot(331, projection='3d')
    # Plot history (if available)
    if history_pos is not None:
        ax1.plot(history_pos[:, 0], history_pos[:, 1], history_pos[:, 2], 'm-', label='History', linewidth=4, alpha=0.8, marker='o', markersize=4)
    # Plot main trajectories
    ax1.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], 'b-', label='Original Trajectory', linewidth=3, alpha=0.8)
    ax1.plot(reconstructed_pos[:, 0], reconstructed_pos[:, 1], reconstructed_pos[:, 2], 'r.-', label='Reconstructed Trajectory', linewidth=2, alpha=0.8)
    ax1.plot(sampled_pos[:, 0], sampled_pos[:, 1], sampled_pos[:, 2], 'g.-', label='Sampled Guided Trajectory', linewidth=2, alpha=0.8)
    
    # Plot target (if available)
    if target_pos is not None:
        ax1.scatter(target_pos[0], target_pos[1], target_pos[2], c='yellow', s=200, marker='*', label='Target Waypoint', edgecolors='black', linewidth=2)
    
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
            
            # Use 'gray' or a fixed color for all obstacles
            ax1.plot_surface(obs_x, obs_y, obs_z, alpha=0.3, color='gray')
            # Add a proxy artist for the legend
            if i == 0:
                from matplotlib.patches import Patch
                obstacle_proxies.append(Patch(color='gray', alpha=0.5, label='Obstacles'))

    # Combine all legends
    handles, labels = ax1.get_legend_handles_labels()
    handles.extend(obstacle_proxies)
    labels.extend([p.get_label() for p in obstacle_proxies])
    ax1.legend(handles, labels)
    
    ax1.set_xlim(fixed_min[0], fixed_max[0])
    ax1.set_ylim(fixed_min[1], fixed_max[1])
    ax1.set_zlim(fixed_min[2], fixed_max[2])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory')
    ax1.grid(True)
    
    # 2. X-Y projection with obstacles
    ax2 = fig.add_subplot(332)
    if history_pos is not None:
        ax2.plot(history_pos[:, 0], history_pos[:, 1], 'm-', label='History', linewidth=4, alpha=0.8, marker='o', markersize=4)
    ax2.plot(original_pos[:, 0], original_pos[:, 1], 'b-', label='Original', linewidth=3, alpha=0.8)
    ax2.plot(reconstructed_pos[:, 0], reconstructed_pos[:, 1], 'r.-', label='Reconstructed', linewidth=2, alpha=0.8)
    ax2.plot(sampled_pos[:, 0], sampled_pos[:, 1], 'g.-', label='Sampled Guided', linewidth=2, alpha=0.8)
    if target_pos is not None:
        ax2.scatter(target_pos[0], target_pos[1], c='yellow', s=200, marker='*', label='Target', edgecolors='black', linewidth=2)
    
    # Plot obstacles in 2D projection
    if obstacles is not None:
        colors = ['red', 'orange', 'purple', 'brown', 'pink']
        for i, obstacle in enumerate(obstacles):
            center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
            radius = obstacle['radius']
            color = colors[i % len(colors)]
            circle = plt.Circle((center[0], center[1]), radius, color=color, alpha=0.3, label=f'Obstacle {i+1}' if i < 3 else "")
            ax2.add_patch(circle)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.set_title('X-Y Projection with Obstacles')
    ax2.grid(True)
    ax2.axis('equal') # Keep aspect ratio for circular obstacles

    # 3. X-Z projection with obstacles
    ax3 = fig.add_subplot(333)
    if history_pos is not None:
        ax3.plot(history_pos[:, 0], history_pos[:, 2], 'm-', label='History', linewidth=4, alpha=0.8, marker='o', markersize=4)
    ax3.plot(original_pos[:, 0], original_pos[:, 2], 'b-', label='Original', linewidth=3, alpha=0.8)
    ax3.plot(reconstructed_pos[:, 0], reconstructed_pos[:, 2], 'r.-', label='Reconstructed', linewidth=2, alpha=0.8)
    ax3.plot(sampled_pos[:, 0], sampled_pos[:, 2], 'g.-', label='Sampled Guided', linewidth=2, alpha=0.8)
    if target_pos is not None:
        ax3.scatter(target_pos[0], target_pos[2], c='yellow', s=200, marker='*', label='Target', edgecolors='black', linewidth=2)
    
    # Plot obstacles in X-Z projection
    if obstacles is not None:
        colors = ['red', 'orange', 'purple', 'brown', 'pink']
        for i, obstacle in enumerate(obstacles):
            center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
            radius = obstacle['radius']
            color = colors[i % len(colors)]
            # Draw a circle on the X-Z plane
            rect = plt.Rectangle((center[0] - radius, center[2] - radius), 2*radius, 2*radius, 
                                angle=0.0, color=color, alpha=0.3)
            ax3.add_patch(rect)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.legend()
    ax3.set_title('X-Z Projection with Obstacles')
    ax3.grid(True)
    ax3.axis('equal') # Keep aspect ratio
    
    # 4. Y-Z projection with obstacles
    ax4 = fig.add_subplot(334)
    if history_pos is not None:
        ax4.plot(history_pos[:, 1], history_pos[:, 2], 'm-', label='History', linewidth=4, alpha=0.8, marker='o', markersize=4)
    ax4.plot(original_pos[:, 1], original_pos[:, 2], 'b-', label='Original', linewidth=3, alpha=0.8)
    ax4.plot(reconstructed_pos[:, 1], reconstructed_pos[:, 2], 'r.-', label='Reconstructed', linewidth=2, alpha=0.8)
    ax4.plot(sampled_pos[:, 1], sampled_pos[:, 2], 'g.-', label='Sampled Guided', linewidth=2, alpha=0.8)
    if target_pos is not None:
        ax4.scatter(target_pos[1], target_pos[2], c='yellow', s=200, marker='*', label='Target', edgecolors='black', linewidth=2)
        
    # Plot obstacles in Y-Z projection
    if obstacles is not None:
        colors = ['red', 'orange', 'purple', 'brown', 'pink']
        for i, obstacle in enumerate(obstacles):
            center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
            radius = obstacle['radius']
            color = colors[i % len(colors)]
            # Draw a circle on the Y-Z plane
            rect = plt.Rectangle((center[1] - radius, center[2] - radius), 2*radius, 2*radius, 
                                angle=0.0, color=color, alpha=0.3)
            ax4.add_patch(rect)

    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.legend()
    ax4.set_title('Y-Z Projection with Obstacles')
    ax4.grid(True)
    ax4.axis('equal') # Keep aspect ratio
    
    # 5. X Position over time
    ax5 = fig.add_subplot(335)
    ax5.plot(time_steps, original_pos[:, 0], 'b-', label='Original X', linewidth=2)
    ax5.plot(time_steps, reconstructed_pos[:, 0], 'r--', label='Reconstructed X', linewidth=2)
    ax5.plot(time_steps, sampled_pos[:, 0], 'g-.', label='Sampled Guided X', linewidth=2)
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('X Position')
    ax5.legend()
    ax5.set_title('X Position Over Time')
    ax5.grid(True)
    
    # 6. Y Position over time
    ax6 = fig.add_subplot(336)
    ax6.plot(time_steps, original_pos[:, 1], 'b-', label='Original Y', linewidth=2)
    ax6.plot(time_steps, reconstructed_pos[:, 1], 'r--', label='Reconstructed Y', linewidth=2)
    ax6.plot(time_steps, sampled_pos[:, 1], 'g-.', label='Sampled Guided Y', linewidth=2)
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Y Position')
    ax6.legend()
    ax6.set_title('Y Position Over Time')
    ax6.grid(True)

    # 7. Z Position over time
    ax7 = fig.add_subplot(337)
    ax7.plot(time_steps, original_pos[:, 2], 'b-', label='Original Z', linewidth=2)
    ax7.plot(time_steps, reconstructed_pos[:, 2], 'r--', label='Reconstructed Z', linewidth=2)
    ax7.plot(time_steps, sampled_pos[:, 2], 'g-.', label='Sampled Guided Z', linewidth=2)
    ax7.set_xlabel('Time Step')
    ax7.set_ylabel('Z Position')
    ax7.legend()
    ax7.set_title('Z Position Over Time')
    ax7.grid(True)
    
    # 8. Speed comparison
    original_speed = original[0, :, 0].detach().cpu().numpy()
    reconstructed_speed = sampled_unguided_denorm[0, :, 0].detach().cpu().numpy()
    sampled_speed = sampled_guided_denorm[0, :, 0].detach().cpu().numpy()
    ax8 = fig.add_subplot(338)
    ax8.plot(time_steps, original_speed, 'b-', label='Original Speed', linewidth=2)
    ax8.plot(time_steps, reconstructed_speed, 'r--', label='Reconstructed Speed', linewidth=2)
    ax8.plot(time_steps, sampled_speed, 'g-.', label='Sampled Guided Speed', linewidth=2)
    ax8.set_xlabel('Time Step')
    ax8.set_ylabel('Speed')
    ax8.legend()
    ax8.set_title('Speed Over Time')
    ax8.grid(True)
    
    # 9. Error analysis
    recon_error = np.linalg.norm(reconstructed_pos - original_pos, axis=1)
    sampled_error = np.linalg.norm(sampled_pos - original_pos, axis=1)
    ax9 = fig.add_subplot(339)
    ax9.plot(time_steps, recon_error, 'r--', label='Unguided Error', linewidth=2)
    ax9.plot(time_steps, sampled_error, 'g-.', label='Guided Error', linewidth=2)
    ax9.axhline(np.mean(recon_error), color='r', linestyle=':', label=f'Mean Unguided ({np.mean(recon_error):.2f})')
    ax9.axhline(np.mean(sampled_error), color='g', linestyle=':', label=f'Mean Guided ({np.mean(sampled_error):.2f})')
    ax9.set_xlabel('Time Step')
    ax9.set_ylabel('L2 Position Error')
    ax9.legend()
    ax9.set_title('L2 Position Error w.r.t Original')
    ax9.grid(True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if show_flag:
        plt.show()
    else:
        # Save as SVG with descriptive filename
        filename = f"Figs/test_sample_{step_idx:03d}_results.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight')
        plt.close() # Close the figure to free memory


def test_model_performance(model, trajectories_norm, mean, std, num_test_samples=3, show_flag=True):
    """testing with obstacle-aware transformer"""
    print("\nTesting obstacle-aware model performance...")
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
            obstacles = generate_random_obstacles(x_0_denorm[0], num_obstacles_range=(0, 3), radius_range=(0.8, 2.5), check_collision=True, device=device)
            
            # Set obstacles data for model input
            model.set_obstacles_data(obstacles)
            
            print(f"\n{'='*60}")
            print(f"TEST SAMPLE {i+1}")
            print(f"Generated {len(obstacles)} random obstacles")
            print(f"Obstacle information integrated into transformer via MLP encoder")
            print(f"{'='*60}")
            
            # 1. Sample Un-Guided Trajectory (to test base model/reconstruction)
            sampled_unguided_norm = model.sample(
                target_norm, action, history, batch_size=1, 
                enable_guidance=False, plot_all_steps=False
            )
            sampled_unguided_denorm = denormalize_trajectories(sampled_unguided_norm, mean, std)

            # 2. Sample CBF-Guided Trajectory
            # Use a smaller number of steps for guided sampling visualization
            config.set_show_flag(False) # Ensure plotting is off during sampling loop
            
            sampled_guided_norm = model.sample(
                target_norm, action, history, batch_size=1, 
                enable_guidance=True, guidance_gamma=config.guidance_gamma, 
                plot_all_steps=False # Plotting is too slow, rely on final plot
            )
            sampled_guided_denorm = denormalize_trajectories(sampled_guided_norm, mean, std)
            
            # 3. Final Plotting (denormalized)
            plot_test_results(
                x_0_denorm, 
                sampled_unguided_denorm, 
                sampled_guided_denorm, 
                denormalize_trajectories(history, mean, std) if history is not None else None,
                target_denorm,
                obstacles,
                show_flag,
                step_idx=i+1
            )


# Update the main execution
if __name__ == "__main__":
    print("Training Obstacle-Aware AeroDM with Transformer Integration and Obstacle-Aware Loss...")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    config = Config()
    
    # Loss function based on flag
    use_obstacle_loss = config.enable_obstacle_encoding
    criterion = AeroDMLoss(
        config, 
        enable_obstacle_term=use_obstacle_loss, 
        safe_extra_factor=0.2, # Safety buffer as fraction of radius (e.g., 20%)
        z_weight=1.5, # Extra weight for Z-axis (height) in aviation
        obstacle_weight=10.0, # Weight for obstacle term
        continuity_weight=5.0 # Weight for continuity term
    )
    print(f"Using AeroDMLoss (obstacle term: {use_obstacle_loss})")
    
    # Training parameters
    num_epochs = 50
    batch_size = 8
    num_trajectories = 10000
    
    print("Generating training data with obstacle-aware transformer...")
    trajectories = generate_aerobatic_trajectories(
        num_trajectories=num_trajectories, 
        seq_len=config.seq_len + config.history_len
    )
    
    # Split trajectories into train and test sets (80/20 split)
    torch.manual_seed(42) # For reproducibility
    indices = torch.randperm(num_trajectories)
    train_size = int(0.8 * num_trajectories)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_trajectories = trajectories[train_indices]
    test_trajectories = trajectories[test_indices]
    
    # Normalize on train set and apply to test
    train_norm, mean, std = normalize_trajectories(train_trajectories)
    test_norm = (test_trajectories - mean) / std
    
    train_norm = train_norm.to(device)
    test_norm = test_norm.to(device)
    # Move to device for consistency in testing
    mean = mean.to(device)
    std = std.to(device)
    
    # Model and Optimizer setup
    model = AeroDM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Use train_size for training loop
    train_size = train_trajectories.shape[0]
    losses = {'total': [], 'position': [], 'vel': [], 'obstacle': [], 'continuity': []}
    mode_str = "with obstacle-aware loss" if use_obstacle_loss else "with basic loss"
    print(f"Starting training {mode_str}...")

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0
        epoch_position_loss = 0
        epoch_vel_loss = 0
        epoch_obstacle_loss = 0 if use_obstacle_loss else None
        epoch_continuity_loss = 0
        num_batches = 0
        
        indices = torch.randperm(train_size)
        
        for i in range(0, train_size, batch_size):
            if i + batch_size > train_size:
                actual_batch_size = train_size - i
            else:
                actual_batch_size = batch_size
            
            batch_indices = indices[i:i+actual_batch_size]
            full_traj = train_norm[batch_indices] # (B, T_full, D)
            
            # Split into history and sequence-to-predict
            history = full_traj[:, :config.history_len, :]
            x_0 = full_traj[:, config.history_len:, :] # (B, T_pred, D)
            
            # Generate condition inputs
            target = generate_target_waypoints(x_0)
            action = generate_action_styles(actual_batch_size, config.action_dim, device=device)
            
            # Sample time step t
            t = torch.randint(0, config.diffusion_steps, (actual_batch_size,), device=device).long()
            
            # Forward diffusion (q_sample)
            noise = torch.randn_like(x_0)
            x_t, _ = model.diffusion_process.q_sample(x_0, t, noise)
            
            # Generate obstacles for this batch (only needed if obstacle loss is enabled)
            obstacles_for_batch = None
            if use_obstacle_loss:
                # Use a small subset of denormalized trajectories for obstacle placement
                # NOTE: This is slow. In a real system, obstacles would be loaded from a dataset.
                obstacles_for_batch = []
                for b in range(actual_batch_size):
                    traj_denorm = denormalize_trajectories(full_traj[b:b+1, config.history_len:, :], mean, std)
                    obstacles_for_batch.append(generate_random_obstacles(
                        traj_denorm[0], 
                        num_obstacles_range=(1, 3), 
                        radius_range=(0.8, 2.5), 
                        check_collision=True, 
                        device=device
                    ))

            # Model prediction (reverse process: predict x_0 from x_t)
            pred_x0 = model(x_t, t, target, action, history, obstacles_data=obstacles_for_batch)

            # Calculate loss based on flag
            total_loss, position_loss, vel_loss, obstacle_loss, continuity_loss = criterion(
                pred_x0, 
                x_0, 
                obstacles_for_batch if use_obstacle_loss else None, 
                mean, 
                std, 
                history
            )
            
            # Backward propagation
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            epoch_total_loss += total_loss.item()
            epoch_position_loss += position_loss.item()
            epoch_vel_loss += vel_loss.item()
            if use_obstacle_loss:
                epoch_obstacle_loss += obstacle_loss.item()
            epoch_continuity_loss += continuity_loss.item()
            num_batches += 1

        # Calculate average losses for the epoch
        if num_batches > 0:
            avg_total = epoch_total_loss / num_batches
            avg_position = epoch_position_loss / num_batches
            avg_vel = epoch_vel_loss / num_batches
            avg_continuity = epoch_continuity_loss / num_batches

            losses['total'].append(avg_total)
            losses['position'].append(avg_position)
            losses['vel'].append(avg_vel)
            losses['continuity'].append(avg_continuity)
            
            if use_obstacle_loss:
                avg_obstacle = epoch_obstacle_loss / num_batches
                losses['obstacle'].append(avg_obstacle)
        
        # Print progress with appropriate loss components
        if epoch % 5 == 0:
            if use_obstacle_loss:
                print(f"Epoch {epoch}: Total Loss: {avg_total:.4f}, "
                      f"Position: {avg_position:.4f}, Vel: {avg_vel:.4f}, Obstacle: {avg_obstacle:.4f}, Continuity: {avg_continuity:.4f}")
            else:
                print(f"Epoch {epoch}: Total Loss: {avg_total:.4f}, "
                      f"Position: {avg_position:.4f}, Vel: {avg_vel:.4f}, Continuity: {avg_continuity:.4f}")
                
    print(f"Training completed after {num_epochs} epochs.")
    
    # ------------------
    # Evaluation and Testing
    # ------------------
    # Optional: Display a few generated trajectories before normalization/splitting
    # plot_trajectories_demo(trajectories[:18], rows=3, cols=6)
    
    # Run the visualization test
    test_model_performance(model, test_norm, mean, std, num_test_samples=3, show_flag=config.show_flag)
