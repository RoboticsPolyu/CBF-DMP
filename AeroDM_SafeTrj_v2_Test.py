#AeroDM + Barrier Function Guidance + Obstacle Encoding

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


from Deformation import generate_aerobatic_trajectories
from Deformation import generate_aerobatic_trajectories_deformation
from circular_trajectories import generate_circular_end_trajectories
from distribute_trajectories import generate_distributed_trajectories


# Configuration parameters based on the paper
class Config:
    # Model dimensions
    latent_dim = 128
    obs_latent_dim = 128
    num_layers = 4
    num_heads = 4
    dropout = 0.1

    # Diffusion parameters
    diffusion_steps = 100
    beta_start = 0.0001
    beta_end = 0.02
    
    # Sequence parameters
    seq_len = 60  # N_a = 60 time steps; 6s-long future trajectory sequence
    state_dim = 7  # x_i ∈ R^7: s(1) + p(3) + r(3) + style(1)
    history_len = 20  # 20-frame historical observations

    # Condition dimensions
    target_dim = 3  # p_t ∈ R^3
    action_dim = 14   # 14 maneuver styles

    # Obstacle parameters
    max_obstacles = 10  # Maximum number of obstacles to process
    obstacle_feat_dim = 4  # [x, y, z, radius]
    enable_obstacle_encoding = False  # Toggle obstacle encoding in the model
    use_obstacle_loss = enable_obstacle_encoding  # Toggle obstacle loss term in training

    # CBF Guidance parameters (from CoDiG paper)
    # enable_cbf_guidance = True  # Disabled by default; toggle for inference
    guidance_gamma = 2000.0  # Base gamma for barrier guidance
    
    safe_extra_factor=0.2 # Safety buffer as fraction of radius (e.g., 20%)
    last_xyz_weight=5.0 # Extra weight for final timestep's position error
    xyz_weight=1.0 # Extra weight for Z-axis (height) in aviation
    diff_vel_weight=0.0 # Weight for velocity term
    other_weight=1.0 # Weight for other losses
    obstacle_weight=10.0 # Weight for obstacle term
    continuity_weight=5.0 # Weight for continuity term
    acc_weight=1.0 # Weight for acceleration term

    drop_style_prob = 0.1  # Probability of dropping style information during training for robustness
    guidance_scale = 2.0  # Classifier-free guidance scale for sampling (tune for best results)

    # Plotting control
    show_flag = True  # Set to False to save plots as SVG instead of displaying
    
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
            nn.Linear(config.obstacle_feat_dim, config.obs_latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.obs_latent_dim * 2, config.obs_latent_dim),
            nn.ReLU(),
            nn.Linear(config.obs_latent_dim, config.obs_latent_dim)
        )

    def forward(self, obstacles_data):
        """
        Process multiple obstacles and generate a global obstacle embedding.
        
        Args:
            obstacles_data: List of lists, where each inner list contains obstacle dicts for a batch sample
        
        Returns:
            global_features: Global obstacle embedding tensor of shape [batch_size, obs_latent_dim]
        """
        if obstacles_data is None or len(obstacles_data) == 0:
            # Return zero embedding if no obstacles
            batch_size = 1 if obstacles_data is None else len(obstacles_data)
            return torch.zeros(batch_size, self.config.obs_latent_dim, device=next(self.parameters()).device)
        
        device = next(self.parameters()).device
        # print(" -------- device:", device)
        batch_size = len(obstacles_data)
        
        # Preprocessing: Prepare a fixed number of obstacles for each sample
        batch_obstacle_tensors = []
        valid_counts = []  # Record the number of effective obstacles for each sample
        
        for sample_obstacles in obstacles_data:
            if not sample_obstacles:
                # Empty sample, create zero tensor
                obstacle_tensor = torch.zeros(self.config.max_obstacles, self.config.obstacle_feat_dim, device=device)
                valid_counts.append(0)
            else:
                # Extracting obstacle features
                obstacle_tensors = []
                for obstacle in sample_obstacles:
                    center = obstacle['center'].to(device)
                    radius = obstacle['radius']
                    obstacle_feat = torch.cat([
                        center,
                        torch.tensor([radius], device=device, dtype=center.dtype)
                    ])
                    obstacle_tensors.append(obstacle_feat)
                
               # Stack and handle quantity limits
                if obstacle_tensors:
                    obstacle_tensor = torch.stack(obstacle_tensors)
                    valid_count = len(obstacle_tensors)
                    
                    if valid_count < self.config.max_obstacles:
                        padding = torch.zeros(self.config.max_obstacles - valid_count, 
                                            self.config.obstacle_feat_dim, device=device)
                        obstacle_tensor = torch.cat([obstacle_tensor, padding], dim=0)
                    elif valid_count > self.config.max_obstacles:
                        obstacle_tensor = obstacle_tensor[:self.config.max_obstacles]
                        valid_count = self.config.max_obstacles
                    
                    valid_counts.append(valid_count)
                else:
                    obstacle_tensor = torch.zeros(self.config.max_obstacles, self.config.obstacle_feat_dim, device=device)
                    valid_counts.append(0)
            
            batch_obstacle_tensors.append(obstacle_tensor)
        
        # Batch process all samples
        batch_obstacle_tensor = torch.stack(batch_obstacle_tensors)  # [batch_size, max_obstacles, obstacle_feat_dim]
        
        # Reshaping for batch processing
        batch_size, max_obs, feat_dim = batch_obstacle_tensor.shape
        flattened_obstacles = batch_obstacle_tensor.view(-1, feat_dim)  # [batch_size * max_obstacles, feat_dim]
        
        # Batch encode all obstacles
        encoded_obstacles = self.obstacle_mlp(flattened_obstacles)  # [batch_size * max_obstacles, obs_latent_dim]
        
        # Restore to original structure
        encoded_obstacles = encoded_obstacles.view(batch_size, max_obs, -1)  # [batch_size, max_obstacles, obs_latent_dim]
        
        # Create an effective mask (to exclude filled obstacles)
        valid_mask = torch.zeros(batch_size, max_obs, device=device)
        for i, count in enumerate(valid_counts):
            if count > 0:
                valid_mask[i, :count] = 1.0
        
        # Perform average pooling on valid obstacles
        masked_embeddings = encoded_obstacles * valid_mask.unsqueeze(-1)  # Apply mask
        sum_embeddings = masked_embeddings.sum(dim=1)  # Sum [batch_size, obs_latent_dim]
        valid_counts_tensor = torch.tensor(valid_counts, device=device).float().clamp(min=1.0)  # Avoid division by zero
        
        # Calculate the average
        global_features = sum_embeddings / valid_counts_tensor.unsqueeze(-1)  # [batch_size, obs_latent_dim]
        
        return global_features

# Attention-based Obstacle Encoder for small obstacle sets
class AttentionObstacleEncoder(nn.Module):
    """
    Attention-based obstacle encoder for small obstacle sets (<10 obstacles).
    Uses self-attention to model interactions between obstacles and handles
    variable numbers of obstacles naturally without padding.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Project obstacle features to latent space
        # Input: [x, y, z, radius] -> Output: [obs_latent_dim]
        self.obstacle_proj = nn.Sequential(
            nn.Linear(config.obstacle_feat_dim, config.obs_latent_dim),
            nn.ReLU(),
            nn.Linear(config.obs_latent_dim, config.obs_latent_dim)
        )
        
        # Multi-head attention for obstacle-obstacle interactions
        # Allows obstacles to "see" each other and understand spatial relationships
        self.attention = nn.MultiheadAttention(
            config.obs_latent_dim,
            num_heads=4,
            batch_first=True,
            dropout=config.dropout
        )
        
        # Learnable positional encoding for obstacles (optional but helpful)
        # Since obstacles are unordered, this helps the model distinguish them
        max_obs = config.max_obstacles
        self.pos_encoding = nn.Parameter(torch.randn(1, max_obs, config.obs_latent_dim) * 0.1)
        
        # Global feature extraction after attention
        self.global_proj = nn.Sequential(
            nn.Linear(config.obs_latent_dim * 2, config.obs_latent_dim),  # *2 for concat of mean+max
            nn.ReLU(),
            nn.Linear(config.obs_latent_dim, config.obs_latent_dim),
            nn.LayerNorm(config.obs_latent_dim)  # Add layer norm for stability
        )
        
    def forward(self, obstacles_data):
        """
        Forward pass for obstacle encoding.
        
        Args:
            obstacles_data: List of lists, each inner list contains obstacle dicts
                           Each dict: {'center': tensor([x,y,z]), 'radius': float}
        
        Returns:
            global_features: Tensor of shape [batch_size, obs_latent_dim]
                            Global obstacle embedding for each sample
        """
        batch_size = len(obstacles_data)
        device = next(self.parameters()).device
        
        batch_embeddings = []
        
        for batch_idx, sample_obs in enumerate(obstacles_data):
            num_obstacles = len(sample_obs)
            
            # Case 1: No obstacles in this sample
            if num_obstacles == 0:
                batch_embeddings.append(
                    torch.zeros(self.config.obs_latent_dim, device=device)
                )
                continue
            
            # Extract obstacle features: [center_x, center_y, center_z, radius]
            obstacle_features = []
            for obstacle in sample_obs:
                center = obstacle['center'].to(device)
                radius = obstacle['radius']
                # Ensure radius is a tensor with correct dtype
                radius_tensor = torch.tensor([radius], device=device, dtype=center.dtype)
                feat = torch.cat([center, radius_tensor])
                obstacle_features.append(feat)
            
            # Stack obstacles for this sample: [num_obs, obstacle_feat_dim]
            obs_tensor = torch.stack(obstacle_features)
            
            # Project to latent space: [num_obs, obs_latent_dim]
            obs_emb = self.obstacle_proj(obs_tensor)
            
            # Add positional encoding (use first num_obs positions)
            # This helps the model maintain consistency despite unordered input
            pos_enc = self.pos_encoding[:, :num_obstacles, :].to(device)
            obs_emb = obs_emb.unsqueeze(0) + pos_enc  # [1, num_obs, obs_latent_dim]
            
            # Apply self-attention to model obstacle interactions
            # Each obstacle attends to all others to understand spatial relationships
            attn_output, attn_weights = self.attention(
                query=obs_emb,
                key=obs_emb,
                value=obs_emb
            )  # attn_output: [1, num_obs, obs_latent_dim]
            
            # Remove batch dimension
            obs_emb = attn_output.squeeze(0)  # [num_obs, obs_latent_dim]
            
            # Global pooling: combine mean and max pooling
            # Mean pooling captures the "average" obstacle context
            mean_pool = obs_emb.mean(dim=0)  # [obs_latent_dim]
            # Max pooling captures the "most critical" obstacle
            max_pool = obs_emb.max(dim=0)[0]  # [obs_latent_dim]
            
            # Concatenate both pooling results
            global_feat = torch.cat([mean_pool, max_pool])  # [obs_latent_dim * 2]
            
            # Final projection to get global obstacle embedding
            global_feat = self.global_proj(global_feat)  # [obs_latent_dim]
            
            batch_embeddings.append(global_feat)
        
        # Stack all batch embeddings: [batch_size, obs_latent_dim]
        return torch.stack(batch_embeddings)
    
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
        # self.obstacle_encoder = ObstacleEncoder(config)
        self.obstacle_encoder = AttentionObstacleEncoder(config)

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
            # print("obstacles are encoded.")
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
        
        # During training, randomly drop style information to improve robustness
        if self.training:
            drop_mask = torch.rand(x.size(0), device=x.device) < self.config.drop_style_prob
            action = action.clone()
            action[drop_mask] = 0  

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

# Barrier Function Computation for Multiple Obstacles
def compute_barrier_and_grad(x, config, mean, std, obstacles_data=None, safety_margin = 0.20):
    """
    Compute barrier V and its gradient ∇V for the trajectory x.
    Fixed version with proper gradient computation.
    """
    # Denormalize positions for barrier computation
    pos_denorm = x[:, :, 1:4] * std[0, 0, 1:4] + mean[0, 0, 1:4]
    batch_size, seq_len, _ = pos_denorm.shape
    
    # Initialize gradient tensor for denormalized positions
    grad_pos_denorm = torch.zeros_like(pos_denorm)
    
    pos_denorm = pos_denorm.clone().detach()
    
    # Initialize barrier value
    V_total = torch.zeros(batch_size, device=x.device)
    
    # Process each obstacle
    if obstacles_data is not None:
        for batch_idx in range(batch_size):
            # Get obstacles for this sample
            batch_obs = obstacles_data[batch_idx] if batch_idx < len(obstacles_data) else []
            
            for obstacle in batch_obs:
                center = obstacle['center'].to(x.device)
                radius = obstacle['radius'] + safety_margin  # Add safety margin
                
                # Euclidean distances
                distances = torch.norm(pos_denorm[batch_idx] - center.unsqueeze(0), dim=1)
                
                # Closeness function: r - distance
                h = radius * radius - distances * distances
                # h = (p-o)^T*(p-o) - r^2
                # P_free(x) = Φ(z) = Φ( h(x) / σ_h(x) )
                # log(P_free) = log(Φ(z)) ; ∇log(P_free) = ( φ(z) / Φ(z) ) * ∇h / σ_h   
                # φ(z) = exp(-0.5*z^2) / sqrt(2π)
                
                # Barrier violation term: max(0, h)^2
                violation = torch.clamp(h, min=0.0)
                V_obstacle = torch.sum(violation) # sum( min(0, r^2 - d^2  ) r^2 - d^2 = r^2 - ||p - c||^2
                V_total[batch_idx] += V_obstacle
                
                # Compute gradient for this obstacle
                violation_mask = (h > 0).float().unsqueeze(1)
                direction_vec = pos_denorm[batch_idx] - center.unsqueeze(0)
                
                epsilon = 1e-6  
                grad_dist = direction_vec / (distances.unsqueeze(1) + epsilon)
                grad_V_obs = -2 * violation.unsqueeze(1) * grad_dist

                grad_pos_denorm[batch_idx] = grad_pos_denorm[batch_idx] + grad_V_obs
    
    # Map gradient back to normalized space
    grad_x = torch.zeros_like(x, device=x.device)
    std_scaled = std[0, 0, 1:4].to(x.device)
    grad_x[:, :, 1:4] = grad_pos_denorm / std_scaled
    
    # V_total is sum of barrier violations
    V_avg = V_total.mean()
    
    return V_avg, grad_x

# Diffusion Process with Obstacle-Aware Sampling
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
    
    def p_sample(self, model, x_t, t, target, action, history=None, enable_guidance=True, guidance_gamma=None, 
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
            
            # print(f"Action: {action[0, :5].cpu().numpy()}...")  # Print first 5 elements

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
                V, grad_V = compute_barrier_and_grad(pred_x0, self.config, mean, std, obstacles_data)
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
                print("s_norm: ", s_norm, "grad_norm: ", grad_norm,  "gamma_t: ", gamma_t, "sigma_t: ", sigma_t)

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
    
    # def set_obstacles_data(self, obstacles_data):
    #     """Set obstacles data for CBF guidance and model input"""
    #     self.obstacles_data = obstacles_data
    #     print(f"Set {len(obstacles_data)} obstacles for model input and CBF guidance")
    
    def set_obstacles_data(self, obstacles_data):
        """Set obstacles data: must be list of list of dicts"""
        if obstacles_data is None:
            self.obstacles_data = None
        else:
            self.obstacles_data = obstacles_data
        print(f"Set obstacles_data: {len(self.obstacles_data) if self.obstacles_data else 0} batches")

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
            # plot_step = plot_all_steps or (t_step % max(1, self.config.diffusion_steps // 5) == 0) or t_step == 0
            
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

# Unified Loss Function for AeroDM with Obstacle Awareness
class AeroDMLoss(nn.Module):
    """
    Unified loss function for AeroDM training.
    Combines position, velocity, speed, attitude, and optional obstacle avoidance losses.
    Supports switching obstacle term via flag; always returns 4 values for consistency.
    Fixes: Proper safety margin for obstacles, Z-weighting, normalization by avg obstacles.
    """
    def __init__(self, config, enable_obstacle_term=False, safe_extra_factor=0.2, last_xyz_weight=1.5, xyz_weight=1.5, diff_vel_weight=1.0, other_weight=1.0, obstacle_weight=10.0, continuity_weight=15.0, acc_weight=1.0):
        super().__init__()
        self.config = config
        # Flag to enable/disable obstacle distance penalty in total loss
        self.enable_obstacle_term = enable_obstacle_term
        # Safety buffer beyond obstacle surface (as fraction of radius, e.g., 0.2 = 20%)
        self.safe_extra_factor = safe_extra_factor
        # Extra weight for the last point's position loss (critical for trajectory endpoint accuracy)
        self.last_xyz_weight = last_xyz_weight
        # Extra weight for Z-axis losses (height is critical in aviation trajectories)
        self.xyz_weight = xyz_weight
        # Weight for velocity loss
        self.diff_vel_weight = diff_vel_weight
        # Weight for other losses
        self.other_weight = other_weight
        # Scaling factor for the entire obstacle loss term
        self.obstacle_weight = obstacle_weight
        # Weight for continuity loss
        self.continuity_weight = continuity_weight
        # Weight for acceleration loss
        self.acc_weight = acc_weight
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
            obstacle_loss = obstacle_loss / (batch_size * avg_num_obs) # obstacle_loss = obstacle_loss / (batch_size * seq_len * avg_num_obs)
        
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
        pred_attitude = pred_trajectory[:, :, 4:10]  # (B, T, 6) - attitude (roll/pitch/yaw approx)
        gt_attitude = gt_trajectory[:, :, 4:10]
        
        # Position losses: Per-dimension MSE
        x_loss = self.xyz_weight * self.mse_loss(pred_pos[:, :, 0], gt_pos[:, :, 0])
        y_loss = self.xyz_weight * self.mse_loss(pred_pos[:, :, 1], gt_pos[:, :, 1])
        # Z loss with extra weight for height accuracy
        z_loss = self.xyz_weight * self.mse_loss(pred_pos[:, :, 2], gt_pos[:, :, 2])
        
        # Last time-step losses (higher weight for endpoint accuracy)
        last_x_loss = self.xyz_weight * self.mse_loss(pred_pos[:, -1, 0], gt_pos[:, -1, 0])
        last_y_loss = self.xyz_weight * self.mse_loss(pred_pos[:, -1, 1], gt_pos[:, -1, 1])
        last_z_loss = self.xyz_weight * self.mse_loss(pred_pos[:, -1, 2], gt_pos[:, -1, 2])
        
        last_xyz_loss = last_x_loss + last_y_loss + last_z_loss
        # Combine: Base + 10x last point
        position_loss = x_loss + y_loss + z_loss
        
        # Velocity loss: From position differences (assumes uniform dt=1)
        if seq_len > 1:
            # Compute deltas: (B, T-1, 3)
            pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
            gt_vel = gt_pos[:, 1:, :] - gt_pos[:, :-1, :]
            # Per-dimension MSE on velocities
            vel_x_loss = self.xyz_weight * self.mse_loss(pred_vel[:, :, 0], gt_vel[:, :, 0])
            vel_y_loss = self.xyz_weight * self.mse_loss(pred_vel[:, :, 1], gt_vel[:, :, 1])
            vel_z_loss = self.xyz_weight * self.mse_loss(pred_vel[:, :, 2], gt_vel[:, :, 2])
            vel_loss = vel_x_loss + vel_y_loss + vel_z_loss
        else:
            # No velocity if single timestep
            vel_loss = torch.tensor(0.0, device=device)
        
        if seq_len >= 3:
            # vel = (B, T-1, 3)
            vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
            # acc = (B, T-2, 3) - acceleration
            acc = vel[:, 1:, :] - vel[:, :-1, :]
            # Smoothness loss = mean squared acceleration
            acc_smoothness = acc.pow(2).mean()

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
            # if history.size(1) > 1:
            #     last_history_vel = history[:, -1, 1:4] - history[:, -2, 1:4]
            #     first_pred_vel = pred_trajectory[:, 0, 1:4] - last_history_pos # Approx
            #     continuity_loss += self.mse_loss(first_pred_vel, last_history_vel)
        
        # Total weighted loss
        total_loss = self.xyz_weight * last_xyz_loss + self.xyz_weight * position_loss + self.diff_vel_weight * vel_loss + self.other_weight * other_loss + self.obstacle_weight * obstacle_loss + self.continuity_weight * continuity_loss + self.acc_weight * acc_smoothness
        
        return total_loss, position_loss, vel_loss, obstacle_loss, continuity_loss

# def normalize_trajectories(trajectories):
#     """Normalize each dimension to zero mean and unit variance"""
#     mean = trajectories.mean(dim=(0, 1), keepdim=True)
#     std = trajectories.std(dim=(0, 1), keepdim=True)
    
#     std = torch.where(std < 1e-8, torch.ones_like(std), std) # avoid division by zero
#     return (trajectories - mean) / std, mean, std

def normalize_trajectories(trajectories, mean=None, std=None):
    """
    Normalize all dimensions to zero mean and unit variance, 
    but then restore the original style dimension values.
    
    Args:
        trajectories: Shape (batch, seq_len, state_dim) where state_dim includes style
    
    Returns:
        normalized_trajectories: State dimensions normalized, style restored to original
        mean: Mean for ALL dimensions (including style)
        std: Std for ALL dimensions (including style)
    """
    # Save original style values (last dimension)
    original_style = trajectories[:, :, -1:]  # Shape: (batch, seq_len, 1)
    
    if mean is None:
            # Normalize everything (including style)
            mean = trajectories.mean(dim=(0, 1), keepdim=True)
    if std is None:
            std = trajectories.std(dim=(0, 1), keepdim=True)
            std = torch.where(std < 1e-8, torch.ones_like(std), std)
    
    normalized = (trajectories - mean) / std
    
    # Restore original style values (overwrite the normalized style dimension)
    normalized[:, :, -1:] = original_style
    
    return normalized, mean, std

# Note: Denormalization should be applied only to the state variables (x,y,z) and not to the style dimension.
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

def generate_colliding_obstacles(trajectory, num_obstacles_range=(1, 5), radius_range=(0.5, 2.0), 
                               min_collision_points=1, max_attempts=100, device='cpu'):
    """
    Generates spherical obstacles that are guaranteed to collide with the trajectory.
    Ensures each obstacle intersects with at least min_collision_points trajectory points.
    Returns list of obstacle dictionaries with center and radius.
    """
    num_obstacles = np.random.randint(num_obstacles_range[0], num_obstacles_range[1] + 1)
    obstacles = []
    
    # Extract trajectory positions (x, y, z) and move to CPU for numpy operations
    traj_pos = trajectory[:, 1:4].cpu().numpy()
    
    # Compute bounding box of the trajectory
    min_bounds = traj_pos.min(axis=0)
    max_bounds = traj_pos.max(axis=0)
    bounds_range = max_bounds - min_bounds
    
    # Expand the placement area
    expanded_min = min_bounds - 0.5 * bounds_range
    expanded_max = max_bounds + 0.5 * bounds_range
    
    print(f"Generating {num_obstacles} obstacles that collide with trajectory")
    
    for i in range(num_obstacles):
        attempts = 0
        placed = False
        
        while not placed and attempts < max_attempts:
            # Sample a random radius
            radius = np.random.uniform(radius_range[0], radius_range[1])
            
            # Strategy 1: Place obstacle around a random trajectory point
            random_point_idx = np.random.randint(0, len(traj_pos))
            random_point = traj_pos[random_point_idx]
            
            # Add some random offset to the center (but ensure collision)
            max_offset = radius * 0.8  # Ensure the point stays inside the sphere
            offset = np.random.uniform(-max_offset, max_offset, 3)
            center = random_point + offset
            
            # Check if this obstacle collides with enough trajectory points
            distances = np.linalg.norm(traj_pos - center, axis=1)
            collision_points = np.sum(distances <= radius)
            
            if collision_points >= min_collision_points:
                # Check for collisions with existing obstacles
                collision_with_obstacles = False
                for prev_obstacle in obstacles:
                    prev_center = prev_obstacle['center'].cpu().numpy()
                    prev_radius = prev_obstacle['radius']
                    dist = np.linalg.norm(center - prev_center)
                    if dist < radius + prev_radius:
                        collision_with_obstacles = True
                        break
                
                if not collision_with_obstacles:
                    obstacle = {
                        'center': torch.tensor(center, dtype=torch.float32, device=device),
                        'radius': float(radius),
                        'id': i,
                        'collision_points': int(collision_points)
                    }
                    obstacles.append(obstacle)
                    print(f"Placed Obstacle {i}: center={center}, radius={radius:.3f}, collides with {collision_points} points")
                    placed = True
            
            attempts += 1
        
        if not placed:
            print(f"Warning: Could not place obstacle {i} with trajectory collision after {max_attempts} attempts.")
    
    return obstacles

# Alternative version that guarantees collisions but allows obstacle overlap
def generate_guaranteed_colliding_obstacles(trajectory, 
                                            num_obstacles_range=(1, 5), 
                                            radius_range=(0.5, 2.0), 
                                            allow_obstacle_overlap=True, 
                                            device='cpu'):
    """
    Generates obstacles guaranteed to collide with trajectory.
    If allow_obstacle_overlap is True, obstacles can overlap with each other.
    """
    num_obstacles = np.random.randint(num_obstacles_range[0], num_obstacles_range[1] + 1)
    obstacles = []
    
    traj_pos = trajectory[:, 1:4].cpu().numpy()
    
    print(f"Generating {num_obstacles} guaranteed colliding obstacles")
    
    for i in range(num_obstacles):
        # Sample radius
        radius = np.random.uniform(radius_range[0], radius_range[1])
        
        # Choose a random trajectory point to collide with
        collision_point_idx = np.random.randint(0, len(traj_pos))
        collision_point = traj_pos[collision_point_idx]
        
        # Generate center such that the collision point is inside the sphere
        # The center can be anywhere within 'radius' distance from the collision point
        direction = np.random.uniform(-1, 1, 3)
        direction = direction / (np.linalg.norm(direction) + 1e-8)  # Normalize
        distance_from_point = np.random.uniform(0, radius * 0.9)  # Keep point inside sphere
        center = collision_point + direction * distance_from_point
        
        if not allow_obstacle_overlap:
            # Check for collisions with existing obstacles
            collision_detected = False
            for prev_obstacle in obstacles:
                prev_center = prev_obstacle['center'].cpu().numpy()
                prev_radius = prev_obstacle['radius']
                dist = np.linalg.norm(center - prev_center)
                if dist < radius + prev_radius:
                    collision_detected = True
                    break
            
            if collision_detected:
                # Try a few more times with different centers
                for attempt in range(10):
                    direction = np.random.uniform(-1, 1, 3)
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                    distance_from_point = np.random.uniform(0, radius * 0.9)
                    center = collision_point + direction * distance_from_point
                    
                    collision_detected = False
                    for prev_obstacle in obstacles:
                        prev_center = prev_obstacle['center'].cpu().numpy()
                        prev_radius = prev_obstacle['radius']
                        dist = np.linalg.norm(center - prev_center)
                        if dist < radius + prev_radius:
                            collision_detected = True
                            break
                    
                    if not collision_detected:
                        break
        
        # Calculate actual collision points for reporting
        distances = np.linalg.norm(traj_pos - center, axis=1)
        collision_points = np.sum(distances <= radius)
        
        obstacle = {
            'center': torch.tensor(center, dtype=torch.float32, device=device),
            'radius': float(radius),
            'id': i,
            'collision_points': int(collision_points)
        }
        obstacles.append(obstacle)
        print(f"Placed Obstacle {i}: center={center}, radius={radius:.3f}, collides with {collision_points} points")
    
    return obstacles

# Utility function to verify collisions
def verify_obstacle_collisions(obstacles, trajectory):
    """Verify that all obstacles actually collide with the trajectory"""
    traj_pos = trajectory[:, 1:4].cpu().numpy()
    
    print("\n=== Collision Verification ===")
    all_collide = True
    for i, obstacle in enumerate(obstacles):
        center = obstacle['center'].cpu().numpy()
        radius = obstacle['radius']
        
        distances = np.linalg.norm(traj_pos - center, axis=1)
        collision_points = np.sum(distances <= radius)
        
        print(f"Obstacle {i}: {collision_points} collision points")
        if collision_points == 0:
            print(f"  WARNING: Obstacle {i} has NO collisions!")
            all_collide = False
    
    if all_collide:
        print("SUCCESS: All obstacles collide with trajectory")
    else:
        print("WARNING: Some obstacles do not collide with trajectory")
    
    return all_collide

def generate_random_obstacles(trajectory, num_obstacles_range=(1, 5), radius_range=(0.5, 2.0), check_collision=False, device='cpu'):
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
    expanded_min = min_bounds - 0.3 * bounds_range
    expanded_max = max_bounds + 0.3 * bounds_range
    
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
            # center = np.random.uniform(expanded_min, expanded_max)
            center = traj_pos[np.random.randint(10, len(traj_pos))] + [0.3, 0.3, 0.3]
            # Sample a random radius within the given range
            radius = np.random.uniform(radius_range[0], radius_range[1])

        # Create obstacle dictionary with proper tensor
        obstacle = {
            'center': torch.tensor(center, dtype=torch.float32, device=device),
            'radius': float(radius),  # Ensure radius is a float, not tensor
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
    # Precompute all data at once to avoid repeated operations
    original_pos = original[0, :, 1:4].detach().cpu().numpy()
    reconstructed_pos = sampled_unguided_denorm[0, :, 1:4].detach().cpu().numpy()
    sampled_pos = sampled_guided_denorm[0, :, 1:4].detach().cpu().numpy()
    
    # Extract speeds
    original_speed = original[0, :, 0].detach().cpu().numpy()
    reconstructed_speed = sampled_unguided_denorm[0, :, 0].detach().cpu().numpy()
    sampled_speed = sampled_guided_denorm[0, :, 0].detach().cpu().numpy()
    
    time_steps = np.arange(len(original_pos))
    history_pos = history[0, :, 1:4].detach().cpu().numpy() if history is not None else None
    target_pos = target[0, :].detach().cpu().numpy() if target is not None else None

    # Create figure with optimized layout
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(f'AeroDM Trajectory Generation Results (Test Sample {step_idx})', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Define consistent styling
    STYLES = {
        'history': {'color': 'magenta', 'linewidth': 4, 'alpha': 0.8, 'marker': 'o', 'markersize': 4},
        'original': {'color': 'blue', 'linewidth': 3, 'alpha': 0.9},
        'reconstructed': {'color': 'red', 'linewidth': 2, 'alpha': 0.8, 'linestyle': '--', 'marker': '.'},
        'sampled': {'color': 'green', 'linewidth': 2, 'alpha': 0.8, 'linestyle': '-.', 'marker': '.'},
        'target': {'color': 'yellow', 's': 200, 'marker': '*', 'edgecolors': 'black', 'linewidth': 2}
    }
    
    # Fixed bounds for consistent scaling
    fixed_bounds = {
        'xlim': (-20, 20),
        'ylim': (-20, 20), 
        'zlim': (-20, 20)
    }

    # 1. 3D trajectory plot
    ax1 = fig.add_subplot(331, projection='3d')
    plot_3d_trajectory(ax1, original_pos, reconstructed_pos, sampled_pos, history_pos, target_pos, obstacles, STYLES, fixed_bounds)
    
    # 2-4. 2D Projections
    projections = [
        (332, 'X-Y Projection', 0, 1, 'X', 'Y'),
        (333, 'X-Z Projection', 0, 2, 'X', 'Z'), 
        (334, 'Y-Z Projection', 1, 2, 'Y', 'Z')
    ]
    
    for subplot_idx, title, dim1, dim2, xlabel, ylabel in projections:
        ax = fig.add_subplot(subplot_idx)
        plot_2d_projection(ax, original_pos, reconstructed_pos, sampled_pos, history_pos, 
                          target_pos, obstacles, STYLES, dim1, dim2, title, xlabel, ylabel)

    # 5-7. Position over time
    positions = [
        (335, 'X Position Over Time', 0, 'X Position'),
        (336, 'Y Position Over Time', 1, 'Y Position'), 
        (337, 'Z Position Over Time', 2, 'Z Position')
    ]
    
    for subplot_idx, title, dim, ylabel in positions:
        ax = fig.add_subplot(subplot_idx)
        plot_position_time(ax, time_steps, original_pos, reconstructed_pos, sampled_pos, 
                          dim, title, ylabel, STYLES)

    # 8. Speed comparison
    ax8 = fig.add_subplot(338)
    plot_speed_comparison(ax8, time_steps, original_speed, reconstructed_speed, sampled_speed, STYLES)

    # 9. Error analysis
    ax9 = fig.add_subplot(339)
    plot_error_analysis(ax9, time_steps, original_pos, reconstructed_pos, sampled_pos)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    if show_flag:
        plt.show()
    else:
        filename = f"Figs/test_sample_{step_idx:03d}_results.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
        plt.close()

# Helper functions for modular plotting
def plot_3d_trajectory(ax, original_pos, reconstructed_pos, sampled_pos, history_pos, target_pos, obstacles, styles, bounds):
    """Plot 3D trajectory with obstacles"""
    # # Plot trajectories
    # if history_pos is not None:
    #     ax.plot(history_pos[:, 0], history_pos[:, 1], history_pos[:, 2], 
    #             label='History', **styles['history'])
    
    # ax.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], 
    #         label='Original Trajectory', **styles['original'])
    # ax.plot(reconstructed_pos[:, 0], reconstructed_pos[:, 1], reconstructed_pos[:, 2], 
    #         label='Reconstructed Trajectory', **styles['reconstructed'])
    # ax.plot(sampled_pos[:, 0], sampled_pos[:, 1], sampled_pos[:, 2], 
    #         label='Sampled Guided Trajectory', **styles['sampled'])
    
    # # Plot target
    # if target_pos is not None:
    #     ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
    #               label='Target Waypoint', **styles['target'])
    
        # Plot trajectories
    if history_pos is not None:
        ax.plot(history_pos[:, 0], history_pos[:, 1], history_pos[:, 2], 
                label='His', **styles['history'])
    ax.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], 
            label='Trj', **styles['original'])
    ax.plot(reconstructed_pos[:, 0], reconstructed_pos[:, 1], reconstructed_pos[:, 2], 
            label='Pred', **styles['reconstructed'])
    ax.plot(sampled_pos[:, 0], sampled_pos[:, 1], sampled_pos[:, 2], 
            label='CBF', **styles['sampled'])
    
    # Plot target
    if target_pos is not None:
        ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
                  label='Tar', **styles['target'])
        
    # Plot obstacles
    obstacle_proxies = plot_3d_obstacles(ax, obstacles)
    
    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    if obstacle_proxies:
        handles.extend(obstacle_proxies)
        labels.extend([p.get_label() for p in obstacle_proxies])
    ax.legend(handles, labels, loc='upper right', fontsize=8)
    
    ax.set_xlim(bounds['xlim'])
    ax.set_ylim(bounds['ylim'])
    ax.set_zlim(bounds['zlim'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory')
    ax.grid(True, alpha=0.3)

def plot_3d_obstacles(ax, obstacles):
    """Plot 3D obstacles and return legend proxies"""
    if not obstacles:
        return []
    
    obstacle_proxies = []
    colors = plt.cm.Set3(np.linspace(0, 1, len(obstacles)))
    
    for i, obstacle in enumerate(obstacles):
        center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
        radius = obstacle['radius']
        
        # Create sphere
        u = np.linspace(0, 2 * np.pi, 12)  # Reduced resolution for performance
        v = np.linspace(0, np.pi, 8)
        obs_x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        obs_y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        obs_z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(obs_x, obs_y, obs_z, alpha=0.3, color=colors[i])
        
        if i == 0:
            from matplotlib.patches import Patch
            obstacle_proxies.append(Patch(color=colors[i], alpha=0.5, label='Obstacles'))
    
    return obstacle_proxies

def plot_2d_projection(ax, original_pos, reconstructed_pos, sampled_pos, history_pos, 
                      target_pos, obstacles, styles, dim1, dim2, title, xlabel, ylabel):
    """Plot 2D projection with obstacles"""
    # Plot trajectories
    if history_pos is not None:
        ax.plot(history_pos[:, dim1], history_pos[:, dim2], 
                label='History', **styles['history'])
    
    ax.plot(original_pos[:, dim1], original_pos[:, dim2], 
            label='Original', **styles['original'])
    ax.plot(reconstructed_pos[:, dim1], reconstructed_pos[:, dim2], 
            label='Reconstructed', **styles['reconstructed'])
    ax.plot(sampled_pos[:, dim1], sampled_pos[:, dim2], 
            label='Sampled Guided', **styles['sampled'])
    
    # Plot target
    if target_pos is not None:
        ax.scatter(target_pos[dim1], target_pos[dim2], 
                  label='Target', **styles['target'])
    
    # Plot obstacles
    plot_2d_obstacles(ax, obstacles, dim1, dim2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

def plot_2d_obstacles(ax, obstacles, dim1, dim2):
    """Plot 2D obstacles"""
    if not obstacles:
        return
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(obstacles)))
    
    for i, obstacle in enumerate(obstacles):
        center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
        radius = obstacle['radius']
        
        circle = plt.Circle((center[dim1], center[dim2]), radius, 
                          color=colors[i], alpha=0.4, 
                          label=f'Obstacle {i+1}' if i < 3 else "")
        ax.add_patch(circle)

def plot_position_time(ax, time_steps, original_pos, reconstructed_pos, sampled_pos, 
                      dim, title, ylabel, styles):
    """Plot position over time for a specific dimension"""
    ax.plot(time_steps, original_pos[:, dim], 
            label=f'Original', **{k: v for k, v in styles['original'].items() if k != 'marker'})
    ax.plot(time_steps, reconstructed_pos[:, dim], 
            label=f'Reconstructed', **{k: v for k, v in styles['reconstructed'].items() if k != 'marker'})
    ax.plot(time_steps, sampled_pos[:, dim], 
            label=f'Sampled Guided', **{k: v for k, v in styles['sampled'].items() if k != 'marker'})
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

def plot_speed_comparison(ax, time_steps, original_speed, reconstructed_speed, sampled_speed, styles):
    """Plot speed comparison over time"""
    ax.plot(time_steps, original_speed, 
            label='Original Speed', **{k: v for k, v in styles['original'].items() if k != 'marker'})
    ax.plot(time_steps, reconstructed_speed, 
            label='Reconstructed Speed', **{k: v for k, v in styles['reconstructed'].items() if k != 'marker'})
    ax.plot(time_steps, sampled_speed, 
            label='Sampled Guided Speed', **{k: v for k, v in styles['sampled'].items() if k != 'marker'})
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Speed')
    ax.legend(fontsize=8)
    ax.set_title('Speed Over Time')
    ax.grid(True, alpha=0.3)

def plot_error_analysis(ax, time_steps, original_pos, reconstructed_pos, sampled_pos):
    """Plot error analysis"""
    recon_error = np.linalg.norm(reconstructed_pos - original_pos, axis=1)
    sampled_error = np.linalg.norm(sampled_pos - original_pos, axis=1)
    
    ax.plot(time_steps, recon_error, 'r--', label='Unguided Error', linewidth=2, alpha=0.8)
    ax.plot(time_steps, sampled_error, 'g-.', label='Guided Error', linewidth=2, alpha=0.8)
    
    # Add mean lines with annotations
    mean_recon = np.mean(recon_error)
    mean_sampled = np.mean(sampled_error)
    
    ax.axhline(mean_recon, color='r', linestyle=':', alpha=0.7, 
               label=f'Mean Unguided ({mean_recon:.2f})')
    ax.axhline(mean_sampled, color='g', linestyle=':', alpha=0.7,
               label=f'Mean Guided ({mean_sampled:.2f})')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('L2 Position Error')
    ax.legend(fontsize=8)
    ax.set_title('L2 Position Error w.r.t Original')
    ax.grid(True, alpha=0.3)

# Main Testing Function
def test_model_performance(model, trajectories_norm, mean, std, num_test_samples=3, show_flag=True):
    """testing with obstacle-aware transformer"""
    print("\nTesting obstacle-aware model performance...")
    config = model.config
    device = next(model.parameters()).device
    
    # Define style names mapping
    style_names = {
        0: 'power_loop',
        1: 'barrel_roll',
        2: 'split_s',
        3: 'immelmann',
        4: 'wall_ride',
        5: 'eight_figure',
        6: 'star',
        7: 'half_moon',
        8: 'sphinx',
        9: 'clover',
        10: 'spiral_inward',
        11: 'spiral_outward',
        12: 'spiral_vertical_up',
        13: 'spiral_vertical_down'
    }
    
    mean_state = mean[..., :-1]  # Shape: (1, 1, 10)
    std_state = std[..., :-1]    # Shape: (1, 1, 10)

    # Set normalization parameters
    model.set_normalization_params(mean, std)
    
    model.eval()
    with torch.no_grad():
        for i in range(min(num_test_samples, trajectories_norm.shape[0])):
            # Prepare test sample
            full_traj = trajectories_norm[i:i+1]
            
            style_info = full_traj[:, :, -1:]  # Shape: (B, T_full, 1)
            state_without_style = full_traj[:, :, :-1]  # Shape: (B, T_full, state_dim-1)

            # Split into history and sequence-to-predict
            history = state_without_style[:, :config.history_len, :]
            x_0 = state_without_style[:, config.history_len:config.history_len+config.seq_len, :]
            target_norm = generate_target_waypoints(x_0)

            # Denormalize for obstacle generation and plotting
            x_0_denorm = denormalize_trajectories(x_0, mean_state, std_state)
            target_denorm = target_norm * std_state[0, 0, 1:4] + mean_state[0, 0, 1:4]
            
            # Extract style information
            history_style = style_info[:, 0, 0]  # Style from history segment (first timestep)
            pred_style = style_info[:, -1, 0]  # Style from prediction segment
            
            style_indices = pred_style.long()
            action = F.one_hot(style_indices, num_classes=config.action_dim).float()

            # Generate random obstacles
            obstacles = generate_random_obstacles(x_0_denorm[0], 
                                                  num_obstacles_range=(3, 5), 
                                                  radius_range=(0.5, 1.0), 
                                                  check_collision=False, 
                                                  device=device)
            
            # Set obstacles data for model input
            model.set_obstacles_data([obstacles])
            
            # Get style names for display
            hist_idx = history_style.item()
            pred_idx = pred_style.item()
            
            print(f"\n{'='*60}")
            print(f"TEST SAMPLE {i+1}")
            print(f"History Style: {style_names.get(hist_idx, f'Style_{hist_idx}')} (idx={hist_idx})")
            print(f"Prediction Style: {style_names.get(pred_idx, f'Style_{pred_idx}')} (idx={pred_idx})")
            print(f"Generated {len(obstacles)} random obstacles")
            print(f"{'='*60}")

            # 1. Sample Un-Guided Trajectory
            sampled_unguided_norm = model.sample(
                target_norm, action, 
                history, 
                batch_size=1, 
                enable_guidance=False, 
                plot_all_steps=False
            )
            sampled_unguided_denorm = denormalize_trajectories(sampled_unguided_norm, mean_state, std_state)

            # 2. Sample CBF-Guided Trajectory
            config.set_show_flag(False)
            
            sampled_guided_norm = model.sample(
                target_norm, 
                action, 
                history, 
                batch_size=1, 
                enable_guidance=True, 
                guidance_gamma=config.guidance_gamma, 
                plot_all_steps=False
            )
            sampled_guided_denorm = denormalize_trajectories(sampled_guided_norm, mean_state, std_state)
            
            # 3. Final Plotting with style information
            plot_test_results(
                x_0_denorm, 
                sampled_unguided_denorm, 
                sampled_guided_denorm, 
                denormalize_trajectories(history, mean_state, std_state) if history is not None else None,
                target_denorm,
                obstacles,
                show_flag,
                step_idx=i+1,
                history_style=history_style,
                pred_style=pred_style,
                style_names=style_names
            )

# def test_model_performance(model, trajectories_norm, mean, std, num_test_samples=3, show_flag=True):
#     """testing with obstacle-aware transformer"""
#     print("\nTesting obstacle-aware model performance...")
#     config = model.config
#     device = next(model.parameters()).device
    
#     mean_state = mean[..., :-1]  # Shape: (1, 1, 10)
#     std_state = std[..., :-1]    # Shape: (1, 1, 10)

#     # Set normalization parameters
#     model.set_normalization_params(mean, std)
    
#     model.eval()
#     with torch.no_grad():
#         for i in range(min(num_test_samples, trajectories_norm.shape[0])):
#             # Prepare test sample
#             full_traj = trajectories_norm[i:i+1]
            

#             # history = generate_history_segments(full_traj, config.history_len, device=device)
#             # x_0 = full_traj[:, config.history_len:, :]
#             # style_indices = x_0[:, 0, -1].long()  # Use style index from first timestep
#             # action = F.one_hot(style_indices, num_classes=config.action_dim).float()

#             style_info = full_traj[:, :, -1:]  # Shape: (B, T_full, 1)
#             state_without_style = full_traj[:, :, :-1]  # Shape: (B, T_full, state_dim-1)

#             # Split into history and sequence-to-predict
#             history = state_without_style[:, :config.history_len, :]
#             x_0 = state_without_style[:, config.history_len:config.history_len+config.seq_len, :]
#             target_norm = generate_target_waypoints(x_0)

#             # Denormalize for obstacle generation and plotting
#             x_0_denorm = denormalize_trajectories(x_0, mean_state, std_state)
#             target_denorm = target_norm * std_state[0, 0, 1:4] + mean_state[0, 0, 1:4]
            
#             style_index = style_info[:, 0, 0]  # Shape: (B,) - take first timestep, first feature
#             style_indices = style_index.long()
#             action = F.one_hot(style_indices, num_classes=config.action_dim).float()  # Shape: (B, action_dim)

#             # Generate random obstacles
#             obstacles = generate_random_obstacles(x_0_denorm[0], 
#                                                   num_obstacles_range=(3, 5), 
#                                                   radius_range=(0.5, 1.0), 
#                                                   check_collision=False, 
#                                                   device=device)
            
#             # obstacles = generate_guaranteed_colliding_obstacles(x_0_denorm[0], num_obstacles_range=(0, 3), radius_range=(0.8, 2.5), allow_obstacle_overlap=True, device=device)

#             # Set obstacles data for model input
#             model.set_obstacles_data([obstacles])
            
#             print(f"\n{'='*60}")
#             print(f"TEST SAMPLE {i+1}")
#             print(f"Generated {len(obstacles)} random obstacles")
#             print(f"Obstacle information integrated into transformer via MLP encoder")
#             print(f"{'='*60}")
            
#             # 1. Sample Un-Guided Trajectory (to test base model/reconstruction)
#             sampled_unguided_norm = model.sample(
#                 target_norm, action, 
#                 history, 
#                 batch_size=1, 
#                 enable_guidance=False, 
#                 plot_all_steps=False
#             )
#             sampled_unguided_denorm = denormalize_trajectories(sampled_unguided_norm, mean_state, std_state)

#             # 2. Sample CBF-Guided Trajectory
#             # Use a smaller number of steps for guided sampling visualization
#             config.set_show_flag(False) # Ensure plotting is off during sampling loop
            
#             sampled_guided_norm = model.sample(
#                 target_norm, 
#                 action, 
#                 history, 
#                 batch_size=1, 
#                 enable_guidance=True, 
#                 guidance_gamma=config.guidance_gamma, 
#                 plot_all_steps=False # Plotting is too slow, rely on final plot
#             )
#             sampled_guided_denorm = denormalize_trajectories(sampled_guided_norm, mean_state, std_state)
            
#             # 3. Final Plotting (denormalized)
#             plot_test_results(
#                 x_0_denorm, 
#                 sampled_unguided_denorm, 
#                 sampled_guided_denorm, 
#                 denormalize_trajectories(history, mean_state, std_state) if history is not None else None,
#                 target_denorm,
#                 obstacles,
#                 show_flag,
#                 step_idx=i+1
#             )

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

def generate_trj_demos():
    # Generate example enhanced circular trajectories for demonstration
    print("Generating example enhanced circular trajectories...")
    demo_trajectories = generate_aerobatic_trajectories(num_trajectories=18, seq_len=60)
    
    # Extract style indices from the trajectories (last dimension)
    # Style index is stored as the last element in the state vector
    style_indices = demo_trajectories[:, 0, -1].long().numpy()
    
    # Define style names mapping (same as in generate_aerobatic_trajectories)
    style_names = {
        0: 'power_loop',
        1: 'barrel_roll',
        2: 'split_s',
        3: 'immelmann',
        4: 'wall_ride',
        5: 'eight_figure',
        6: 'star',
        7: 'half_moon',
        8: 'sphinx',
        9: 'clover',
        10: 'spiral_inward',
        11: 'spiral_outward',
        12: 'spiral_vertical_up',
        13: 'spiral_vertical_down'
    }
    
    # Get style names for each trajectory
    trajectory_styles = [style_names.get(idx, 'unknown') for idx in style_indices]
    
    # Visualize some training data with z-axis focus
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Enhanced Circular Trajectories with Style Information', fontsize=16, fontweight='bold')
    
    for i in range(6):
        # First row: Trajectories 1-6
        ax = fig.add_subplot(3, 6, i+1, projection='3d')
        trajectory = demo_trajectories[i, :, 1:4].numpy()
        style = trajectory_styles[i]
        
        # Color coding based on style
        if 'loop' in style:
            color = 'blue'
        elif 'roll' in style:
            color = 'red'
        elif 'spiral' in style:
            color = 'green'
        elif 'figure' in style:
            color = 'purple'
        else:
            color = 'orange'
        
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                color=color, linewidth=2, alpha=0.8)
        ax.set_title(f'Traj {i+1}: {style}', fontsize=9, pad=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True, alpha=0.3)
        
        # Second row: Trajectories 7-12
        ax = fig.add_subplot(3, 6, i+7, projection='3d')
        trajectory = demo_trajectories[i+6, :, 1:4].numpy()
        style = trajectory_styles[i+6]
        
        if 'loop' in style:
            color = 'blue'
        elif 'roll' in style:
            color = 'red'
        elif 'spiral' in style:
            color = 'green'
        elif 'figure' in style:
            color = 'purple'
        else:
            color = 'orange'
        
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                color=color, linewidth=2, alpha=0.8)
        ax.set_title(f'Traj {i+7}: {style}', fontsize=9, pad=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True, alpha=0.3)
        
        # Third row: Trajectories 13-18
        ax = fig.add_subplot(3, 6, i+13, projection='3d')
        trajectory = demo_trajectories[i+12, :, 1:4].numpy()
        style = trajectory_styles[i+12]
        
        if 'loop' in style:
            color = 'blue'
        elif 'roll' in style:
            color = 'red'
        elif 'spiral' in style:
            color = 'green'
        elif 'figure' in style:
            color = 'purple'
        else:
            color = 'orange'
        
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                color=color, linewidth=2, alpha=0.8)
        ax.set_title(f'Traj {i+13}: {style}', fontsize=9, pad=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True, alpha=0.3)
    
    # Add legend for style-color mapping
    plt.figtext(0.5, 0.01, 
                'Color Legend: Blue=Loops, Red=Rolls, Green=Spirals, Purple=Figures, Orange=Others',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to accommodate title and legend
    plt.show()
    
    # Print style distribution
    print("\n=== Style Distribution in Generated Trajectories ===")
    style_counts = {}
    for style in trajectory_styles:
        style_counts[style] = style_counts.get(style, 0) + 1
    
    for style, count in style_counts.items():
        print(f"{style}: {count} trajectories")
    
    print(f"Total: {len(trajectory_styles)} trajectories")

# Test the effect of different actions (maneuver styles) on trajectory generation
def test_action_effect(model, trajectories_norm, mean, std, num_test_samples=5, show_flag=True):
    """Test the effect of different actions (maneuver styles) on trajectory generation"""
    print("\nTesting action (maneuver style) effects on trajectory generation...")
    config = model.config
    device = next(model.parameters()).device
    
    mean_state = mean[..., :-1]
    std_state = std[..., :-1]

    # Set normalization parameters
    model.set_normalization_params(mean, std)
    
    # Define style names for display
    style_names = {
        0: 'power_loop',
        1: 'barrel_roll',
        2: 'split_s',
        3: 'immelmann',
        4: 'wall_ride',
        5: 'eight_figure',
        6: 'star',
        7: 'half_moon',
        8: 'sphinx',
        9: 'clover',
        10: 'spiral_inward',
        11: 'spiral_outward',
        12: 'spiral_vertical_up',
        13: 'spiral_vertical_down'
    }
    
    # Select a subset of styles to test
    test_styles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # All styles
    
    model.eval()
    with torch.no_grad():
        # Use a fixed test sample from the dataset
        for i in range(min(num_test_samples, trajectories_norm.shape[0])):
            full_traj = trajectories_norm[i:i+1]
            
            # Extract history from the test sample
            state_without_style = full_traj[:, :, :-1]
            history = state_without_style[:, :config.history_len, :]
            
            # Get the target from the ground truth trajectory
            x_0 = state_without_style[:, config.history_len:config.history_len+config.seq_len, :]
            target_norm = generate_target_waypoints(x_0)
            target_denorm = target_norm * std_state[0, 0, 1:4] + mean_state[0, 0, 1:4]
            
            # Generate fixed obstacles for consistent comparison
            x_0_denorm = denormalize_trajectories(x_0, mean_state, std_state)
            obstacles = generate_random_obstacles(
                x_0_denorm[0], 
                num_obstacles_range=(3, 5), 
                radius_range=(0.5, 1.0), 
                check_collision=False, 
                device=device
            )
            model.set_obstacles_data([obstacles])
            
            print(f"\n{'='*80}")
            print(f"TEST SAMPLE {i+1} - Comparing All Maneuver Styles")
            print(f"Target Waypoint: {target_denorm[0].cpu().numpy()}")
            print(f"Number of obstacles: {len(obstacles)}")
            print(f"{'='*80}")
            
            # Store results for all styles
            all_trajectories = {}
            all_trajectories_norm = {}
            
            # Generate trajectories for each test style
            for style_idx in test_styles:
                # Create one-hot action for this style
                action = F.one_hot(torch.tensor([style_idx]), num_classes=config.action_dim).float().to(device)
                style_name = style_names.get(style_idx, f'style_{style_idx}')
                
                # Sample with CBF guidance
                sampled_norm = model.sample(
                    target_norm, 
                    action, 
                    history, 
                    batch_size=1, 
                    enable_guidance=True, 
                    guidance_gamma=config.guidance_gamma, 
                    plot_all_steps=False
                )
                
                sampled_denorm = denormalize_trajectories(sampled_norm, mean_state, std_state)
                all_trajectories[style_name] = sampled_denorm[0].cpu().numpy()
                all_trajectories_norm[style_name] = sampled_norm[0].cpu().numpy()
            
            # Plot comparison of all styles
            plot_action_comparison(
                all_trajectories, 
                target_denorm[0].cpu().numpy(),
                obstacles,
                history[0].cpu().numpy() if history is not None else None,
                mean_state, std_state,
                show_flag,
                sample_idx=i+1
            )
            
            # Plot style statistics
            plot_style_statistics(all_trajectories, show_flag, sample_idx=i+1)

def plot_action_comparison(all_trajectories, target, obstacles, history, mean_state, std_state, show_flag, sample_idx=1):
    """Plot comparison of trajectories generated with different actions/styles"""
    
    num_styles = len(all_trajectories)
    cols = min(4, num_styles)
    rows = (num_styles + cols - 1) // cols
    
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    fig.suptitle(f'Effect of Different Maneuver Styles on Trajectory Generation (Sample {sample_idx})', 
                 fontsize=16, fontweight='bold')
    
    # Color map for different styles
    colors = plt.cm.tab20(np.linspace(0, 1, num_styles))
    
    # Fixed bounds for consistent scaling
    bounds = {
        'xlim': (-20, 20),
        'ylim': (-20, 20),
        'zlim': (-10, 30)
    }
    
    for idx, (style_name, trajectory) in enumerate(all_trajectories.items()):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        
        # Plot trajectory
        ax.plot(trajectory[:, 1], trajectory[:, 2], trajectory[:, 3], 
                color=colors[idx], linewidth=2, alpha=0.8, label=style_name)
        
        # Mark start and end points
        ax.scatter(trajectory[0, 1], trajectory[0, 2], trajectory[0, 3], 
                  color=colors[idx], s=50, marker='o', edgecolors='black', label='Start')
        ax.scatter(trajectory[-1, 1], trajectory[-1, 2], trajectory[-1, 3], 
                  color=colors[idx], s=50, marker='s', edgecolors='black', label='End')
        
        # Plot obstacles
        if obstacles:
            for obstacle in obstacles:
                center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
                radius = obstacle['radius']
                
                # Create sphere
                u = np.linspace(0, 2 * np.pi, 10)
                v = np.linspace(0, np.pi, 10)
                x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='red')
        
        # Plot target
        ax.scatter(target[0], target[1], target[2], color='gold', s=100, 
                  marker='*', edgecolors='black', linewidth=2, label='Target')
        
        ax.set_xlim(bounds['xlim'])
        ax.set_ylim(bounds['ylim'])
        ax.set_zlim(bounds['zlim'])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{style_name}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Only show legend for first subplot to avoid clutter
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if show_flag:
        plt.show()
    else:
        filename = f"Figs/action_comparison_sample_{sample_idx:03d}.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight')
        plt.close()

def plot_style_statistics(all_trajectories, show_flag, sample_idx=1):
    """Plot statistical comparison of different styles"""
    
    num_styles = len(all_trajectories)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Style Statistics Comparison (Sample {sample_idx})', fontsize=14, fontweight='bold')
    
    style_names = list(all_trajectories.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, num_styles))
    
    # 1. Trajectory length (total distance traveled)
    ax1 = axes[0, 0]
    lengths = []
    for style_name, traj in all_trajectories.items():
        positions = traj[:, 1:4]  # x, y, z
        diffs = np.diff(positions, axis=0)
        total_length = np.sum(np.linalg.norm(diffs, axis=1))
        lengths.append(total_length)
    
    bars = ax1.bar(style_names, lengths, color=colors)
    ax1.set_ylabel('Total Path Length')
    ax1.set_title('Trajectory Length by Style')
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    for bar, length in zip(bars, lengths):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{length:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    # 2. Maximum altitude (Z)
    ax2 = axes[0, 1]
    max_altitudes = []
    for style_name, traj in all_trajectories.items():
        max_z = np.max(traj[:, 3])  # Z is index 3 (after speed at index 0, x,y,z at 1,2,3)
        max_altitudes.append(max_z)
    
    bars = ax2.bar(style_names, max_altitudes, color=colors)
    ax2.set_ylabel('Maximum Altitude (Z)')
    ax2.set_title('Maximum Altitude by Style')
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    for bar, alt in zip(bars, max_altitudes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{alt:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    # 3. Speed profile (mean speed)
    ax3 = axes[1, 0]
    mean_speeds = []
    for style_name, traj in all_trajectories.items():
        speeds = traj[:, 0]  # Speed is at index 0
        mean_speed = np.mean(speeds)
        mean_speeds.append(mean_speed)
    
    bars = ax3.bar(style_names, mean_speeds, color=colors)
    ax3.set_ylabel('Mean Speed')
    ax3.set_title('Mean Speed by Style')
    ax3.tick_params(axis='x', rotation=45, labelsize=8)
    for bar, speed in zip(bars, mean_speeds):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{speed:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    # 4. Curvature (how winding the path is)
    ax4 = axes[1, 1]
    curvatures = []
    for style_name, traj in all_trajectories.items():
        positions = traj[:, 1:4]
        if len(positions) >= 3:
            # Approximate curvature using three points
            curv = []
            for i in range(1, len(positions) - 1):
                v1 = positions[i] - positions[i-1]
                v2 = positions[i+1] - positions[i]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                    curv.append(angle)
            if curv:
                curvatures.append(np.mean(curv))
            else:
                curvatures.append(0)
        else:
            curvatures.append(0)
    
    bars = ax4.bar(style_names, curvatures, color=colors)
    ax4.set_ylabel('Mean Turning Angle (rad)')
    ax4.set_title('Path Curvature by Style')
    ax4.tick_params(axis='x', rotation=45, labelsize=8)
    
    plt.tight_layout()
    
    if show_flag:
        plt.show()
    else:
        filename = f"Figs/style_statistics_sample_{sample_idx:03d}.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight')
        plt.close()

def test_single_action_effect(model, trajectories_norm, mean, std, style_idx=0, num_test_samples=3, show_flag=True):
    """Test and visualize a single action style with detailed plots"""
    print(f"\nTesting detailed effect of style {style_idx}...")
    config = model.config
    device = next(model.parameters()).device
    
    mean_state = mean[..., :-1]
    std_state = std[..., :-1]
    
    style_names = {
        0: 'power_loop', 1: 'barrel_roll', 2: 'split_s', 3: 'immelmann',
        4: 'wall_ride', 5: 'eight_figure', 6: 'star', 7: 'half_moon',
        8: 'sphinx', 9: 'clover', 10: 'spiral_inward', 11: 'spiral_outward',
        12: 'spiral_vertical_up', 13: 'spiral_vertical_down'
    }
    style_name = style_names.get(style_idx, f'style_{style_idx}')
    
    model.set_normalization_params(mean, std)
    model.eval()
    
    with torch.no_grad():
        for i in range(min(num_test_samples, trajectories_norm.shape[0])):
            full_traj = trajectories_norm[i:i+1]
            
            state_without_style = full_traj[:, :, :-1]
            history = state_without_style[:, :config.history_len, :]
            
            x_0 = state_without_style[:, config.history_len:config.history_len+config.seq_len, :]
            target_norm = generate_target_waypoints(x_0)
            target_denorm = target_norm * std_state[0, 0, 1:4] + mean_state[0, 0, 1:4]
            
            # Create action for the specified style
            action = F.one_hot(torch.tensor([style_idx]), num_classes=config.action_dim).float().to(device)
            
            # Generate obstacles
            x_0_denorm = denormalize_trajectories(x_0, mean_state, std_state)
            obstacles = generate_random_obstacles(
                x_0_denorm[0], 
                num_obstacles_range=(3, 5), 
                radius_range=(0.5, 1.0), 
                check_collision=False, 
                device=device
            )
            model.set_obstacles_data([obstacles])
            
            print(f"\n{'='*60}")
            print(f"Testing Style: {style_name} (idx={style_idx}) - Sample {i+1}")
            print(f"Target: {target_denorm[0].cpu().numpy()}")
            print(f"{'='*60}")
            
            # Generate both guided and unguided trajectories
            sampled_guided_norm = model.sample(
                target_norm, action, history, batch_size=1,
                enable_guidance=True, guidance_gamma=config.guidance_gamma, plot_all_steps=False
            )
            sampled_unguided_norm = model.sample(
                target_norm, action, history, batch_size=1,
                enable_guidance=False, plot_all_steps=False
            )
            
            sampled_guided_denorm = denormalize_trajectories(sampled_guided_norm, mean_state, std_state)
            sampled_unguided_denorm = denormalize_trajectories(sampled_unguided_norm, mean_state, std_state)
            
            # Plot detailed comparison
            plot_detailed_style_comparison(
                sampled_guided_denorm[0].cpu().numpy(),
                sampled_unguided_denorm[0].cpu().numpy(),
                target_denorm[0].cpu().numpy(),
                history[0].cpu().numpy() if history is not None else None,
                obstacles,
                style_name,
                show_flag,
                sample_idx=i+1
            )

def plot_detailed_style_comparison(guided_traj, unguided_traj, target, history, obstacles, style_name, show_flag, sample_idx=1):
    """Plot detailed comparison between guided and unguided trajectories for a specific style"""
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Style: {style_name} - Guided vs Unguided Trajectory Comparison (Sample {sample_idx})', 
                 fontsize=16, fontweight='bold')
    
    # Extract positions (x,y,z from indices 1,2,3)
    guided_pos = guided_traj[:, 1:4]
    unguided_pos = unguided_traj[:, 1:4]
    
    # Extract speeds (index 0)
    guided_speed = guided_traj[:, 0]
    unguided_speed = unguided_traj[:, 0]
    
    time_steps = np.arange(len(guided_pos))
    
    # 1. 3D trajectory comparison
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.plot(guided_pos[:, 0], guided_pos[:, 1], guided_pos[:, 2], 
             'g-', linewidth=2, label='Guided (CBF)', alpha=0.8)
    ax1.plot(unguided_pos[:, 0], unguided_pos[:, 1], unguided_pos[:, 2], 
             'r--', linewidth=2, label='Unguided', alpha=0.8)
    
    if history is not None:
        history_pos = history[:, 1:4]
        ax1.plot(history_pos[:, 0], history_pos[:, 1], history_pos[:, 2], 
                 'm-', linewidth=3, label='History', alpha=0.7)
    
    if obstacles:
        for obstacle in obstacles:
            center = obstacle['center'].cpu().numpy() if hasattr(obstacle['center'], 'cpu') else obstacle['center']
            radius = obstacle['radius']
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='red')
    
    ax1.scatter(target[0], target[1], target[2], color='gold', s=100, marker='*', label='Target')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('3D Trajectory Comparison')
    
    # 2-4. Position components over time
    components = [
        (232, 'X Position', 0, 'X'),
        (233, 'Y Position', 1, 'Y'),
        (234, 'Z Position', 2, 'Z')
    ]
    
    for idx, (subplot, title, dim, label) in enumerate(components):
        ax = fig.add_subplot(subplot)
        ax.plot(time_steps, guided_pos[:, dim], 'g-', linewidth=2, label='Guided')
        ax.plot(time_steps, unguided_pos[:, dim], 'r--', linewidth=2, label='Unguided')
        if history is not None:
            history_time = np.arange(-len(history), 0)
            ax.plot(history_time, history[:, dim+1], 'm-', linewidth=2, label='History', alpha=0.7)
        ax.axhline(y=target[dim], color='gold', linestyle=':', linewidth=2, label='Target')
        ax.set_xlabel('Time Step')
        ax.set_ylabel(label)
        ax.legend()
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    # 5. Speed comparison
    ax5 = fig.add_subplot(235)
    ax5.plot(time_steps, guided_speed, 'g-', linewidth=2, label='Guided Speed')
    ax5.plot(time_steps, unguided_speed, 'r--', linewidth=2, label='Unguided Speed')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Speed')
    ax5.legend()
    ax5.set_title('Speed Profile')
    ax5.grid(True, alpha=0.3)
    
    # 6. Position error
    ax6 = fig.add_subplot(236)
    error = np.linalg.norm(guided_pos - unguided_pos, axis=1)
    ax6.plot(time_steps, error, 'b-', linewidth=2)
    ax6.fill_between(time_steps, 0, error, alpha=0.3)
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('L2 Distance')
    ax6.set_title('Position Difference (Guided vs Unguided)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show_flag:
        plt.show()
    else:
        filename = f"Figs/style_{style_name}_sample_{sample_idx:03d}.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight')
        plt.close()

def random_concatenate_trajectories(trajectories, history_len, seq_len, num_concatenated=None):
    """
    Randomly concatenate different trajectories to generate new augmented trajectories.
    
    Method: Select first [0:history_len] from trajectory 1 + [history_len:history_len+seq_len] from trajectory 2.
    The style (last dimension) is inherited from trajectory 2 since it determines the maneuver type.
    
    Args:
        trajectories (torch.Tensor): Original trajectories tensor of shape (num_trajectories, total_len, state_dim)
        history_len (int): Length of history segment to take from first trajectory
        seq_len (int): Length of future segment to take from second trajectory
        num_concatenated (int, optional): Number of concatenated trajectories to generate. 
                                          If None, defaults to half of original trajectories.
    
    Returns:
        torch.Tensor: Concatenated trajectories with same shape as input
    """
    num_original = trajectories.shape[0]
    total_len = history_len + seq_len
    
    if num_concatenated is None:
        num_concatenated = num_original // 2
    
    device = trajectories.device
    
    concatenated_trajectories = []
    
    for i in range(num_concatenated):
        # Randomly select two different trajectories
        idx1 = torch.randint(0, num_original, (1,)).item()
        idx2 = torch.randint(0, num_original, (1,)).item()
        
        # Ensure they are different trajectories
        while idx1 == idx2 and num_original > 1:
            idx2 = torch.randint(0, num_original, (1,)).item()
        
        # Extract segments
        # Trajectory 1: first history_len steps
        history_segment = trajectories[idx1, :history_len, :]  # (history_len, state_dim)
        
        # Trajectory 2: from history_len to history_len+seq_len steps
        future_segment = trajectories[idx2, history_len:history_len+seq_len, :]  # (seq_len, state_dim)
        
        # Concatenate along time dimension
        new_trajectory = torch.cat([history_segment, future_segment], dim=0)  # (total_len, state_dim)
        
        concatenated_trajectories.append(new_trajectory)
    
    # Stack all concatenated trajectories
    concatenated_trajectories = torch.stack(concatenated_trajectories, dim=0)
    
    return concatenated_trajectories


def augment_trajectories_with_concatenation(trajectories, history_len, seq_len, 
                                            concat_ratio=0.5, smooth_transition=True):
    """
    Augment trajectory dataset by randomly concatenating different trajectories.
    
    This creates new trajectories by combining history from one trajectory with 
    future from another, which helps the model learn to handle diverse history-future
    combinations and improves generalization.
    
    Args:
        trajectories (torch.Tensor): Original trajectories of shape (N, total_len, state_dim)
        history_len (int): Length of history segment
        seq_len (int): Length of future prediction segment  
        concat_ratio (float): Ratio of concatenated trajectories to original (default: 0.5)
        smooth_transition (bool): If True, apply smoothing at the connection point to avoid 
                                  abrupt changes (default: True)
    
    Returns:
        torch.Tensor: Augmented trajectories (original + concatenated)
    """
    num_original = trajectories.shape[0]
    num_concatenated = int(num_original * concat_ratio)
    
    print(f"Augmenting trajectories: {num_original} original + {num_concatenated} concatenated")
    
    # Generate concatenated trajectories
    concatenated = random_concatenate_trajectories(
        trajectories, history_len, seq_len, num_concatenated
    )
    
    if smooth_transition and history_len > 0 and seq_len > 0:
        # Apply smoothing at the connection boundary to avoid abrupt velocity changes
        # Smooth over a small window around the connection point
        smooth_window = min(3, history_len, seq_len)
        
        for i in range(num_concatenated):
            # Get positions at boundary (indices 1:4 are x,y,z)
            pos_before = concatenated[i, history_len-1, 1:4].clone()
            pos_after = concatenated[i, history_len, 1:4].clone()
            
            # Linear interpolation for smooth transition
            for j in range(1, smooth_window + 1):
                alpha = j / (smooth_window + 1)
                # Blend positions near boundary
                if history_len - j >= 0:
                    original_pos = concatenated[i, history_len - j, 1:4].clone()
                    target_pos = pos_before * (1 - alpha * 0.3) + pos_after * (alpha * 0.3)
                    concatenated[i, history_len - j, 1:4] = original_pos * 0.7 + target_pos * 0.3
                
                if history_len + j - 1 < concatenated.shape[1]:
                    original_pos = concatenated[i, history_len + j - 1, 1:4].clone()
                    target_pos = pos_before * (alpha * 0.3) + pos_after * (1 - alpha * 0.3)
                    concatenated[i, history_len + j - 1, 1:4] = original_pos * 0.7 + target_pos * 0.3
    
    # Combine original and concatenated trajectories
    augmented_trajectories = torch.cat([trajectories, concatenated], dim=0)
    
    print(f"Total trajectories after augmentation: {augmented_trajectories.shape[0]}")
    
    return augmented_trajectories


def compute_trajectory_connection_transform(history_end_pos, history_end_vel, future_start_pos):
    """
    Compute translation and rotation to smoothly connect two trajectory segments.
    
    Args:
        history_end_pos (torch.Tensor): Final position of history segment (3,)
        history_end_vel (torch.Tensor): Final velocity of history segment (3,)
        future_start_pos (torch.Tensor): First position of future segment (3,)
    
    Returns:
        tuple: (translation vector, rotation matrix) to apply to future segment
    """
    # Translation: align future start to history end
    translation = history_end_pos - future_start_pos
    
    # For simplicity, we use identity rotation (no rotation)
    # Full rotation alignment would require more complex computation
    rotation = torch.eye(3, device=history_end_pos.device)
    
    return translation, rotation


def apply_smooth_connection(history_segment, future_segment, smooth_window=5):
    """
    Apply smooth blending at the connection boundary between history and future segments.
    
    Uses cosine-weighted blending to ensure C1 continuity (position and velocity continuous).
    
    Args:
        history_segment (torch.Tensor): History segment of shape (history_len, state_dim)
        future_segment (torch.Tensor): Future segment of shape (seq_len, state_dim)
        smooth_window (int): Number of frames to blend on each side of the connection
    
    Returns:
        tuple: (adjusted_history_segment, adjusted_future_segment)
    """
    history_len = history_segment.shape[0]
    seq_len = future_segment.shape[0]
    
    # Ensure window doesn't exceed available frames
    window = min(smooth_window, history_len, seq_len)
    
    if window < 2:
        return history_segment, future_segment
    
    # Get positions at the boundary
    history_pos = history_segment[:, 1:4]  # (history_len, 3)
    future_pos = future_segment[:, 1:4]    # (seq_len, 3)
    
    # Compute velocities at boundary (using finite differences)
    if history_len >= 2:
        history_vel = history_pos[-1] - history_pos[-2]
    else:
        history_vel = torch.zeros(3, device=history_pos.device)
    
    if seq_len >= 2:
        future_vel = future_pos[1] - future_pos[0]
    else:
        future_vel = torch.zeros(3, device=future_pos.device)
    
    # Compute desired connection velocity (average of both sides for smoothness)
    connection_vel = (history_vel + future_vel) / 2.0
    
    # Adjust history segment near the end
    adjusted_history = history_segment.clone()
    adjusted_future = future_segment.clone()
    
    for i in range(1, window + 1):
        # Cosine blending weight: smooth transition from 0 to 1
        alpha = 0.5 * (1 - torch.cos(torch.tensor(i / (window + 1) * 3.14159, device=history_pos.device)))
        
        # Adjust history positions near the end
        hist_idx = history_len - window + i - 1
        if hist_idx >= 0 and hist_idx < history_len:
            # Target position based on smooth velocity integration
            target_pos = history_pos[-1] - connection_vel * (window - i + 1)
            # Blend between original and target
            adjusted_history[hist_idx, 1:4] = (1 - alpha) * history_pos[hist_idx] + alpha * target_pos
        
        # Adjust future positions near the start
        fut_idx = i - 1
        if fut_idx < seq_len:
            # Target position based on smooth velocity integration
            target_pos = history_pos[-1] + connection_vel * i
            # Blend between original and target
            adjusted_future[fut_idx, 1:4] = (1 - alpha) * future_pos[fut_idx] + alpha * target_pos
    
    # Ensure exact continuity at the boundary
    adjusted_future[0, 1:4] = history_pos[-1]
    
    # Update velocities (speed component at index 0)
    adjusted_history = update_speed_from_positions(adjusted_history)
    adjusted_future = update_speed_from_positions(adjusted_future)
    
    return adjusted_history, adjusted_future


def update_speed_from_positions(trajectory):
    """
    Update speed component (index 0) based on position differences.
    
    Args:
        trajectory (torch.Tensor): Trajectory of shape (seq_len, state_dim)
    
    Returns:
        torch.Tensor: Trajectory with updated speed values
    """
    traj = trajectory.clone()
    positions = traj[:, 1:4]
    
    if positions.shape[0] >= 2:
        # Compute speeds as magnitude of velocity
        velocities = positions[1:] - positions[:-1]
        speeds = torch.norm(velocities, dim=1)
        
        # Assign speeds (first frame uses second frame's speed)
        traj[0, 0] = speeds[0]
        traj[1:, 0] = speeds
    
    return traj


def random_concatenate_trajectories_smooth(trajectories, history_len, seq_len, 
                                           num_concatenated=None, smooth_window=5):
    """
    Randomly concatenate different trajectories with smooth connection.
    
    Method: 
    1. Select history segment from trajectory 1: [0:history_len]
    2. Select future segment from trajectory 2: [history_len:history_len+seq_len]
    3. Translate future segment to align with history end point
    4. Apply smooth blending at the connection boundary
    
    Args:
        trajectories (torch.Tensor): Original trajectories of shape (N, total_len, state_dim)
        history_len (int): Length of history segment
        seq_len (int): Length of future prediction segment
        num_concatenated (int, optional): Number of concatenated trajectories to generate
        smooth_window (int): Number of frames to blend at the connection
    
    Returns:
        torch.Tensor: Smoothly concatenated trajectories
    """
    num_original = trajectories.shape[0]
    
    if num_concatenated is None:
        num_concatenated = num_original // 2
    
    device = trajectories.device
    concatenated_trajectories = []
    
    for i in range(num_concatenated):
        # Randomly select two different trajectories
        idx1 = torch.randint(0, num_original, (1,)).item()
        idx2 = torch.randint(0, num_original, (1,)).item()
        
        # Ensure they are different trajectories
        while idx1 == idx2 and num_original > 1:
            idx2 = torch.randint(0, num_original, (1,)).item()
        
        style1 = trajectories[idx1, 0, -1].item()
        style2 = trajectories[idx2, 0, -1].item()
        
        if style1 != style2:
            # different_style_count += 1
            print(f"Concatenating different styles: {style1} -> {style2}")
    
        # print(f"Total concatenations with different styles: {different_style_count}/{num_concatenated}")

        # Extract segments
        history_segment = trajectories[idx1, :history_len, :].clone()  # (history_len, state_dim)
        future_segment = trajectories[idx2, history_len:history_len+seq_len, :].clone()  # (seq_len, state_dim)
        
        # Get boundary positions
        history_end_pos = history_segment[-1, 1:4]  # (3,)
        future_start_pos = future_segment[0, 1:4]   # (3,)
        
        # Compute translation to align future start with history end
        translation = history_end_pos - future_start_pos
        
        # Apply translation to all positions in future segment
        future_segment[:, 1:4] = future_segment[:, 1:4] + translation.unsqueeze(0)
        
        # Apply smooth blending at the connection
        history_segment, future_segment = apply_smooth_connection(
            history_segment, future_segment, smooth_window
        )
        
        # Concatenate along time dimension
        new_trajectory = torch.cat([history_segment, future_segment], dim=0)  # (total_len, state_dim)
        
        concatenated_trajectories.append(new_trajectory)
    
    # Stack all concatenated trajectories
    concatenated_trajectories = torch.stack(concatenated_trajectories, dim=0)
    
    return concatenated_trajectories


def augment_trajectories_with_smooth_concatenation(trajectories, history_len, seq_len,
                                                   concat_ratio=0.5, smooth_window=5,
                                                   validate_continuity=True):
    """
    Augment trajectory dataset by smoothly concatenating different trajectories.
    
    This creates new trajectories by combining history from one trajectory with 
    future from another, ensuring smooth position and velocity continuity at the boundary.
    
    Args:
        trajectories (torch.Tensor): Original trajectories of shape (N, total_len, state_dim)
        history_len (int): Length of history segment
        seq_len (int): Length of future prediction segment  
        concat_ratio (float): Ratio of concatenated trajectories to original (default: 0.5)
        smooth_window (int): Number of frames for smooth blending at connection
        validate_continuity (bool): If True, print continuity validation statistics
    
    Returns:
        torch.Tensor: Augmented trajectories (original + concatenated)
    """
    num_original = trajectories.shape[0]
    num_concatenated = int(num_original * concat_ratio)
    
    print(f"Augmenting trajectories with smooth concatenation:")
    print(f"  - Original trajectories: {num_original}")
    print(f"  - Concatenated trajectories: {num_concatenated}")
    print(f"  - Smooth window: {smooth_window}")
    
    # Generate smoothly concatenated trajectories
    concatenated = random_concatenate_trajectories_smooth(
        trajectories, history_len, seq_len, num_concatenated, smooth_window
    )
    
    if validate_continuity:
        validate_connection_continuity(concatenated, history_len)
    
    # Combine original and concatenated trajectories
    augmented_trajectories = torch.cat([trajectories, concatenated], dim=0)
    
    print(f"  - Total trajectories after augmentation: {augmented_trajectories.shape[0]}")
    
    return concatenated


def validate_connection_continuity(concatenated_trajectories, history_len):
    """
    Validate the continuity at the connection point of concatenated trajectories.
    
    Computes position jump and velocity discontinuity at the boundary.
    
    Args:
        concatened_trajectories (torch.Tensor): Concatenated trajectories
        history_len (int): Length of history segment (boundary index)
    """
    num_traj = concatenated_trajectories.shape[0]
    device = concatenated_trajectories.device
    
    position_jumps = []
    velocity_jumps = []
    
    with torch.no_grad():
        for i in range(num_traj):
            # Get positions around boundary
            pos_before = concatenated_trajectories[i, history_len-1, 1:4]
            pos_at = concatenated_trajectories[i, history_len, 1:4]
            pos_after = concatenated_trajectories[i, history_len+1, 1:4] if history_len+1 < concatenated_trajectories.shape[1] else pos_at
            
            # Position jump at boundary (should be near zero)
            pos_jump = torch.norm(pos_at - pos_before)
            position_jumps.append(pos_jump.item())
            
            # Velocity continuity
            vel_before = pos_at - pos_before
            vel_after = pos_after - pos_at
            vel_jump = torch.norm(vel_after - vel_before)
            velocity_jumps.append(vel_jump.item())
    
    # Compute statistics
    pos_jumps = torch.tensor(position_jumps)
    vel_jumps = torch.tensor(velocity_jumps)
    
    print(f"\nConnection Continuity Validation:")
    print(f"  - Position jump at boundary:")
    print(f"      Mean: {pos_jumps.mean():.6f}, Max: {pos_jumps.max():.6f}, Std: {pos_jumps.std():.6f}")
    print(f"  - Velocity discontinuity at boundary:")
    print(f"      Mean: {vel_jumps.mean():.6f}, Max: {vel_jumps.max():.6f}, Std: {vel_jumps.std():.6f}")
    
    # Check if continuity is good (position jump < 1e-4)
    if pos_jumps.mean() < 1e-4:
        print(f"  ✓ Position continuity is excellent")
    elif pos_jumps.mean() < 1e-2:
        print(f"  ✓ Position continuity is good")
    else:
        print(f"  ⚠ Position continuity may need improvement")


def plot_test_results(original, sampled_unguided_denorm, sampled_guided_denorm, history, target, 
                      obstacles=None, show_flag=True, step_idx=0, 
                      history_style=None, pred_style=None, style_names=None):
    """
    Plot test results including original, reconstructed (unguided), and guided samples.
    Supports 3D, 2D projections, time-series plots, and style information display.
    
    Args:
        original: Original trajectory (1, seq_len, state_dim)
        sampled_unguided_denorm: Unguided sampled trajectory (1, seq_len, state_dim)
        sampled_guided_denorm: Guided sampled trajectory (1, seq_len, state_dim)
        history: History segment (1, history_len, state_dim)
        target: Target waypoint (1, 3)
        obstacles: List of obstacle dictionaries
        show_flag: Whether to display plot or save to file
        step_idx: Sample index for filename
        history_style: Style index of history segment (int or tensor)
        pred_style: Style index of prediction segment (int or tensor)
        style_names: Dictionary mapping style indices to names
    """
    # Precompute all data at once to avoid repeated operations
    original_pos = original[0, :, 1:4].detach().cpu().numpy()
    reconstructed_pos = sampled_unguided_denorm[0, :, 1:4].detach().cpu().numpy()
    sampled_pos = sampled_guided_denorm[0, :, 1:4].detach().cpu().numpy()
    
    # Extract speeds
    original_speed = original[0, :, 0].detach().cpu().numpy()
    reconstructed_speed = sampled_unguided_denorm[0, :, 0].detach().cpu().numpy()
    sampled_speed = sampled_guided_denorm[0, :, 0].detach().cpu().numpy()
    
    time_steps = np.arange(len(original_pos))
    history_pos = history[0, :, 1:4].detach().cpu().numpy() if history is not None else None
    target_pos = target[0, :].detach().cpu().numpy() if target is not None else None
    
    # Process style information
    history_style_name = "Unknown"
    pred_style_name = "Unknown"
    if style_names is not None:
        if history_style is not None:
            hist_idx = history_style.item() if hasattr(history_style, 'item') else history_style
            history_style_name = style_names.get(hist_idx, f"Style_{hist_idx}")
        if pred_style is not None:
            pred_idx = pred_style.item() if hasattr(pred_style, 'item') else pred_style
            pred_style_name = style_names.get(pred_idx, f"Style_{pred_idx}")

    # Create figure with optimized layout (added one more subplot for style info)
    fig = plt.figure(figsize=(24, 18))
    
    # Main title with style information
    title_text = f'AeroDM Trajectory Generation Results (Test Sample {step_idx})'
    if history_style is not None or pred_style is not None:
        title_text += f'\nHistory Style: {history_style_name} | Prediction Style: {pred_style_name}'
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
    
    # Define consistent styling
    STYLES = {
        'history': {'color': 'magenta', 'linewidth': 4, 'alpha': 0.8, 'marker': 'o', 'markersize': 4},
        'original': {'color': 'blue', 'linewidth': 3, 'alpha': 0.9},
        'reconstructed': {'color': 'red', 'linewidth': 2, 'alpha': 0.8, 'linestyle': '--', 'marker': '.'},
        'sampled': {'color': 'green', 'linewidth': 2, 'alpha': 0.8, 'linestyle': '-.', 'marker': '.'},
        'target': {'color': 'yellow', 's': 200, 'marker': '*', 'edgecolors': 'black', 'linewidth': 2}
    }
    
    # Fixed bounds for consistent scaling
    fixed_bounds = {
        'xlim': (-20, 20),
        'ylim': (-20, 20), 
        'zlim': (-20, 20)
    }

    # 1. 3D trajectory plot
    ax1 = fig.add_subplot(341, projection='3d')
    plot_3d_trajectory(ax1, original_pos, reconstructed_pos, sampled_pos, history_pos, target_pos, obstacles, STYLES, fixed_bounds)
    
    # 2-4. 2D Projections
    projections = [
        (342, 'X-Y Projection', 0, 1, 'X', 'Y'),
        (343, 'X-Z Projection', 0, 2, 'X', 'Z'), 
        (344, 'Y-Z Projection', 1, 2, 'Y', 'Z')
    ]
    
    for subplot_idx, title, dim1, dim2, xlabel, ylabel in projections:
        ax = fig.add_subplot(subplot_idx)
        plot_2d_projection(ax, original_pos, reconstructed_pos, sampled_pos, history_pos, 
                          target_pos, obstacles, STYLES, dim1, dim2, title, xlabel, ylabel)

    # 5-7. Position over time
    positions = [
        (345, 'X Position Over Time', 0, 'X Position'),
        (346, 'Y Position Over Time', 1, 'Y Position'), 
        (347, 'Z Position Over Time', 2, 'Z Position')
    ]
    
    for subplot_idx, title, dim, ylabel in positions:
        ax = fig.add_subplot(subplot_idx)
        plot_position_time(ax, time_steps, original_pos, reconstructed_pos, sampled_pos, 
                          history_pos, dim, title, ylabel, STYLES)

    # 8. Speed comparison
    ax8 = fig.add_subplot(348)
    plot_speed_comparison(ax8, time_steps, original_speed, reconstructed_speed, sampled_speed, STYLES)

    # 9. Error analysis
    ax9 = fig.add_subplot(349)
    plot_error_analysis(ax9, time_steps, original_pos, reconstructed_pos, sampled_pos)
    
    # 10. Style information display
    ax10 = fig.add_subplot(3, 4, 10)
    plot_style_information(ax10, history_style_name, pred_style_name, style_names, 
                           history_pos, original_pos, history_style, pred_style)
    
    # 11. Trajectory statistics
    ax11 = fig.add_subplot(3, 4, 11)
    plot_trajectory_statistics(ax11, original_pos, reconstructed_pos, sampled_pos)
    
    # 12. Style distribution or additional info
    ax12 = fig.add_subplot(3, 4, 12)
    plot_connection_info(ax12, history_pos, original_pos, history_len=history_pos.shape[0] if history_pos is not None else 0)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    if show_flag:
        plt.show()
    else:
        filename = f"Figs/test_sample_{step_idx:03d}_results.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
        plt.close()


def plot_style_information(ax, history_style_name, pred_style_name, style_names, 
                           history_pos, original_pos, history_style, pred_style):
    """
    Plot style information for history and prediction segments.
    
    Args:
        ax: Matplotlib axis
        history_style_name: Name of history style
        pred_style_name: Name of prediction style
        style_names: Dictionary of all style names
        history_pos: History positions array
        original_pos: Original prediction positions array
        history_style: History style index
        pred_style: Prediction style index
    """
    ax.axis('off')
    
    # Create style information text
    info_lines = []
    info_lines.append("=" * 40)
    info_lines.append("STYLE INFORMATION")
    info_lines.append("=" * 40)
    info_lines.append("")
    
    # History segment style
    if history_style is not None:
        hist_idx = history_style.item() if hasattr(history_style, 'item') else history_style
        info_lines.append(f"📜 HISTORY SEGMENT STYLE:")
        info_lines.append(f"   Index: {hist_idx}")
        info_lines.append(f"   Name:  {history_style_name}")
        if history_pos is not None:
            info_lines.append(f"   Length: {len(history_pos)} frames")
    else:
        info_lines.append(f"📜 HISTORY SEGMENT: None")
    
    info_lines.append("")
    info_lines.append("-" * 40)
    info_lines.append("")
    
    # Prediction segment style
    if pred_style is not None:
        pred_idx = pred_style.item() if hasattr(pred_style, 'item') else pred_style
        info_lines.append(f"🎯 PREDICTION SEGMENT STYLE:")
        info_lines.append(f"   Index: {pred_idx}")
        info_lines.append(f"   Name:  {pred_style_name}")
        info_lines.append(f"   Length: {len(original_pos)} frames")
    else:
        info_lines.append(f"🎯 PREDICTION SEGMENT: Unknown")
    
    info_lines.append("")
    info_lines.append("=" * 40)
    
    # Check if styles match
    if history_style is not None and pred_style is not None:
        hist_idx = history_style.item() if hasattr(history_style, 'item') else history_style
        pred_idx = pred_style.item() if hasattr(pred_style, 'item') else pred_style
        if hist_idx == pred_idx:
            info_lines.append("✓ Styles MATCH")
        else:
            info_lines.append("⚠ Styles DIFFERENT")
    
    # Color code based on style match
    if history_style is not None and pred_style is not None:
        hist_idx = history_style.item() if hasattr(history_style, 'item') else history_style
        pred_idx = pred_style.item() if hasattr(pred_style, 'item') else pred_style
        if hist_idx == pred_idx:
            box_color = 'lightgreen'
        else:
            box_color = 'lightyellow'
    else:
        box_color = 'lightgray'
    
    # Display text in a box
    info_text = '\n'.join(info_lines)
    ax.text(0.5, 0.5, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))


def plot_trajectory_statistics(ax, original_pos, reconstructed_pos, sampled_pos):
    """
    Plot trajectory statistics comparison.
    
    Args:
        ax: Matplotlib axis
        original_pos: Original trajectory positions
        reconstructed_pos: Reconstructed (unguided) positions
        sampled_pos: Sampled guided positions
    """
    # Compute statistics
    def compute_stats(pos):
        total_length = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
        max_height = np.max(pos[:, 2])
        mean_speed = np.mean(np.linalg.norm(np.diff(pos, axis=0), axis=1))
        return total_length, max_height, mean_speed
    
    orig_len, orig_height, orig_speed = compute_stats(original_pos)
    recon_len, recon_height, recon_speed = compute_stats(reconstructed_pos)
    sampled_len, sampled_height, sampled_speed = compute_stats(sampled_pos)
    
    # Create bar chart
    categories = ['Path Length', 'Max Height', 'Mean Speed']
    orig_values = [orig_len, orig_height, orig_speed]
    recon_values = [recon_len, recon_height, recon_speed]
    sampled_values = [sampled_len, sampled_height, sampled_speed]
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax.bar(x - width, orig_values, width, label='Original', color='blue', alpha=0.7)
    bars2 = ax.bar(x, recon_values, width, label='Unguided', color='red', alpha=0.7)
    bars3 = ax.bar(x + width, sampled_values, width, label='Guided', color='green', alpha=0.7)
    
    ax.set_ylabel('Value')
    ax.set_title('Trajectory Statistics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')


def plot_connection_info(ax, history_pos, original_pos, history_len=0):
    """
    Plot connection information between history and prediction.
    
    Args:
        ax: Matplotlib axis
        history_pos: History positions array
        original_pos: Original prediction positions array
        history_len: Length of history segment
    """
    ax.axis('off')
    
    info_lines = []
    info_lines.append("=" * 40)
    info_lines.append("CONNECTION INFORMATION")
    info_lines.append("=" * 40)
    info_lines.append("")
    
    if history_pos is not None and len(history_pos) > 0 and len(original_pos) > 0:
        # Compute position jump at connection
        history_end = history_pos[-1]
        pred_start = original_pos[0]
        pos_jump = np.linalg.norm(pred_start - history_end)
        
        # Compute velocity continuity
        if len(history_pos) >= 2:
            history_vel = history_pos[-1] - history_pos[-2]
        else:
            history_vel = np.zeros(3)
        
        if len(original_pos) >= 2:
            pred_vel = original_pos[1] - original_pos[0]
        else:
            pred_vel = np.zeros(3)
        
        vel_jump = np.linalg.norm(pred_vel - history_vel)
        
        info_lines.append(f"History length: {len(history_pos)}")
        info_lines.append(f"Prediction length: {len(original_pos)}")
        info_lines.append("")
        info_lines.append(f"Position jump at connection:")
        info_lines.append(f"  {pos_jump:.4f}")
        info_lines.append("")
        info_lines.append(f"Velocity discontinuity:")
        info_lines.append(f"  {vel_jump:.4f}")
        
        # Evaluate connection quality
        info_lines.append("")
        if pos_jump < 0.01:
            info_lines.append("✓ Position: EXCELLENT")
        elif pos_jump < 0.1:
            info_lines.append("✓ Position: GOOD")
        else:
            info_lines.append("⚠ Position: POOR")
            
        if vel_jump < 0.1:
            info_lines.append("✓ Velocity: SMOOTH")
        elif vel_jump < 0.5:
            info_lines.append("✓ Velocity: ACCEPTABLE")
        else:
            info_lines.append("⚠ Velocity: ABRUPT")
    else:
        info_lines.append("No connection information")
        info_lines.append("(Missing history or prediction)")
    
    info_lines.append("")
    info_lines.append("=" * 40)
    
    info_text = '\n'.join(info_lines)
    ax.text(0.5, 0.5, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))


def plot_position_time(ax, time_steps, original_pos, reconstructed_pos, sampled_pos, 
                       history_pos, dim, title, ylabel, styles):
    """Plot position over time for a specific dimension, including history."""
    # Plot history if available
    if history_pos is not None:
        history_time = np.arange(-len(history_pos), 0)
        ax.plot(history_time, history_pos[:, dim], 
                label='History', **{k: v for k, v in styles['history'].items() if k != 'marker'})
    
    ax.plot(time_steps, original_pos[:, dim], 
            label=f'Original', **{k: v for k, v in styles['original'].items() if k != 'marker'})
    ax.plot(time_steps, reconstructed_pos[:, dim], 
            label=f'Reconstructed', **{k: v for k, v in styles['reconstructed'].items() if k != 'marker'})
    ax.plot(time_steps, sampled_pos[:, dim], 
            label=f'Sampled Guided', **{k: v for k, v in styles['sampled'].items() if k != 'marker'})
    
    # Add vertical line at connection point
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='Connection')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


# the main execution
if __name__ == "__main__":

    # generate_trj_demos()

    print("Training Obstacle-Aware AeroDM with Transformer Integration and Obstacle-Aware Loss...")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    config = Config()
    
    # Loss function based on flag
    criterion = AeroDMLoss(
        config, 
        enable_obstacle_term = config.use_obstacle_loss, # not use obstacle term in loss for this experiment
        safe_extra_factor=config.safe_extra_factor, # Safety buffer as fraction of radius (e.g., 20%)
        last_xyz_weight=config.last_xyz_weight, # Extra weight for final timestep's position error
        xyz_weight=config.xyz_weight, # Extra weight for Z-axis (height) in aviation
        diff_vel_weight=config.diff_vel_weight, # Weight for velocity term
        other_weight=config.other_weight, # Weight for other losses
        obstacle_weight=config.obstacle_weight, # Weight for obstacle term
        continuity_weight=config.continuity_weight, # Weight for continuity term
        acc_weight=config.acc_weight # Weight for acceleration term
    )

    print(f"Using AeroDMLoss (obstacle term: {config.use_obstacle_loss})")
    
    # Training parameters
    num_epochs = 100
    batch_size = 32
    num_trajectories = 3000
    
    print("Generating training data with obstacle-aware transformer...")
    trajectories = generate_aerobatic_trajectories(
        num_trajectories=num_trajectories, 
        seq_len=config.seq_len + config.history_len
    )

    # trajectories = generate_aerobatic_trajectories_deformation(    
    #     num_trajectories=num_trajectories, 
    #     seq_len=config.seq_len + config.history_len
    # )

    # trajectories = generate_circular_end_trajectories(    
    #     num_trajectories=num_trajectories, 
    #     seq_len=config.seq_len + config.history_len
    # )

    # trajectories = generate_distributed_trajectories(    
    #     num_trajectories=num_trajectories, 
    #     seq_len=config.seq_len + config.history_len
    # )

    # Augment trajectories with random concatenation
    # trajectories = augment_trajectories_with_concatenation(
    #     trajectories, 
    #     config.history_len, 
    #     config.seq_len,
    #     concat_ratio=0.5,  # Add 50% more trajectories through concatenation
    #     smooth_transition=True
    # )

    # Augment trajectories with smooth concatenation
    trajectories = augment_trajectories_with_smooth_concatenation(
        trajectories, 
        config.history_len, 
        config.seq_len,
        concat_ratio=1.0,      # Add 50% more trajectories
        smooth_window=5,        # Blend over 5 frames
        validate_continuity=True # Print continuity statistics
    )
    

    # Split trajectories into train and test sets (80/20 split)
    torch.manual_seed(42) # For reproducibility
    indices = torch.randperm(num_trajectories)
    train_size = int(0.9 * num_trajectories)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_trajectories = trajectories[train_indices]
    test_trajectories = trajectories[test_indices]
    
    # Normalize on train set and apply to test
    train_norm, mean, std = normalize_trajectories(train_trajectories)
    test_norm, _, _ = normalize_trajectories(test_trajectories, mean=mean, std=std)
    
    # Move data to device
    train_norm = train_norm.to(device)
    test_norm = test_norm.to(device)
    mean = mean.to(device)
    std = std.to(device)
    
    # Model and Optimizer setup
    model = AeroDM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.set_normalization_params(mean, std)

    # Use train_size for training loop
    train_size = train_trajectories.shape[0]
    losses = {'total': [], 'position': [], 'vel': [], 'obstacle': [], 'continuity': []}
    mode_str = "with obstacle-aware loss" if config.use_obstacle_loss else "with basic loss"
    print(f"Starting training {mode_str}...")
    start_time = time.time()

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0
        epoch_position_loss = 0
        epoch_vel_loss = 0
        epoch_obstacle_loss = 0 if config.use_obstacle_loss else None
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
            
            # Extract style index (last dimension)
            style_info = full_traj[:, :, -1:]  # Shape: (B, T_full, 1)
            state_without_style = full_traj[:, :, :-1]  # Shape: (B, T_full, state_dim-1)

            # Split into history and sequence-to-predict
            history = state_without_style[:, :config.history_len, :]
            x_0 = state_without_style[:, config.history_len:config.history_len+config.seq_len, :]

            # Generate condition inputs
            target = generate_target_waypoints(x_0)

            # FIXED: Use the style index from the FIRST timestep of the FULL trajectory
            # Since style is constant, any timestep works, but use the first for clarity
            # style_index = style_info[:, 0, 0]  # Shape: (B,) - take first timestep, first feature
            style_index = style_info[:, -1, 0]  # Take the last timestep's style index, which should be the same as the first if consistent
            style_indices = style_index.long()
            action = F.one_hot(style_indices, num_classes=config.action_dim).float()  # Shape: (B, action_dim)
            
            # Print action information
            # print(f"Style indices: {style_indices.cpu().numpy()}")
            # print(f"Action shape: {action.shape}")
            # print(f"Action (one-hot):\n{action.cpu().numpy()}")

            # Sample time step t
            t = torch.randint(0, config.diffusion_steps, (actual_batch_size,), device=device).long()
            
            # Forward diffusion (q_sample)
            noise = torch.randn_like(x_0)
            x_t, _ = model.diffusion_process.q_sample(x_0, t, noise)
            
            # Generate obstacles for this batch (only needed if obstacle loss is enabled)
            obstacles_for_batch = None
            # In the training loop, modify the obstacles generation:
            if config.use_obstacle_loss or config.enable_obstacle_encoding:
                obstacles_for_batch = []
                for b in range(actual_batch_size):
                    traj_denorm = denormalize_trajectories(full_traj[b:b+1, config.history_len:, :], mean, std)
                    obstacles = generate_random_obstacles(
                        traj_denorm[0], 
                        num_obstacles_range=(3, 5), 
                        radius_range=(0.5, 1.0), 
                        check_collision=False, 
                        device=device
                    )
                    obstacles_for_batch.append(obstacles) 

            # Model prediction (reverse process: predict x_0 from x_t)
            pred_x0 = model(x_t, t, target, action, history, obstacles_data=obstacles_for_batch)

            # Calculate loss based on flag
            total_loss, position_loss, vel_loss, obstacle_loss, continuity_loss = criterion(
                pred_x0, 
                x_0, 
                obstacles_for_batch if config.use_obstacle_loss else None, 
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
            if config.use_obstacle_loss:
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
            
            if config.use_obstacle_loss:
                avg_obstacle = epoch_obstacle_loss / num_batches
                losses['obstacle'].append(avg_obstacle)
            else:
                avg_obstacle = 0.0

        # if epoch % 5 == 0 or epoch == num_epochs - 1:
        progress_str = format_progress(epoch, num_epochs, start_time, avg_total, avg_position, 
                                    avg_vel, avg_obstacle, avg_continuity, config.use_obstacle_loss)
        sys.stdout.write(progress_str)
        sys.stdout.flush()
        
        if epoch == num_epochs - 1:
            print() 
            
    print(f"Training completed after {num_epochs} epochs.")
    print(f"\nTraining finished in {time.time()-start_time:.2f} seconds.")
    # ------------------
    # Evaluation and Testing
    # ------------------
    # Optional: Display a few generated trajectories before normalization/splitting
    # plot_trajectories_demo(trajectories[:18], rows=3, cols=6)
    
     # Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': losses,
        'mean': mean,
        'std': std,
        'use_obstacle_loss': config.use_obstacle_loss  # Save the training mode
    }
    
    if config.use_obstacle_loss:
        torch.save(checkpoint, "model/obstacle_aware_aerodm_v2_test.pth")
    else:
        torch.save(checkpoint, "model/aerodm_v2_test.pth")

    # Run the visualization test
    test_model_performance(model, test_norm, mean, std, num_test_samples=100, show_flag=config.show_flag)

    # Run the action effect test with detailed plots for each style
    # test_action_effect(model, test_norm, mean, std, num_test_samples=3, show_flag=config.show_flag)
