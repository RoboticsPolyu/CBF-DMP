#AeroDM + Barrier Function Guidance + Obstacle Encoding with Flow Matching

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from AeroDM.AeroDM_SafeTrj_v2_Test  import plot_test_results
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

from Deformation import generate_aerobatic_trajectories
from Deformation import generate_aerobatic_trajectories_pvR
from Deformation import augment_trajectories_with_smooth_concatenation
from Deformation import generate_aerobatic_trajectories_deformation
from Test.circular_trajectories import generate_circular_end_trajectories
from Test.distribute_trajectories import generate_distributed_trajectories

# Configuration parameters based on the paper
class Config:
    # Model dimensions
    latent_dim = 128
    obs_latent_dim = 128
    num_layers = 4
    num_heads = 4
    dropout = 0.1

    # Flow Matching parameters (replaces diffusion steps)
    num_time_steps = 100  # Number of time steps for ODE solver
    ode_solver = 'euler'  # 'euler' or 'rk4' for ODE integration
    
    # Sequence parameters
    seq_len = 60  # N_a = 60 time steps; 6s-long future trajectory sequence
    state_dim = 10  # x_i ∈ R^10: p(3) + v(3) + r(4) = 10 (style handled separately)
    history_len = 20  # 20-frame historical observations

    # Condition dimensions
    target_dim = 4  # p_t ∈ R^3 + valid flag (1)
    action_dim = 14   # 14 maneuver styles

    # Obstacle parameters
    max_obstacles = 10  # Maximum number of obstacles to process
    obstacle_feat_dim = 4  # [x, y, z, radius]
    enable_obstacle_encoding = False  # Toggle obstacle encoding in the model
    use_obstacle_loss = enable_obstacle_encoding  # Toggle obstacle loss term in training

    # CBF Guidance parameters (from CoDiG paper)
    guidance_gamma = 2000.0  # Base gamma for barrier guidance
    safe_extra_factor = 0.2  # Safety buffer as fraction of radius (e.g., 20%)
    
    last_xyz_weight = 5.0  # Extra weight for final timestep's position error
    xyz_weight = 1.0  # Extra weight for Z-axis (height) in aviation
    vel_weight = 1.0  # Weight for velocity term
    other_weight = 1.0  # Weight for other losses
    obstacle_weight = 1.0  # Weight for obstacle term
    continuity_weight = 5.0  # Weight for continuity term
    acc_weight = 10.0  # Weight for acceleration term
    delta_T = 0.1  # Time step duration (0.1s for 10Hz control frequency)

    drop_style_prob = 0.1  # Probability of dropping style information during training
    drop_target_prob = 0.1  # Probability of dropping target waypoint information
    
    # Plotting control
    show_flag = False  # Set to False to save plots as SVG instead of displaying

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
        
        self.obstacle_mlp = nn.Sequential(
            nn.Linear(config.obstacle_feat_dim, config.obs_latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.obs_latent_dim * 2, config.obs_latent_dim),
            nn.ReLU(),
            nn.Linear(config.obs_latent_dim, config.obs_latent_dim)
        )

    def forward(self, obstacles_data):
        if obstacles_data is None or len(obstacles_data) == 0:
            batch_size = 1 if obstacles_data is None else len(obstacles_data)
            return torch.zeros(batch_size, self.config.obs_latent_dim, device=next(self.parameters()).device)
        
        device = next(self.parameters()).device
        batch_size = len(obstacles_data)
        
        batch_obstacle_tensors = []
        valid_counts = []
        
        for sample_obstacles in obstacles_data:
            if not sample_obstacles:
                obstacle_tensor = torch.zeros(self.config.max_obstacles, self.config.obstacle_feat_dim, device=device)
                valid_counts.append(0)
            else:
                obstacle_tensors = []
                for obstacle in sample_obstacles:
                    center = obstacle['center'].to(device)
                    radius = obstacle['radius']
                    obstacle_feat = torch.cat([
                        center,
                        torch.tensor([radius], device=device, dtype=center.dtype)
                    ])
                    obstacle_tensors.append(obstacle_feat)
                
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
        
        batch_obstacle_tensor = torch.stack(batch_obstacle_tensors)
        batch_size, max_obs, feat_dim = batch_obstacle_tensor.shape
        flattened_obstacles = batch_obstacle_tensor.view(-1, feat_dim)
        encoded_obstacles = self.obstacle_mlp(flattened_obstacles)
        encoded_obstacles = encoded_obstacles.view(batch_size, max_obs, -1)
        
        valid_mask = torch.zeros(batch_size, max_obs, device=device)
        for i, count in enumerate(valid_counts):
            if count > 0:
                valid_mask[i, :count] = 1.0
        
        masked_embeddings = encoded_obstacles * valid_mask.unsqueeze(-1)
        sum_embeddings = masked_embeddings.sum(dim=1)
        valid_counts_tensor = torch.tensor(valid_counts, device=device).float().clamp(min=1.0)
        global_features = sum_embeddings / valid_counts_tensor.unsqueeze(-1)
        
        return global_features

# Attention-based Obstacle Encoder for small obstacle sets
class AttentionObstacleEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.obstacle_proj = nn.Sequential(
            nn.Linear(config.obstacle_feat_dim, config.obs_latent_dim),
            nn.ReLU(),
            nn.Linear(config.obs_latent_dim, config.obs_latent_dim)
        )
        
        self.attention = nn.MultiheadAttention(
            config.obs_latent_dim,
            num_heads=4,
            batch_first=True,
            dropout=config.dropout
        )
        
        max_obs = config.max_obstacles
        self.pos_encoding = nn.Parameter(torch.randn(1, max_obs, config.obs_latent_dim) * 0.1)
        
        self.global_proj = nn.Sequential(
            nn.Linear(config.obs_latent_dim * 2, config.obs_latent_dim),
            nn.ReLU(),
            nn.Linear(config.obs_latent_dim, config.obs_latent_dim),
            nn.LayerNorm(config.obs_latent_dim)
        )
        
    def forward(self, obstacles_data):
        batch_size = len(obstacles_data)
        device = next(self.parameters()).device
        
        batch_embeddings = []
        
        for batch_idx, sample_obs in enumerate(obstacles_data):
            num_obstacles = len(sample_obs)
            
            if num_obstacles == 0:
                batch_embeddings.append(
                    torch.zeros(self.config.obs_latent_dim, device=device)
                )
                continue
            
            obstacle_features = []
            for obstacle in sample_obs:
                center = obstacle['center'].to(device)
                radius = obstacle['radius']
                radius_tensor = torch.tensor([radius], device=device, dtype=center.dtype)
                feat = torch.cat([center, radius_tensor])
                obstacle_features.append(feat)
            
            obs_tensor = torch.stack(obstacle_features)
            obs_emb = self.obstacle_proj(obs_tensor)
            
            pos_enc = self.pos_encoding[:, :num_obstacles, :].to(device)
            obs_emb = obs_emb.unsqueeze(0) + pos_enc
            
            attn_output, attn_weights = self.attention(
                query=obs_emb,
                key=obs_emb,
                value=obs_emb
            )
            
            obs_emb = attn_output.squeeze(0)
            
            mean_pool = obs_emb.mean(dim=0)
            max_pool = obs_emb.max(dim=0)[0]
            global_feat = torch.cat([mean_pool, max_pool])
            global_feat = self.global_proj(global_feat)
            
            batch_embeddings.append(global_feat)
        
        return torch.stack(batch_embeddings)
    
# Condition embedding module
class ConditionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Time embedding for flow matching (t in [0,1])
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
        
        # Learned null embeddings for missing conditions
        self.null_target_embed = nn.Parameter(torch.randn(config.latent_dim))
        self.null_action_embed = nn.Parameter(torch.randn(config.latent_dim))
        self.null_obstacle_embed = nn.Parameter(torch.randn(config.latent_dim))
        
        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(config.action_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )
        
        # Obstacle embedding
        self.obstacle_encoder = AttentionObstacleEncoder(config)

        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.latent_dim * 4, config.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim * 2, config.latent_dim)
        )

    def forward(self, t, target=None, action=None, obstacles_data=None):
        batch_size = t.shape[0]
        device = t.device
        
        t_emb = self.t_embed(t.unsqueeze(-1).float())
        
        if target is not None:
            target_emb = self.target_embed(target)
        else:
            target_emb = self.null_target_embed.unsqueeze(0).expand(batch_size, -1)
        
        if action is not None:
            action_emb = self.action_embed(action)
        else:
            action_emb = self.null_action_embed.unsqueeze(0).expand(batch_size, -1)
        
        if obstacles_data is not None and self.config.enable_obstacle_encoding:
            obstacle_emb = self.obstacle_encoder(obstacles_data)
        else:
            obstacle_emb = self.null_obstacle_embed.unsqueeze(0).expand(batch_size, -1)
        
        combined_emb = torch.cat([t_emb, target_emb, action_emb, obstacle_emb], dim=-1)
        cond_emb = self.fusion_layer(combined_emb)
        
        return cond_emb

class FlowMatchingTransformer(nn.Module):
    """
    Transformer model that predicts the velocity field v(x_t, t) for flow matching.
    Given a noisy sample x_t at time t, predicts the velocity that pushes it toward the data distribution.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.state_dim, config.latent_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.latent_dim)
        
        # Condition embedding (includes timestep t, target, action, obstacles)
        self.cond_embed = ConditionEmbedding(config)
        
        # Transformer decoder layers
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
        
        # Output projection to predict velocity field
        self.output_proj = nn.Linear(config.latent_dim, config.state_dim)

    def forward(self, x, t, target=None, action=None, history=None, obstacles_data=None):
        """
        Predict velocity field v(x_t, t) for given samples.
        
        Args:
            x: Current samples at time t (batch, seq_len, state_dim)
            t: Time step (batch,) in [0,1]
            target: Target waypoint (batch, target_dim)
            action: One-hot action style (batch, action_dim)
            history: Historical trajectory (batch, history_len, state_dim)
            obstacles_data: List of obstacle dictionaries
        
        Returns:
            velocity: Predicted vector field (batch, seq_len, state_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Set default values if conditions are missing
        if target is None:
            target = torch.ones(batch_size, self.config.target_dim, device=device) * 1e-6
        
        if action is None:
            action = torch.zeros(batch_size, self.config.action_dim, device=device)
    
        # Randomly drop conditions during training for robustness
        if self.training:
            if action is not None:
                drop_mask = torch.rand(batch_size, device=device) < self.config.drop_style_prob
                action = action.clone()
                action[drop_mask] = 0
            if target is not None:
                drop_mask = torch.rand(batch_size, device=device) < self.config.drop_target_prob
                target = target.clone()
                target[drop_mask] = 1e-6

        # Project input to latent space
        x_proj = self.input_proj(x)
        
        # Concatenate history if provided
        if history is not None:
            if history.size(0) != batch_size:
                if history.size(0) == 1:
                    history = history.repeat(batch_size, 1, 1)
            history_proj = self.input_proj(history)
            transformer_input = torch.cat([history_proj, x_proj], dim=1)
            total_seq_len = history_proj.size(1) + seq_len
        else:
            transformer_input = x_proj
            total_seq_len = seq_len
        
        # Add positional encoding
        transformer_input = self.pos_encoding(transformer_input.transpose(0, 1)).transpose(0, 1)
        
        # Generate causal mask for autoregressive generation
        memory_mask = self._generate_square_subsequent_mask(total_seq_len).to(device)
        
        # Get condition embedding
        cond_emb = self.cond_embed(t, target, action, obstacles_data)
        cond_seq = cond_emb.unsqueeze(1).expand(-1, total_seq_len, -1)
        
        # Add condition to input
        transformer_input = transformer_input + cond_seq
        
        # Apply transformer with causal masking
        transformer_output = self.transformer(
            tgt=transformer_input,
            memory=transformer_input,
            tgt_mask=memory_mask,
            memory_mask=memory_mask
        )
        
        # Extract only the current sequence (exclude history)
        if history is not None:
            current_output = transformer_output[:, -seq_len:, :]
        else:
            current_output = transformer_output
        
        # Output velocity field
        velocity = self.output_proj(current_output)
        return velocity

    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive attention."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

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

def generate_target_waypoints(trajectory):
    # Target is the final position (indices 1:4 for x, y, z)
    target_pos = trajectory[:, -1, 1:4]  # (batch, 3)
    
    # Add validity flag (1 = valid target)
    valid_flag = torch.ones(target_pos.shape[0], 1, device=target_pos.device)
    
    # Concatenate: [x, y, z, valid]
    return torch.cat([target_pos, valid_flag], dim=-1)

# Note: Denormalization should be applied only to the state variables (x,y,z) and not to the style dimension.
def denormalize_trajectories(trajectories_norm, mean, std):
    return trajectories_norm * std + mean

def denormalize_target(target_norm, mean, std):
    # Extract position normalization parameters (indices 1:4 for x,y,z)
    pos_mean = mean[0, 0, 1:4]  # Shape: (3,)
    pos_std = std[0, 0, 1:4]    # Shape: (3,)
    
    # Extract x,y,z from target
    target_pos = target_norm[..., :3]  # Shape: (batch, 3)
    target_valid = target_norm[..., 3:]  # Shape: (batch, 1)
    
    # Denormalize only the position part
    target_pos_denorm = target_pos * pos_std + pos_mean
    
    # Concatenate with original valid flag
    return torch.cat([target_pos_denorm, target_valid], dim=-1)

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


# ============================================================================
# CORRECTED FLOW MATCHING PROCESS
# ============================================================================

class FlowMatchingProcess:
    """
    Flow Matching process that learns to transport noise to data via ODE integration.
    
    Key insight: For optimal transport path x_t = (1-t)*x_0 + t*x_1,
    the conditional vector field is u_t(x_t|x_0,x_1) = x_1 - x_0.
    
    Training: Sample t ~ Uniform(0,1), x_0 ~ N(0,I), compute x_t and u_t,
              train model to predict u_t from (x_t, t).
    
    Sampling: Start from x_0 ~ N(0,I), integrate ODE dx/dt = v_theta(x_t, t) from t=0 to t=1.
    """
    
    def __init__(self, config):
        self.config = config
        
    def sample_conditional_flow(self, x_1, t):
        """
        Sample from the conditional probability path.
        
        Args:
            x_1: Real data samples at t=1 (batch, seq_len, state_dim)
            t: Time steps (batch,) in [0,1]
        
        Returns:
            x_t: Interpolated samples at time t (batch, seq_len, state_dim)
            u_t: Conditional vector field (batch, seq_len, state_dim)
        """
        batch_size, seq_len, state_dim = x_1.shape
        device = x_1.device
        
        # Sample noise from prior distribution x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_1)
        
        # Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        # This is the optimal transport path between Gaussian and data distribution
        t_expanded = t.view(-1, 1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        
        # Conditional vector field: u_t = x_1 - x_0
        # This is the velocity that pushes x_t toward x_1
        u_t = x_1 - x_0
        
        return x_t, u_t
    
    @torch.no_grad()
    def sample(self, model, target=None, action=None, history=None, 
               enable_guidance=True, guidance_gamma=None, mean=None, std=None, 
               obstacles_data=None, method='euler', num_steps=100):
        """
        Generate samples by solving the ODE from t=0 to t=1.
        
        Args:
            model: Flow matching model that predicts velocity field
            target: Target waypoint
            action: Action style
            history: Historical trajectory
            enable_guidance: Whether to use CBF guidance
            guidance_gamma: Guidance strength
            mean, std: Normalization parameters
            obstacles_data: Obstacle information
            method: ODE solver ('euler' or 'rk4')
            num_steps: Number of integration steps
        
        Returns:
            x_1: Generated trajectory at t=1
        """
        device = next(model.parameters()).device
        batch_size = target.shape[0] if target is not None else 1
        
        # Initialize from prior distribution x_0 ~ N(0, I)
        x_t = torch.randn(batch_size, self.config.seq_len, self.config.state_dim, device=device)
        
        # Time discretization
        dt = 1.0 / num_steps
        
        print(f"\n{'='*60}")
        print("FLOW MATCHING ODE SOLVER")
        print(f"Initial: x_0 ~ N(0, I) | Mean: {x_t.mean().item():.4f}, Std: {x_t.std().item():.4f}")
        print(f"Integration steps: {num_steps} | Method: {method}")
        print(f"CBF Guidance: {enable_guidance}")
        print(f"{'='*60}")
        
        # Integrate ODE from t=0 to t=1
        for step in range(num_steps):
            t_current = step / num_steps  # Current time in [0, 1)
            t_next = (step + 1) / num_steps
            
            # Create time tensor
            t_tensor = torch.full((batch_size,), t_current, device=device, dtype=torch.float32)
            
            # Predict velocity field
            v_t = model(x_t, t_tensor, target, action, history, obstacles_data)
            
            # Apply CBF guidance if enabled
            if enable_guidance and guidance_gamma is not None and mean is not None and std is not None:
                # Guidance strength decreases as t increases (stronger at the end)
                gamma_t = guidance_gamma * (1.0 - t_current)
                
                # Compute barrier gradient
                V, grad_V = compute_barrier_and_grad_logistic(
                    x_t, self.config, mean, std, obstacles_data
                )
                
                # Guided velocity field: push away from obstacles
                v_t = v_t - gamma_t * grad_V
                
                # Print guidance info periodically
                if step % max(1, num_steps // 10) == 0:
                    print(f"  Step {step:3d}/{num_steps} | t={t_current:.3f} | "
                          f"V={V.item():.4f} | gamma={gamma_t:.2f}")
            
            # ODE integration
            if method == 'euler':
                # Euler method: x_{t+dt} = x_t + dt * v_t
                x_t = x_t + dt * v_t
                
            elif method == 'rk4':
                # 4th order Runge-Kutta for better accuracy
                # k1 = v(x_t, t)
                k1 = v_t
                
                # k2 = v(x_t + dt/2 * k1, t + dt/2)
                t_mid = t_current + dt / 2
                t_mid_tensor = torch.full((batch_size,), t_mid, device=device, dtype=torch.float32)
                x_mid = x_t + (dt / 2) * k1
                k2 = model(x_mid, t_mid_tensor, target, action, history, obstacles_data)
                
                # k3 = v(x_t + dt/2 * k2, t + dt/2)
                x_mid = x_t + (dt / 2) * k2
                k3 = model(x_mid, t_mid_tensor, target, action, history, obstacles_data)
                
                # k4 = v(x_t + dt * k3, t + dt)
                t_next_tensor = torch.full((batch_size,), t_next, device=device, dtype=torch.float32)
                x_next = x_t + dt * k3
                k4 = model(x_next, t_next_tensor, target, action, history, obstacles_data)
                
                # Combine: x_{t+dt} = x_t + dt/6 * (k1 + 2k2 + 2k3 + k4)
                x_t = x_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Progress update
            if (step + 1) % max(1, num_steps // 5) == 0 or step == num_steps - 1:
                print(f"  Progress: {(step+1)/num_steps*100:.0f}% | "
                      f"x_t mean: {x_t.mean().item():.4f}, std: {x_t.std().item():.4f}")
        
        print(f"{'='*60}")
        print(f"Generation complete! Final x_1 stats - Mean: {x_t.mean().item():.4f}, Std: {x_t.std().item():.4f}")
        print(f"{'='*60}\n")
        
        return x_t


# ============================================================================
# AERO-FM MODEL (Replaces AeroDM)
# ============================================================================

class AeroFM(nn.Module):
    """
    Aerobatic trajectory generation using Flow Matching instead of Diffusion.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.flow_model = FlowMatchingTransformer(config)
        self.flow_process = FlowMatchingProcess(config)
        self.mean = None
        self.std = None
        self.obstacles_data = None
        
    def forward(self, x_t, t, target=None, action=None, history=None, obstacles_data=None):
        """Predict velocity field."""
        return self.flow_model(x_t, t, target, action, history, obstacles_data)
    
    def set_normalization_params(self, mean, std):
        """Set normalization parameters for CBF guidance."""
        self.mean = mean
        self.std = std
    
    def set_obstacles_data(self, obstacles_data):
        """Set obstacles data for the model."""
        if obstacles_data is None:
            self.obstacles_data = None
        else:
            self.obstacles_data = obstacles_data
        print(f"Set obstacles_data: {len(self.obstacles_data) if self.obstacles_data else 0} batches")

    def sample(self, target=None, action=None, history=None, batch_size=1, 
               enable_guidance=True, guidance_gamma=None, method='euler', num_steps=100):
        """
        Generate trajectory samples using flow matching.
        
        Args:
            target: Target waypoint (batch, 4)
            action: One-hot action style (batch, action_dim)
            history: Historical trajectory (batch, history_len, state_dim)
            batch_size: Batch size (if target/action not provided)
            enable_guidance: Whether to use CBF guidance
            guidance_gamma: Guidance strength
            method: ODE solver method ('euler' or 'rk4')
            num_steps: Number of integration steps
        
        Returns:
            Generated trajectory (batch, seq_len, state_dim)
        """
        device = next(self.parameters()).device
        
        # Set default conditions if not provided
        if target is None:
            target = torch.ones(batch_size, self.config.target_dim, device=device) * 1e-6
        if action is None:
            action = torch.zeros(batch_size, self.config.action_dim, device=device)
        
        # Solve ODE to generate trajectory
        x_1 = self.flow_process.sample(
            self.flow_model, target, action, history, 
            enable_guidance, guidance_gamma, self.mean, self.std, 
            self.obstacles_data, method, num_steps
        )
        
        return x_1


# ============================================================================
# FLOW MATCHING LOSS
# ============================================================================

class AeroFMLoss(nn.Module):
    """
    Loss function for Flow Matching training.
    
    The main loss is MSE between predicted velocity and true conditional velocity:
    L = E_{t~U(0,1), x_0~N(0,I), x_1~data} [||v_theta(x_t, t) - (x_1 - x_0)||^2]
    
    Additional losses (position, velocity, continuity) are also included.
    """
    def __init__(self, config, enable_obstacle_term=False, safe_extra_factor=0.2, 
                 last_xyz_weight=1.5, xyz_weight=1.5, vel_weight=1.0, other_weight=1.0, 
                 obstacle_weight=10.0, continuity_weight=15.0, acc_weight=1.0):
        super().__init__()
        self.config = config
        self.enable_obstacle_term = enable_obstacle_term
        self.safe_extra_factor = safe_extra_factor
        self.last_xyz_weight = last_xyz_weight
        self.xyz_weight = xyz_weight
        self.vel_weight = vel_weight
        self.other_weight = other_weight
        self.obstacle_weight = obstacle_weight
        self.continuity_weight = continuity_weight
        self.acc_weight = acc_weight
        self.mse_loss = nn.MSELoss()
    
    def compute_obstacle_distance_loss(self, pred_trajectory, obstacles_data, mean, std):
        """
        Compute obstacle avoidance loss for the generated trajectory.
        Penalizes trajectories that enter safety buffer around obstacles.
        """
        if not self.enable_obstacle_term or not obstacles_data or len(obstacles_data) == 0:
            return torch.tensor(0.0, device=pred_trajectory.device, requires_grad=True)
        
        batch_size, seq_len, _ = pred_trajectory.shape
        device = pred_trajectory.device
        obstacle_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_num_obs = 0
        
        # Denormalize positions
        pos_std = std[0, 0, 1:4]
        pos_mean = mean[0, 0, 1:4]
        pred_pos_denorm = pred_trajectory[:, :, 1:4] * pos_std + pos_mean
        
        for batch_idx in range(batch_size):
            batch_obs = obstacles_data[batch_idx] if batch_idx < len(obstacles_data) else []
            total_num_obs += len(batch_obs)
            
            for obstacle in batch_obs:
                center = obstacle['center'].to(device)
                radius = obstacle['radius']
                safe_extra = self.safe_extra_factor * radius
                
                distances = torch.norm(pred_pos_denorm[batch_idx] - center.unsqueeze(0), dim=1)
                surface_dist = distances - radius
                closeness_penalty = torch.clamp(safe_extra - surface_dist, min=0.0)
                obstacle_loss = obstacle_loss + torch.sum(closeness_penalty ** 2)
        
        avg_num_obs = total_num_obs / max(batch_size, 1.0)
        if avg_num_obs > 0:
            obstacle_loss = obstacle_loss / (batch_size * avg_num_obs)
        
        return torch.clamp(obstacle_loss, min=0.0)
    
    def forward(self, pred_velocity, gt_velocity, pred_trajectory=None, gt_trajectory=None, 
                obstacles_data=None, mean=None, std=None, history=None):
        """
        Compute total loss for flow matching training.
        
        Args:
            pred_velocity: Predicted vector field from model
            gt_velocity: Ground truth conditional vector field (x_1 - x_0)
            pred_trajectory: Generated trajectory (for auxiliary losses)
            gt_trajectory: Ground truth trajectory (for auxiliary losses)
            obstacles_data: Obstacle information
            mean, std: Normalization parameters
            history: Historical trajectory for continuity loss
        
        Returns:
            total_loss, velocity_loss, position_loss, obstacle_loss, continuity_loss
        """
        batch_size, seq_len, state_dim = pred_velocity.shape
        device = pred_velocity.device
        
        # Primary flow matching loss: MSE between predicted and true vector fields
        velocity_loss = self.mse_loss(pred_velocity, gt_velocity)
        
        # Auxiliary trajectory losses (optional, for better quality)
        position_loss = torch.tensor(0.0, device=device)
        vel_loss = torch.tensor(0.0, device=device)
        other_loss = torch.tensor(0.0, device=device)
        obstacle_loss = torch.tensor(0.0, device=device)
        continuity_loss = torch.tensor(0.0, device=device)
        acc_smoothness = torch.tensor(0.0, device=device)
        
        if pred_trajectory is not None and gt_trajectory is not None:
            # Extract components
            pred_pos = pred_trajectory[:, :, 1:4]
            gt_pos = gt_trajectory[:, :, 1:4]
            pred_speed = pred_trajectory[:, :, 0:1]
            gt_speed = gt_trajectory[:, :, 0:1]
            pred_vel = pred_trajectory[:, :, 4:7]
            gt_vel = gt_trajectory[:, :, 4:7]
            pred_attitude = pred_trajectory[:, :, 7:10]
            gt_attitude = gt_trajectory[:, :, 7:10]
            
            # Position losses with Z-weighting
            x_loss = self.xyz_weight * self.mse_loss(pred_pos[:, :, 0], gt_pos[:, :, 0])
            y_loss = self.xyz_weight * self.mse_loss(pred_pos[:, :, 1], gt_pos[:, :, 1])
            z_loss = self.xyz_weight * self.mse_loss(pred_pos[:, :, 2], gt_pos[:, :, 2])
            
            # Last timestep losses (higher weight for endpoint accuracy)
            last_x_loss = self.xyz_weight * self.mse_loss(pred_pos[:, -1, 0], gt_pos[:, -1, 0])
            last_y_loss = self.xyz_weight * self.mse_loss(pred_pos[:, -1, 1], gt_pos[:, -1, 1])
            last_z_loss = self.xyz_weight * self.mse_loss(pred_pos[:, -1, 2], gt_pos[:, -1, 2])
            last_xyz_loss = last_x_loss + last_y_loss + last_z_loss
            
            position_loss = x_loss + y_loss + z_loss + self.last_xyz_weight * last_xyz_loss
            
            # Velocity loss
            if seq_len > 1:
                vel_x_loss = self.xyz_weight * self.mse_loss(pred_vel[:, :, 0], gt_vel[:, :, 0])
                vel_y_loss = self.xyz_weight * self.mse_loss(pred_vel[:, :, 1], gt_vel[:, :, 1])
                vel_z_loss = self.xyz_weight * self.mse_loss(pred_vel[:, :, 2], gt_vel[:, :, 2])
                vel_loss = vel_x_loss + vel_y_loss + vel_z_loss
            
            # Acceleration smoothness
            if seq_len >= 3:
                vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
                acc = vel[:, 1:, :] - vel[:, :-1, :]
                acc_smoothness = acc.pow(2).mean()
            
            # Speed and attitude losses
            speed_loss = self.mse_loss(pred_speed, gt_speed)
            attitude_loss = self.mse_loss(pred_attitude, gt_attitude)
            other_loss = speed_loss + attitude_loss
            
            # Obstacle loss
            if self.enable_obstacle_term and obstacles_data is not None and mean is not None and std is not None:
                obstacle_loss = self.compute_obstacle_distance_loss(pred_trajectory, obstacles_data, mean, std)
            
            # Continuity loss (smooth connection with history)
            if history is not None and pred_trajectory.size(1) > 0:
                last_history_pos = history[:, -1, 1:4]
                first_pred_pos = pred_trajectory[:, 0, 1:4]
                continuity_loss = self.mse_loss(first_pred_pos, last_history_pos)
        
        # Total weighted loss
        total_loss = (velocity_loss + 
                     self.xyz_weight * position_loss + 
                     self.vel_weight * vel_loss + 
                     self.other_weight * other_loss + 
                     self.obstacle_weight * obstacle_loss + 
                     self.continuity_weight * continuity_loss + 
                     self.acc_weight * acc_smoothness)
        
        return total_loss, velocity_loss, position_loss, obstacle_loss, continuity_loss


# ============================================================================
# BARRIER FUNCTION (unchanged from original)
# ============================================================================

def compute_barrier_and_grad_logistic(x, config, mean, std, obstacles_data=None, safety_margin=0.20, sigma=0.5):
    """
    Compute barrier V and its gradient ∇V using logistic approach.
    V = -log(P_safe) where P_safe is probability of being safe.
    """
    # Denormalize positions
    pos_denorm = x[:, :, 1:4] * std[0, 0, 1:4] + mean[0, 0, 1:4]
    batch_size, seq_len, _ = pos_denorm.shape
    
    grad_pos_denorm = torch.zeros_like(pos_denorm)
    pos_denorm = pos_denorm.clone().detach()
    V_total = torch.zeros(batch_size, device=x.device)
    
    if obstacles_data is not None:
        for batch_idx in range(batch_size):
            batch_obs = obstacles_data[batch_idx] if batch_idx < len(obstacles_data) else []
            
            for obstacle in batch_obs:
                center = obstacle['center'].to(x.device)
                radius = obstacle['radius'] + safety_margin
                
                # Distance to obstacle center
                delta = pos_denorm[batch_idx] - center.unsqueeze(0)
                dist_sq = torch.sum(delta * delta, dim=1)
                R_sq = radius * radius
                
                # Safe probability: P_safe = sigmoid((d^2 - R^2)/sigma)
                logit_safe = (dist_sq - R_sq) / sigma
                P_safe = torch.sigmoid(logit_safe)
                
                # Barrier: V = -log(P_safe)
                V_obstacle = -torch.log(P_safe + 1e-8)
                V_total[batch_idx] += torch.sum(V_obstacle)
                
                # Gradient: dV/dp = -(1-P_safe) * (2*delta)/sigma
                grad_factor = -(1 - P_safe) * 2.0 / sigma
                grad_V_obs = grad_factor.unsqueeze(1) * delta
                grad_pos_denorm[batch_idx] = grad_pos_denorm[batch_idx] + grad_V_obs
    
    # Map gradient back to normalized space
    grad_x = torch.zeros_like(x, device=x.device)
    std_scaled = std[0, 0, 1:4].to(x.device)
    grad_x[:, :, 1:4] = grad_pos_denorm / std_scaled
    
    V_avg = V_total.mean()
    return V_avg, grad_x


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_flow_matching_model():
    """Train the Flow Matching model on aerobatic trajectories."""
    
    print("="*70)
    print("TRAINING FLOW MATCHING AERO-FM MODEL")
    print("="*70)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    config = Config()
    
    # Loss function
    criterion = AeroFMLoss(
        config, 
        enable_obstacle_term=config.use_obstacle_loss,
        safe_extra_factor=config.safe_extra_factor,
        last_xyz_weight=config.last_xyz_weight,
        xyz_weight=config.xyz_weight,
        vel_weight=config.vel_weight,
        other_weight=config.other_weight,
        obstacle_weight=config.obstacle_weight,
        continuity_weight=config.continuity_weight,
        acc_weight=config.acc_weight
    )
    
    print(f"Using AeroFMLoss (obstacle term: {config.use_obstacle_loss})")
    
    # Training parameters
    num_epochs = 100
    batch_size = 32
    num_trajectories = 2000
    
    print("Generating training data...")
    trajectories = generate_aerobatic_trajectories_pvR(
        num_trajectories=num_trajectories, 
        seq_len=config.seq_len + config.history_len,
        delta_T=config.delta_T
    )
    
    # Split into train/test (90/10)
    torch.manual_seed(42)
    indices = torch.randperm(trajectories.shape[0])
    train_size = int(0.9 * trajectories.shape[0])
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_trajectories = trajectories[train_indices]
    test_trajectories = trajectories[test_indices]
    
    # Normalize
    train_norm, mean, std = normalize_trajectories(train_trajectories)
    test_norm, _, _ = normalize_trajectories(test_trajectories, mean=mean, std=std)
    
    # Move to device
    train_norm = train_norm.to(device)
    test_norm = test_norm.to(device)
    mean = mean.to(device)
    std = std.to(device)
    
    # Initialize model and optimizer
    model = AeroFM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.set_normalization_params(mean, std)
    
    print(f"Training samples: {train_size}, Test samples: {test_trajectories.shape[0]}")
    print(f"Starting training for {num_epochs} epochs...")
    print("-"*70)
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0
        epoch_velocity_loss = 0
        epoch_position_loss = 0
        epoch_obstacle_loss = 0 if config.use_obstacle_loss else None
        epoch_continuity_loss = 0
        num_batches = 0
        
        # Shuffle indices
        indices = torch.randperm(train_size)
        
        for i in range(0, train_size, batch_size):
            actual_batch_size = min(batch_size, train_size - i)
            batch_indices = indices[i:i+actual_batch_size]
            
            # Get batch data
            full_traj = train_norm[batch_indices]
            
            # Extract style (last dimension) and state (all except style)
            style_info = full_traj[:, :, -1:]
            state_without_style = full_traj[:, :, :-1]
            
            # Split into history and prediction target
            history = state_without_style[:, :config.history_len, :]
            x_1 = state_without_style[:, config.history_len:config.history_len+config.seq_len, :]
            
            # Generate conditions
            target = generate_target_waypoints(x_1)
            style_index = style_info[:, -1, 0]  # Use last timestep's style
            action = F.one_hot(style_index.long(), num_classes=config.action_dim).float()
            
            # Sample random time t ~ Uniform(0, 1)
            t = torch.rand(actual_batch_size, device=device).float()
            
            # Sample from conditional flow
            x_t, u_t = model.flow_process.sample_conditional_flow(x_1, t)
            
            # Generate obstacles for this batch (if needed)
            obstacles_for_batch = None
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
            
            # Predict velocity field
            pred_u = model(x_t, t, target, action, history, obstacles_for_batch)
            
            # Compute loss
            total_loss, velocity_loss, position_loss, obstacle_loss, continuity_loss = criterion(
                pred_u, u_t, x_1, x_1,  # Use x_1 as ground truth for trajectory
                obstacles_for_batch if config.use_obstacle_loss else None,
                mean, std, history
            )
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            epoch_total_loss += total_loss.item()
            epoch_velocity_loss += velocity_loss.item()
            epoch_position_loss += position_loss.item()
            if config.use_obstacle_loss:
                epoch_obstacle_loss += obstacle_loss.item()
            epoch_continuity_loss += continuity_loss.item()
            num_batches += 1
        
        # Compute average losses
        avg_total = epoch_total_loss / num_batches
        avg_velocity = epoch_velocity_loss / num_batches
        avg_position = epoch_position_loss / num_batches
        avg_continuity = epoch_continuity_loss / num_batches
        
        if config.use_obstacle_loss:
            avg_obstacle = epoch_obstacle_loss / num_batches
        else:
            avg_obstacle = 0.0
        
        # Print progress
        elapsed = time.time() - start_time
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Time: {elapsed:.1f}s | "
                  f"Total: {avg_total:.6f} | Vel: {avg_velocity:.6f} | "
                  f"Pos: {avg_position:.6f} | Obs: {avg_obstacle:.6f}")
    
    print(f"\nTraining completed in {time.time()-start_time:.2f} seconds.")
    
    # Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'mean': mean.cpu(),
        'std': std.cpu(),
        'use_obstacle_loss': config.use_obstacle_loss,
        'config': config
    }
    
    import os
    os.makedirs('model', exist_ok=True)
    torch.save(checkpoint, "model/aerofm_model.pth")
    print("Model saved to model/aerofm_model.pth")
    
    return model, test_norm, mean, std


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_flow_matching_model(model, test_norm, mean, std, num_samples=5):
    """Test the trained Flow Matching model."""
    
    print("\n" + "="*70)
    print("TESTING FLOW MATCHING MODEL")
    print("="*70)
    
    device = next(model.parameters()).device
    config = model.config
    
    mean_state = mean[..., :-1].to(device)
    std_state = std[..., :-1].to(device)
    model.set_normalization_params(mean_state, std_state)
    
    style_names = {
        0: 'power_loop', 1: 'barrel_roll', 2: 'split_s', 3: 'immelmann',
        4: 'wall_ride', 5: 'eight_figure', 6: 'star', 7: 'half_moon',
        8: 'sphinx', 9: 'clover', 10: 'spiral_inward', 11: 'spiral_outward',
        12: 'spiral_vertical_up', 13: 'spiral_vertical_down'
    }
    
    model.eval()
    
    with torch.no_grad():
        for i in range(min(num_samples, test_norm.shape[0])):
            print(f"\n--- Test Sample {i+1} ---")
            
            # Prepare test sample
            full_traj = test_norm[i:i+1]
            style_info = full_traj[:, :, -1:]
            state_without_style = full_traj[:, :, :-1]
            
            history = state_without_style[:, :config.history_len, :]
            x_1_gt = state_without_style[:, config.history_len:config.history_len+config.seq_len, :]
            target = generate_target_waypoints(x_1_gt)
            
            # Denormalize for visualization
            x_1_gt_denorm = denormalize_trajectories(x_1_gt, mean_state, std_state)
            target_denorm = denormalize_target(target, mean_state, std_state)
            
            # Get action
            style_index = style_info[:, -1, 0]
            action = F.one_hot(style_index.long(), num_classes=config.action_dim).float()
            
            # Generate obstacles
            obstacles = generate_random_obstacles(
                x_1_gt_denorm[0], 
                num_obstacles_range=(3, 5), 
                radius_range=(0.5, 1.0), 
                device=device
            )
            model.set_obstacles_data([obstacles])
            
            # Generate trajectory using flow matching
            sampled_norm = model.sample(
                target, action, history, batch_size=1,
                enable_guidance=False, 
                guidance_gamma=config.guidance_gamma,
                method='euler', 
                num_steps=300
            )
            
            sampled_denorm = denormalize_trajectories(sampled_norm, mean_state, std_state)
            
            # Print statistics
            print(f"  Style: {style_names.get(style_index.item(), 'unknown')}")
            print(f"  Obstacles: {len(obstacles)}")
            print(f"  GT trajectory shape: {x_1_gt_denorm.shape}")
            print(f"  Generated trajectory shape: {sampled_denorm.shape}")
            print(f"  Generated - Mean: {sampled_denorm.mean().item():.4f}, Std: {sampled_denorm.std().item():.4f}")
            
            # Optional: Plot results (if you have plotting functions)
            # plot_comparison(x_1_gt_denorm, sampled_denorm, history, target_denorm, obstacles, i)
            plot_test_results(
                    x_1_gt_denorm, 
                    sampled_denorm, 
                    sampled_denorm, 
                    denormalize_trajectories(history, mean_state, std_state) if history is not None else None,
                    target_denorm,
                    obstacles,
                    True
                )
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FLOW MATCHING AERO-FM FOR AEROBATIC TRAJECTORY GENERATION")
    print("="*70)
    print("\nKey differences from Diffusion:")
    print("  1. Training: Learn velocity field v(x_t, t) instead of noise prediction")
    print("  2. Sampling: ODE integration from t=0 to t=1 (Euler or RK4)")
    print("  3. Path: Linear interpolation x_t = (1-t)*x_0 + t*x_1")
    print("  4. Benefits: Fewer sampling steps, simpler objective, better theoretical guarantees")
    print("="*70 + "\n")
    
    # Train the model
    model, test_norm, mean, std = train_flow_matching_model()
    
    # Test the model
    test_flow_matching_model(model, test_norm, mean, std, num_samples=15)
    
    print("\nFlow Matching model training and testing complete!")