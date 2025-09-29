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

# Diffusion process with linear noise schedule
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
    
    def p_sample(self, model, x_t, t, target, action, history=None):
        """Reverse diffusion process: p(x_{t-1} | x_t)"""
        with torch.no_grad():
            pred_x0 = model(x_t, t, target, action, history)
            
            batch_size = x_t.size(0)
            x_prev = torch.zeros_like(x_t)
            
            for i in range(batch_size):
                t_val = t[i].item() if t.dim() > 0 else t.item()
                
                if t_val > 0:
                    alpha_bar_t = self.alpha_bars[t_val]
                    alpha_bar_t_prev = self.alpha_bars[t_val-1] if t_val > 0 else torch.tensor(1.0, device=x_t.device)
                    beta_t = self.betas[t_val]
                    
                    coeff_x0 = (alpha_bar_t_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                    coeff_xt = ((1 - alpha_bar_t_prev) * self.alphas[t_val].sqrt()) / (1 - alpha_bar_t)
                    
                    mean = coeff_x0 * pred_x0[i] + coeff_xt * x_t[i]
                    
                    noise = torch.randn_like(x_t[i])
                    variance = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * beta_t
                    x_prev[i] = mean + variance.sqrt() * noise
                else:
                    x_prev[i] = pred_x0[i]
            
            return x_prev

# Complete Aerobatic Diffusion Model (AeroDM)
class AeroDM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.diffusion_model = DiffusionTransformer(config)
        self.diffusion_process = DiffusionProcess(config)
        
    def forward(self, x_t, t, target, action, history=None):
        return self.diffusion_model(x_t, t, target, action, history)
    
    def sample(self, target, action, history=None, batch_size=1):
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
        
        # Reverse diffusion process
        for t_step in reversed(range(self.config.diffusion_steps)):
            t_batch = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
            x_t = self.diffusion_process.p_sample(
                self.diffusion_model, x_t, t_batch, target, action, history
            )
        
        return x_t

# Improved Loss Function with Balanced Z-Axis Learning
class ImprovedAeroDMLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred_trajectory, gt_trajectory):
        # Separate position dimensions for balanced learning
        pred_pos = pred_trajectory[:, :, 1:4]  # x, y, z positions
        gt_pos = gt_trajectory[:, :, 1:4]
        
        # Individual dimension losses with weighting
        x_loss = self.mse_loss(pred_pos[:, :, 0], gt_pos[:, :, 0])
        y_loss = self.mse_loss(pred_pos[:, :, 1], gt_pos[:, :, 1])
        z_loss = self.mse_loss(pred_pos[:, :, 2], gt_pos[:, :, 2])
        
        # Weight z-axis loss more heavily to balance learning
        position_loss = x_loss + y_loss + z_loss  # Increased z weight
        
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

def generate_enhanced_circular_trajectories(num_trajectories=100, seq_len=60, radius=10.0, height=50.0):
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

def analyze_z_axis_performance(original, reconstructed, sampled):
    """Specifically analyze z-axis learning performance"""
    original_z = original[0, :, 3].detach().cpu().numpy()  # z is at index 3 (1: speed, 2-4: x,y,z)
    reconstructed_z = reconstructed[0, :, 3].detach().cpu().numpy()
    sampled_z = sampled[0, :, 3].detach().cpu().numpy()
    
    z_error_recon = np.mean(np.abs(reconstructed_z - original_z))
    z_error_sampled = np.mean(np.abs(sampled_z - original_z))
    
    print(f"Z-axis Mean Absolute Error - Reconstruction: {z_error_recon:.4f}")
    print(f"Z-axis Mean Absolute Error - Sampling: {z_error_sampled:.4f}")
    
    # Plot z-axis specifically
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.plot(original_z, 'b-', label='Original Z', linewidth=2)
    plt.plot(reconstructed_z, 'r--', label='Reconstructed Z', linewidth=2)
    plt.plot(sampled_z, 'g-.', label='Sampled Z', linewidth=2)
    plt.title('Z-axis Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(132)
    plt.plot(np.abs(reconstructed_z - original_z), 'r-', label='Recon Error', linewidth=2)
    plt.plot(np.abs(sampled_z - original_z), 'g-', label='Sample Error', linewidth=2)
    plt.title('Z-axis Absolute Error')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(133)
    plt.bar(['Reconstruction', 'Sampling'], [z_error_recon, z_error_sampled])
    plt.title('Z-axis Mean Absolute Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_circular_trajectory_comparison(original, reconstructed, sampled, history=None, target=None, title="Circular Trajectory Comparison"):
    """Enhanced visualization with history, target, and different colors"""
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
    
    # 1. 3D trajectory plot with history and target
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
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('3D Trajectory Comparison with History & Target')
    ax1.grid(True)
    
    # 2. XY plane projection
    ax2 = fig.add_subplot(342)
    
    if history_pos is not None:
        ax2.plot(history_pos[:, 0], history_pos[:, 1], 'm-', label='History', linewidth=4, marker='o', markersize=4)
    
    ax2.plot(original_pos[:, 0], original_pos[:, 1], 'b-', label='Original', linewidth=3)
    ax2.plot(reconstructed_pos[:, 0], reconstructed_pos[:, 1], 'r.-', label='Reconstructed', linewidth=2)
    ax2.plot(sampled_pos[:, 0], sampled_pos[:, 1], 'g.-', label='Sampled', linewidth=2)
    
    if target_pos is not None:
        ax2.scatter(target_pos[0], target_pos[1], c='yellow', s=150, marker='*', 
                   label='Target', edgecolors='black', linewidth=2)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.set_title('XY Plane Projection')
    ax2.grid(True)
    ax2.axis('equal')
    
    # 3. XZ plane projection (Focus on Z-axis)
    ax3 = fig.add_subplot(343)
    
    if history_pos is not None:
        ax3.plot(history_pos[:, 0], history_pos[:, 2], 'm-', label='History', linewidth=4, marker='o', markersize=4)
    
    ax3.plot(original_pos[:, 0], original_pos[:, 2], 'b-', label='Original', linewidth=3)
    ax3.plot(reconstructed_pos[:, 0], reconstructed_pos[:, 2], 'r--', label='Reconstructed', linewidth=2)
    ax3.plot(sampled_pos[:, 0], sampled_pos[:, 2], 'g-.', label='Sampled', linewidth=2)
    
    if target_pos is not None:
        ax3.scatter(target_pos[0], target_pos[2], c='yellow', s=150, marker='*', 
                   label='Target', edgecolors='black', linewidth=2)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.legend()
    ax3.set_title('XZ Plane Projection (Z-axis Focus)')
    ax3.grid(True)
    
    # 4. YZ plane projection (Focus on Z-axis)
    ax4 = fig.add_subplot(344)
    
    if history_pos is not None:
        ax4.plot(history_pos[:, 1], history_pos[:, 2], 'm-', label='History', linewidth=4, marker='o', markersize=4)
    
    ax4.plot(original_pos[:, 1], original_pos[:, 2], 'b-', label='Original', linewidth=3)
    ax4.plot(reconstructed_pos[:, 1], reconstructed_pos[:, 2], 'r--', label='Reconstructed', linewidth=2)
    ax4.plot(sampled_pos[:, 1], sampled_pos[:, 2], 'g-.', label='Sampled', linewidth=2)
    
    if target_pos is not None:
        ax4.scatter(target_pos[1], target_pos[2], c='yellow', s=150, marker='*', 
                   label='Target', edgecolors='black', linewidth=2)
    
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.legend()
    ax4.set_title('YZ Plane Projection (Z-axis Focus)')
    ax4.grid(True)
    
    # 5. Z-axis only comparison
    time_steps = np.arange(original_pos.shape[0])
    ax5 = fig.add_subplot(345)
    
    # Add history time steps if available
    if history_pos is not None:
        history_time = np.arange(-len(history_pos), 0)
        ax5.plot(history_time, history_pos[:, 2], 'm-', label='History Z', linewidth=3, marker='o', markersize=4)
    
    ax5.plot(time_steps, original_pos[:, 2], 'b-', label='Original Z', linewidth=3)
    ax5.plot(time_steps, reconstructed_pos[:, 2], 'r--', label='Reconstructed Z', linewidth=2)
    ax5.plot(time_steps, sampled_pos[:, 2], 'g-.', label='Sampled Z', linewidth=2)
    
    if target_pos is not None:
        ax5.axhline(y=target_pos[2], color='yellow', linestyle=':', linewidth=2, label='Target Z')
    
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Z Value')
    ax5.legend()
    ax5.set_title('Z-axis Values Over Time')
    ax5.grid(True)
    
    # 6. Timeline visualization showing history -> prediction
    ax6 = fig.add_subplot(346)
    
    # Combine history and current trajectories for timeline view
    all_time_steps = []
    all_original_z = []
    all_reconstructed_z = []
    all_sampled_z = []
    
    if history_pos is not None:
        history_time = np.arange(-len(history_pos), 0)
        all_time_steps.extend(history_time)
        all_original_z.extend(history_pos[:, 2])
        all_reconstructed_z.extend(history_pos[:, 2])  # History is same for all
        all_sampled_z.extend(history_pos[:, 2])
    
    all_time_steps.extend(time_steps)
    all_original_z.extend(original_pos[:, 2])
    all_reconstructed_z.extend(reconstructed_pos[:, 2])
    all_sampled_z.extend(sampled_pos[:, 2])
    
    ax6.plot(all_time_steps, all_original_z, 'b-', label='Original', linewidth=3)
    ax6.plot(all_time_steps, all_reconstructed_z, 'r--', label='Reconstructed', linewidth=2)
    ax6.plot(all_time_steps, all_sampled_z, 'g-.', label='Sampled', linewidth=2)
    
    # Add vertical line separating history from prediction
    if history_pos is not None:
        ax6.axvline(x=-0.5, color='black', linestyle='--', alpha=0.7, label='History/Prediction Boundary')
    
    if target_pos is not None:
        ax6.axhline(y=target_pos[2], color='yellow', linestyle=':', linewidth=2, label='Target Z')
    
    ax6.set_xlabel('Time Step (Negative: History, Positive: Prediction)')
    ax6.set_ylabel('Z Value')
    ax6.legend()
    ax6.set_title('Complete Timeline: History + Prediction')
    ax6.grid(True)
    
    # 7. Position error with separate z-error
    recon_error = np.linalg.norm(reconstructed_pos - original_pos, axis=1)
    sampled_error = np.linalg.norm(sampled_pos - original_pos, axis=1)
    recon_z_error = np.abs(reconstructed_pos[:, 2] - original_pos[:, 2])
    sampled_z_error = np.abs(sampled_pos[:, 2] - original_pos[:, 2])
    
    ax7 = fig.add_subplot(347)
    ax7.plot(time_steps, recon_error, 'r-', label='Recon Total Error', linewidth=2)
    ax7.plot(time_steps, sampled_error, 'g-', label='Sample Total Error', linewidth=2)
    ax7.plot(time_steps, recon_z_error, 'r--', label='Recon Z Error', linewidth=2, alpha=0.7)
    ax7.plot(time_steps, sampled_z_error, 'g--', label='Sample Z Error', linewidth=2, alpha=0.7)
    ax7.set_xlabel('Time Step')
    ax7.set_ylabel('Error')
    ax7.legend()
    ax7.set_title('Error Analysis (Z-error highlighted)')
    ax7.grid(True)
    
    # 8. Velocity comparison
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
    ax8.set_title('Speed Comparison')
    ax8.grid(True)
    
    # 9. Error statistics including history-target relationship
    ax9 = fig.add_subplot(349)
    errors = {
        'Recon Total': np.mean(recon_error),
        'Sample Total': np.mean(sampled_error),
        'Recon Z': np.mean(recon_z_error),
        'Sample Z': np.mean(sampled_z_error)
    }
    
    # Add target distance if available
    target_distance = None
    if target_pos is not None:
        final_original_pos = original_pos[-1]
        target_distance = np.linalg.norm(final_original_pos - target_pos)
        errors['Target Dist'] = target_distance
    
    colors = ['red', 'green', 'darkred', 'darkgreen', 'orange']
    ax9.bar(range(len(errors)), list(errors.values()), color=colors[:len(errors)])
    ax9.set_xticks(range(len(errors)))
    ax9.set_xticklabels(list(errors.keys()), rotation=45)
    ax9.set_ylabel('Mean Error / Distance')
    ax9.set_title('Error & Target Distance Comparison')
    ax9.grid(True, alpha=0.3)
    
    # 10. Target achievement analysis
    ax10 = fig.add_subplot(3, 4, 10)
    
    if target_pos is not None:
        # Calculate distance to target over time
        recon_target_dist = np.linalg.norm(reconstructed_pos - target_pos, axis=1)
        sampled_target_dist = np.linalg.norm(sampled_pos - target_pos, axis=1)
        original_target_dist = np.linalg.norm(original_pos - target_pos, axis=1)
        
        ax10.plot(time_steps, original_target_dist, 'b-', label='Original to Target', linewidth=2)
        ax10.plot(time_steps, recon_target_dist, 'r--', label='Recon to Target', linewidth=2)
        ax10.plot(time_steps, sampled_target_dist, 'g-.', label='Sampled to Target', linewidth=2)
        ax10.set_xlabel('Time Step')
        ax10.set_ylabel('Distance to Target')
        ax10.legend()
        ax10.set_title('Distance to Target Over Time')
        ax10.grid(True)
    
    # 11. History continuity analysis
    ax11 = fig.add_subplot(3, 4, 11)
    
    if history_pos is not None:
        # Check continuity between history and predictions
        history_end = history_pos[-1]
        recon_start = reconstructed_pos[0]
        sampled_start = sampled_pos[0]
        original_start = original_pos[0]
        
        continuity_errors = {
            'Original': np.linalg.norm(original_start - history_end),
            'Reconstructed': np.linalg.norm(recon_start - history_end),
            'Sampled': np.linalg.norm(sampled_start - history_end)
        }
        
        colors = ['blue', 'red', 'green']
        ax11.bar(range(len(continuity_errors)), list(continuity_errors.values()), color=colors)
        ax11.set_xticks(range(len(continuity_errors)))
        ax11.set_xticklabels(list(continuity_errors.keys()), rotation=45)
        ax11.set_ylabel('Continuity Error')
        ax11.set_title('History-Prediction Continuity')
        ax11.grid(True, alpha=0.3)
    
    # 12. Summary statistics
    ax12 = fig.add_subplot(3, 4, 12)
    
    summary_stats = []
    stat_labels = []
    
    # Basic statistics
    summary_stats.extend([np.mean(original_pos[:, 2]), np.std(original_pos[:, 2])])
    stat_labels.extend(['Mean Z', 'Std Z'])
    
    # Error statistics
    summary_stats.extend([np.mean(recon_z_error), np.mean(sampled_z_error)])
    stat_labels.extend(['Recon Z Err', 'Sample Z Err'])
    
    if target_pos is not None:
        summary_stats.append(target_distance)
        stat_labels.append('Final Target Dist')
    
    ax12.bar(range(len(summary_stats)), summary_stats, color='lightblue', alpha=0.7)
    ax12.set_xticks(range(len(summary_stats)))
    ax12.set_xticklabels(stat_labels, rotation=45)
    ax12.set_ylabel('Values')
    ax12.set_title('Summary Statistics')
    ax12.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed error analysis
    print("\n=== Detailed Error Analysis ===")
    print(f"Reconstruction - Total Error: {np.mean(recon_error):.4f}, Z-error: {np.mean(recon_z_error):.4f}")
    print(f"Sampling - Total Error: {np.mean(sampled_error):.4f}, Z-error: {np.mean(sampled_z_error):.4f}")
    
    if target_pos is not None:
        print(f"Final Target Distance - Original: {np.linalg.norm(original_pos[-1] - target_pos):.4f}")
        print(f"Final Target Distance - Reconstructed: {np.linalg.norm(reconstructed_pos[-1] - target_pos):.4f}")
        print(f"Final Target Distance - Sampled: {np.linalg.norm(sampled_pos[-1] - target_pos):.4f}")
    
    if history_pos is not None:
        print(f"History-Prediction Continuity Error - Original: {np.linalg.norm(original_pos[0] - history_pos[-1]):.4f}")
        print(f"History-Prediction Continuity Error - Reconstructed: {np.linalg.norm(reconstructed_pos[0] - history_pos[-1]):.4f}")
        print(f"History-Prediction Continuity Error - Sampled: {np.linalg.norm(sampled_pos[0] - history_pos[-1]):.4f}")

# Update the test function to pass history and target
def test_model_performance(model, trajectories_norm, mean, std, num_test_samples=3):
    """Enhanced testing with history and target visualization"""
    print("\nTesting model performance with history and target visualization...")
    config = Config()
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        for i in range(min(num_test_samples, trajectories_norm.shape[0])):
            # Prepare test sample
            full_traj = trajectories_norm[i:i+1]  # (1, 65, 10) full trajectory
            target_norm = generate_target_waypoints(full_traj)  # From full trajectory end
            action = generate_action_styles(1, config.action_dim, device=device)
            history = generate_history_segments(full_traj, config.history_len, device=device)  # First 5 steps
            x_0 = full_traj[:, config.history_len:, :]  # FIXED: Last 60 steps (prediction sequence)
            
            # Denormalize target for plotting only (model uses normalized)
            target_denorm = target_norm * std[0, 0, 1:4] + mean[0, 0, 1:4]  # Position dims only
            
            # Add noise and reconstruct
            t = torch.randint(0, config.diffusion_steps, (1,), device=device)
            noisy_x, _ = model.diffusion_process.q_sample(x_0, t)
            reconstructed = model(noisy_x, t, target_norm, action, history)  # Pass normalized to model
            
            # Sample new trajectory
            sampled = model.sample(target_norm, action, history, batch_size=1)  # Normalized input to model
            
            # Denormalize for visualization
            x_0_denorm = denormalize_trajectories(x_0, mean, std)
            reconstructed_denorm = denormalize_trajectories(reconstructed, mean, std)
            sampled_denorm = denormalize_trajectories(sampled, mean, std)
            history_denorm = denormalize_trajectories(history, mean, std)
            
            # Visualize results with history and denormalized target
            plot_circular_trajectory_comparison(
                x_0_denorm, reconstructed_denorm, sampled_denorm, 
                history=history_denorm, target=target_denorm,
                title=f"Enhanced Circular Trajectory Test Sample {i+1}\n(History: Magenta, Target: Yellow Star)"
            )
            
            # Z-axis specific analysis
            analyze_z_axis_performance(x_0_denorm, reconstructed_denorm, sampled_denorm)
    
    model.train()

def train_improved_aerodm():
    """Enhanced training function with z-axis improvements"""
    config = Config()
    model = AeroDM(config)
    
    # Set device to MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    # Move diffusion process tensors to device
    model.diffusion_process.betas = model.diffusion_process.betas.to(device)
    model.diffusion_process.alphas = model.diffusion_process.alphas.to(device)
    model.diffusion_process.alpha_bars = model.diffusion_process.alpha_bars.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = ImprovedAeroDMLoss(config)  # Use improved loss
    
    # Training parameters
    num_epochs = 50
    batch_size = 8
    num_trajectories = 20000
    
    print("Generating enhanced circular trajectory data...")
    # Generate enhanced training data
    trajectories = generate_enhanced_circular_trajectories(
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
    
    # Plot training losses
    plot_training_losses(losses)
    
    # Test model performance
    test_model_performance(model, trajectories_norm, mean, std)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': losses,
    }, "model/improved_aerodm_checkpoint.pth")

    return model, losses, trajectories, mean, std

def load_trained_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load trained model"""
    config = Config()
    model = AeroDM(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Move diffusion process parameters to device
    model.diffusion_process.betas = model.diffusion_process.betas.to(device)
    model.diffusion_process.alphas = model.diffusion_process.alphas.to(device)
    model.diffusion_process.alpha_bars = model.diffusion_process.alpha_bars.to(device)
    
    print(f"Model loaded successfully! Using device: {device}")
    return model, config

def run_model_validation(checkpoint_path, num_test_cases=5):
    """Run model validation"""
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_trained_model(checkpoint_path, device)
    
    print("Generating enhanced circular trajectory data...")
    # Generate enhanced training data
    trajectories = generate_enhanced_circular_trajectories(
        num_trajectories=num_test_cases,
        seq_len=config.seq_len + config.history_len
    )
    
    # Normalize trajectories
    trajectories_norm, mean, std = normalize_trajectories(trajectories)
    trajectories_norm = trajectories_norm.to(device)
    mean = mean.to(device)
    std = std.to(device)
    
    # Test model performance
    test_model_performance(model, trajectories_norm, mean, std)

# Main execution
if __name__ == "__main__":
    # Specify your model file path
    checkpoint_path = "model/improved_aerodm_checkpoint.pth"  # Change to your model path
    
    try:
        # Run validation
        test_cases, trajectories = run_model_validation(checkpoint_path, num_test_cases=5)
        
        print("\n" + "="*60)
        print("Model validation completed!")
        print("="*60)
        
    except FileNotFoundError:
        print(f"Error: Model file '{checkpoint_path}' not found")
        print("Please ensure the model file path is correct, or use the following to download an example model:")
        print("1. Check file path")
        print("2. If file doesn't exist, you need to train the model first")
        
    except Exception as e:
        print(f"Error during validation: {e}")
        print("Possible reasons:")
        print("1. Model file corrupted")
        print("2. Model structure doesn't match code")
        print("3. Device incompatibility")