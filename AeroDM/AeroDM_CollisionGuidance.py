import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
import warnings

# Configuration parameters based on the paper
class Config:
    # Model dimensions
    latent_dim = 256
    num_layers = 4
    num_heads = 4
    dropout = 0.1
    
    # Diffusion process parameters
    diffusion_steps = 30
    beta_start = 0.0001
    beta_end = 0.02
    
    # Sequence parameters
    seq_len = 60  # N_a = 60 time steps
    state_dim = 10  # x_i ∈ R^10: s(1) + p(3) + r(6)
    history_len = 5  # 5 frames of historical observation
    
    # Condition dimensions
    target_dim = 3  # p_t ∈ R^3
    action_dim = 5   # 5 maneuver styles
    
    # Obstacle avoidance parameters (new)
    batch_size_sampling = 50  # Reduce batch size for faster demonstration
    collision_threshold = 0.5  # Collision threshold
    safety_margin = 0.3  # Safety margin

# Obstacle representation class
class Obstacle:
    def __init__(self, center: np.ndarray, radius: float, obs_type: str = "sphere"):
        self.center = np.array(center)
        self.radius = radius
        self.type = obs_type
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate distance from point to obstacle"""
        return np.linalg.norm(point - self.center) - self.radius

# Environment class
class Environment:
    def __init__(self, bounds: np.ndarray):
        self.bounds = bounds  # [x_min, x_max, y_min, y_max, z_min, z_max]
        self.obstacles = []
    
    def add_obstacle(self, center: List[float], radius: float):
        """Add spherical obstacle"""
        self.obstacles.append(Obstacle(np.array(center), radius))
    
    def check_collision(self, trajectory: np.ndarray, safety_margin: float = 0.0) -> bool:
        """Check if trajectory collides with obstacles"""
        if len(trajectory) == 0:
            return False
            
        for point in trajectory:
            for obstacle in self.obstacles:
                if obstacle.distance_to_point(point) < safety_margin:
                    return True
        return False
    
    def compute_collision_cost(self, trajectory: np.ndarray) -> float:
        """Calculate collision cost for trajectory"""
        if len(trajectory) == 0:
            return 0.0
            
        total_cost = 0.0
        for point in trajectory:
            min_distance = float('inf')
            for obstacle in self.obstacles:
                distance = obstacle.distance_to_point(point)
                min_distance = min(min_distance, distance)
            
            # Higher cost for closer distances (using exponential decay)
            if min_distance < 0:
                cost = 10.0  # High cost for collision
            else:
                cost = np.exp(-min_distance)  # Higher cost for closer distances
            
            total_cost += cost
        
        return total_cost / len(trajectory)

# Classifier-guided collision cost function
class CollisionGuidance:
    def __init__(self, environment: Environment, config: Config):
        self.env = environment
        self.config = config
    
    def compute_guidance_scores(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Calculate collision cost scores for batch of trajectories"""
        batch_size = trajectories.shape[0]
        scores = torch.zeros(batch_size)
        
        for i in range(batch_size):
            # Extract position information (seq_len, 3)
            positions = trajectories[i, :, :3].detach().cpu().numpy()
            
            # Calculate collision cost
            collision_cost = self.env.compute_collision_cost(positions)
            
            # Convert to score (lower cost is better)
            scores[i] = -collision_cost  # Negative because we want to maximize score (minimize cost)
        
        return scores

# Coarse collision checking module
class CoarseCollisionChecker:
    def __init__(self, environment: Environment, config: Config):
        self.env = environment
        self.config = config
    
    def check_trajectory(self, trajectory: torch.Tensor) -> bool:
        """Perform coarse collision check for single trajectory"""
        positions = trajectory[:, :3].detach().cpu().numpy()
        
        if len(positions) == 0:
            return True
            
        # Check collision for each point
        for point in positions:
            for obstacle in self.env.obstacles:
                if obstacle.distance_to_point(point) < self.config.safety_margin:
                    return False  # Collision exists
        
        # Check interpolated path between consecutive points
        for i in range(len(positions) - 1):
            # Linear interpolation to check intermediate points
            num_interp = 5
            for j in range(1, num_interp):
                alpha = j / num_interp
                interp_point = (1 - alpha) * positions[i] + alpha * positions[i + 1]
                
                for obstacle in self.env.obstacles:
                    if obstacle.distance_to_point(interp_point) < self.config.safety_margin:
                        return False
        
        return True  # No collision

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
                    alpha_bar_t_prev = self.alpha_bars[t_val-1] if t_val > 0 else torch.tensor(1.0)
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

# Modified AeroDM class with obstacle avoidance functionality
class AeroDMWithObstacleAvoidance(AeroDM):
    def __init__(self, config: Config, environment: Environment):
        super().__init__(config)
        self.config = config
        self.environment = environment
        self.collision_guidance = CollisionGuidance(environment, config)
        self.collision_checker = CoarseCollisionChecker(environment, config)
    
    def sample_with_obstacle_avoidance(self, target: torch.Tensor, action: torch.Tensor, 
                                     history: Optional[torch.Tensor] = None, 
                                     max_attempts: int = 3) -> torch.Tensor:
        """Trajectory sampling with obstacle avoidance"""
        device = next(self.parameters()).device
        
        print(f"Starting obstacle avoidance trajectory generation, batch size: {self.config.batch_size_sampling}")
        
        for attempt in range(max_attempts):
            # Batch generate candidate trajectories
            candidate_trajectories = self._batch_sample_candidates(target, action, history)
            
            # Calculate collision cost scores
            with torch.no_grad():
                guidance_scores = self.collision_guidance.compute_guidance_scores(candidate_trajectories)
            
            # Select best candidate
            best_idx = torch.argmax(guidance_scores)
            best_trajectory = candidate_trajectories[best_idx:best_idx+1]
            best_score = guidance_scores[best_idx]
            
            print(f"Attempt {attempt + 1}: Best collision score = {best_score.item():.4f}")
            
            # Coarse collision check
            if self.collision_checker.check_trajectory(best_trajectory[0]):
                print("Found collision-free trajectory!")
                return best_trajectory
            else:
                print("Best candidate failed coarse collision check, continuing...")
        
        print(f"Failed to find safe trajectory after {max_attempts} attempts, returning best candidate")
        return best_trajectory
    
    def _batch_sample_candidates(self, target: torch.Tensor, action: torch.Tensor, 
                               history: Optional[torch.Tensor]) -> torch.Tensor:
        """Batch generate candidate trajectories"""
        batch_size = self.config.batch_size_sampling
        device = next(self.parameters()).device
        
        # Expand conditions to batch size
        target_batch = target.repeat(batch_size, 1) if target.size(0) == 1 else target
        action_batch = action.repeat(batch_size, 1) if action.size(0) == 1 else action
        
        if history is not None:
            history_batch = history.repeat(batch_size, 1, 1) if history.size(0) == 1 else history
        else:
            history_batch = None
        
        # Initialize noise
        x_t = torch.randn(batch_size, self.config.seq_len, self.config.state_dim).to(device)
        
        # Reverse diffusion process
        for t_step in reversed(range(self.config.diffusion_steps)):
            t_batch = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
            x_t = self.diffusion_process.p_sample(
                self.diffusion_model, x_t, t_batch, target_batch, action_batch, history_batch
            )
        
        return x_t
    
    def generate_long_horizon_trajectory(self, target_sequence: List[torch.Tensor], 
                                       action_sequence: List[torch.Tensor],
                                       num_primitives: int = 3) -> Dict[str, Any]:
        """Generate long-horizon multi-primitive trajectory"""
        device = next(self.parameters()).device
        all_primitives = []
        history = None
        
        success_count = 0
        collision_info = []
        
        for i in range(num_primitives):
            print(f"Generating aerobatic primitive {i+1}...")
            
            # Select target
            target_idx = i % len(target_sequence)
            action_idx = i % len(action_sequence)
            
            target = target_sequence[target_idx].to(device)
            action = action_sequence[action_idx].to(device)
            
            # Sample with obstacle avoidance
            primitive = self.sample_with_obstacle_avoidance(target, action, history)
            
            # Check final result
            primitive_pos = primitive[0, :, :3].detach().cpu().numpy()
            collision_free = not self.environment.check_collision(primitive_pos, self.config.safety_margin)
            
            if collision_free:
                success_count += 1
                collision_info.append(True)
                print(f"Primitive {i+1}: Collision-free ✓")
            else:
                collision_info.append(False)
                print(f"Primitive {i+1}: Potential collision ⚠")
            
            # Update history
            if history is None:
                history = primitive[:, -self.config.history_len:, :]
            else:
                combined = torch.cat([history, primitive], dim=1)
                history = combined[:, -self.config.history_len:, :]
            
            all_primitives.append(primitive)
        
        # Concatenate all primitives
        if all_primitives:
            full_trajectory = torch.cat(all_primitives, dim=1)
        else:
            full_trajectory = torch.empty(1, 0, self.config.state_dim)
        
        return {
            'trajectory': full_trajectory,
            'primitives': all_primitives,
            'success_rate': success_count / num_primitives if num_primitives > 0 else 0.0,
            'collision_info': collision_info,
            'num_primitives': num_primitives
        }

# Visualization function: Display obstacles and trajectories
def plot_trajectories_with_obstacles(primitives, environment, title="Aerobatic Trajectories with Obstacles"):
    """Plot trajectories with obstacles"""
    if not primitives:
        print("No trajectory data to plot")
        return
    
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot obstacles
    for i, obstacle in enumerate(environment.obstacles):
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = obstacle.center[0] + obstacle.radius * np.outer(np.cos(u), np.sin(v))
        y = obstacle.center[1] + obstacle.radius * np.outer(np.sin(u), np.sin(v))
        z = obstacle.center[2] + obstacle.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x, y, z, color='red', alpha=0.3, label='Obstacle' if i == 0 else "")
    
    # Plot each primitive's trajectory
    colors = ['b-', 'g-', 'c-', 'm-', 'y-']
    for i, primitive in enumerate(primitives):
        primitive_pos = primitive[0, :, :3].detach().cpu().numpy()
        color = colors[i % len(colors)]
        label = f'Primitive {i+1}'
        ax1.plot(primitive_pos[:, 0], primitive_pos[:, 1], primitive_pos[:, 2], 
                color, label=label, linewidth=2)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('3D Trajectories with Obstacles')
    
    # 2D projection (X-Y plane)
    ax2 = fig.add_subplot(222)
    for obstacle in environment.obstacles:
        circle = plt.Circle((obstacle.center[0], obstacle.center[1]), obstacle.radius, 
                          color='red', alpha=0.3, label='Obstacle')
        ax2.add_patch(circle)
    
    for i, primitive in enumerate(primitives):
        primitive_pos = primitive[0, :, :3].detach().cpu().numpy()
        color = colors[i % len(colors)]
        label = f'Primitive {i+1}'
        ax2.plot(primitive_pos[:, 0], primitive_pos[:, 1], color, label=label, linewidth=2)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.set_title('X-Y Plane Projection')
    ax2.grid(True)
    ax2.set_aspect('equal')
    
    # Position vs time (show only first primitive)
    if primitives:
        first_primitive = primitives[0][0, :, :3].detach().cpu().numpy()
        time_steps = np.arange(first_primitive.shape[0])
        
        ax3 = fig.add_subplot(223)
        ax3.plot(time_steps, first_primitive[:, 0], 'b-', label='X Position', linewidth=2)
        ax3.plot(time_steps, first_primitive[:, 1], 'g-', label='Y Position', linewidth=2)
        ax3.plot(time_steps, first_primitive[:, 2], 'r-', label='Z Position', linewidth=2)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Position')
        ax3.legend()
        ax3.set_title('Position Changes for First Primitive')
        ax3.grid(True)
    
    # Velocity magnitude
    if primitives:
        all_velocities = []
        for primitive in primitives:
            primitive_pos = primitive[0, :, :3].detach().cpu().numpy()
            velocity = np.linalg.norm(primitive_pos[1:] - primitive_pos[:-1], axis=1)
            all_velocities.extend(velocity)
        
        ax4 = fig.add_subplot(224)
        ax4.plot(range(len(all_velocities)), all_velocities, 'b-', linewidth=2)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Velocity Magnitude')
        ax4.set_title('Overall Velocity Changes')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

# Demonstrate obstacle avoidance functionality
def demonstrate_obstacle_avoidance():
    """Demonstrate obstacle avoidance functionality"""
    config = Config()
    
    # Create environment and add obstacles
    env = Environment(bounds=np.array([-10, 10, -10, 10, -5, 5]))
    
    # Add some obstacles
    env.add_obstacle([2, 1, 0], 1.0)    # Factory equipment
    env.add_obstacle([-1, -2, 0], 0.8)  # Pillar
    env.add_obstacle([4, -1, 0], 1.2)   # Machine
    env.add_obstacle([-3, 3, 0], 0.6)   # Small obstacle
    
    # Create model with obstacle avoidance
    model = AeroDMWithObstacleAvoidance(config, env)
    
    # Create sample inputs
    targets = [
        torch.tensor([[5.0, 5.0, 2.0]]),  # Target 1
        torch.tensor([[3.0, -3.0, 1.0]]), # Target 2  
        torch.tensor([[-2.0, 2.0, 1.5]])  # Target 3
    ]
    
    actions = [torch.randn(1, config.action_dim) for _ in range(3)]
    
    print("Starting obstacle avoidance trajectory generation...")
    
    # Generate trajectories with obstacle avoidance
    result = model.generate_long_horizon_trajectory(
        target_sequence=targets,
        action_sequence=actions,
        num_primitives=3
    )
    
    print(f"Generation completed! Success rate: {result['success_rate']:.1%}")
    print(f"Collision information: {result['collision_info']}")
    
    # Visualize results - only plot generated primitives
    plot_trajectories_with_obstacles(result['primitives'], env)
    
    return model, result

if __name__ == "__main__":
    print("Demonstrating AeroDM with obstacle avoidance...")
    model, result = demonstrate_obstacle_avoidance()