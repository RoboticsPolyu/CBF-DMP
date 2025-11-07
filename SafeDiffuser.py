import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    """Trajectory dataset for training"""
    def __init__(self, num_samples=1000, trajectory_len=30, state_dim=6):
        self.num_samples = num_samples
        self.trajectory_len = trajectory_len
        self.state_dim = state_dim
        self.data = self._generate_dataset()
    
    def _generate_dataset(self):
        """Generate synthetic trajectory dataset"""
        data = []
        for _ in range(self.num_samples):
            # Generate smooth trajectories using sinusoidal patterns
            trajectory = np.zeros((self.trajectory_len, self.state_dim))
            
            # Position components (smooth curves)
            for i in range(3):  # x, y, z
                freq = np.random.uniform(0.5, 2.0)
                phase = np.random.uniform(0, 2*np.pi)
                amplitude = np.random.uniform(1.0, 3.0)
                
                for t in range(self.trajectory_len):
                    trajectory[t, i] = amplitude * np.sin(freq * t * 0.2 + phase)
            
            # Velocity components (derivatives of position)
            for i in range(3):
                for t in range(1, self.trajectory_len-1):
                    trajectory[t, i+3] = (trajectory[t+1, i] - trajectory[t-1, i]) / 0.2  # Finite difference
            
            # Add some noise
            trajectory += np.random.normal(0, 0.1, trajectory.shape)
            data.append(trajectory)
        
        return np.array(data)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])

class SimpleDiffusionModel(nn.Module):
    """Simplified Diffusion Model with improved architecture"""
    def __init__(self, trajectory_len=50, state_dim=6, hidden_dim=256):
        super().__init__()
        self.trajectory_len = trajectory_len
        self.state_dim = state_dim
        
        # Improved time step embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 512)
        )
        
        # Positional encoding for trajectory points
        self.positional_encoding = nn.Parameter(torch.randn(1, trajectory_len, 64))
        
        # Main network with residual connections
        self.input_projection = nn.Linear(state_dim + 512 + 64, hidden_dim)
        
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(4)
        ])
        
        self.output_projection = nn.Linear(hidden_dim, state_dim)
        
        self.activation = nn.SiLU()
        
    def forward(self, x, t):
        """
        Args:
            x: Input trajectory [batch, trajectory_len, state_dim]
            t: Time step [batch]
        """
        batch_size, trajectory_len, state_dim = x.shape
        
        # Time embedding
        t_embed = self.time_embedding(t.float().unsqueeze(-1))  # [batch, 512]
        t_embed = t_embed.unsqueeze(1).repeat(1, trajectory_len, 1)  # [batch, trajectory_len, 512]
        
        # Positional encoding
        pos_embed = self.positional_encoding.repeat(batch_size, 1, 1)  # [batch, trajectory_len, 64]
        
        # Concatenate inputs
        x_combined = torch.cat([x, t_embed, pos_embed], dim=-1)  # [batch, trajectory_len, state_dim+512+64]
        
        # Project input
        h = self.input_projection(x_combined)
        h = self.activation(h)
        
        # Residual blocks
        for residual_block in self.residual_blocks:
            residual = residual_block(h)
            h = self.activation(h + residual)
        
        # Output projection
        output = self.output_projection(h)
        
        return output

class SafeTrajectoryDiffusion:
    def __init__(self, trajectory_len=50, state_dim=6, num_obstacles=3, device='cuda'):
        """
        Safe Trajectory Diffusion Model
        
        Args:
            trajectory_len: Trajectory length (n)
            state_dim: Dimension of each point (m=6: x,y,z,vx,vy,vz)
            num_obstacles: Number of obstacles (K)
            device: Computing device
        """
        self.trajectory_len = trajectory_len
        self.state_dim = state_dim
        self.num_obstacles = num_obstacles
        self.device = device
        
        # Generate random obstacles
        self.obstacles = self._generate_obstacles()
        
        # Improved diffusion model
        self.diffusion_model = SimpleDiffusionModel(
            trajectory_len=trajectory_len,
            state_dim=state_dim
        ).to(device)
        
        # Diffusion parameters
        self.num_diffusion_steps = 1000
        self.betas = torch.linspace(1e-4, 0.02, self.num_diffusion_steps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Training parameters
        self.optimizer = optim.AdamW(self.diffusion_model.parameters(), lr=1e-4, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        
    def _generate_obstacles(self):
        """Generate random spherical obstacles"""
        obstacles = []
        for i in range(self.num_obstacles):
            obstacle = {
                'position': np.random.uniform(-3, 3, 3),  # Obstacle center position
                'radius': np.random.uniform(0.5, 1.5)     # Obstacle radius
            }
            obstacles.append(obstacle)
        return obstacles
    
    def train_step(self, clean_trajectories):
        """
        Single training step for diffusion model
        
        Args:
            clean_trajectories: Clean trajectory samples [batch_size, trajectory_len, state_dim]
        """
        batch_size = clean_trajectories.shape[0]
        
        # Sample random diffusion steps
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,)).to(self.device)
        
        # Sample random noise
        noise = torch.randn_like(clean_trajectories)
        
        # Add noise to clean trajectories
        alpha_bars_t = self.alpha_bars[t].view(-1, 1, 1)
        noisy_trajectories = torch.sqrt(alpha_bars_t) * clean_trajectories + torch.sqrt(1 - alpha_bars_t) * noise
        
        # Predict noise
        noise_pred = self.diffusion_model(noisy_trajectories, t)
        
        # Calculate loss
        loss = self.criterion(noise_pred, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataloader, num_epochs=100, save_interval=10):
        """
        Full training procedure
        
        Args:
            dataloader: DataLoader for training data
            num_epochs: Number of training epochs
            save_interval: Interval for saving model checkpoints
        """
        print("Starting training...")
        self.diffusion_model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, clean_trajectories in enumerate(dataloader):
                clean_trajectories = clean_trajectories.to(self.device)
                
                # Training step
                loss = self.train_step(clean_trajectories)
                epoch_loss += loss
                num_batches += 1
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} | '
                          f'Batch: {batch_idx:03d}/{len(dataloader):03d} | '
                          f'Loss: {loss:.6f}')
            
            # Update learning rate
            self.scheduler.step()
            
            # Record average epoch loss
            avg_epoch_loss = epoch_loss / num_batches
            self.train_losses.append(avg_epoch_loss)
            
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} | '
                  f'Average Loss: {avg_epoch_loss:.6f} | '
                  f'LR: {self.scheduler.get_last_lr()[0]:.2e}')
            
            # Save model checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
        
        print("Training completed!")
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.diffusion_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'obstacles': self.obstacles
        }
        torch.save(checkpoint, f'diffusion_checkpoint_epoch_{epoch}.pth')
        print(f"Checkpoint saved for epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.obstacles = checkpoint['obstacles']
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    def plot_training_loss(self):
        """Plot training loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Diffusion Model Training Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.show()
    
    # CBF constraint methods (same as before, but included for completeness)
    def cbf_velocity_constraint(self, trajectory, max_velocity=5.0):
        """CBF velocity constraint"""
        velocities = trajectory[..., 3:]
        velocity_norms = torch.norm(velocities, dim=-1)
        constraint_value = max_velocity**2 - velocity_norms**2
        gradient = torch.zeros_like(trajectory)
        gradient[..., 3:] = -2 * velocities
        return constraint_value, gradient
    
    def cbf_obstacle_constraint_robust(self, trajectory, safety_margin=0.2):
        """Robust obstacle constraint"""
        batch_size = trajectory.shape[0]
        constraint_value = torch.zeros(batch_size, self.trajectory_len, self.num_obstacles).to(self.device)
        gradient = torch.zeros_like(trajectory)
        
        for k, obstacle in enumerate(self.obstacles):
            positions = trajectory[..., :3]
            obstacle_pos = torch.tensor(obstacle['position']).to(self.device)
            radius = obstacle['radius']
            
            diff = positions - obstacle_pos
            distances = torch.norm(diff, dim=-1)
            constraint_value[..., k] = distances - (radius + safety_margin)
            
            safe_distances = distances.clone()
            safe_distances[safe_distances < 1e-6] = 1e-6
            grad_contrib = diff / safe_distances.unsqueeze(-1)
            gradient[..., :3] += grad_contrib
        
        return constraint_value, gradient
    
    def cbf_obstacle_constraint_relaxed(self, trajectory, diffusion_step, safety_margin=0.2):
        """Relaxed obstacle constraint"""
        time_weight = 1.0 - (diffusion_step / self.num_diffusion_steps)
        relaxation = 0.5 * time_weight
        constraint_value, gradient = self.cbf_obstacle_constraint_robust(trajectory, safety_margin)
        constraint_value = constraint_value + relaxation
        return constraint_value, gradient, relaxation
    
    def cbf_obstacle_constraint_timevarying(self, trajectory, diffusion_step, safety_margin=0.2):
        """Time-varying obstacle constraint"""
        time_varying_margin = safety_margin * (1.0 - diffusion_step / self.num_diffusion_steps) + 0.1
        constraint_value, gradient = self.cbf_obstacle_constraint_robust(trajectory, time_varying_margin)
        return constraint_value, gradient, time_varying_margin
    
    def qp_safety_correction(self, trajectory, diffusion_step, cbf_type='robust'):
        """QP safety correction"""
        trajectory_flat = trajectory.view(-1)
        original_trajectory = trajectory.clone()
        
        def objective(x):
            x_tensor = torch.tensor(x, dtype=torch.float32).view_as(trajectory).to(self.device)
            return torch.norm(x_tensor - original_trajectory).item()
        
        def constraints(x):
            x_tensor = torch.tensor(x, dtype=torch.float32).view_as(trajectory).to(self.device)
            constraint_values = []
            
            vel_constraint, _ = self.cbf_velocity_constraint(x_tensor)
            constraint_values.extend(vel_constraint.view(-1).cpu().numpy())
            
            if cbf_type == 'robust':
                obs_constraint, _ = self.cbf_obstacle_constraint_robust(x_tensor)
            elif cbf_type == 'relaxed':
                obs_constraint, _, _ = self.cbf_obstacle_constraint_relaxed(x_tensor, diffusion_step)
            elif cbf_type == 'timevarying':
                obs_constraint, _, _ = self.cbf_obstacle_constraint_timevarying(x_tensor, diffusion_step)
            
            constraint_values.extend(obs_constraint.view(-1).cpu().numpy())
            return np.array(constraint_values)
        
        result = minimize(
            objective,
            trajectory_flat.cpu().numpy(),
            constraints={'type': 'ineq', 'fun': constraints},
            method='SLSQP',
            options={'maxiter': 50}
        )
        
        if result.success:
            return torch.tensor(result.x, dtype=torch.float32).view_as(trajectory).to(self.device)
        else:
            print("QP solution failed, returning original trajectory")
            return original_trajectory
    
    def generate_trajectory(self, batch_size=1, cbf_type='robust'):
        """Generate safe trajectory"""
        self.diffusion_model.eval()
        
        # Start from Gaussian noise
        x = torch.randn(batch_size, self.trajectory_len, self.state_dim).to(self.device)
        
        # Reverse diffusion process
        for step in range(self.num_diffusion_steps - 1, -1, -1):
            with torch.no_grad():
                noise_pred = self.diffusion_model(x, torch.tensor([step]).to(self.device))
            
            alpha_bar = self.alpha_bars[step]
            if step > 0:
                alpha_bar_prev = self.alpha_bars[step-1]
            else:
                alpha_bar_prev = torch.tensor(1.0)
            
            # DDPM sampling
            x = (1 / torch.sqrt(self.alphas[step])) * (
                x - ((1 - self.alphas[step]) / torch.sqrt(1 - alpha_bar)) * noise_pred
            )
            
            if step > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(1 - alpha_bar_prev) * noise
            
            # Apply CBF safety correction in last steps
            if step < 50:
                x = self.qp_safety_correction(x, step, cbf_type)
        
        return x.cpu().numpy()

def visualize_trajectory(trajectory, obstacles, title="Safe Trajectory"):
    """Visualize trajectory and obstacles"""
    fig = plt.figure(figsize=(12, 10))
    
    ax1 = fig.add_subplot(221, projection='3d')
    positions = trajectory[0, :, :3]
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Trajectory')
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    
    for i, obstacle in enumerate(obstacles):
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = obstacle['radius'] * np.outer(np.cos(u), np.sin(v)) + obstacle['position'][0]
        y = obstacle['radius'] * np.outer(np.sin(u), np.sin(v)) + obstacle['position'][1]
        z = obstacle['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle['position'][2]
        ax1.plot_surface(x, y, z, color='r', alpha=0.3)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'{title} - 3D View')
    ax1.legend()
    
    ax2 = fig.add_subplot(222)
    velocities = np.linalg.norm(trajectory[0, :, 3:], axis=1)
    ax2.plot(velocities, 'g-', linewidth=2)
    ax2.axhline(y=5.0, color='r', linestyle='--', label='Velocity Limit')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Velocity')
    ax2.set_title('Velocity Profile')
    ax2.legend()
    ax2.grid(True)
    
    ax3 = fig.add_subplot(223)
    ax3.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
    ax3.scatter(positions[0, 0], positions[0, 1], c='g', s=100)
    ax3.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100)
    
    for obstacle in obstacles:
        circle = plt.Circle(obstacle['position'][:2], obstacle['radius'], color='r', alpha=0.3)
        ax3.add_patch(circle)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('XY Plane Projection')
    ax3.axis('equal')
    
    ax4 = fig.add_subplot(224)
    ax4.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2)
    ax4.scatter(positions[0, 0], positions[0, 2], c='g', s=100)
    ax4.scatter(positions[-1, 0], positions[-1, 2], c='r', s=100)
    
    for obstacle in obstacles:
        circle = plt.Circle(obstacle['position'][::2], obstacle['radius'], color='r', alpha=0.3)
        ax4.add_patch(circle)
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Z')
    ax4.set_title('XZ Plane Projection')
    ax4.axis('equal')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = TrajectoryDataset(num_samples=5000, trajectory_len=30, state_dim=6)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Initialize model
    diffusion_model = SafeTrajectoryDiffusion(
        trajectory_len=30,
        state_dim=6,
        num_obstacles=3,
        device=device
    )
    
    # Option 1: Train new model
    print("Starting training...")
    start_time = time.time()
    diffusion_model.train(dataloader, num_epochs=100, save_interval=20)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Option 2: Load pre-trained model (uncomment to use)
    # diffusion_model.load_checkpoint('diffusion_checkpoint_epoch_100.pth')
    
    # Plot training loss
    diffusion_model.plot_training_loss()
    
    # Generate and visualize trajectories with different CBF constraints
    cbf_types = ['robust', 'relaxed', 'timevarying']
    
    for cbf_type in cbf_types:
        print(f"\nGenerating trajectory using {cbf_type} CBF constraint...")
        
        trajectory = diffusion_model.generate_trajectory(
            batch_size=1,
            cbf_type=cbf_type
        )
        
        # Validate constraints
        trajectory_tensor = torch.tensor(trajectory).to(diffusion_model.device)
        
        vel_constraint, _ = diffusion_model.cbf_velocity_constraint(trajectory_tensor)
        vel_violation = torch.sum(vel_constraint < 0).item()
        print(f"Velocity constraint violation points: {vel_violation}")
        
        obs_constraint, _ = diffusion_model.cbf_obstacle_constraint_robust(trajectory_tensor)
        obs_violation = torch.sum(obs_constraint < 0).item()
        print(f"Obstacle constraint violation points: {obs_violation}")
        
        visualize_trajectory(trajectory, diffusion_model.obstacles, 
                           title=f"{cbf_type.capitalize()} CBF Constrained Trajectory")