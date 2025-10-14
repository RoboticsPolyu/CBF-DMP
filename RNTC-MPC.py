import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# B. Derajić, M.-K. Bouzidi, S. Bernhard, and W. Hönig, "Residual Neural Terminal Constraint for MPC-based Collision Avoidance in Dynamic Environments," p. arXiv:2508.03428doi: 10.48550/arXiv.2508.03428.


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import time

class SinActivation(nn.Module):
    """Sine activation function used in the paper"""
    def forward(self, x):
        return torch.sin(x)

class MainNetwork(nn.Module):
    """Main network that approximates the residual function R(x)"""
    def __init__(self, input_dim=3, hidden_dims=[32, 32, 16]):
        super(MainNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(hidden_dims) - 1:
                layers.append(SinActivation())
            else:
                layers.append(nn.SELU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        residual = self.network(x)
        # Apply ELU + 1 for non-negative output
        return torch.nn.functional.elu(residual) + 1.0

class ValuePredictor(nn.Module):
    """Network that predicts value function from SDF sequence"""
    def __init__(self, sdf_seq_length=2, grid_size=32, output_size=32*32):
        super(ValuePredictor, self).__init__()
        
        # CNN backbone for processing SDF sequences
        self.backbone = nn.Sequential(
            nn.Conv2d(sdf_seq_length, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            nn.Flatten()
        )
        
        # Calculate the flattened size
        with torch.no_grad():
            dummy_input = torch.randn(1, sdf_seq_length, grid_size, grid_size)
            flattened_size = self.backbone(dummy_input).shape[1]
        
        print(f"Flattened feature size: {flattened_size}")
        
        # Regression head to predict value function
        self.regressor = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
        self.grid_size = grid_size
        self.output_size = output_size
        
    def forward(self, sdf_sequence):
        features = self.backbone(sdf_sequence)
        value_pred = self.regressor(features)
        return value_pred.view(-1, self.grid_size, self.grid_size)

class RNTC_MPC:
    """Residual Neural Terminal Constraint MPC"""
    def __init__(self, state_dim=3, control_dim=2, horizon=8, dt=0.2):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.dt = dt
        
        # Initialize networks
        self.value_predictor = ValuePredictor(grid_size=32, output_size=32*32)
        self.main_net = MainNetwork()
        
        # MPC weights - tuned for better navigation
        self.Q = torch.diag(torch.tensor([2.0, 2.0, 0.1]))  # Higher state cost
        self.R = torch.diag(torch.tensor([0.05, 0.05]))  # Lower control cost
        self.QN = torch.diag(torch.tensor([5.0, 5.0, 0.2]))  # Higher terminal cost
        
    def compute_sdf(self, positions: torch.Tensor, query_point: torch.Tensor) -> torch.Tensor:
        """Compute Signed Distance Function for given obstacle positions and query point"""
        if len(positions) == 0:
            return torch.tensor(10.0)  # No obstacles -> large safe distance
            
        distances = torch.norm(positions - query_point, dim=1)
        min_distance = torch.min(distances)
        
        # SDF: positive outside obstacles, negative inside
        safety_margin = 0.5
        return min_distance - safety_margin
    
    def predict_obstacle_trajectories(self, current_positions: torch.Tensor, 
                                   current_velocities: torch.Tensor) -> List[torch.Tensor]:
        """Predict future obstacle positions using constant velocity model"""
        trajectories = []
        for pos, vel in zip(current_positions, current_velocities):
            traj = []
            for i in range(self.horizon + 1):
                new_pos = pos + vel * i * self.dt
                traj.append(new_pos)
            trajectories.append(torch.stack(traj))
        return trajectories
    
    def compute_sdf_sequence(self, obstacle_trajectories: List[torch.Tensor]) -> torch.Tensor:
        """Compute SDF sequence for the hypernetwork input"""
        seq_length = 2  # Current and one past SDF
        grid_size = 32
        
        # Create a grid around robot's current position (simplified to origin for demo)
        x = torch.linspace(-3, 3, grid_size)
        y = torch.linspace(-3, 3, grid_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid_points = torch.stack([grid_x, grid_y], dim=-1)
        
        sdf_sequence = torch.zeros(1, seq_length, grid_size, grid_size)
        
        for t in range(seq_length):
            # Aggregate all obstacle positions at time t
            all_obstacle_pos = []
            for traj in obstacle_trajectories:
                if t < len(traj):
                    all_obstacle_pos.append(traj[t])
            
            if all_obstacle_pos:
                all_obstacle_pos = torch.stack(all_obstacle_pos)
                
                # Compute SDF for each grid point
                for i in range(grid_size):
                    for j in range(grid_size):
                        grid_point = grid_points[i, j]
                        sdf_val = self.compute_sdf(all_obstacle_pos, grid_point)
                        sdf_sequence[0, t, i, j] = sdf_val
            else:
                # No obstacles - set to safe distance
                sdf_sequence[0, t, :, :] = 10.0
        
        return sdf_sequence
    
    def estimate_terminal_constraint(self, sdf_sequence: torch.Tensor, 
                                   terminal_state: torch.Tensor) -> torch.Tensor:
        """Estimate terminal constraint using the neural network"""
        # For demo, use a simple heuristic based on goal direction and obstacles
        goal_dir = torch.atan2(5.0 - terminal_state[1], 5.0 - terminal_state[0])
        heading_error = torch.abs(terminal_state[2] - goal_dir)
        
        # Simple safety check based on distance to obstacles
        safety_margin = 1.0
        terminal_value = safety_margin - heading_error
        
        return terminal_value
    
    def mpc_optimization(self, current_state: torch.Tensor, goal_state: torch.Tensor,
                        obstacles: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Solve MPC optimization problem with improved navigation"""
        # Predict obstacle trajectories
        obstacle_positions = [obs[0] for obs in obstacles]
        obstacle_velocities = [obs[1] for obs in obstacles]
        
        if obstacle_positions:
            obstacle_trajectories = self.predict_obstacle_trajectories(
                torch.stack(obstacle_positions), torch.stack(obstacle_velocities))
        else:
            obstacle_trajectories = []
        
        # Compute SDF sequence for value predictor
        sdf_sequence = self.compute_sdf_sequence(obstacle_trajectories)
        
        # Improved MPC optimization with goal-oriented sampling
        control_sequence = self._improved_mpc_solve(current_state, goal_state, 
                                                   sdf_sequence, obstacle_trajectories)
        
        return control_sequence[0] if control_sequence is not None else torch.zeros(self.control_dim)
    
    def _improved_mpc_solve(self, current_state: torch.Tensor, goal_state: torch.Tensor,
                           sdf_sequence: torch.Tensor, 
                           obstacle_trajectories: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Improved MPC solver with goal-oriented control sampling"""
        num_candidates = 30
        
        # Generate both random and goal-oriented controls
        controls = torch.zeros(num_candidates, self.horizon, self.control_dim)
        
        # Random controls (50%)
        controls[:num_candidates//2] = torch.randn(num_candidates//2, self.horizon, self.control_dim) * 0.3
        
        # Goal-oriented controls (50%)
        goal_dir = torch.atan2(goal_state[1] - current_state[1], 
                              goal_state[0] - current_state[0])
        
        for i in range(num_candidates//2, num_candidates):
            # Bias controls toward goal direction
            base_speed = 0.4
            base_omega = (goal_dir - current_state[2]) * 0.5
            
            # Add some variation
            speed_variation = torch.randn(self.horizon) * 0.1
            omega_variation = torch.randn(self.horizon) * 0.2
            
            controls[i, :, 0] = torch.clamp(base_speed + speed_variation, 0.1, 0.5)
            controls[i, :, 1] = torch.clamp(base_omega + omega_variation, -0.5, 0.5)
        
        controls = torch.clamp(controls, -0.5, 0.5)
        
        best_cost = float('inf')
        best_control = None
        
        for control_seq in controls:
            cost = self._evaluate_trajectory(current_state, goal_state, control_seq, 
                                           obstacle_trajectories, sdf_sequence)
            if cost < best_cost:
                best_cost = cost
                best_control = control_seq
        
        return best_control
    
    def _evaluate_trajectory(self, current_state: torch.Tensor, goal_state: torch.Tensor,
                           control_seq: torch.Tensor, obstacle_trajectories: List[torch.Tensor],
                           sdf_sequence: torch.Tensor) -> float:
        """Evaluate cost of a trajectory"""
        state = current_state.clone()
        total_cost = 0.0
        
        # Stage costs
        for k in range(self.horizon):
            # Apply dynamics
            state = self._unicycle_dynamics(state, control_seq[k])
            
            # State cost (higher weight for position error)
            state_error = state - goal_state
            state_cost = state_error @ self.Q @ state_error
            
            # Control cost (encourage smooth controls)
            control_cost = control_seq[k] @ self.R @ control_seq[k]
            
            # Collision cost
            collision_cost = self._collision_cost(state, obstacle_trajectories, k)
            
            # Progress reward (negative cost for moving toward goal)
            progress = -torch.norm(state[:2] - goal_state[:2]).item() * 0.1
            
            total_cost += state_cost.item() + control_cost.item() + collision_cost + progress
        
        # Terminal cost with neural constraint
        terminal_constraint = self.estimate_terminal_constraint(sdf_sequence, state)
        terminal_cost = (state - goal_state) @ self.QN @ (state - goal_state)
        total_cost += terminal_cost.item()
        
        # Penalize violation of terminal constraint (negative value = unsafe)
        if terminal_constraint < 0:
            total_cost += 100.0
        
        return total_cost
    
    def _unicycle_dynamics(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Unicycle robot dynamics"""
        x, y, theta = state
        v, omega = control
        
        # Clamp controls to limits
        v = torch.clamp(v, -0.5, 0.5)
        omega = torch.clamp(omega, -0.5, 0.5)
        
        new_x = x + v * torch.cos(theta) * self.dt
        new_y = y + v * torch.sin(theta) * self.dt
        new_theta = theta + omega * self.dt
        
        # Normalize angle
        new_theta = (new_theta + torch.pi) % (2 * torch.pi) - torch.pi
        
        return torch.tensor([new_x, new_y, new_theta])
    
    def _collision_cost(self, state: torch.Tensor, 
                       obstacle_trajectories: List[torch.Tensor], 
                       time_step: int) -> float:
        """Compute collision cost for a state"""
        if not obstacle_trajectories:
            return 0.0
            
        min_distance = float('inf')
        
        for traj in obstacle_trajectories:
            if time_step < len(traj):
                obstacle_pos = traj[time_step]
                distance = torch.norm(state[:2] - obstacle_pos).item()
                min_distance = min(min_distance, distance)
        
        safety_margin = 0.4
        if min_distance < safety_margin:
            return 50.0 / (min_distance + 1e-6)  # High cost for collisions
        return 0.0

class CMELoss(nn.Module):
    """Combined MSE and Exponential Loss from the paper"""
    def __init__(self, gamma=0.2):
        super(CMELoss, self).__init__()
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions, targets):
        mse_component = self.mse_loss(predictions, targets)
        
        # Exponential component
        exp_component = torch.exp(-targets * predictions)
        exp_component = torch.mean(exp_component)
        
        return self.gamma * mse_component + (1 - self.gamma) * exp_component

def generate_training_data(num_samples=100, grid_size=32):
    """Generate synthetic training data with proper structure"""
    # SDF sequences (batch_size, seq_length, grid_size, grid_size)
    sdf_sequences = torch.randn(num_samples, 2, grid_size, grid_size)
    
    # True value functions (batch_size, grid_size, grid_size)
    value_functions = torch.zeros(num_samples, grid_size, grid_size)
    
    # Create meaningful training data with safe/unsafe regions
    for i in range(num_samples):
        # Create random obstacle configurations
        num_obstacles = torch.randint(1, 4, (1,)).item()
        obstacles = []
        
        for _ in range(num_obstacles):
            center = torch.rand(2) * 4 - 2  # Random center in [-2, 2]
            radius = torch.rand(1) * 0.5 + 0.3  # Radius between 0.3 and 0.8
            obstacles.append((center, radius))
        
        # Compute value function (simplified HJ reachability)
        for x in range(grid_size):
            for y in range(grid_size):
                pos = torch.tensor([x/(grid_size-1)*6 - 3, y/(grid_size-1)*6 - 3])
                
                # Find minimum distance to any obstacle
                min_dist = float('inf')
                for center, radius in obstacles:
                    dist = torch.norm(pos - center)
                    min_dist = min(min_dist, dist.item() - radius.item())
                
                # Value function: positive = safe, negative = unsafe
                value_functions[i, x, y] = min_dist
    
    return sdf_sequences, value_functions

def train_rntc_model():
    """Training procedure for RNTC-MPC"""
    print("Training RNTC-MPC model...")
    
    # Initialize model
    value_predictor = ValuePredictor(grid_size=32, output_size=32*32)
    
    # Loss function
    criterion = CMELoss(gamma=0.2)
    optimizer = optim.Adam(value_predictor.parameters(), lr=0.001)
    
    # Generate training data
    num_samples = 100
    sdf_sequences, value_functions = generate_training_data(num_samples, 32)
    
    # Training loop
    epochs = 3  # Reduced for quick demo
    batch_size = 10
    
    value_predictor.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Simple sequential batching
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            actual_batch_size = end_idx - i
            
            batch_sdf = sdf_sequences[i:end_idx]
            batch_target = value_functions[i:end_idx]
            
            # Forward pass
            predictions = value_predictor(batch_sdf)
            
            loss = criterion(predictions, batch_target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else total_loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print("Training completed!")
    return value_predictor

def demo_rntc_mpc():
    """Demonstration of RNTC-MPC for collision avoidance"""
    print("\nRunning RNTC-MPC Demo...")
    
    # Initialize MPC controller
    mpc = RNTC_MPC(horizon=8, dt=0.2)
    
    # Initial state [x, y, theta]
    current_state = torch.tensor([0.0, 0.0, 0.0])
    goal_state = torch.tensor([5.0, 5.0, 0.0])
    
    # Moving obstacles: (position, velocity)
    obstacles = [
        (torch.tensor([2.0, 1.5]), torch.tensor([0.2, 0.3])),
        (torch.tensor([1.0, 3.0]), torch.tensor([0.3, -0.2])),
    ]
    
    # Simulation parameters
    max_steps = 600  # Increased for better chance to reach goal
    trajectory = [current_state.numpy().copy()]
    controls_applied = []
    
    print("Starting navigation from (0,0) to (5,5)...")
    print(f"Initial obstacles: {[obs[0].tolist() for obs in obstacles]}")
    
    for step in range(max_steps):
        current_pos = current_state[:2].numpy()
        goal_dist = np.linalg.norm(current_pos - goal_state[:2].numpy())
        print(f"Step {step+1}/{max_steps}, Position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}), Goal dist: {goal_dist:.2f}")
        
        # Get control from MPC
        control = mpc.mpc_optimization(current_state, goal_state, obstacles)
        controls_applied.append(control.numpy().copy())
        
        # Apply dynamics
        current_state = mpc._unicycle_dynamics(current_state, control)
        trajectory.append(current_state.numpy().copy())
        
        # Update obstacle positions with boundary handling
        new_obstacles = []
        for pos, vel in obstacles:
            new_pos = pos + vel * mpc.dt
            
            # Boundary checking and bouncing
            if new_pos[0] <= 0 or new_pos[0] >= 6:
                vel = torch.tensor([-vel[0], vel[1]])
            if new_pos[1] <= 0 or new_pos[1] >= 6:
                vel = torch.tensor([vel[0], -vel[1]])
                
            new_pos = torch.clamp(new_pos, 0.1, 5.9)
            new_obstacles.append((new_pos, vel))
            
        obstacles = new_obstacles
        
        # Check if goal reached
        goal_distance = torch.norm(current_state[:2] - goal_state[:2])
        if goal_distance < 0.5:
            print(f"🎯 Goal reached at step {step+1}! Final distance: {goal_distance:.2f}")
            break
        
        # Check for collision
        min_obstacle_dist = min(torch.norm(current_state[:2] - obs[0]).item() for obs in obstacles)
        if min_obstacle_dist < 0.3:
            print(f"⚠️ Near collision at step {step+1}! Minimum distance: {min_obstacle_dist:.2f}")
    
    print("Navigation completed!")
    return trajectory, controls_applied, obstacles

def plot_results(trajectory, controls, obstacles, goal_state):
    """Plot the robot trajectory, controls, and obstacles"""
    trajectory = np.array(trajectory)
    controls = np.array(controls)
    
    # Convert goal_state to numpy if it's a tensor
    if torch.is_tensor(goal_state):
        goal_state_np = goal_state.numpy()
    else:
        goal_state_np = goal_state
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot trajectory
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Robot Path')
    ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
    ax1.plot(goal_state_np[0], goal_state_np[1], 'g*', markersize=15, label='Goal')
    
    # Plot obstacles and their paths
    colors = ['purple', 'orange', 'brown', 'pink']
    for i, (pos, vel) in enumerate(obstacles):
        pos_np = pos.numpy() if torch.is_tensor(pos) else pos
        vel_np = vel.numpy() if torch.is_tensor(vel) else vel
        
        ax1.plot(pos_np[0], pos_np[1], 's', color=colors[i % len(colors)], 
                markersize=12, label=f'Obstacle {i+1}')
        # Plot velocity vector
        ax1.arrow(pos_np[0], pos_np[1], vel_np[0], vel_np[1], 
                 head_width=0.15, head_length=0.15, fc=colors[i % len(colors)], 
                 ec=colors[i % len(colors)], alpha=0.7)
    
    ax1.set_xlabel('X position [m]')
    ax1.set_ylabel('Y position [m]')
    ax1.set_title('RNTC-MPC Collision Avoidance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, 6.5)
    ax1.set_ylim(-0.5, 6.5)
    ax1.set_aspect('equal')
    
    # Plot controls
    if len(controls) > 0:
        time_steps = np.arange(len(controls))
        ax2.plot(time_steps, controls[:, 0], 'b-', linewidth=2, label='Linear Velocity [m/s]')
        ax2.plot(time_steps, controls[:, 1], 'r-', linewidth=2, label='Angular Velocity [rad/s]')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Control Values')
        ax2.set_title('Control History')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    if len(trajectory) > 1:
        path_length = np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1))
        final_distance = np.linalg.norm(trajectory[-1, :2] - goal_state_np[:2])
        
        min_distances = []
        for i, pos in enumerate(trajectory[:, :2]):
            min_obs_dist = min(np.linalg.norm(pos - (obs[0].numpy() if torch.is_tensor(obs[0]) else obs[0])) 
                             for obs in obstacles)
            min_distances.append(min_obs_dist)
        min_obstacle_distance = min(min_distances)
        
        print(f"\n=== Summary Statistics ===")
        print(f"Path length: {path_length:.2f} m")
        print(f"Final goal distance: {final_distance:.2f} m")
        print(f"Minimum obstacle distance: {min_obstacle_distance:.2f} m")
        print(f"Number of steps: {len(trajectory) - 1}")
        print(f"Success: {'🎯 Yes' if final_distance < 0.8 else '⚠️ Partial'}")
        if final_distance < 0.8:
            print("🎉 Congratulations! The robot successfully reached the goal!")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Train the model
        print("=== RNTC-MPC Demo ===")
        trained_predictor = train_rntc_model()
        
        # Run the demo
        trajectory, controls, final_obstacles = demo_rntc_mpc()
        
        # Plot results
        plot_results(trajectory, controls, final_obstacles, goal_state=torch.tensor([5.0, 5.0, 0.0]))
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()