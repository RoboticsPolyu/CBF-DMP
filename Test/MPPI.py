import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import time

class EfficientMPPIPathPlanner:
    def __init__(self, config):
        self.config = config
        self.state_dim = 4  # [x, y, vx, vy]
        self.control_dim = 2  # [ax, ay]
        
    def dynamics(self, state, control):
        """Simple double integrator dynamics"""
        dt = self.config['dt']
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        B = np.array([
            [0.5*dt**2, 0],
            [0, 0.5*dt**2],
            [dt, 0],
            [0, dt]
        ])
        return A @ state + B @ control
    
    def cost_function(self, trajectory, goal, obstacles):
        """Calculate total cost for a trajectory"""
        states = trajectory['states']
        controls = trajectory['controls']
        
        # Goal cost (quadratic distance to goal)
        goal_cost = 0
        for i, state in enumerate(states):
            # Progressive weighting - higher weight for later states
            weight_multiplier = 1.0 + (i / len(states)) * 2.0
            dist_to_goal = np.linalg.norm(state[:2] - goal)
            goal_cost += self.config['goal_weight'] * weight_multiplier * dist_to_goal**2
        
        # Control cost
        control_cost = 0
        for control in controls:
            control_cost += self.config['control_weight'] * np.sum(control**2)
        
        # Obstacle cost
        obstacle_cost = 0
        for state in states:
            for obstacle in obstacles:
                pos = state[:2]
                dist = self.distance_to_box(pos, obstacle)
                
                if dist < 0:  # Inside obstacle
                    obstacle_cost += self.config['obstacle_weight'] * 1000
                elif dist < self.config['safety_margin']:
                    # Smooth penalty that increases as distance decreases
                    penalty = (self.config['safety_margin'] - dist) ** 2
                    obstacle_cost += self.config['obstacle_weight'] * penalty
        
        # Terminal cost
        terminal_cost = self.config['terminal_weight'] * np.linalg.norm(states[-1][:2] - goal)**2
        
        # Path length cost (encourage shorter paths)
        path_length = 0
        for i in range(1, len(states)):
            path_length += np.linalg.norm(states[i][:2] - states[i-1][:2])
        path_cost = self.config['path_weight'] * path_length
        
        return goal_cost + control_cost + obstacle_cost + terminal_cost + path_cost
    
    def distance_to_box(self, point, box):
        """Calculate minimum distance from point to box obstacle"""
        x, y = point
        x_min, y_min, x_max, y_max = box
        
        # If point is inside box, return negative distance
        if x_min <= x <= x_max and y_min <= y <= y_max:
            dx = min(x - x_min, x_max - x)
            dy = min(y - y_min, y_max - y)
            return -min(dx, dy)
        
        # Find closest point on box boundary
        closest_x = max(x_min, min(x, x_max))
        closest_y = max(y_min, min(y, y_max))
        
        return np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
    
    def generate_guided_rollouts(self, initial_state, nominal_controls, goal, obstacles):
        """Generate trajectories with intelligent sampling strategies"""
        K = self.config['num_samples']
        T = self.config['horizon']
        
        all_trajectories = []
        
        # Split samples into different strategies
        num_exploratory = int(K * 0.3)  # 30% exploratory
        num_goal_directed = int(K * 0.4)  # 40% goal-directed
        num_refinement = K - num_exploratory - num_goal_directed  # 30% refinement
        
        # Strategy 1: Exploratory sampling (wide exploration)
        for k in range(num_exploratory):
            states = [initial_state.copy()]
            controls = []
            current_state = initial_state.copy()
            
            # Higher noise for exploration
            exploration_noise = self.config['noise_sigma'] * 1.5 * np.random.randn(T, self.control_dim)
            
            for t in range(T):
                if t < len(nominal_controls):
                    control = nominal_controls[t] + exploration_noise[t]
                else:
                    control = exploration_noise[t]
                
                control = np.clip(control, 
                                -self.config['control_limits'], 
                                self.config['control_limits'])
                
                next_state = self.dynamics(current_state, control)
                states.append(next_state)
                controls.append(control)
                current_state = next_state
            
            all_trajectories.append({
                'states': states,
                'controls': controls,
                'noise': exploration_noise,
                'type': 'exploratory'
            })
        
        # Strategy 2: Goal-directed sampling
        goal_direction = goal - initial_state[:2]
        goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-6)
        
        for k in range(num_goal_directed):
            states = [initial_state.copy()]
            controls = []
            current_state = initial_state.copy()
            
            # Bias noise toward goal direction
            base_noise = self.config['noise_sigma'] * np.random.randn(T, self.control_dim)
            
            # Add goal bias
            for t in range(T):
                goal_bias = 0.3 * np.array([goal_direction[0], goal_direction[1]])
                biased_noise = base_noise[t] + goal_bias
                
                if t < len(nominal_controls):
                    control = nominal_controls[t] + biased_noise
                else:
                    control = biased_noise
                
                control = np.clip(control, 
                                -self.config['control_limits'], 
                                self.config['control_limits'])
                
                next_state = self.dynamics(current_state, control)
                states.append(next_state)
                controls.append(control)
                current_state = next_state
            
            all_trajectories.append({
                'states': states,
                'controls': controls,
                'noise': base_noise,
                'type': 'goal_directed'
            })
        
        # Strategy 3: Refinement sampling (low noise around current best)
        for k in range(num_refinement):
            states = [initial_state.copy()]
            controls = []
            current_state = initial_state.copy()
            
            # Lower noise for refinement
            refinement_noise = self.config['noise_sigma'] * 0.5 * np.random.randn(T, self.control_dim)
            
            for t in range(T):
                if t < len(nominal_controls):
                    control = nominal_controls[t] + refinement_noise[t]
                else:
                    control = refinement_noise[t]
                
                control = np.clip(control, 
                                -self.config['control_limits'], 
                                self.config['control_limits'])
                
                next_state = self.dynamics(current_state, control)
                states.append(next_state)
                controls.append(control)
                current_state = next_state
            
            all_trajectories.append({
                'states': states,
                'controls': controls,
                'noise': refinement_noise,
                'type': 'refinement'
            })
        
        return all_trajectories
    
    def adaptive_noise_sigma(self, iteration, max_iterations, initial_cost, current_cost):
        """Adaptively adjust noise based on performance"""
        base_sigma = self.config['noise_sigma']
        
        # Reduce noise as we converge
        iteration_factor = 1.0 - (iteration / max_iterations) * 0.7
        
        # Increase noise if we're stuck
        if iteration > 10 and current_cost > initial_cost * 0.8:
            stuck_factor = 1.5
        else:
            stuck_factor = 1.0
            
        return base_sigma * iteration_factor * stuck_factor
    
    def update_controls(self, trajectories, costs, nominal_controls):
        """Update nominal controls using MPPI update rule"""
        T = self.config['horizon']
        costs = np.array(costs)
        
        # Find minimum cost for numerical stability
        min_cost = np.min(costs)
        cost_range = np.max(costs) - min_cost
        
        if cost_range > 1e-6:
            # Shift and scale costs
            shifted_costs = costs - min_cost
            normalized_costs = shifted_costs / cost_range
        else:
            normalized_costs = np.ones_like(costs) * 0.5
        
        # Compute weights with numerical stability
        max_cost_for_exp = np.max(normalized_costs) if np.max(normalized_costs) > 0 else 1.0
        exp_values = np.exp(-self.config['temperature'] * normalized_costs / max_cost_for_exp)
        
        # Handle numerical issues
        if np.any(np.isnan(exp_values)) or np.any(np.isinf(exp_values)):
            exp_values = np.ones_like(costs)
            
        weights = exp_values / (np.sum(exp_values) + 1e-10)
        
        # Initialize new nominal controls
        new_nominal_controls = np.zeros((T, self.control_dim))
        
        # Weighted average of controls
        for t in range(T):
            for k in range(len(trajectories)):
                new_nominal_controls[t] += weights[k] * trajectories[k]['controls'][t]
        
        return new_nominal_controls, weights
    
    def plan_path(self, start, goal, obstacles, max_iterations=50):
        """Main planning function"""
        # Initialize state and controls
        initial_state = np.array([start[0], start[1], 0, 0])
        nominal_controls = np.zeros((self.config['horizon'], self.control_dim))
        
        best_trajectory = None
        best_cost = float('inf')
        all_iterations_data = []
        
        initial_cost = None
        
        for iteration in range(max_iterations):
            # Adaptive noise
            current_noise_sigma = self.adaptive_noise_sigma(
                iteration, max_iterations, 
                initial_cost if initial_cost is not None else 1000, 
                best_cost
            )
            self.config['noise_sigma'] = current_noise_sigma
            
            # Generate guided rollouts
            trajectories = self.generate_guided_rollouts(
                initial_state, nominal_controls, goal, obstacles
            )
            
            # Evaluate costs
            costs = []
            for traj in trajectories:
                cost = self.cost_function(traj, goal, obstacles)
                costs.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_trajectory = traj
            
            if initial_cost is None:
                initial_cost = np.mean(costs)
            
            # Update controls
            nominal_controls, weights = self.update_controls(trajectories, costs, nominal_controls)
            
            # Store iteration data
            all_iterations_data.append({
                'iteration': iteration,
                'trajectories': trajectories,
                'costs': costs,
                'weights': weights,
                'best_trajectory': best_trajectory.copy() if best_trajectory else trajectories[0].copy(),
                'nominal_controls': nominal_controls.copy(),
                'noise_sigma': current_noise_sigma
            })
            
            # Print progress
            if iteration % 5 == 0:
                avg_cost = np.mean(costs)
                std_cost = np.std(costs)
                print(f"Iteration {iteration}, Best: {best_cost:.2f}, Avg: {avg_cost:.2f} ± {std_cost:.2f}, Noise: {current_noise_sigma:.3f}")
            
            # Early stopping criteria
            if (np.linalg.norm(best_trajectory['states'][-1][:2] - goal) < 0.3 and 
                best_cost < 10.0):
                print(f"Converged at iteration {iteration}")
                break
        
        return best_trajectory, all_iterations_data

def plot_sampling_strategies(iteration_data, start, goal, obstacles, iteration_num):
    """Plot trajectories colored by sampling strategy"""
    trajectories = iteration_data['trajectories']
    
    plt.figure(figsize=(15, 5))
    
    # Color map for different strategies
    strategy_colors = {
        'exploratory': 'red',
        'goal_directed': 'blue', 
        'refinement': 'green'
    }
    
    # Plot 1: Color by strategy
    plt.subplot(1, 3, 1)
    
    # Plot obstacles
    for obstacle in obstacles:
        x_min, y_min, x_max, y_max = obstacle
        plt.fill_between([x_min, x_max], y_min, y_max, color='gray', alpha=0.3)
    
    # Plot start and goal
    plt.plot(start[0], start[1], 'ko', markersize=12, label='Start', markeredgecolor='black')
    plt.plot(goal[0], goal[1], 'k*', markersize=15, label='Goal', markeredgecolor='black')
    
    # Plot trajectories by strategy
    for traj in trajectories:
        states = traj['states']
        path_x = [state[0] for state in states]
        path_y = [state[1] for state in states]
        
        color = strategy_colors.get(traj['type'], 'black')
        alpha = 0.3 if traj['type'] == 'exploratory' else 0.5
        
        plt.plot(path_x, path_y, color=color, alpha=alpha, linewidth=1, label=traj['type'] if traj['type'] not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.title(f'Iteration {iteration_num}\nSampling Strategies')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    # Avoid duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # Plot 2: Color by cost
    plt.subplot(1, 3, 2)
    
    # Plot obstacles
    for obstacle in obstacles:
        x_min, y_min, x_max, y_max = obstacle
        plt.fill_between([x_min, x_max], y_min, y_max, color='gray', alpha=0.3)
    
    plt.plot(start[0], start[1], 'ko', markersize=12, markeredgecolor='black')
    plt.plot(goal[0], goal[1], 'k*', markersize=15, markeredgecolor='black')
    
    costs = iteration_data['costs']
    min_cost, max_cost = min(costs), max(costs)
    
    for i, traj in enumerate(trajectories):
        states = traj['states']
        path_x = [state[0] for state in states]
        path_y = [state[1] for state in states]
        
        if max_cost > min_cost:
            normalized_cost = (costs[i] - min_cost) / (max_cost - min_cost)
        else:
            normalized_cost = 0.5
            
        color = plt.cm.viridis(normalized_cost)
        plt.plot(path_x, path_y, color=color, alpha=0.6, linewidth=1)
    
    plt.title(f'Iteration {iteration_num}\nCost Coloring\n(Green=Low, Purple=High)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('X')
    
    # Plot 3: Best trajectory and statistics
    plt.subplot(1, 3, 3)
    
    # Plot obstacles
    for obstacle in obstacles:
        x_min, y_min, x_max, y_max = obstacle
        plt.fill_between([x_min, x_max], y_min, y_max, color='gray', alpha=0.3)
    
    plt.plot(start[0], start[1], 'ko', markersize=12, markeredgecolor='black')
    plt.plot(goal[0], goal[1], 'k*', markersize=15, markeredgecolor='black')
    
    # Plot best trajectory
    best_trajectory = iteration_data['best_trajectory']
    best_states = best_trajectory['states']
    best_x = [state[0] for state in best_states]
    best_y = [state[1] for state in best_states]
    plt.plot(best_x, best_y, 'r-', linewidth=3, label='Best Trajectory')
    plt.plot(best_x, best_y, 'ro', markersize=2, alpha=0.6)
    
    # Add statistics
    stats_text = f'Best Cost: {min(costs):.2f}\nAvg Cost: {np.mean(costs):.2f}\nStd Cost: {np.std(costs):.2f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title(f'Iteration {iteration_num}\nBest Trajectory')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('X')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Updated configuration
config = {
    'horizon': 60,  # Reduced horizon for efficiency
    'num_samples': 200,  # Fewer but better samples
    'dt': 0.1,
    'noise_sigma': 0.5,  # Lower base noise
    'temperature': 10.0,  # Higher temperature for better exploration
    'goal_weight': 2.0,
    'control_weight': 0.1,
    'obstacle_weight': 20.0,  # Higher obstacle penalty
    'terminal_weight': 3.0,
    'path_weight': 0.5,  # New: path length cost
    'safety_margin': 0.5,
    'control_limits': 1.5
}

# Define scenario with more obstacles
start_point = np.array([0, 0])
goal_point = np.array([8, 8])

# obstacles = [
#     [2, 1, 3, 6],    # Vertical obstacle
#     [4, 3, 7, 4],    # Horizontal obstacle  
#     [5, 6, 6, 9],    # Square obstacle
#     [1, 7, 3, 8]     # Small obstacle
# ]

obstacles = [
    [2, 1, 3, 6],    # Vertical obstacle
    [4, 3, 7, 4]
]

# Create efficient planner and plan path
planner = EfficientMPPIPathPlanner(config)

print("Starting Efficient MPPI path planning...")
print("Sampling strategies:")
print("- 30% Exploratory (Red): Wide exploration")
print("- 40% Goal-directed (Blue): Biased toward goal")  
print("- 30% Refinement (Green): Low-noise refinement")

start_time = time.time()
best_trajectory, all_iterations_data = planner.plan_path(start_point, goal_point, obstacles, max_iterations=15)
end_time = time.time()

print(f"\nPlanning completed in {end_time - start_time:.2f} seconds")
print(f"Final position: {best_trajectory['states'][-1][:2]}")
print(f"Distance to goal: {np.linalg.norm(best_trajectory['states'][-1][:2] - goal_point):.2f}")

# Plot key iterations to see improved sampling
print("\nPlotting key iterations with sampling strategies...")
key_iterations = [0, 2, 4, 7, len(all_iterations_data)-1]
for iter_num in key_iterations:
    if iter_num < len(all_iterations_data):
        plot_sampling_strategies(all_iterations_data[iter_num], start_point, goal_point, obstacles, iter_num)

# Final result
plt.figure(figsize=(10, 8))
for obstacle in obstacles:
    x_min, y_min, x_max, y_max = obstacle
    plt.fill_between([x_min, x_max], y_min, y_max, color='red', alpha=0.3, label='Obstacles')

plt.plot(start_point[0], start_point[1], 'go', markersize=15, label='Start', markeredgecolor='black')
plt.plot(goal_point[0], goal_point[1], 'bo', markersize=15, label='Goal', markeredgecolor='black')

best_states = best_trajectory['states']
best_x = [state[0] for state in best_states]
best_y = [state[1] for state in best_states]
plt.plot(best_x, best_y, 'g-', linewidth=4, label='Optimized Path')
plt.plot(best_x, best_y, 'go', markersize=3, alpha=0.6)

plt.title('Efficient MPPI - Final Optimized Path')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()