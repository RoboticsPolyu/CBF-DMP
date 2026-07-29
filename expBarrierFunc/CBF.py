"""
Simulation of Safety Probability Based on Control Barrier Functions (CBF)
Implementation of the probabilistic safety function:
P_free,i(x) = Φ((h_i(x) + τ * max(ḣ_i(x), -γ h_i(x))) / σ_h(x))
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import norm
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

class CBFProbabilisticSafety:
    """
    Implementation of probabilistic safety function based on CBF
    with Gaussian uncertainty
    """
    
    def __init__(self, obstacles, robot_radius=0.2, gamma=2.0, tau=0.5, 
                 sigma_position=0.1, sigma_velocity=0.05):
        """
        Initialize CBF probabilistic safety simulator
        
        Parameters:
        -----------
        obstacles : list of tuples
            Each tuple: (center_x, center_y, radius)
        robot_radius : float
            Radius of the robot (for collision checking)
        gamma : float
            CBF decay rate parameter (γ > 0)
        tau : float
            Lookahead time scale parameter (seconds)
        sigma_position : float
            Standard deviation of position uncertainty
        sigma_velocity : float
            Standard deviation of velocity uncertainty
        """
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        self.gamma = gamma
        self.tau = tau
        self.sigma_position = sigma_position
        self.sigma_velocity = sigma_velocity
        
        # State estimation covariance matrix
        self.Sigma = np.diag([sigma_position**2, sigma_position**2])
        
        # Color maps for visualization
        self.cmap_safety = plt.cm.RdYlGn  # Red-Yellow-Green for safety probability
        self.cmap_h = plt.cm.RdBu_r       # Red-Blue for h(x) values
        
    def h_function(self, position, obstacle):
        """
        Compute CBF candidate function h(x) for a single obstacle
        
        h(x) = ||x - o||² - (R + r)²
        Positive: outside obstacle, Negative: inside obstacle
        
        Parameters:
        -----------
        position : array-like (2,)
            Robot position [x, y]
        obstacle : tuple
            (center_x, center_y, radius)
            
        Returns:
        --------
        float
            Value of h(x)
        """
        center = np.array(obstacle[:2])
        radius = obstacle[2]
        pos = np.array(position)
        
        # Squared distance to obstacle center
        dist_sq = np.sum((pos - center)**2)
        
        # Squared safety margin (obstacle radius + robot radius)
        safety_margin_sq = (radius + self.robot_radius)**2
        
        return dist_sq - safety_margin_sq
    
    def grad_h(self, position, obstacle):
        """
        Compute gradient of h(x): ∇h(x) = 2(x - o)
        
        Parameters:
        -----------
        position : array-like (2,)
            Robot position
        obstacle : tuple
            (center_x, center_y, radius)
            
        Returns:
        --------
        ndarray (2,)
            Gradient vector ∇h(x)
        """
        center = np.array(obstacle[:2])
        pos = np.array(position)
        
        return 2.0 * (pos - center)
    
    def h_dot(self, position, velocity, obstacle):
        """
        Compute time derivative of h(x): ḣ(x) = ∇h(x)·v
        
        Parameters:
        -----------
        position : array-like (2,)
            Robot position
        velocity : array-like (2,)
            Robot velocity [vx, vy]
        obstacle : tuple
            (center_x, center_y, radius)
            
        Returns:
        --------
        float
            ḣ(x) = ∇h(x)·v
        """
        grad_h = self.grad_h(position, obstacle)
        vel = np.array(velocity)
        
        return np.dot(grad_h, vel)
    
    def sigma_h(self, position, obstacle):
        """
        Compute uncertainty in h(x): σ_h(x) = sqrt(∇h(x)^T Σ ∇h(x))
        
        Parameters:
        -----------
        position : array-like (2,)
            Robot position
        obstacle : tuple
            (center_x, center_y, radius)
            
        Returns:
        --------
        float
            Standard deviation of h(x) due to position uncertainty
        """
        grad_h = self.grad_h(position, obstacle)
        
        # Compute variance: ∇h^T Σ ∇h
        var_h = grad_h.T @ self.Sigma @ grad_h
        
        return np.sqrt(var_h)
    
    def cbf_safety_margin(self, h_val, h_dot_val):
        """
        Compute CBF-based safety margin: max(ḣ(x), -γ h(x))
        
        This implements the CBF constraint in a way that ensures
        safety probability increases when constraint is satisfied
        
        Parameters:
        -----------
        h_val : float
            Value of h(x)
        h_dot_val : float
            Value of ḣ(x)
            
        Returns:
        --------
        float
            Safety margin term
        """
        # CBF constraint: ḣ(x) ≥ -γ h(x)
        cbf_constraint = -self.gamma * h_val
        
        # Use the maximum between actual ḣ and CBF constraint
        # This ensures we consider the "safest" scenario
        return max(h_dot_val, cbf_constraint)
    
    def safety_probability_single(self, position, velocity, obstacle):
        """
        Compute safety probability for a single obstacle
        
        P_free,i(x) = Φ((h_i(x) + τ * max(ḣ_i(x), -γ h_i(x))) / σ_h_i(x))
        
        Parameters:
        -----------
        position : array-like (2,)
            Robot position
        velocity : array-like (2,)
            Robot velocity
        obstacle : tuple
            (center_x, center_y, radius)
            
        Returns:
        --------
        dict
            Dictionary containing probability and intermediate values
        """
        # Compute all components
        h_val = self.h_function(position, obstacle)
        h_dot_val = self.h_dot(position, velocity, obstacle)
        sigma_h_val = self.sigma_h(position, obstacle)
        
        # Safety margin based on CBF constraint
        safety_margin = self.cbf_safety_margin(h_val, h_dot_val)
        
        # Avoid division by zero
        if sigma_h_val < 1e-10:
            sigma_h_val = 1e-10
        
        # Compute argument for normal CDF
        z = (h_val + self.tau * safety_margin) / sigma_h_val
        
        # Safety probability = Φ(z)
        prob_safe = norm.cdf(z)
        
        # Clip to valid probability range [0, 1]
        prob_safe = np.clip(prob_safe, 0.0, 1.0)
        
        return {
            'probability': prob_safe,
            'h_value': h_val,
            'h_dot': h_dot_val,
            'safety_margin': safety_margin,
            'sigma_h': sigma_h_val,
            'z_score': z
        }
    
    def total_safety_probability(self, position, velocity, method='product'):
        """
        Compute total safety probability considering all obstacles
        
        Parameters:
        -----------
        position : array-like (2,)
            Robot position
        velocity : array-like (2,)
            Robot velocity
        method : str
            Method to combine probabilities:
            - 'product': Multiply individual probabilities (assumes independence)
            - 'min': Take minimum probability (worst-case)
            - 'softmin': Soft minimum using exponential weights
            
        Returns:
        --------
        dict
            Dictionary containing total probability and individual results
        """
        individual_results = []
        
        for i, obstacle in enumerate(self.obstacles):
            result = self.safety_probability_single(position, velocity, obstacle)
            result['obstacle_id'] = i
            individual_results.append(result)
        
        # Extract individual probabilities
        probs = [r['probability'] for r in individual_results]
        
        # Combine probabilities based on specified method
        if method == 'product':
            total_prob = np.prod(probs) if probs else 1.0
        elif method == 'min':
            total_prob = min(probs) if probs else 1.0
        elif method == 'softmin':
            # Soft minimum: weighted average with exponential weights
            weights = np.exp(-np.array(probs) / 0.1)
            weights = weights / np.sum(weights)
            total_prob = np.sum(weights * probs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'total_probability': total_prob,
            'individual_results': individual_results,
            'method': method
        }
    
    def cbf_qp_controller(self, position, nominal_velocity, relaxation=0.1):
        """
        Simple CBF-QP controller that ensures CBF constraints are satisfied
        
        min ||v - v_nom||² s.t. ḣ(x) ≥ -γ h(x) - relaxation for all obstacles
        
        Parameters:
        -----------
        position : array-like (2,)
            Current position
        nominal_velocity : array-like (2,)
            Nominal desired velocity
        relaxation : float
            Relaxation term for constraint (allows small violations)
            
        Returns:
        --------
        ndarray (2,)
            Safe velocity that satisfies CBF constraints
        """
        from scipy.optimize import minimize
        
        v_nom = np.array(nominal_velocity)
        
        # Objective function: minimize deviation from nominal velocity
        def objective(v):
            return np.sum((v - v_nom)**2)
        
        # Constraints: ḣ(x) ≥ -γ h(x) - relaxation
        constraints = []
        
        for obstacle in self.obstacles:
            h_val = self.h_function(position, obstacle)
            grad_h_val = self.grad_h(position, obstacle)
            
            # CBF constraint function
            def make_constraint(grad=grad_h_val, h=h_val, gamma=self.gamma, relax=relaxation):
                return lambda v: np.dot(grad, v) + gamma * h + relax
            
            constraints.append({'type': 'ineq', 'fun': make_constraint()})
        
        # Velocity bounds (maximum speed)
        max_speed = 2.0
        bounds = [(-max_speed, max_speed), (-max_speed, max_speed)]
        
        # Initial guess: nominal velocity
        v0 = v_nom.copy()
        
        # Solve optimization problem
        try:
            result = minimize(objective, v0, bounds=bounds, constraints=constraints)
            
            if result.success:
                safe_velocity = result.x
            else:
                # If optimization fails, scale down nominal velocity
                safe_velocity = v_nom * 0.5
                print(f"QP failed: {result.message}, using scaled velocity")
                
        except Exception as e:
            print(f"QP exception: {e}")
            safe_velocity = v_nom * 0.5
        
        return safe_velocity
    
    def compute_probability_field(self, x_range=(-5, 5), y_range=(-5, 5), 
                                  grid_size=50, velocity=None):
        """
        Compute safety probability field over a grid
        
        Parameters:
        -----------
        x_range : tuple
            (x_min, x_max) for grid
        y_range : tuple
            (y_min, y_max) for grid
        grid_size : int
            Number of grid points in each dimension
        velocity : array-like or None
            Velocity vector. If None, use zero velocity
            
        Returns:
        --------
        tuple
            (X, Y, P_total, P_individual, H_values)
        """
        if velocity is None:
            velocity = [0.0, 0.0]
        
        # Create grid
        x = np.linspace(x_range[0], x_range[1], grid_size)
        y = np.linspace(y_range[0], y_range[1], grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Initialize arrays
        P_total = np.zeros_like(X)
        H_values = np.zeros((grid_size, grid_size, len(self.obstacles)))
        P_individual = np.zeros((grid_size, grid_size, len(self.obstacles)))
        
        # Compute at each grid point
        for i in range(grid_size):
            for j in range(grid_size):
                position = [X[i, j], Y[i, j]]
                
                # Total safety probability
                result = self.total_safety_probability(position, velocity, method='product')
                P_total[i, j] = result['total_probability']
                
                # Store individual values
                for k, obs_result in enumerate(result['individual_results']):
                    H_values[i, j, k] = obs_result['h_value']
                    P_individual[i, j, k] = obs_result['probability']
        
        return X, Y, P_total, P_individual, H_values
    
    def visualize_probability_field(self, velocity=None, grid_size=60, 
                                   show_individual=False):
        """
        Create comprehensive visualization of safety probability field
        
        Parameters:
        -----------
        velocity : array-like or None
            Velocity vector for probability computation
        grid_size : int
            Resolution of visualization grid
        show_individual : bool
            Whether to show individual obstacle probabilities
        """
        if velocity is None:
            velocity = [0.5, 0.0]  # Default: moving right
        
        # Compute probability field
        X, Y, P_total, P_individual, H_values = self.compute_probability_field(
            grid_size=grid_size, velocity=velocity
        )
        
        # Determine number of subplots
        n_obstacles = len(self.obstacles)
        if show_individual and n_obstacles > 0:
            n_rows = 2
            n_cols = max(2, (n_obstacles + 1) // 2 + 1)
            fig_size = (5 * n_cols, 5 * n_rows)
        else:
            n_rows, n_cols = 2, 3
            fig_size = (15, 10)
        
        fig = plt.figure(figsize=fig_size)
        
        # 1. Total safety probability (main plot)
        ax1 = plt.subplot(n_rows, n_cols, 1)
        contour1 = ax1.contourf(X, Y, P_total, levels=20, cmap=self.cmap_safety, vmin=0, vmax=1)
        plt.colorbar(contour1, ax=ax1, label='Safety Probability')
        
        # Draw obstacles
        for i, obstacle in enumerate(self.obstacles):
            center = obstacle[:2]
            radius = obstacle[2]
            
            # Actual obstacle
            circle_actual = Circle(center, radius, fill=True, alpha=0.7, 
                                   color='black', label=f'Obstacle {i+1}')
            ax1.add_patch(circle_actual)
            
            # Safety boundary (obstacle + robot radius)
            circle_safety = Circle(center, radius + self.robot_radius, 
                                   fill=False, linestyle='--', linewidth=2,
                                   color='blue', label='Safety boundary')
            ax1.add_patch(circle_safety)
        
        ax1.set_xlabel('X position')
        ax1.set_ylabel('Y position')
        ax1.set_title(f'Total Safety Probability\nv={velocity}, γ={self.gamma}, τ={self.tau}')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Remove duplicate labels
        handles, labels = ax1.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax1.legend(unique_labels.values(), unique_labels.keys())
        
        # 2. Minimum h(x) value across obstacles
        ax2 = plt.subplot(n_rows, n_cols, 2)
        h_min = np.min(H_values, axis=2)
        contour2 = ax2.contourf(X, Y, h_min, levels=20, cmap=self.cmap_h)
        plt.colorbar(contour2, ax=ax2, label='min h(x)')
        
        # Draw obstacles
        for obstacle in self.obstacles:
            center = obstacle[:2]
            radius = obstacle[2]
            circle = Circle(center, radius + self.robot_radius, 
                           fill=False, linestyle='--', linewidth=2, color='red')
            ax2.add_patch(circle)
        
        ax2.set_xlabel('X position')
        ax2.set_ylabel('Y position')
        ax2.set_title('Minimum CBF Function h(x)')
        ax2.set_aspect('equal')
        
        # 3. Safety probability along a horizontal slice
        ax3 = plt.subplot(n_rows, n_cols, 3)
        slice_y = 0.0  # Horizontal slice at y = 0
        
        # Find row index closest to slice_y
        y_values = np.linspace(-5, 5, grid_size)
        slice_idx = np.argmin(np.abs(y_values - slice_y))
        
        # Plot total probability along slice
        ax3.plot(X[slice_idx, :], P_total[slice_idx, :], 'b-', linewidth=2, label='Total P_safe')
        
        # Plot individual probabilities if not too many
        if n_obstacles <= 4:
            for k in range(n_obstacles):
                ax3.plot(X[slice_idx, :], P_individual[slice_idx, :, k], '--', 
                        label=f'Obstacle {k+1}')
        
        # Mark obstacle regions
        for obstacle in self.obstacles:
            center_x, center_y, radius = obstacle
            if abs(center_y - slice_y) < radius + self.robot_radius + 1.0:
                # This obstacle affects the slice
                ax3.axvspan(center_x - radius - self.robot_radius,
                           center_x + radius + self.robot_radius,
                           alpha=0.2, color='red', label='Collision region')
        
        ax3.set_xlabel('X position')
        ax3.set_ylabel('Safety Probability')
        ax3.set_title(f'Safety Probability at y={slice_y}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.05, 1.05)
        
        # 4. 3D surface plot of safety probability
        ax4 = plt.subplot(n_rows, n_cols, 4, projection='3d')
        surf = ax4.plot_surface(X, Y, P_total, cmap=self.cmap_safety, 
                               linewidth=0, antialiased=True, alpha=0.8)
        
        # Add contour lines on the XY plane
        ax4.contour(X, Y, P_total, zdir='z', offset=0, cmap=self.cmap_safety, alpha=0.5)
        
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('P_safe')
        ax4.set_title('3D Safety Probability Surface')
        ax4.view_init(elev=30, azim=45)
        
        # 5. CBF-QP controller velocity field
        ax5 = plt.subplot(n_rows, n_cols, 5)
        
        # Sample grid for velocity vectors
        sample_step = max(1, grid_size // 15)
        X_sample = X[::sample_step, ::sample_step]
        Y_sample = Y[::sample_step, ::sample_step]
        
        # Compute safe velocities at sample points
        V_safe_x = np.zeros_like(X_sample)
        V_safe_y = np.zeros_like(Y_sample)
        
        nominal_velocity = np.array([1.0, 0])  # Always try to go right
        
        for i in range(X_sample.shape[0]):
            for j in range(X_sample.shape[1]):
                position = [X_sample[i, j], Y_sample[i, j]]
                safe_vel = self.cbf_qp_controller(position, nominal_velocity)
                V_safe_x[i, j] = safe_vel[0]
                V_safe_y[i, j] = safe_vel[1]
        
        # Plot velocity field
        ax5.quiver(X_sample, Y_sample, V_safe_x, V_safe_y, color='green', 
                  alpha=0.7, scale=20, width=0.003)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            center = obstacle[:2]
            radius = obstacle[2]
            circle = Circle(center, radius + self.robot_radius, 
                           fill=False, linestyle='--', linewidth=2, color='red')
            ax5.add_patch(circle)
        
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_title('CBF-QP Safe Velocity Field')
        ax5.set_aspect('equal')
        ax5.grid(True, alpha=0.3)
        
        # 6. Parameter sensitivity analysis
        ax6 = plt.subplot(n_rows, n_cols, 6)
        
        # Test different gamma values at a specific point
        test_position = [0, 0]
        test_velocity = [0.5, 0]
        
        gamma_values = np.linspace(0.1, 5.0, 50)
        probs_gamma = []
        
        original_gamma = self.gamma
        for gamma_val in gamma_values:
            self.gamma = gamma_val
            result = self.total_safety_probability(test_position, test_velocity)
            probs_gamma.append(result['total_probability'])
        self.gamma = original_gamma  # Restore original value
        
        ax6.plot(gamma_values, probs_gamma, 'b-', linewidth=2)
        ax6.axvline(x=self.gamma, color='r', linestyle='--', label=f'Current γ={self.gamma}')
        ax6.set_xlabel('CBF parameter γ')
        ax6.set_ylabel('Safety Probability')
        ax6.set_title(f'Sensitivity to γ at position {test_position}')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Show individual obstacle probabilities if requested
        if show_individual and n_obstacles > 0:
            for k in range(n_obstacles):
                ax_idx = n_cols + k + 1  # Start from second row
                if ax_idx <= n_rows * n_cols:
                    ax = plt.subplot(n_rows, n_cols, ax_idx)
                    
                    contour = ax.contourf(X, Y, P_individual[:, :, k], 
                                         levels=20, cmap=self.cmap_safety, vmin=0, vmax=1)
                    plt.colorbar(contour, ax=ax, label='P_safe')
                    
                    # Draw this specific obstacle
                    obstacle = self.obstacles[k]
                    center = obstacle[:2]
                    radius = obstacle[2]
                    circle = Circle(center, radius + self.robot_radius, 
                                   fill=False, linestyle='--', linewidth=2, color='red')
                    ax.add_patch(circle)
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_title(f'Obstacle {k+1} Safety Probability')
                    ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def simulate_trajectory(self, start_pos, goal_pos, dt=0.1, total_time=10.0):
        """
        Simulate robot trajectory using CBF-QP controller
        
        Parameters:
        -----------
        start_pos : array-like (2,)
            Starting position
        goal_pos : array-like (2,)
            Goal position
        dt : float
            Time step for simulation
        total_time : float
            Total simulation time
            
        Returns:
        --------
        dict
            Simulation results including positions, velocities, and safety probabilities
        """
        # Initialize trajectory
        time_points = np.arange(0, total_time, dt)
        n_steps = len(time_points)
        
        positions = np.zeros((n_steps, 2))
        velocities = np.zeros((n_steps, 2))
        safety_probs = np.zeros(n_steps)
        
        positions[0] = np.array(start_pos)
        
        # Simple goal attraction controller (nominal)
        def nominal_controller(pos, goal):
            # Simple proportional controller toward goal
            k_p = 1.0
            desired_vel = k_p * (np.array(goal) - np.array(pos))
            
            # Limit maximum speed
            max_speed = 2.0
            speed = np.linalg.norm(desired_vel)
            if speed > max_speed:
                desired_vel = desired_vel / speed * max_speed
            
            return desired_vel
        
        # Simulate
        for i in range(1, n_steps):
            current_pos = positions[i-1]
            
            # Compute nominal velocity (toward goal)
            v_nom = nominal_controller(current_pos, goal_pos)
            
            # Compute safe velocity using CBF-QP
            v_safe = self.cbf_qp_controller(current_pos, v_nom)
            
            # Update position (simple Euler integration)
            positions[i] = current_pos + v_safe * dt
            velocities[i] = v_safe
            
            # Compute safety probability at current state
            result = self.total_safety_probability(current_pos, v_safe)
            safety_probs[i-1] = result['total_probability']
        
        # Compute safety probability at final position
        result = self.total_safety_probability(positions[-1], velocities[-1])
        safety_probs[-1] = result['total_probability']
        
        return {
            'time': time_points,
            'positions': positions,
            'velocities': velocities,
            'safety_probs': safety_probs,
            'dt': dt,
            'goal_pos': goal_pos
        }
    
    def animate_trajectory(self, traj_result, save_animation=False, filename='trajectory.mp4'):
        """
        Animate the simulated trajectory
        
        Parameters:
        -----------
        traj_result : dict
            Result from simulate_trajectory()
        save_animation : bool
            Whether to save animation to file
        filename : str
            Filename for saved animation
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        time = traj_result['time']
        positions = traj_result['positions']
        safety_probs = traj_result['safety_probs']
        goal_pos = traj_result['goal_pos']
        
        # Create initial plots
        ax1 = axes[0, 0]  # Trajectory plot
        ax2 = axes[0, 1]  # Safety probability over time
        ax3 = axes[1, 0]  # Instantaneous probability field
        ax4 = axes[1, 1]  # Velocity magnitude
        
        # Plot obstacles on ax1 and ax3
        for ax in [ax1, ax3]:
            for obstacle in self.obstacles:
                center = obstacle[:2]
                radius = obstacle[2]
                circle_actual = Circle(center, radius, fill=True, alpha=0.7, color='black')
                circle_safety = Circle(center, radius + self.robot_radius, 
                                      fill=False, linestyle='--', linewidth=2, color='blue')
                ax.add_patch(circle_actual)
                ax.add_patch(circle_safety)
        
        # Initialize trajectory line
        traj_line, = ax1.plot([], [], 'b-', linewidth=2, label='Trajectory')
        robot_point, = ax1.plot([], [], 'ro', markersize=10, label='Robot')
        goal_point, = ax1.plot(goal_pos[0], goal_pos[1], 'g*', markersize=15, label='Goal')
        
        ax1.set_xlim(-5, 5)
        ax1.set_ylim(-5, 5)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Robot Trajectory with CBF-QP Control')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Safety probability plot
        prob_line, = ax2.plot([], [], 'r-', linewidth=2)
        current_prob_point, = ax2.plot([], [], 'ro', markersize=8)
        ax2.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='Safety threshold (0.95)')
        ax2.set_xlim(0, time[-1])
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Safety Probability')
        ax2.set_title('Safety Probability Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Compute probability field for current frame
        current_idx = 0
        current_pos = positions[current_idx]
        current_vel = traj_result['velocities'][current_idx]
        
        X, Y, P_total, _, _ = self.compute_probability_field(
            grid_size=30, velocity=current_vel
        )
        
        # Probability field plot
        contour = ax3.contourf(X, Y, P_total, levels=20, cmap=self.cmap_safety, vmin=0, vmax=1)
        plt.colorbar(contour, ax=ax3, label='P_safe')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title(f'Instantaneous Safety Probability Field\nv={current_vel}')
        ax3.set_aspect('equal')
        
        # Velocity magnitude plot
        vel_magnitudes = np.linalg.norm(traj_result['velocities'], axis=1)
        vel_line, = ax4.plot([], [], 'b-', linewidth=2, label='Velocity magnitude')
        current_vel_point, = ax4.plot([], [], 'bo', markersize=8)
        ax4.set_xlim(0, time[-1])
        ax4.set_ylim(0, max(vel_magnitudes) * 1.1)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity magnitude')
        ax4.set_title('Robot Velocity Over Time')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        def update(frame):
            # Update trajectory
            traj_line.set_data(positions[:frame+1, 0], positions[:frame+1, 1])
            robot_point.set_data([positions[frame, 0]], [positions[frame, 1]])
            
            # Update safety probability
            prob_line.set_data(time[:frame+1], safety_probs[:frame+1])
            current_prob_point.set_data([time[frame]], [safety_probs[frame]])
            
            # Update probability field for current velocity
            current_vel = traj_result['velocities'][frame]
            _, _, P_total_new, _, _ = self.compute_probability_field(
                grid_size=30, velocity=current_vel
            )
            
            # Clear and redraw contour
            ax3.clear()
            for obstacle in self.obstacles:
                center = obstacle[:2]
                radius = obstacle[2]
                circle_actual = Circle(center, radius, fill=True, alpha=0.7, color='black')
                circle_safety = Circle(center, radius + self.robot_radius, 
                                      fill=False, linestyle='--', linewidth=2, color='blue')
                ax3.add_patch(circle_actual)
                ax3.add_patch(circle_safety)
            
            contour_new = ax3.contourf(X, Y, P_total_new, levels=20, 
                                      cmap=self.cmap_safety, vmin=0, vmax=1)
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_title(f'Instantaneous Safety Probability Field\nv={current_vel.round(2)}')
            ax3.set_aspect('equal')
            
            # Update velocity plot
            vel_line.set_data(time[:frame+1], vel_magnitudes[:frame+1])
            current_vel_point.set_data([time[frame]], [vel_magnitudes[frame]])
            
            return traj_line, robot_point, prob_line, current_prob_point, vel_line, current_vel_point
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(time), interval=50, blit=False
        )
        
        if save_animation:
            print(f"Saving animation to {filename}...")
            anim.save(filename, writer='ffmpeg', fps=20)
        
        plt.tight_layout()
        plt.show()
        
        return anim

# Main demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("CBF-BASED PROBABILISTIC SAFETY SIMULATION")
    print("=" * 70)
    print("\nFormula: P_free,i(x) = Φ((h_i(x) + τ·max(ḣ_i(x), -γ h_i(x))) / σ_h(x))")
    
    # Define obstacles (center_x, center_y, radius)
    obstacles = [
        (-3, 1, 1.0),
        (0, -2, 0.8),
        (2, 2, 1.2),
        (-2, -3, 0.7)
    ]
    
    # Initialize simulator
    simulator = CBFProbabilisticSafety(
        obstacles=obstacles,
        robot_radius=0.2,
        gamma=2.0,          # CBF decay rate
        tau=0.5,            # Lookahead time scale
        sigma_position=0.1, # Position uncertainty
        sigma_velocity=0.05 # Velocity uncertainty
    )
    
    # Test specific scenarios
    print("\n1. Testing Specific Scenarios:")
    print("-" * 40)
    
    test_scenarios = [
        ("Center of environment", [0, 0], [0.5, 0]),
        ("Near obstacle 1", [-3, 2.5], [0, -0.5]),
        ("Between obstacles", [0, 0], [1, 0]),
        ("Far from obstacles", [4, 4], [0, 0]),
    ]
    
    for description, position, velocity in test_scenarios:
        result = simulator.total_safety_probability(position, velocity)
        
        print(f"\n{description}:")
        print(f"  Position: {position}, Velocity: {velocity}")
        print(f"  Total safety probability: {result['total_probability']:.6f}")
        
        for obs_result in result['individual_results']:
            obs_id = obs_result['obstacle_id'] + 1
            h_val = obs_result['h_value']
            h_dot = obs_result['h_dot']
            margin = obs_result['safety_margin']
            sigma_h = obs_result['sigma_h']
            prob = obs_result['probability']
            
            status = "DANGER" if prob < 0.5 else "OK" if prob < 0.9 else "SAFE"
            print(f"  Obstacle {obs_id}: h={h_val:.3f}, ḣ={h_dot:.3f}, "
                  f"margin={margin:.3f}, σ_h={sigma_h:.3f}, P={prob:.4f} [{status}]")
    
    # Visualize probability field
    print("\n2. Generating probability field visualization...")
    print("-" * 40)
    
    fig = simulator.visualize_probability_field(
        velocity=[0.5, 0],  # Moving right
        grid_size=60,
        show_individual=False
    )
    
    # Simulate trajectory
    print("\n3. Simulating trajectory with CBF-QP control...")
    print("-" * 40)
    
    traj_result = simulator.simulate_trajectory(
        start_pos=[-4, 0],
        goal_pos=[4, 0],
        dt=0.1,
        total_time=8.0
    )
    
    # Analyze trajectory safety
    min_safety = np.min(traj_result['safety_probs'])
    mean_safety = np.mean(traj_result['safety_probs'])
    time_below_threshold = np.sum(traj_result['safety_probs'] < 0.95) * traj_result['dt']
    
    print(f"Trajectory safety analysis:")
    print(f"  Minimum safety probability: {min_safety:.6f}")
    print(f"  Mean safety probability: {mean_safety:.6f}")
    print(f"  Time below 0.95 threshold: {time_below_threshold:.2f}s")
    
    # Animate trajectory
    print("\n4. Animating trajectory (close window to continue)...")
    print("-" * 40)
    
    anim = simulator.animate_trajectory(
        traj_result,
        save_animation=False,  # Set to True to save as video
        filename='cbf_safety_trajectory.mp4'
    )
    
    # Interactive parameter exploration
    print("\n5. Interactive Parameter Exploration")
    print("-" * 40)
    print("\nYou can modify these parameters and rerun sections:")
    print("  - gamma: Controls how aggressively CBF enforces safety")
    print("  - tau: Lookahead time scale (affects 'future awareness')")
    print("  - sigma_position: Position uncertainty (affects probability spread)")
    print("  - Robot velocity: Affects ḣ(x) and thus safety probability")
    
    print("\n" + "=" * 70)
    print("Simulation complete! Modify parameters and rerun for different behaviors.")
    print("=" * 70)