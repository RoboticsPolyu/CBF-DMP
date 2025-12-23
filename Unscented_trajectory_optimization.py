import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.patches as patches

class UnscentedVehicleTrajectoryOptimization:
    """
    Vehicle trajectory optimization using Unscented Transform
    Vehicle model: Similar to Zermelo problem
    """
    
    def __init__(self):
        # Vehicle parameters
        self.dt = 0.05  # Time step
        self.tf = 3.0   # Final time
        
        # Uncertain parameters: wind field (p, q)
        self.p_nominal = 1.0    # Baseline value
        self.q_nominal = -1.0   # Baseline value
        self.p_std = 0.2        # Standard deviation
        self.q_std = 0.1        # Standard deviation
        
        # Initial state
        self.x0 = 2.25
        self.y0 = 1.0
        
        # Target position
        self.x_target = 0.0
        self.y_target = 0.0
        
        # Unscented transform parameters
        self.n_sigma = 15  # Number of sigma points
        self.sigma_points = None
        self.weights = None
        
        # Generate sigma points
        self._generate_sigma_points()
    
    def _generate_sigma_points(self):
        """Generate sigma points for unscented transform"""
        n = 2  # Parameter dimension (p, q)
        
        # Mean
        mean = np.array([self.p_nominal, self.q_nominal])
        
        # Covariance
        cov = np.diag([self.p_std**2, self.q_std**2])
        
        # Calculate sigma points (simplified version)
        kappa = 3 - n
        lambda_ = 3 - n
        
        # Square root of covariance matrix
        sqrt_cov = np.linalg.cholesky((n + lambda_) * cov)
        
        # Generate sigma points
        self.sigma_points = np.zeros((2*n + 1, 2))
        self.weights = np.zeros(2*n + 1)
        
        # First point: mean
        self.sigma_points[0] = mean
        self.weights[0] = lambda_ / (n + lambda_)
        
        # Other points
        for i in range(n):
            self.sigma_points[1 + i] = mean + sqrt_cov[i]
            self.sigma_points[1 + n + i] = mean - sqrt_cov[i]
            
            self.weights[1 + i] = 1 / (2 * (n + lambda_))
            self.weights[1 + n + i] = 1 / (2 * (n + lambda_))
    
    def vehicle_dynamics(self, state, t, u, p, q):
        """
        Vehicle dynamics equation
        state: [x, y]
        u: [u1, u2] control input
        p, q: wind field parameters
        """
        x, y = state
        
        # Control input constraint: unit circle
        u_norm = np.sqrt(u[0]**2 + u[1]**2)
        if u_norm > 1.0:
            u = u / u_norm
        
        # Dynamics: dx/dt = p*y + u1, dy/dt = q*x + u2
        dxdt = p * y + u[0]
        dydt = q * x + u[1]
        
        return np.array([dxdt, dydt])
    
    def simulate_trajectory(self, u_func, p, q, x0=None, y0=None):
        """
        Simulate single trajectory
        u_func: control function u(t)
        p, q: wind field parameters
        """
        if x0 is None:
            x0 = self.x0
        if y0 is None:
            y0 = self.y0
        
        # Time points
        t_eval = np.linspace(0, self.tf, int(self.tf/self.dt) + 1)
        
        # Store results
        states = np.zeros((len(t_eval), 2))
        controls = np.zeros((len(t_eval), 2))
        states[0] = [x0, y0]
        
        # Numerical integration (Euler method)
        for i in range(len(t_eval)-1):
            t = t_eval[i]
            state = states[i]
            
            # Get control input
            u = u_func(t)
            controls[i] = u
            
            # Euler integration
            dstate = self.vehicle_dynamics(state, t, u, p, q)
            states[i+1] = state + dstate * self.dt
        
        # Last control
        controls[-1] = u_func(t_eval[-1])
        
        return t_eval, states, controls
    
    def linear_control_policy(self, params):
        """Linear control policy: u(t) = a*t + b"""
        a1, b1, a2, b2 = params
        
        def u_func(t):
            u1 = a1 * t + b1
            u2 = a2 * t + b2
            return np.array([u1, u2])
        
        return u_func
    
    def cost_function_deterministic(self, params):
        """Cost function for deterministic baseline problem"""
        # Linear control policy
        u_func = self.linear_control_policy(params)
        
        # Simulate with nominal parameters
        t, states, _ = self.simulate_trajectory(u_func, self.p_nominal, self.q_nominal)
        
        # Terminal cost: distance to target + time penalty
        xf, yf = states[-1]
        distance_cost = (xf - self.x_target)**2 + (yf - self.y_target)**2
        time_cost = 0.1 * self.tf  # Penalty for longer time
        
        # Control smoothness penalty
        control_penalty = 0.01 * np.sum(params**2)
        
        return distance_cost + time_cost + control_penalty
    
    def cost_function_unscented_mean(self, params):
        """Cost function for unscented mean control problem"""
        # Linear control policy
        u_func = self.linear_control_policy(params)
        
        total_cost = 0
        
        # Simulate for all sigma points
        for i, (p, q) in enumerate(self.sigma_points):
            t, states, _ = self.simulate_trajectory(u_func, p, q)
            
            # Terminal cost for this sigma point
            xf, yf = states[-1]
            distance_cost = (xf - self.x_target)**2 + (yf - self.y_target)**2
            
            # Weighted sum
            total_cost += self.weights[i] * distance_cost
        
        # Add control smoothness penalty
        control_penalty = 0.01 * np.sum(params**2)
        
        return total_cost + control_penalty
    
    def cost_function_unscented_covariance(self, params):
        """Cost function for unscented covariance control problem"""
        # Linear control policy
        u_func = self.linear_control_policy(params)
        
        # Store final states for all sigma points
        final_states = []
        
        # Simulate for all sigma points
        for i, (p, q) in enumerate(self.sigma_points):
            t, states, _ = self.simulate_trajectory(u_func, p, q)
            final_states.append(states[-1])
        
        final_states = np.array(final_states)
        
        # Calculate weighted mean
        weighted_mean = np.zeros(2)
        for i in range(len(self.sigma_points)):
            weighted_mean += self.weights[i] * final_states[i]
        
        # Calculate weighted covariance (trace)
        covariance_trace = 0
        for i in range(len(self.sigma_points)):
            diff = final_states[i] - weighted_mean
            covariance_trace += self.weights[i] * np.sum(diff**2)
        
        # Add penalty for mean deviation from target
        mean_deviation = np.sum(weighted_mean**2)
        
        # Add control smoothness penalty
        control_penalty = 0.01 * np.sum(params**2)
        
        return covariance_trace + 10 * mean_deviation + control_penalty
    
    def optimize_deterministic(self):
        """Solve deterministic baseline problem"""
        print("Solving deterministic baseline problem...")
        
        # Initial guess for control parameters
        params0 = np.array([0.0, -0.5, 0.0, -0.5])  # [a1, b1, a2, b2]
        
        # Bounds for parameters
        bounds = [(-2, 2), (-1, 1), (-2, 2), (-1, 1)]
        
        # Optimize
        result = minimize(self.cost_function_deterministic, params0, 
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 100, 'disp': True})
        
        print(f"Optimization success: {result.success}")
        print(f"Optimal parameters: {result.x}")
        print(f"Optimal cost: {result.fun}")
        
        return result.x
    
    def optimize_unscented_mean(self):
        """Solve unscented mean control problem"""
        print("\nSolving unscented mean control problem...")
        
        params0 = np.array([0.0, -0.5, 0.0, -0.5])
        bounds = [(-2, 2), (-1, 1), (-2, 2), (-1, 1)]
        
        result = minimize(self.cost_function_unscented_mean, params0,
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 100, 'disp': True})
        
        print(f"Optimization success: {result.success}")
        print(f"Optimal parameters: {result.x}")
        print(f"Optimal cost: {result.fun}")
        
        return result.x
    
    def optimize_unscented_covariance(self):
        """Solve unscented covariance control problem"""
        print("\nSolving unscented covariance control problem...")
        
        params0 = np.array([0.0, -0.5, 0.0, -0.5])
        bounds = [(-2, 2), (-1, 1), (-2, 2), (-1, 1)]
        
        result = minimize(self.cost_function_unscented_covariance, params0,
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 100, 'disp': True})
        
        print(f"Optimization success: {result.success}")
        print(f"Optimal parameters: {result.x}")
        print(f"Optimal cost: {result.fun}")
        
        return result.x
    
    def monte_carlo_analysis(self, params, n_samples=1000):
        """
        Monte Carlo simulation for given control parameters
        """
        u_func = self.linear_control_policy(params)
        
        # Generate random samples from normal distribution
        p_samples = np.random.normal(self.p_nominal, self.p_std, n_samples)
        q_samples = np.random.normal(self.q_nominal, self.q_std, n_samples)
        
        # Store final states
        final_states = []
        trajectories = []
        
        # Simulate for all samples
        for i in range(n_samples):
            t, states, controls = self.simulate_trajectory(u_func, p_samples[i], q_samples[i])
            final_states.append(states[-1])
            
            # Store every 10th trajectory for visualization
            if i % (n_samples//20) == 0:
                trajectories.append(states)
        
        final_states = np.array(final_states)
        
        # Calculate statistics
        mean_final = np.mean(final_states, axis=0)
        cov_final = np.cov(final_states.T)
        
        # Calculate risk (probability of missing target)
        # Define target region as circle with radius 0.2
        distances = np.sqrt((final_states[:, 0] - self.x_target)**2 + 
                           (final_states[:, 1] - self.y_target)**2)
        target_radius = 0.2
        success_rate = np.mean(distances <= target_radius)
        risk = 1 - success_rate
        
        return {
            'final_states': final_states,
            'mean_final': mean_final,
            'cov_final': cov_final,
            'risk': risk,
            'success_rate': success_rate,
            'trajectories': trajectories,
            'p_samples': p_samples,
            'q_samples': q_samples
        }
    
    def plot_results(self, deterministic_params, mean_params, cov_params):
        """
        Plot comparison of different optimization strategies
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Colors for different strategies
        colors = ['blue', 'green', 'red']
        labels = ['Deterministic', 'Unscented Mean', 'Unscented Covariance']
        all_params = [deterministic_params, mean_params, cov_params]
        
        # Monte Carlo analysis for each strategy
        mc_results = []
        for params in all_params:
            mc_results.append(self.monte_carlo_analysis(params, n_samples=500))
        
        # 1. Plot trajectories (sample)
        ax = axes[0, 0]
        for i, (params, color, label) in enumerate(zip(all_params, colors, labels)):
            u_func = self.linear_control_policy(params)
            t, states, _ = self.simulate_trajectory(u_func, self.p_nominal, self.q_nominal)
            ax.plot(states[:, 0], states[:, 1], color=color, label=label, linewidth=2)
        
        ax.scatter(self.x0, self.y0, color='black', s=100, label='Start', zorder=5)
        ax.scatter(self.x_target, self.y_target, color='red', s=100, label='Target', zorder=5)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Nominal Trajectories')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        
        # 2. Plot control inputs
        ax = axes[0, 1]
        t = np.linspace(0, self.tf, 100)
        for i, (params, color, label) in enumerate(zip(all_params, colors, labels)):
            u_func = self.linear_control_policy(params)
            controls = np.array([u_func(ti) for ti in t])
            ax.plot(t, controls[:, 0], color=color, label=f'{label} - u1', linestyle='-')
            ax.plot(t, controls[:, 1], color=color, label=f'{label} - u2', linestyle='--')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Control Input')
        ax.set_title('Control Trajectories')
        ax.legend()
        ax.grid(True)
        
        # 3. Plot final state distributions
        ax = axes[0, 2]
        for i, (result, color, label) in enumerate(zip(mc_results, colors, labels)):
            ax.scatter(result['final_states'][:, 0], result['final_states'][:, 1], 
                      alpha=0.3, color=color, label=label, s=10)
        
        # Add target region
        target_circle = patches.Circle((self.x_target, self.y_target), 0.2,
                                      fill=False, color='red', linewidth=2)
        ax.add_patch(target_circle)
        
        ax.scatter(self.x_target, self.y_target, color='red', s=100, zorder=5)
        ax.set_xlabel('Final X Position')
        ax.set_ylabel('Final Y Position')
        ax.set_title('Monte Carlo: Final State Distributions')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        
        # 4. Plot covariance ellipses
        ax = axes[1, 0]
        for i, (result, color, label) in enumerate(zip(mc_results, colors, labels)):
            mean = result['mean_final']
            cov = result['cov_final']
            
            # Plot covariance ellipse
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            
            # Ellipse parameters
            ell_angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
            ell_width = 2 * lambda_[0]
            ell_height = 2 * lambda_[1]
            
            ellipse = patches.Ellipse(mean, ell_width, ell_height, angle=ell_angle,
                                     fill=False, color=color, linewidth=2, alpha=0.8)
            ax.add_patch(ellipse)
            ax.scatter(mean[0], mean[1], color=color, s=50, zorder=5)
        
        target_circle = patches.Circle((self.x_target, self.y_target), 0.2,
                                      fill=False, color='red', linewidth=2)
        ax.add_patch(target_circle)
        
        ax.set_xlabel('Final X Position')
        ax.set_ylabel('Final Y Position')
        ax.set_title('Covariance Ellipses (2σ)')
        ax.grid(True)
        ax.axis('equal')
        
        # 5. Plot risk analysis
        ax = axes[1, 1]
        target_radii = np.linspace(0.05, 0.5, 20)
        risk_values = []
        
        for params in all_params:
            risks = []
            for radius in target_radii:
                # Quick Monte Carlo for each radius
                n_samples = 200
                u_func = self.linear_control_policy(params)
                p_samples = np.random.normal(self.p_nominal, self.p_std, n_samples)
                q_samples = np.random.normal(self.q_nominal, self.q_std, n_samples)
                
                final_x = []
                final_y = []
                
                for i in range(n_samples):
                    t, states, _ = self.simulate_trajectory(u_func, p_samples[i], q_samples[i])
                    final_x.append(states[-1, 0])
                    final_y.append(states[-1, 1])
                
                distances = np.sqrt((np.array(final_x) - self.x_target)**2 + 
                                   (np.array(final_y) - self.y_target)**2)
                success_rate = np.mean(distances <= radius)
                risks.append(1 - success_rate)
            
            risk_values.append(risks)
        
        for i, (risks, color, label) in enumerate(zip(risk_values, colors, labels)):
            ax.plot(target_radii, risks, color=color, label=label, linewidth=2)
        
        ax.set_xlabel('Target Radius')
        ax.set_ylabel('Risk (1 - Success Probability)')
        ax.set_title('Risk vs Target Tolerance')
        ax.legend()
        ax.grid(True)
        
        # 6. Plot parameter distributions
        ax = axes[1, 2]
        x = np.linspace(self.p_nominal - 3*self.p_std, self.p_nominal + 3*self.p_std, 100)
        y = np.linspace(self.q_nominal - 3*self.q_std, self.q_nominal + 3*self.q_std, 100)
        X, Y = np.meshgrid(x, y)
        
        # 2D Gaussian PDF
        Z = (1/(2*np.pi*self.p_std*self.q_std)) * np.exp(-0.5*(
            ((X - self.p_nominal)/self.p_std)**2 + 
            ((Y - self.q_nominal)/self.q_std)**2
        ))
        
        ax.contourf(X, Y, Z, levels=20, cmap='Blues')
        ax.scatter(self.p_nominal, self.q_nominal, color='red', s=100, 
                  label='Nominal', zorder=5)
        
        # Plot sigma points
        ax.scatter(self.sigma_points[:, 0], self.sigma_points[:, 1], 
                  color='orange', s=100, marker='s', label='Sigma Points', zorder=5)
        
        ax.set_xlabel('Parameter p (wind x)')
        ax.set_ylabel('Parameter q (wind y)')
        ax.set_title('Parameter Distribution & Sigma Points')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\n" + "="*50)
        print("COMPARISON OF OPTIMIZATION STRATEGIES")
        print("="*50)
        
        for i, (label, result) in enumerate(zip(labels, mc_results)):
            print(f"\n{label}:")
            print(f"  Final mean: {result['mean_final']}")
            print(f"  Final covariance trace: {np.trace(result['cov_final']):.4f}")
            print(f"  Success rate (radius=0.2): {result['success_rate']:.2%}")
            print(f"  Risk (radius=0.2): {result['risk']:.2%}")
        
        return mc_results

def main():
    """Main function to run the unscented trajectory optimization"""
    # Create optimizer instance
    optimizer = UnscentedVehicleTrajectoryOptimization()
    
    # Step 1: Solve deterministic baseline problem
    print("="*60)
    print("STEP 1: DETERMINISTIC BASELINE OPTIMIZATION")
    print("="*60)
    deterministic_params = optimizer.optimize_deterministic()
    
    # Step 2: Monte Carlo analysis for baseline
    print("\n" + "="*60)
    print("STEP 2: MONTE CARLO ANALYSIS FOR BASELINE")
    print("="*60)
    mc_baseline = optimizer.monte_carlo_analysis(deterministic_params, n_samples=500)
    print(f"Baseline success rate: {mc_baseline['success_rate']:.2%}")
    print(f"Baseline risk: {mc_baseline['risk']:.2%}")
    
    # Step 3: Solve unscented mean control problem
    print("\n" + "="*60)
    print("STEP 3: UNSCTED MEAN CONTROL OPTIMIZATION")
    print("="*60)
    mean_params = optimizer.optimize_unscented_mean()
    
    # Step 4: Monte Carlo analysis for mean control
    print("\n" + "="*60)
    print("STEP 4: MONTE CARLO ANALYSIS FOR MEAN CONTROL")
    print("="*60)
    mc_mean = optimizer.monte_carlo_analysis(mean_params, n_samples=500)
    print(f"Mean control success rate: {mc_mean['success_rate']:.2%}")
    print(f"Mean control risk: {mc_mean['risk']:.2%}")
    
    # Step 5: Solve unscented covariance control problem
    print("\n" + "="*60)
    print("STEP 5: UNSCTED COVARIANCE CONTROL OPTIMIZATION")
    print("="*60)
    cov_params = optimizer.optimize_unscented_covariance()
    
    # Step 6: Monte Carlo analysis for covariance control
    print("\n" + "="*60)
    print("STEP 6: MONTE CARLO ANALYSIS FOR COVARIANCE CONTROL")
    print("="*60)
    mc_cov = optimizer.monte_carlo_analysis(cov_params, n_samples=500)
    print(f"Covariance control success rate: {mc_cov['success_rate']:.2%}")
    print(f"Covariance control risk: {mc_cov['risk']:.2%}")
    
    # Plot all results
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION")
    print("="*60)
    mc_results = optimizer.plot_results(deterministic_params, mean_params, cov_params)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("1. Deterministic optimization: Fast but high risk")
    print("2. Unscented mean control: Better mean accuracy")
    print("3. Unscented covariance control: Best dispersion control")
    print("\nTrade-off: Covariance control reduces dispersion but may require")
    print("longer maneuver time (increased tf would show this).")

if __name__ == "__main__":
    main()