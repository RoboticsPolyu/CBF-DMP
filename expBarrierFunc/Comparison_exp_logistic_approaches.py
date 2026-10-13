import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit
from pathlib import Path

# Portable publication style; individual figures use a local rc_context.
PLOT_STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#98A2B3', 'axes.labelcolor': '#344054',
    'xtick.color': '#475467', 'ytick.color': '#475467',
    'grid.color': '#D0D5DD', 'grid.alpha': 0.45,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
}

class ExponentialBarrierSafety:
    """
    Safety probability using exponential barrier functions
    P_i(x) = exp(-b_i(x)) with various barrier formulations
    """
    
    def __init__(self, obstacles, robot_radius=0.2, epsilon=0.01, 
                 scale_factor=1.0, gamma=2.0):
        """
        Parameters:
        -----------
        obstacles : list
            List of (center_x, center_y, radius)
        robot_radius : float
            Robot radius for obstacle inflation
        epsilon : float
            Small constant to avoid division by zero
        scale_factor : float
            Scaling factor for barrier function
        gamma : float
            Control parameter for barrier steepness
        """
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        self.epsilon = epsilon
        self.scale = scale_factor
        self.gamma = gamma
    
    def barrier_function_1(self, position, obstacle):
        """
        Barrier function 1: b(x) = (R+r)² / (‖x-o‖² + ε)
        
        Simple reciprocal form, smooth everywhere
        """
        center = np.array(obstacle[:2])
        radius = obstacle[2]
        pos = np.array(position)
        
        dist_sq = np.sum((pos - center)**2) + self.epsilon
        safety_radius = radius + self.robot_radius
        
        b = (safety_radius**2) / dist_sq
        
        # Safety probability
        p = np.exp(-self.scale * b)
        
        return p, b, dist_sq
    
    def barrier_function_2(self, position, obstacle):
        """
        Barrier function 2: b(x) = max(0, 1 - ‖x-o‖²/(R+r)²)ᵞ
        
        Zero inside safe region, positive near boundary
        """
        center = np.array(obstacle[:2])
        radius = obstacle[2]
        pos = np.array(position)
        
        dist_sq = np.sum((pos - center)**2)
        safety_radius = radius + self.robot_radius
        threshold = safety_radius**2
        
        if dist_sq >= threshold:
            b = 0.0  # Outside or on boundary
        else:
            # Inside safety region: barrier increases as we approach center
            normalized = 1.0 - dist_sq / threshold
            b = normalized**self.gamma
        
        # Safety probability
        p = np.exp(-self.scale * b)
        
        return p, b, dist_sq
    
    def barrier_function_3(self, position, velocity, obstacle):
        """
        Barrier function 3: Dynamic barrier considering velocity
        
        b(x,v) = max(0, 1 - (‖x-o‖² + α⋅v⋅(x-o))/(R+r)²)
        
        α: velocity influence factor
        """
        center = np.array(obstacle[:2])
        radius = obstacle[2]
        pos = np.array(position)
        vel = np.array(velocity)
        
        # Position component
        dist_sq = np.sum((pos - center)**2)
        
        # Velocity influence (dot product with direction to obstacle)
        dir_to_obs = pos - center
        dir_to_obs_norm = np.linalg.norm(dir_to_obs)
        if dir_to_obs_norm > 1e-10:
            dir_to_obs_unit = dir_to_obs / dir_to_obs_norm
            velocity_influence = np.dot(vel, dir_to_obs_unit)
        else:
            velocity_influence = 0.0
        
        # Combined safety metric (positive if moving toward obstacle)
        safety_metric = dist_sq + 0.5 * velocity_influence
        safety_radius = radius + self.robot_radius
        threshold = safety_radius**2
        
        if safety_metric >= threshold:
            b = 0.0
        else:
            normalized = 1.0 - safety_metric / threshold
            b = max(0, normalized)**2
        
        # Safety probability
        p = np.exp(-self.scale * b)
        
        return p, b, safety_metric
    
    def barrier_function_4(self, position, obstacle):
        """
        Barrier function 4: b(x) = (R+r)² - ‖x-o‖²
        
        Simple reciprocal form, smooth everywhere
        """
        center = np.array(obstacle[:2])
        radius = obstacle[2]
        pos = np.array(position)
        
        dist_sq = np.sum((pos - center)**2) 
        safety_radius = radius + self.robot_radius

        b = (safety_radius**2) - dist_sq

        # Safety probability
        p = np.exp(-self.scale * b)
        
        return p, b, dist_sq
    
    def total_safety_probability(self, position, velocity=None, 
                                 method='product', barrier_type=1):
        """
        Compute total safety probability using specified barrier function
        
        Parameters:
        -----------
        position : array-like
            Robot position [x, y]
        velocity : array-like or None
            Robot velocity [vx, vy]
        method : str
            'product', 'min', or 'softmin'
        barrier_type : int
            1, 2, or 3 (different barrier formulations)
            
        Returns:
        --------
        dict containing total probability and individual values
        """
        if velocity is None:
            velocity = [0.0, 0.0]
        
        individual_results = []
        
        for i, obstacle in enumerate(self.obstacles):
            if barrier_type == 1:
                prob, b_val, dist_metric = self.barrier_function_1(
                    position, obstacle
                )
            elif barrier_type == 2:
                prob, b_val, dist_metric = self.barrier_function_2(
                    position, obstacle
                )
            elif barrier_type == 3:
                prob, b_val, dist_metric = self.barrier_function_3(
                    position, velocity, obstacle
                )
            elif barrier_type == 4:
                prob, b_val, dist_metric = self.barrier_function_4(
                    position, velocity, obstacle
                )
            else:
                raise ValueError(f"Unknown barrier type: {barrier_type}")
            
            individual_results.append({
                'obstacle_id': i,
                'probability': prob,
                'barrier_value': b_val,
                'distance_metric': dist_metric,
                'center': obstacle[:2],
                'radius': obstacle[2]
            })
        
        # Combine probabilities
        probs = [r['probability'] for r in individual_results]
        
        if method == 'product':
            total_prob = np.prod(probs) if probs else 1.0
        elif method == 'min':
            total_prob = min(probs) if probs else 1.0
        elif method == 'softmin':
            beta = 10.0
            weights = np.exp(-beta * np.array(probs))
            weights = weights / np.sum(weights)
            total_prob = np.sum(weights * probs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            'total_probability': total_prob,
            'individual_results': individual_results,
            'method': method,
            'barrier_type': barrier_type
        }
    
    def visualize_comparison(self):
        """Compare different barrier function formulations"""
        
        # Create test grid
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        
        # Single obstacle at origin for visualization
        test_obstacle = (0, 0, 1.0)
        
        # Initialize result arrays
        P1 = np.zeros_like(X)  # Barrier type 1
        P2 = np.zeros_like(X)  # Barrier type 2
        P3 = np.zeros_like(X)  # Barrier type 3
        P4 = np.zeros_like(X)  # Barrier type 4
        B1 = np.zeros_like(X)  # Barrier values
        
        # Test velocity (for type 3)
        test_velocity = [0.5, 0]
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pos = [X[i, j], Y[i, j]]
                
                # Type 1
                p1, b1, _ = self.barrier_function_1(pos, test_obstacle)
                P1[i, j] = p1
                B1[i, j] = b1
                
                # Type 2
                p2, _, _ = self.barrier_function_2(pos, test_obstacle)
                P2[i, j] = p2
                
                # Type 3
                p3, _, _ = self.barrier_function_3(pos, test_velocity, test_obstacle)
                P3[i, j] = p3

                p4, _, _ = self.barrier_function_4(pos, test_obstacle)
                P4[i, j] = p4
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Barrier function values (type 1)
        im1 = axes[0, 0].imshow(B1, extent=[-5, 5, -5, 5], 
                               origin='lower', cmap='viridis')
        # plt.colorbar(im1, ax=axes[0, 0], label='b(x)')
        circle1 = plt.Circle((0, 0), 1.0 + self.robot_radius, 
                            fill=False, linestyle='--', linewidth=2, color='red')
        axes[0, 0].add_patch(circle1)
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].set_title('Barrier Function b(x) = (R+r)²/(‖x-o‖²+ε)')
        axes[0, 0].set_aspect('equal')
        
        # 2. Safety probability (type 1)
        im2 = axes[0, 1].imshow(P1, extent=[-5, 5, -5, 5], 
                               origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
        # plt.colorbar(im2, ax=axes[0, 1], label='P(x) = exp(-b(x))')
        circle2 = plt.Circle((0, 0), 1.0 + self.robot_radius, 
                            fill=False, linestyle='--', linewidth=2, color='blue')
        axes[0, 1].add_patch(circle2)
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')
        axes[0, 1].set_title('Safety Probability: Type 1')
        axes[0, 1].set_aspect('equal')
        
        # 3. Safety probability (type 2)
        im3 = axes[0, 2].imshow(P2, extent=[-5, 5, -5, 5], 
                               origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
        # plt.colorbar(im3, ax=axes[0, 2], label='P(x)')
        circle3 = plt.Circle((0, 0), 1.0 + self.robot_radius, 
                            fill=False, linestyle='--', linewidth=2, color='blue')
        axes[0, 2].add_patch(circle3)
        axes[0, 2].set_xlabel('X')
        axes[0, 2].set_ylabel('Y')
        axes[0, 2].set_title('Safety Probability: Type 2 (with deadzone)')
        axes[0, 2].set_aspect('equal')
        
        # 4. Safety probability (type 3 - dynamic)
        im4 = axes[1, 0].imshow(P3, extent=[-5, 5, -5, 5], 
                               origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
        # plt.colorbar(im4, ax=axes[1, 0], label='P(x,v)')
        circle4 = plt.Circle((0, 0), 1.0 + self.robot_radius, 
                            fill=False, linestyle='--', linewidth=2, color='blue')
        axes[1, 0].add_patch(circle4)
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        axes[1, 0].set_title('Safety Probability: Type 3 (dynamic, v=[0.5,0])')
        axes[1, 0].set_aspect('equal')

        im5 = axes[1, 0].imshow(P4, extent=[-5, 5, -5, 5], 
                               origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
        # plt.colorbar(im5, ax=axes[1, 0], label='P(x,v)')
        circle5 = plt.Circle((0, 0), 1.0 + self.robot_radius, 
                            fill=False, linestyle='--', linewidth=2, color='blue')
        axes[1, 0].add_patch(circle5)
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        axes[1, 0].set_title('Safety Probability: Type 4 (dynamic, v=[0.5,0])')
        axes[1, 0].set_aspect('equal')

        # 5. Radial cross-section
        axes[1, 1].set_position([0.55, 0.05, 0.4, 0.4])  # Adjust position
        
        # Extract radial profile along x-axis
        center_idx = X.shape[1] // 2
        radial_distances = X[center_idx, :]
        
        # Probabilities along x-axis
        p1_profile = P1[center_idx, :]
        p2_profile = P2[center_idx, :]
        p3_profile = P3[center_idx, :]
        p4_profile = P4[center_idx, :]
        
        axes[1, 1].plot(radial_distances, p1_profile, 'b-', linewidth=2, 
                       label='Type 1: P=exp(-(R+r)²/(d²+ε))')
        axes[1, 1].plot(radial_distances, p2_profile, 'r--', linewidth=2, 
                       label='Type 2: P=exp(-max(0,1-d²/(R+r)²)ᵞ)')
        axes[1, 1].plot(radial_distances, p3_profile, 'g-.', linewidth=2, 
                       label='Type 3: Dynamic with velocity')
        axes[1, 1].plot(radial_distances, p4_profile, 'm:', linewidth=2, 
                       label='Type 4: Dynamic with velocity')
        
        # Mark obstacle boundary
        safety_radius = 1.0 + self.robot_radius
        axes[1, 1].axvline(x=-safety_radius, color='k', linestyle=':', alpha=0.5)
        axes[1, 1].axvline(x=safety_radius, color='k', linestyle=':', alpha=0.5)
        axes[1, 1].axvspan(-safety_radius, safety_radius, alpha=0.1, color='red')
        
        axes[1, 1].set_xlabel('Distance along x-axis')
        axes[1, 1].set_ylabel('Safety Probability')
        axes[1, 1].set_title('Radial Cross-section Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(-0.05, 1.05)
        
        # Remove unused subplot
        axes[1, 2].remove()
        
        plt.suptitle('expBarrierFunc/Comparison of Exponential Barrier Function Formulations', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return X, Y, P1, P2, P3, B1
    
    def compare_with_normal_cdf(self, show=True, output_dir=None):
        """Plot the original three curves in a compact paper-ready panel.

        Save vector PDF and 300-dpi PNG next to this script by default.
        The return tuple retains the original reciprocal-exponential values.
        """
        safety_radius = 1.0 + self.robot_radius
        sigma = 0.5
        distances = np.linspace(0, 2 * safety_radius, 500)
        h = distances**2 - safety_radius**2
        exp_probs = np.exp(-safety_radius**2 / (distances**2 + self.epsilon) / sigma)
        exp_probs_max = np.exp(-np.maximum(-h, 0) / sigma)
        cdf_probs = norm.cdf(h / sigma)
        hybrid_probs = expit(h / sigma)
        colors = ['#C46732', '#3467A1', '#228575']
        curves = [exp_probs_max, cdf_probs, hybrid_probs]
        labels = ['Exponential', 'Normal CDF', 'Logistic']
        linestyles = ['-', '--', '-.']
        output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / 'expBarrier_vs_NormalCDF_Comparison'

        paper_style = dict(PLOT_STYLE, **{
            'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif',
            'font.size': 8, 'axes.labelsize': 9,
            'xtick.labelsize': 8, 'ytick.labelsize': 8,
            'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
            'axes.edgecolor': '#555555',
        })
        with plt.rc_context(paper_style):
            fig, ax = plt.subplots(figsize=(3.5, 2.6), layout='constrained')
            ax.set_axisbelow(True)
            ax.grid(True, linewidth=0.4, alpha=0.3)
            ax.axvline(safety_radius, color='#777777', ls=':', lw=1)
            ax.text(safety_radius + 0.035, 0.12, r'$R_s$', color='#555555', fontsize=8)
            for values, color, style, label in zip(curves, colors, linestyles, labels):
                ax.plot(distances, values, color=color, ls=style, lw=1.6, label=label)
            ax.set(xlim=(0, 2 * safety_radius), ylim=(-0.02, 1.04),
                   xlabel=r'Distance $d$ (m)', ylabel='Safety probability')
            ax.set_yticks(np.linspace(0, 1, 5))
            ax.tick_params(direction='out', length=3, width=0.6)
            ax.legend(loc='upper left', frameon=False, fontsize=8, handlelength=2.5)
            fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
            fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white')
            if show:
                plt.show()
            plt.close(fig)

        print(f"Saved comparison: {stem.with_suffix('.pdf')}")
        print('At d = R_s: clipped exponential = 1.0; Normal CDF = logistic = 0.5')
        return distances, exp_probs.tolist(), cdf_probs.tolist(), hybrid_probs.tolist()

if __name__ == "__main__":
    print("="*70)
    print("EXPONENTIAL BARRIER FUNCTION SAFETY PROBABILITY")
    print("="*70)
    print("\nExploring: P(x) = exp(-b(x)) where b(x) is a barrier function")
    
    obstacles = [(0, 0, 1.0)]
    
    exp_safety = ExponentialBarrierSafety(
        obstacles=obstacles,
        robot_radius=0.2,
        epsilon=0.01,
        scale_factor=2.0,  
        gamma=2.0          
    )

    print("\n1. Visualizing different barrier function formulations...")
    # X, Y, P1, P2, P3, B1 = exp_safety.visualize_comparison()
    
    print("\n2. Comparing with Normal CDF approach...")
    distances, exp_probs, cdf_probs, hybrid_probs = exp_safety.compare_with_normal_cdf()
    
    print("\n3. Testing specific positions:")
    print("-"*50)
    
    test_positions = [
        ("Inside obstacle", [0.5, 0]),
        ("At boundary", [1.2, 0]),  # R+r = 1.0+0.2 = 1.2
        ("Near obstacle", [1.5, 0]),
        ("Far away", [3.0, 0])
    ]
    
    for desc, pos in test_positions:
        dist = np.linalg.norm(pos)
        
        result_exp = exp_safety.total_safety_probability(
            pos, velocity=[0.5, 0], barrier_type=1
        )
        
        safety_radius = 1.0 + 0.2
        h_val = dist**2 - safety_radius**2
        sigma = 0.3
        z_score = h_val / sigma
        p_cdf = norm.cdf(z_score)
        
        print(f"\n{desc} at {pos} (distance={dist:.2f}):")
        print(f"  Exponential barrier: P = {result_exp['total_probability']:.6f}")
        print(f"  Normal CDF (σ={sigma}): P = Φ({z_score:.3f}) = {p_cdf:.6f}")
        
        if result_exp['individual_results']:
            barrier_val = result_exp['individual_results'][0]['barrier_value']
            print(f"  Barrier value b(x) = {barrier_val:.4f}")
    
    print("\n4. Numerical properties analysis:")
    print("-"*50)
    
    extreme_positions = [
        ("Very close to center", [0.01, 0]),
        ("Very far", [100, 0])
    ]
    
    for desc, pos in extreme_positions:
        result = exp_safety.total_safety_probability(pos, barrier_type=1)
        p = result['total_probability']
        
        print(f"{desc}:")
        print(f"  Position: {pos}")
        print(f"  Safety probability: {p:.10e}")
        
        if p < 1e-100:
            print(f"  → Effectively 0 (underflow protection needed)")
    
    print("\n5. Gradient calculation for optimization:")
    print("-"*50)
    
    test_point = np.array([2.0, 0.0])
    epsilon_grad = 1e-6
    
    grad_numerical = np.zeros(2)
    for i in range(2):
        # f(x + ε)
        point_plus = test_point.copy()
        point_plus[i] += epsilon_grad
        result_plus = exp_safety.total_safety_probability(point_plus, barrier_type=1)
        f_plus = result_plus['total_probability']
        
        # f(x - ε)
        point_minus = test_point.copy()
        point_minus[i] -= epsilon_grad
        result_minus = exp_safety.total_safety_probability(point_minus, barrier_type=1)
        f_minus = result_minus['total_probability']
        
        grad_numerical[i] = (f_plus - f_minus) / (2 * epsilon_grad)
    
    print(f"At position {test_point}:")
    print(f"  Safety probability: {exp_safety.total_safety_probability(test_point, barrier_type=1)['total_probability']:.6f}")
    print(f"  Numerical gradient: [{grad_numerical[0]:.6f}, {grad_numerical[1]:.6f}]")
    print(f"  Gradient direction: points toward safety increase")
    
    print("\n" + "="*70)
    print("SUMMARY: exp(-b(x)) as a safety probability function")
    print("="*70)
    print("\nAdvantages:")
    print("1. Simple and computationally efficient")
    print("2. Naturally bounded in (0,1] (or [0,1] with modifications)")
    print("3. Smooth and differentiable everywhere (with proper b(x))")
    print("4. Easy to tune with scale parameter")
    
    print("\nDisadvantages:")
    print("1. No direct connection to probability theory")
    print("2. Boundary probability not naturally 0.5")
    print("3. Harder to incorporate uncertainty explicitly")
    print("4. Less theoretical guarantees than CDF-based methods")
    
    print("\nUse when: Computation speed > Theoretical rigor")
    print("Use Φ(z) when: Uncertainty modeling > Computation speed")