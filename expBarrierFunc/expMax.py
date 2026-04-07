import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import norm

class MinBarrierSafetyCorrected:
    """
    Safety probability using corrected versions of barrier functions
    Original idea: b(x) = min{0, d² - (R+r)²}
    But we implement proper probability formulations
    """
    
    def __init__(self, obstacles, robot_radius=0.2, alpha=1.0, beta=1.0, sigma=0.3):
        """
        Parameters:
        -----------
        obstacles : list
            (center_x, center_y, radius)
        robot_radius : float
            Robot radius
        alpha : float
            Scaling factor for sigmoid
        beta : float
            Scaling factor for exponential penalty
        sigma : float
            Uncertainty parameter for CDF
        """
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
    
    def logistic_sigmoid_version(self, position, obstacle):
        """
        Version 1: Logistic sigmoid (recommended)
        
        P(x) = 1 / (1 + exp(-α·(d² - (R+r)²)))
        
        This is the proper probabilistic interpretation
        of the original min barrier idea
        """
        center = np.array(obstacle[:2])
        radius = obstacle[2]
        pos = np.array(position)
        
        d_sq = np.sum((pos - center)**2)
        d_safe_sq = (radius + self.robot_radius)**2
        
        # Logistic function applied to h(x) = d² - d_safe²
        h = d_sq - d_safe_sq
        z = self.alpha * h
        p = 1.0 / (1.0 + np.exp(-z))
        
        return p, h, d_sq, d_safe_sq
    
    def exponential_penalty_version(self, position, obstacle):
        """
        Version 2: Exponential penalty only inside
        
        b(x) = max{0, (R+r)² - d²}
        P(x) = exp(-β·b(x))
        """
        center = np.array(obstacle[:2])
        radius = obstacle[2]
        pos = np.array(position)
        
        d_sq = np.sum((pos - center)**2)
        d_safe_sq = (radius + self.robot_radius)**2
        
        # Penalty only when inside obstacle region
        penalty = np.maximum(0, d_safe_sq - d_sq)
        
        # Safety probability
        p = np.exp(-self.beta * penalty)
        
        # Equivalent h(x) for comparison
        h = d_sq - d_safe_sq
        
        return p, penalty, d_sq, d_safe_sq, h
    
    def normal_cdf_version(self, position, obstacle):
        """
        Version 3: Normal CDF (most theoretically grounded)
        
        P(x) = Φ((d² - (R+r)²)/σ) = Φ(h(x)/σ)
        
        This has proper probabilistic interpretation with Gaussian uncertainty
        """
        center = np.array(obstacle[:2])
        radius = obstacle[2]
        pos = np.array(position)
        
        d_sq = np.sum((pos - center)**2)
        d_safe_sq = (radius + self.robot_radius)**2
        
        # CBF function h(x)
        h = d_sq - d_safe_sq
        
        # Normal CDF transformation
        z = h / self.sigma
        p = norm.cdf(z)
        
        return p, h, d_sq, d_safe_sq, z
    
    def softplus_smoothed_version(self, position, obstacle):
        """
        Version 4: Softplus smoothed barrier
        
        b(x) = log(1 + exp((R+r)² - d²))
        P(x) = exp(-b(x))
        """
        center = np.array(obstacle[:2])
        radius = obstacle[2]
        pos = np.array(position)
        
        d_sq = np.sum((pos - center)**2)
        d_safe_sq = (radius + self.robot_radius)**2
        
        # Softplus barrier
        arg = d_safe_sq - d_sq
        b = np.log(1.0 + np.exp(arg))  # softplus
        
        # Safety probability
        p = np.exp(-b)
        
        # Equivalent h(x)
        h = d_sq - d_safe_sq
        
        return p, b, d_sq, d_safe_sq, h
    
    def compare_formulations(self, grid_size=80):
        """
        Compare all corrected formulations
        """
        # Single obstacle at origin
        test_obstacle = (0, 0, 1.0)
        safety_radius = 1.0 + self.robot_radius
        
        # Create grid
        x = np.linspace(-2, 2, grid_size)
        y = np.linspace(-2, 2, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Initialize result arrays
        P_sigmoid = np.zeros_like(X)
        P_exp_penalty = np.zeros_like(X)
        P_cdf = np.zeros_like(X)
        P_softplus = np.zeros_like(X)
        
        # Barrier/helper function values
        H_values = np.zeros_like(X)  # h(x) = d² - d_safe²
        
        for i in range(grid_size):
            for j in range(grid_size):
                pos = [X[i, j], Y[i, j]]
                
                # Logistic sigmoid
                p_sig, h, _, _ = self.logistic_sigmoid_version(pos, test_obstacle)
                P_sigmoid[i, j] = p_sig
                H_values[i, j] = h
                
                # Exponential penalty
                p_exp, _, _, _, _ = self.exponential_penalty_version(pos, test_obstacle)
                P_exp_penalty[i, j] = p_exp
                
                # Normal CDF
                p_cdf, _, _, _, _ = self.normal_cdf_version(pos, test_obstacle)
                P_cdf[i, j] = p_cdf
                
                # Softplus
                p_sp, _, _, _, _ = self.softplus_smoothed_version(pos, test_obstacle)
                P_softplus[i, j] = p_sp
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Underlying function h(x) = d² - (R+r)²
        im1 = axes[0, 0].imshow(H_values, extent=[-2, 2, -2, 2], 
                               origin='lower', cmap='RdBu_r')
        plt.colorbar(im1, ax=axes[0, 0], label='h(x)')
        circle1 = Circle((0, 0), safety_radius, 
                        fill=False, linestyle='--', linewidth=2, color='red')
        axes[0, 0].add_patch(circle1)
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].set_title('CBF Function: h(x) = d² - (R+r)²\n(Negative inside, 0 at boundary, Positive outside)')
        axes[0, 0].set_aspect('equal')
        
        # 2. Logistic sigmoid: P = 1/(1+exp(-αh))
        im2 = axes[0, 1].imshow(P_sigmoid, extent=[-2, 2, -2, 2], 
                               origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(im2, ax=axes[0, 1], label='P(x)')
        circle2 = Circle((0, 0), safety_radius, 
                        fill=False, linestyle='--', linewidth=2, color='blue')
        axes[0, 1].add_patch(circle2)
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')
        axes[0, 1].set_title(f'Logistic: P=1/(1+exp(-αh)), α={self.alpha}')
        axes[0, 1].set_aspect('equal')
        
        # 3. Exponential penalty: P = exp(-β·max{0,(R+r)²-d²})
        im3 = axes[0, 2].imshow(P_exp_penalty, extent=[-2, 2, -2, 2], 
                               origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(im3, ax=axes[0, 2], label='P(x)')
        circle3 = Circle((0, 0), safety_radius, 
                        fill=False, linestyle='--', linewidth=2, color='blue')
        axes[0, 2].add_patch(circle3)
        axes[0, 2].set_xlabel('X')
        axes[0, 2].set_ylabel('Y')
        axes[0, 2].set_title(f'Exponential Penalty: P=exp(-β·max{{0,(R+r)²-d²}}), β={self.beta}')
        axes[0, 2].set_aspect('equal')
        
        # 4. Normal CDF: P = Φ(h/σ)
        im4 = axes[1, 0].imshow(P_cdf, extent=[-2, 2, -2, 2], 
                               origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(im4, ax=axes[1, 0], label='P(x)')
        circle4 = Circle((0, 0), safety_radius, 
                        fill=False, linestyle='--', linewidth=2, color='blue')
        axes[1, 0].add_patch(circle4)
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        axes[1, 0].set_title(f'Normal CDF: P=Φ(h/σ), σ={self.sigma}')
        axes[1, 0].set_aspect('equal')
        
        # 5. Softplus: P = exp(-log(1+exp((R+r)²-d²)))
        im5 = axes[1, 1].imshow(P_softplus, extent=[-2, 2, -2, 2], 
                               origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(im5, ax=axes[1, 1], label='P(x)')
        circle5 = Circle((0, 0), safety_radius, 
                        fill=False, linestyle='--', linewidth=2, color='blue')
        axes[1, 1].add_patch(circle5)
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        axes[1, 1].set_title('Softplus: P=exp(-log(1+exp((R+r)²-d²)))')
        axes[1, 1].set_aspect('equal')
        
        # 6. Radial cross-section comparison
        ax6 = axes[1, 2]
        
        # Extract along x-axis
        center_idx = grid_size // 2
        radial_dist = X[center_idx, :]
        
        # Extract probability profiles
        p_sig_profile = P_sigmoid[center_idx, :]
        p_exp_profile = P_exp_penalty[center_idx, :]
        p_cdf_profile = P_cdf[center_idx, :]
        p_sp_profile = P_softplus[center_idx, :]
        
        ax6.plot(radial_dist, p_sig_profile, 'r-', linewidth=2, 
                label=f'Logistic (α={self.alpha})')
        ax6.plot(radial_dist, p_exp_profile, 'b--', linewidth=2, 
                label=f'Exp Penalty (β={self.beta})')
        ax6.plot(radial_dist, p_cdf_profile, 'g-.', linewidth=2, 
                label=f'Normal CDF (σ={self.sigma})')
        ax6.plot(radial_dist, p_sp_profile, 'k:', linewidth=2, 
                label='Softplus')
        
        # Mark boundary
        ax6.axvline(x=-safety_radius, color='k', linestyle=':', alpha=0.5)
        ax6.axvline(x=safety_radius, color='k', linestyle=':', alpha=0.5)
        ax6.axhline(y=0.5, color='k', linestyle=':', alpha=0.3)
        ax6.axvspan(-safety_radius, safety_radius, alpha=0.1, color='red', 
                   label='Obstacle region')
        
        ax6.set_xlabel('Distance along x-axis')
        ax6.set_ylabel('Safety Probability')
        ax6.set_title('Radial Cross-section Comparison\nh(x) = d² - (R+r)²')
        ax6.legend(loc='best')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(-0.05, 1.05)
        
        plt.suptitle('Corrected Formulations Based on h(x) = d² - (R+r)²', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return X, Y, H_values, P_sigmoid, P_exp_penalty, P_cdf, P_softplus
    
    def analyze_properties(self):
        """
        Analyze key properties of each formulation
        """
        safety_radius = 1.0 + self.robot_radius
        d_safe_sq = safety_radius**2
        
        print("\n" + "="*70)
        print("PROPERTY ANALYSIS OF CORRECTED FORMULATIONS")
        print("="*70)
        
        # Test points at different regions
        test_cases = [
            ("Inside obstacle", 0.5),      # d < R+r
            ("At boundary", safety_radius), # d = R+r
            ("Outside obstacle", 2.0),      # d > R+r
        ]
        
        for desc, d in test_cases:
            d_sq = d**2
            h = d_sq - d_safe_sq
            
            print(f"\n{desc}: d={d:.2f}, h(x)={h:.4f}")
            print("-"*50)
            
            # Calculate all formulations
            test_obstacle = (0, 0, 1.0)
            pos = [d, 0]  # On x-axis
            
            # Logistic
            p_sig, _, _, _ = self.logistic_sigmoid_version(pos, test_obstacle)
            
            # Exponential penalty
            p_exp, penalty, _, _, _ = self.exponential_penalty_version(pos, test_obstacle)
            
            # Normal CDF
            p_cdf, _, _, _, z = self.normal_cdf_version(pos, test_obstacle)
            
            # Softplus
            p_sp, b_sp, _, _, _ = self.softplus_smoothed_version(pos, test_obstacle)
            
            print(f"Logistic:           P = 1/(1+exp(-{self.alpha}×{h:.4f})) = {p_sig:.6f}")
            print(f"Exponential Penalty: penalty={penalty:.4f}, P=exp(-{self.beta}×{penalty:.4f}) = {p_exp:.6f}")
            print(f"Normal CDF:         z={h:.4f}/{self.sigma:.4f}={z:.4f}, P=Φ({z:.4f}) = {p_cdf:.6f}")
            print(f"Softplus:           b=log(1+exp({d_safe_sq:.4f}-{d_sq:.4f}))={b_sp:.4f}, P=exp(-{b_sp:.4f})={p_sp:.6f}")
    
    def total_safety_probability(self, position, method='product', formulation='logistic'):
        """
        Compute total safety probability for all obstacles
        
        Parameters:
        -----------
        position : array-like
            Robot position [x, y]
        method : str
            'product', 'min', or 'softmin' for combining probabilities
        formulation : str
            'logistic', 'exponential', 'cdf', or 'softplus'
            
        Returns:
        --------
        dict with total probability and individual results
        """
        individual_results = []
        
        for i, obstacle in enumerate(self.obstacles):
            if formulation == 'logistic':
                p_i, h, d_sq, d_safe_sq = self.logistic_sigmoid_version(position, obstacle)
            elif formulation == 'exponential':
                p_i, penalty, d_sq, d_safe_sq, h = self.exponential_penalty_version(position, obstacle)
            elif formulation == 'cdf':
                p_i, h, d_sq, d_safe_sq, z = self.normal_cdf_version(position, obstacle)
            elif formulation == 'softplus':
                p_i, b, d_sq, d_safe_sq, h = self.softplus_smoothed_version(position, obstacle)
            else:
                raise ValueError(f"Unknown formulation: {formulation}")
            
            individual_results.append({
                'obstacle_id': i,
                'probability': p_i,
                'h_value': h,
                'distance': np.sqrt(d_sq),
                'safety_distance': np.sqrt(d_safe_sq),
                'safe': h >= 0
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
            'formulation': formulation
        }
    
    def gradient_analysis(self, position, formulation='logistic'):
        """
        Compute gradient of safety probability for optimization
        """
        if formulation == 'logistic':
            # P = 1/(1+exp(-αh)), h = d² - d_safe²
            # dP/dx = P(1-P) * α * dh/dx
            # dh/dx = 2(x - o)
            
            grad_total = np.zeros(2)
            pos_array = np.array(position)
            
            for obstacle in self.obstacles:
                center = np.array(obstacle[:2])
                radius = obstacle[2]
                
                # Compute h and P
                d_sq = np.sum((pos_array - center)**2)
                d_safe_sq = (radius + self.robot_radius)**2
                h = d_sq - d_safe_sq
                z = self.alpha * h
                P = 1.0 / (1.0 + np.exp(-z))
                
                # Gradient of P
                dP_dh = P * (1 - P) * self.alpha
                dh_dx = 2 * (pos_array - center)
                
                grad_total += dP_dh * dh_dx
            
            return grad_total
        
        elif formulation == 'exponential':
            # P = exp(-β·max{0, d_safe²-d²})
            grad_total = np.zeros(2)
            pos_array = np.array(position)
            
            for obstacle in self.obstacles:
                center = np.array(obstacle[:2])
                radius = obstacle[2]
                
                d_sq = np.sum((pos_array - center)**2)
                d_safe_sq = (radius + self.robot_radius)**2
                
                if d_sq < d_safe_sq:  # Inside penalty region
                    penalty = d_safe_sq - d_sq
                    P = np.exp(-self.beta * penalty)
                    
                    # Gradient
                    dP_dpenalty = -self.beta * P
                    dpenalty_dd_sq = -1
                    dd_sq_dx = 2 * (pos_array - center)
                    
                    grad_total += dP_dpenalty * dpenalty_dd_sq * dd_sq_dx
            
            return grad_total
        
        else:
            # For other formulations, use numerical gradient
            epsilon = 1e-6
            grad = np.zeros(2)
            
            for i in range(2):
                pos_plus = list(position)
                pos_minus = list(position)
                pos_plus[i] += epsilon
                pos_minus[i] -= epsilon
                
                result_plus = self.total_safety_probability(pos_plus, formulation=formulation)
                result_minus = self.total_safety_probability(pos_minus, formulation=formulation)
                
                grad[i] = (result_plus['total_probability'] - 
                          result_minus['total_probability']) / (2 * epsilon)
            
            return grad

# 主程序演示
if __name__ == "__main__":
    print("="*70)
    print("CORRECTED BARRIER FUNCTION FORMULATIONS")
    print("="*70)
    print("\nBased on h(x) = d² - (R+r)² with proper probability transformations")
    
    # 初始化
    obstacles = [(0, 0, 1.0)]  # 单个障碍物在原点
    
    corrected_safety = MinBarrierSafetyCorrected(
        obstacles=obstacles,
        robot_radius=0.2,
        alpha=2.0,    # Sigmoid sharpness
        beta=2.0,     # Exponential penalty strength
        sigma=0.5     # Uncertainty for CDF
    )
    
    # 1. 比较所有修正后的公式
    print("\n1. Comparing all corrected formulations...")
    X, Y, H, P_sig, P_exp, P_cdf, P_sp = corrected_safety.compare_formulations(grid_size=80)
    
    # 2. 分析关键属性
    print("\n2. Analyzing key properties...")
    corrected_safety.analyze_properties()
    
    # 3. 多障碍物场景测试
    print("\n3. Multi-obstacle scenario test...")
    print("-"*50)
    
    # 多个障碍物
    multi_obstacles = [
        (-2, 0, 0.8),
        (0, 1, 0.6),
        (1, -1, 1.0)
    ]
    
    multi_safety = MinBarrierSafetyCorrected(
        obstacles=multi_obstacles,
        robot_radius=0.2,
        alpha=2.0,
        beta=2.0,
        sigma=0.5
    )
    
    # 测试不同点
    test_points = [
        ("Between obstacles", [0, 0]),
        ("Near obstacle 1", [-2, 0.5]),
        ("Inside obstacle 3", [1, -1]),
        ("Far from all", [3, 3])
    ]
    
    formulations = ['logistic', 'exponential', 'cdf', 'softplus']
    
    for desc, pos in test_points:
        print(f"\n{desc} at {pos}:")
        print("-"*40)
        
        for formulation in formulations:
            result = multi_safety.total_safety_probability(pos, formulation=formulation)
            p_total = result['total_probability']
            
            print(f"  {formulation.capitalize()}: P_total = {p_total:.6f}")
            
            # 显示最危险的障碍物
            min_prob = min([r['probability'] for r in result['individual_results']])
            for r in result['individual_results']:
                if r['probability'] == min_prob:
                    status = "INSIDE" if not r['safe'] else "OUTSIDE"
                    print(f"    Most dangerous: Obstacle {r['obstacle_id']+1}, "
                          f"d={r['distance']:.2f}, d_safe={r['safety_distance']:.2f} [{status}], "
                          f"P={r['probability']:.4f}")
                    break
    
    # 4. 梯度分析（用于优化）
    print("\n4. Gradient analysis for optimization...")
    print("-"*50)
    
    test_position = [0.5, 0.5]
    
    for formulation in formulations:
        grad = multi_safety.gradient_analysis(test_position, formulation=formulation)
        grad_norm = np.linalg.norm(grad)
        
        print(f"\n{formulation.capitalize()} formulation at {test_position}:")
        print(f"  Gradient: [{grad[0]:.6f}, {grad[1]:.6f}]")
        print(f"  Norm: {grad_norm:.6f}")
        
        if grad_norm > 1e-10:
            # 归一化梯度方向
            grad_dir = grad / grad_norm
            print(f"  Direction: {grad_dir}")
            
            # 解释方向
            if np.dot(grad_dir, -np.array(test_position)) > 0.7:
                print("  → Points away from origin (toward safety)")
            elif np.dot(grad_dir, np.array(test_position)) > 0.7:
                print("  → Points toward origin (dangerous!)")
    
    # 5. 边界情况下的行为
    print("\n5. Behavior at extreme cases...")
    print("-"*50)
    
    extreme_cases = [
        ("Exactly at obstacle center", [0, 0]),
        ("Very far", [10, 10]),
        ("On boundary of obstacle 1", [-2 + 1.0, 0])  # d = R+r = 0.8+0.2=1.0
    ]
    
    for desc, pos in extreme_cases:
        print(f"\n{desc} at {pos}:")
        
        # 使用logistic公式（最稳定）
        result = multi_safety.total_safety_probability(pos, formulation='logistic')
        
        print(f"  Total safety probability: {result['total_probability']:.10f}")
        
        # 检查是否有数值问题
        if result['total_probability'] == 0:
            print("  WARNING: Probability is exactly 0 (possible underflow)")
        elif result['total_probability'] == 1:
            print("  WARNING: Probability is exactly 1 (possible saturation)")
        
        # 显示各障碍物的贡献
        for r in result['individual_results']:
            if r['probability'] < 0.01:
                print(f"  Obstacle {r['obstacle_id']+1}: P={r['probability']:.2e} (dangerous!)")
    
    print("\n" + "="*70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*70)
    
    print("\nYour original idea: b(x) = min{0, d²-(R+r)²}")
    print("Problem: Directly applying exp(-b(x)) gives invalid probability (>1 inside)")
    
    print("\nCorrect formulations based on h(x) = d²-(R+r)²:")
    print("1. Logistic sigmoid: P = 1/(1+exp(-αh))")
    print("   - Pros: Always in (0,1), P=0.5 at boundary, smooth gradient")
    print("   - Cons: No direct uncertainty modeling")
    
    print("\n2. Normal CDF: P = Φ(h/σ)")
    print("   - Pros: Proper probabilistic interpretation with uncertainty σ")
    print("   - Cons: Requires computing Φ (slightly more expensive)")
    
    print("\n3. Exponential penalty: P = exp(-β·max{0,(R+r)²-d²})")
    print("   - Pros: Simple, P=1 outside obstacle")
    print("   - Cons: P=1 at boundary (zero gradient)")
    
    print("\n4. Softplus: P = exp(-log(1+exp((R+r)²-d²)))")
    print("   - Pros: Smooth approximation, P≈0.5 at boundary")
    print("   - Cons: Slightly more complex")
    
    print("\nRecommendation for practical use:")
    print("- For theoretical work: Use Normal CDF (probabilistically sound)")
    print("- For optimization: Use Logistic sigmoid (good gradients everywhere)")
    print("- For simplicity: Use Exponential penalty (but beware zero gradient at boundary)")
    
    print("\nAll these are proper probability functions: 0 ≤ P(x) ≤ 1")
    print("And all are based on your core idea: h(x) = d² - (R+r)²")