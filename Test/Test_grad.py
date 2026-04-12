import torch
import numpy as np
import matplotlib.pyplot as plt

from AeroDM_SafeTrj_v2_Test import compute_barrier_and_grad

def test_cbf_function():
    """
    Demo to verify the correctness of CBF barrier function computation.
    """
    torch.manual_seed(42)
    
    # ==================== 1. Configuration ====================
    class Config:
        pass
    
    config = Config()
    
    # Test data dimensions
    batch_size = 2
    seq_len = 10      # Trajectory length (timesteps)
    feature_dim = 6   # Assumed features: [?, x, y, z, ?, ?]
    
    # Normalization parameters (simulating training data statistics)
    mean = torch.zeros(1, 1, feature_dim)
    mean[0, 0, 1:4] = torch.tensor([0.0, 0.0, 0.0])  # Position mean at origin
    std = torch.ones(1, 1, feature_dim)
    std[0, 0, 1:4] = torch.tensor([2.0, 2.0, 2.0])   # Position standard deviation
    
    # ==================== 2. Create Test Trajectories ====================
    # Trajectory 1: Straight line through origin (will intersect obstacles)
    t = torch.linspace(-2, 2, seq_len)
    trajectory1 = torch.stack([
        torch.zeros_like(t),      # Dimension 0 (could be time or other)
        t,                        # x-coordinate: from -2 to 2
        torch.zeros_like(t),      # y-coordinate: constant 0
        torch.zeros_like(t),      # z-coordinate: constant 0
        torch.zeros_like(t),      # Other features
        torch.zeros_like(t)       # Other features
    ], dim=1).unsqueeze(0)        # Shape: (1, seq_len, 6)
    
    # Trajectory 2: Circular path around origin (should be safe)
    theta = torch.linspace(0, 2*np.pi, seq_len)
    trajectory2 = torch.stack([
        torch.zeros_like(theta),
        1.5 * torch.cos(theta),   # x = 1.5*cosθ
        1.5 * torch.sin(theta),   # y = 1.5*sinθ
        torch.zeros_like(theta),  # z = 0
        torch.zeros_like(theta),
        torch.zeros_like(theta)
    ], dim=1).unsqueeze(0)
    
    # Combine trajectories
    x = torch.cat([trajectory1, trajectory2], dim=0)  # Shape: (2, seq_len, 6)
    
    print("="*60)
    print("CBF Barrier Function Verification Demo")
    print("="*60)
    print(f"Trajectory shape: {x.shape}")
    print(f"Trajectory 1: Straight line from (-2,0,0) to (2,0,0)")
    print(f"Trajectory 2: Circular path with radius 1.5")
    
    # ==================== 3. Create Obstacle Data ====================
    # Obstacle 1: Center at origin, radius 1.0
    # Obstacle 2: Center at (1.0, 0.0, 0.0), radius 0.5
    obstacles_data = [
        [  # Obstacles for batch 0
            {'center': torch.tensor([0.0, 0.0, 0.0]), 'radius': 1.0},
            {'center': torch.tensor([1.0, 0.0, 0.0]), 'radius': 0.5}
        ],
        [  # Obstacles for batch 1 (can differ from batch 0)
            {'center': torch.tensor([0.0, 0.0, 0.0]), 'radius': 1.0}
        ]
    ]
    
    # ==================== 4. Call Fixed CBF Function ====================
    print("\nCalling fixed CBF function...")
    V, grad_V = compute_barrier_and_grad(x, config, mean, std, obstacles_data)
    
    print(f"\nComputation Results:")
    print(f"Barrier value V = {V.item():.6f}")
    print(f"Gradient shape: {grad_V.shape}")
    print(f"Gradient norm: {torch.norm(grad_V):.6f}")
    
    # ==================== 5. Detailed Analysis ====================
    print("\n" + "="*60)
    print("Detailed Violation Analysis per Trajectory")
    print("="*60)
    
    # Denormalize positions for analysis
    pos_denorm = x[:, :, 1:4] * std[0, 0, 1:4] + mean[0, 0, 1:4]
    
    for batch_idx in range(batch_size):
        print(f"\n--- Trajectory {batch_idx} ---")
        batch_obs = obstacles_data[batch_idx]
        
        total_violations = 0
        for obs_idx, obstacle in enumerate(batch_obs):
            center = obstacle['center']
            radius = obstacle['radius']
            
            # Compute distances to obstacle center
            distances = torch.norm(pos_denorm[batch_idx] - center, dim=1)
            
            # Compute violations
            violations = torch.clamp(radius - distances, min=0.0)
            violation_count = torch.sum(violations > 0).item()
            total_violations += violation_count
            
            print(f"  Obstacle {obs_idx} (center={center.tolist()}, radius={radius}):")
            print(f"    Violating points: {violation_count}/{seq_len}")
            
            # Display violating point positions
            if violation_count > 0:
                viol_indices = torch.where(violations > 0)[0]
                for idx in viol_indices[:3]:  # Show first 3 only
                    pos = pos_denorm[batch_idx, idx]
                    dist = distances[idx]
                    print(f"      Point{idx}: position={pos.tolist()}, distance={dist:.3f}")
                if violation_count > 3:
                    print(f"      ... plus {violation_count-3} more violating points")
    
    # ==================== 6. Gradient Direction Verification ====================
    print("\n" + "="*60)
    print("Gradient Direction Verification")
    print("="*60)
    
    for batch_idx in range(batch_size):
        print(f"\nTrajectory {batch_idx}:")
        
        # Find points with non-zero gradients
        grad_norms = torch.norm(grad_V[batch_idx, :, 1:4], dim=1)
        nonzero_indices = torch.where(grad_norms > 1e-6)[0]
        
        if len(nonzero_indices) == 0:
            print("  No gradient (safe trajectory or no violations)")
            continue
            
        print(f"  Points with gradients: {len(nonzero_indices)}")
        
        # Examine the first point with gradient
        idx = nonzero_indices[0].item()
        grad_at_point = grad_V[batch_idx, idx, 1:4]
        pos_at_point = pos_denorm[batch_idx, idx]
        
        print(f"  Point{idx}: position={pos_at_point.tolist()}")
        print(f"          gradient={grad_at_point.tolist()}")
        print(f"          gradient norm={grad_norms[idx].item():.6f}")
        
        # Check if gradient points away from nearest obstacle
        batch_obs = obstacles_data[batch_idx]
        for obs_idx, obstacle in enumerate(batch_obs):
            center = obstacle['center']
            radius = obstacle['radius']
            distance = torch.norm(pos_at_point - center)
            
            if distance < radius + 0.1:  # Near obstacle
                # Expected direction: away from obstacle center
                expected_dir = (pos_at_point - center) / (distance + 1e-6)
                actual_dir = grad_at_point / (grad_norms[idx] + 1e-6)
                
                cosine_similarity = torch.dot(expected_dir, actual_dir).item()
                print(f"    Relative to obstacle {obs_idx}:")
                print(f"      distance={distance:.3f} (radius={radius})")
                print(f"      expected direction={expected_dir.tolist()}")
                print(f"      actual direction={actual_dir.tolist()}")
                print(f"      cosine similarity={cosine_similarity:.3f} (should be near 1.0)")
    
    # ==================== 7. Numerical Gradient Verification ====================
    print("\n" + "="*60)
    print("Numerical Gradient Verification (Finite Difference)")
    print("="*60)
    
    # Select a test point
    test_batch = 0
    test_time = 5   # Middle point
    test_dim = 1    # x-coordinate
    
    eps = 1e-5  # Small perturbation
    
    # Create perturbed inputs
    x_plus = x.clone()
    x_minus = x.clone()
    x_plus[test_batch, test_time, test_dim] += eps
    x_minus[test_batch, test_time, test_dim] -= eps
    
    # Compute barrier values for perturbed inputs
    V_plus, _ = compute_barrier_and_grad(x_plus, config, mean, std, obstacles_data)
    V_minus, _ = compute_barrier_and_grad(x_minus, config, mean, std, obstacles_data)
    
    # Numerical gradient (finite difference)
    numerical_grad = (V_plus - V_minus) / (2 * eps)
    
    # Analytical gradient (from our function)
    _, grad_analytic = compute_barrier_and_grad(x, config, mean, std, obstacles_data)
    analytic_grad = grad_analytic[test_batch, test_time, test_dim]
    
    print(f"Test point: batch={test_batch}, time={test_time}, dim={test_dim}")
    print(f"Numerical gradient: {numerical_grad.item():.6f}")
    print(f"Analytical gradient: {analytic_grad.item():.6f}")
    print(f"Relative error: {abs(numerical_grad - analytic_grad) / (abs(analytic_grad) + 1e-8):.6f}")
    
    # ==================== 8. Boundary Case Tests ====================
    print("\n" + "="*60)
    print("Boundary Case Tests")
    print("="*60)
    
    # Test 1: No obstacles
    print("Test 1: No obstacles")
    V_no_obs, grad_no_obs = compute_barrier_and_grad(x, config, mean, std, obstacles_data=None)
    print(f"  V value: {V_no_obs.item():.6f} (should be 0)")
    print(f"  Gradient norm: {torch.norm(grad_no_obs).item():.6f} (should be 0)")
    
    # Test 2: Completely safe trajectory (far from all obstacles)
    print("\nTest 2: Completely safe trajectory")
    safe_traj = torch.zeros(1, seq_len, feature_dim)
    safe_traj[0, :, 1:4] = torch.tensor([5.0, 5.0, 5.0])  # Far away
    V_safe, grad_safe = compute_barrier_and_grad(safe_traj, config, mean, std, obstacles_data[0:1])
    print(f"  V value: {V_safe.item():.6f} (should be 0)")
    print(f"  Non-zero gradient points: {torch.sum(torch.norm(grad_safe[:, :, 1:4], dim=2) > 1e-6).item()}")
    
    # ==================== 9. Visualization ====================
    print("\n" + "="*60)
    print("Generating visualization...")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: XY-plane view
    ax1 = axes[0]
    
    # Plot obstacles
    colors = ['red', 'blue']
    for batch_idx in range(min(batch_size, 2)):
        batch_obs = obstacles_data[batch_idx] if batch_idx < len(obstacles_data) else []
        for obs_idx, obstacle in enumerate(batch_obs):
            center = obstacle['center']
            radius = obstacle['radius']
            circle = plt.Circle(center[:2].numpy(), radius, 
                              color=colors[obs_idx], alpha=0.3,
                              label=f'Obstacle{obs_idx}' if batch_idx==0 else None)
            ax1.add_patch(circle)
            ax1.plot(center[0], center[1], 'x', color=colors[obs_idx], markersize=10)
    
    # Plot trajectories
    for i in range(batch_size):
        traj_2d = pos_denorm[i, :, :2].detach().numpy()
        ax1.plot(traj_2d[:, 0], traj_2d[:, 1], 'o-', linewidth=2, 
                label=f'Trajectory{i}', markersize=4)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('XY Plane View')
    ax1.set_aspect('equal', 'box')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Barrier value along trajectory
    ax2 = axes[1]
    
    for i in range(batch_size):
        # Compute barrier value at each timestep
        V_per_step = torch.zeros(seq_len)
        pos_batch = pos_denorm[i]
        batch_obs = obstacles_data[i] if i < len(obstacles_data) else []
        
        for t in range(seq_len):
            V_t = 0.0
            for obstacle in batch_obs:
                center = obstacle['center']
                radius = obstacle['radius']
                distance = torch.norm(pos_batch[t] - center)
                violation = max(0.0, radius - distance)
                V_t += violation ** 2
            V_per_step[t] = V_t
        
        ax2.plot(range(seq_len), V_per_step.numpy(), 'o-', label=f'Trajectory{i}')
    
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Barrier Value V(t)')
    ax2.set_title('Barrier Value Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gradient magnitude
    ax3 = axes[2]
    
    for i in range(batch_size):
        # Compute gradient L2 norm at each timestep
        grad_norm = torch.norm(grad_V[i, :, 1:4], dim=1).detach().numpy()
        ax3.plot(range(seq_len), grad_norm, 's-', label=f'Trajectory{i}', alpha=0.7)
    
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Gradient Magnitude ||∇V||')
    ax3.set_title('Gradient Magnitude Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cbf_verification.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return V, grad_V, x, pos_denorm

if __name__ == "__main__":
    # Run the test
    V, grad_V, x, pos_denorm = test_cbf_function()
    
    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)
    print(f"Total barrier value V: {V.item():.6f}")
    print(f"Gradient tensor shape: {grad_V.shape}")
    print(f"Percentage of non-zero gradient elements: {(torch.abs(grad_V) > 1e-6).float().mean().item()*100:.2f}%")
    print("\nVerification complete! Check 'cbf_verification.png' for visualization.")