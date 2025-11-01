import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

class StableDynamicsNetwork(nn.Module):
    def __init__(self, state_dim=2, hidden_dims=[64, 64]):
        """
        Neural network to learn stable dynamics dx/dt = h(x)
        """
        super().__init__()
        self.state_dim = state_dim
        
        # Build the network layers
        layers = []
        dims = [state_dim] + hidden_dims + [state_dim]
        
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:  # No activation on output layer
                layers.append(nn.Tanh())
                
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def compute_contraction_loss(model, x_batch, beta=0.5):
    """
    Compute contraction loss that enforces stability via Jacobian condition
    Loss = max(0, λ_max(J_sym) + β) where we want λ_max < -β
    """
    batch_size = x_batch.shape[0]
    total_loss = 0.0
    valid_samples = 0
    
    for i in range(batch_size):
        x = x_batch[i].unsqueeze(0).requires_grad_(True)
        
        # Compute Jacobian J = dh/dx
        h = model(x)
        J = torch.zeros(h.shape[1], x.shape[1], device=x.device)
        
        for j in range(h.shape[1]):
            grad = torch.autograd.grad(h[:, j].sum(), x, create_graph=True, retain_graph=True)[0]
            J[j, :] = grad[0]
        
        # Symmetric part of Jacobian
        J_sym = (J + J.T) / 2
        
        # Compute maximum eigenvalue using power iteration (more efficient)
        try:
            # Simple power iteration to find maximum eigenvalue
            v = torch.randn(J_sym.shape[0], 1, device=x.device)
            v = v / torch.norm(v)
            
            for _ in range(20):  # Few iterations usually suffice
                v = torch.mm(J_sym, v)
                v = v / torch.norm(v)
                
            lambda_max = torch.mm(v.T, torch.mm(J_sym, v)).squeeze()
            
            # Penalize if lambda_max > -beta
            if lambda_max > -beta:
                total_loss += torch.nn.functional.relu(lambda_max + beta)
                valid_samples += 1
                
        except:
            # Skip if numerical issues
            continue
    
    return total_loss / max(valid_samples, 1) if valid_samples > 0 else torch.tensor(0.0)

def generate_training_data(num_samples=1000):
    """
    Generate training data for a stable 2D system
    True system: dx1/dt = -x1 + x2, dx2/dt = -x1 - x2
    This has Jacobian with eigenvalues -1 ± i (stable)
    """
    # Random initial conditions
    x_data = torch.randn(num_samples, 2) * 2.0
    
    # True stable dynamics (we'll try to learn this)
    # A = [[-1, 1], [-1, -1]]  # Stable focus
    A = torch.tensor([[-1.0, 1.0], [-1.0, -1.0]])
    
    # Compute derivatives
    x_dot_data = torch.mm(x_data, A.T)
    
    # Add some noise to make it more realistic
    x_dot_data += 0.01 * torch.randn_like(x_dot_data)
    
    return x_data, x_dot_data

def train_model_with_contraction():
    """Train model WITH contraction loss"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = StableDynamicsNetwork(state_dim=2, hidden_dims=[32, 32]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate training data
    x_data, x_dot_data = generate_training_data(2000)
    x_data, x_dot_data = x_data.to(device), x_dot_data.to(device)
    
    # Training parameters
    epochs = 800
    alpha = 0.1  # Weight for contraction loss
    beta = 0.5   # Desired contraction rate
    
    history = {'total_loss': [], 'mse_loss': [], 'contraction_loss': []}
    
    print("Training model WITH contraction loss...")
    for epoch in tqdm(range(epochs)):
        # Mini-batch training
        indices = torch.randperm(len(x_data))[:256]
        x_batch = x_data[indices]
        x_dot_batch = x_dot_data[indices]
        
        # Forward pass
        x_dot_pred = model(x_batch)
        
        # Compute losses
        mse_loss = nn.MSELoss()(x_dot_pred, x_dot_batch)
        contraction_loss = compute_contraction_loss(model, x_batch, beta)
        total_loss = mse_loss + alpha * contraction_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Store history
        history['total_loss'].append(total_loss.item())
        history['mse_loss'].append(mse_loss.item())
        history['contraction_loss'].append(contraction_loss.item())
        
    return model, history

def train_model_without_contraction():
    """Train model WITHOUT contraction loss (baseline)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = StableDynamicsNetwork(state_dim=2, hidden_dims=[32, 32]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate training data
    x_data, x_dot_data = generate_training_data(2000)
    x_data, x_dot_data = x_data.to(device), x_dot_data.to(device)
    
    # Training parameters
    epochs = 800
    
    history = {'mse_loss': []}
    
    print("Training model WITHOUT contraction loss...")
    for epoch in tqdm(range(epochs)):
        # Mini-batch training
        indices = torch.randperm(len(x_data))[:256]
        x_batch = x_data[indices]
        x_dot_batch = x_dot_data[indices]
        
        # Forward pass
        x_dot_pred = model(x_batch)
        
        # Compute loss (only MSE)
        mse_loss = nn.MSELoss()(x_dot_pred, x_dot_batch)
        
        # Backward pass
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
        
        # Store history
        history['mse_loss'].append(mse_loss.item())
        
    return model, history

def simulate_trajectory(model, x0, steps=1000, dt=0.01):
    """Simulate trajectory for a given initial condition"""
    trajectory = [x0.cpu().numpy()]
    x = x0.clone()
    
    with torch.no_grad():
        for _ in range(steps):
            dx = model(x.unsqueeze(0)).squeeze()
            x = x + dt * dx
            
            # Check for numerical instability
            if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                break
                
            trajectory.append(x.cpu().numpy())
            
    return np.array(trajectory)

def comprehensive_verification(model, device):
    """
    Comprehensive verification of contraction loss effectiveness
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE VERIFICATION")
    print("="*60)
    
    # 1. Jacobian Eigenvalue Analysis
    print("\n1. JACOBIAN EIGENVALUE ANALYSIS")
    print("-" * 40)
    
    test_points = torch.randn(100, 2, device=device) * 2.0
    max_eigenvalues = []
    contraction_rates = []
    
    for x in test_points:
        x = x.unsqueeze(0).requires_grad_(True)
        h = model(x)
        
        # Compute Jacobian
        J = torch.autograd.functional.jacobian(lambda x: model(x).sum(dim=0), x)
        J = J.squeeze().cpu().detach().numpy()
        
        # Symmetric part
        J_sym = (J + J.T) / 2
        eigvals = np.linalg.eigvalsh(J_sym)
        max_eig = np.max(eigvals)
        max_eigenvalues.append(max_eig)
        
        # Contraction rate estimate
        contraction_rate = -max_eig if max_eig < 0 else 0
        contraction_rates.append(contraction_rate)
    
    max_eigenvalues = np.array(max_eigenvalues)
    contraction_rates = np.array(contraction_rates)
    
    print(f"Max eigenvalue statistics:")
    print(f"  Mean: {max_eigenvalues.mean():.4f}")
    print(f"  Std:  {max_eigenvalues.std():.4f}")
    print(f"  Min:  {max_eigenvalues.min():.4f}")
    print(f"  Max:  {max_eigenvalues.max():.4f}")
    print(f"  % Stable points (λ_max < 0): {np.mean(max_eigenvalues < 0) * 100:.1f}%")
    print(f"  Average contraction rate: {contraction_rates.mean():.4f}")

    # 2. Trajectory Convergence Analysis
    print("\n2. TRAJECTORY CONVERGENCE ANALYSIS")
    print("-" * 40)
    
    # Test multiple initial conditions
    initial_conditions = [
        torch.tensor([2.0, 2.0], device=device),
        torch.tensor([-1.5, 1.5], device=device),
        torch.tensor([0.5, -2.0], device=device),
        torch.tensor([-2.0, -0.5], device=device)
    ]
    
    convergence_times = []
    final_distances = []
    
    for x0 in initial_conditions:
        traj = simulate_trajectory(model, x0)
        final_point = traj[-1]
        distance_to_origin = np.linalg.norm(final_point)
        final_distances.append(distance_to_origin)
        
        # Find convergence time (time to reach within 0.1 of origin)
        distances = np.linalg.norm(traj, axis=1)
        convergence_mask = distances < 0.1
        if np.any(convergence_mask):
            convergence_time = np.argmax(convergence_mask) * 0.01
        else:
            convergence_time = 10.0  # max time
        convergence_times.append(convergence_time)
    
    print(f"Convergence statistics:")
    print(f"  Mean convergence time: {np.mean(convergence_times):.2f}s")
    print(f"  Mean final distance: {np.mean(final_distances):.4f}")
    print(f"  % trajectories converged (final_dist < 0.1): {np.mean(np.array(final_distances) < 0.1) * 100:.1f}%")

    # 3. Pairwise Trajectory Analysis (Direct Contraction Test)
    print("\n3. PAIRWISE TRAJECTORY CONTRACTION")
    print("-" * 40)
    
    def analyze_pairwise_contraction(model, x01, x02):
        traj1 = simulate_trajectory(model, x01, steps=500, dt=0.01)
        traj2 = simulate_trajectory(model, x02, steps=500, dt=0.01)
        
        distances = np.linalg.norm(traj1 - traj2, axis=1)
        times = np.arange(len(distances)) * 0.01
        
        # Fit exponential decay: d(t) ≈ d0 * exp(-βt)
        if len(distances) > 10 and distances[0] > 1e-6:
            log_distances = np.log(distances[:100] + 1e-8)
            times_fit = times[:100]
            coeffs = np.polyfit(times_fit, log_distances, 1)
            estimated_beta = -coeffs[0]
        else:
            estimated_beta = 0.0
            
        return distances, estimated_beta
    
    # Test multiple pairs
    contraction_rates_pairwise = []
    for i in range(5):
        x01 = torch.randn(2, device=device) * 2
        x02 = torch.randn(2, device=device) * 2
        distances, beta = analyze_pairwise_contraction(model, x01, x02)
        contraction_rates_pairwise.append(beta)
    
    print(f"Pairwise contraction rates:")
    print(f"  Mean: {np.mean(contraction_rates_pairwise):.4f}")
    print(f"  Std:  {np.std(contraction_rates_pairwise):.4f}")
    print(f"  Min:  {np.min(contraction_rates_pairwise):.4f}")
    print(f"  Max:  {np.max(contraction_rates_pairwise):.4f}")

    return {
        'max_eigenvalues': max_eigenvalues,
        'convergence_times': convergence_times,
        'contraction_rates': contraction_rates_pairwise,
        'final_distances': final_distances
    }

def comparative_analysis(model_with, model_without, device):
    """
    Compare models trained with and without contraction loss
    """
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS")
    print("="*60)
    
    # Test points
    test_points = torch.randn(200, 2, device=device) * 3.0
    
    metrics = {}
    
    for name, model in [('With Contraction', model_with), 
                       ('Without Contraction', model_without)]:
        
        print(f"\n{name}:")
        print("-" * 30)
        
        # Eigenvalue analysis
        max_eigs = []
        for x in test_points:
            x = x.unsqueeze(0).requires_grad_(True)
            J = torch.autograd.functional.jacobian(lambda x: model(x).sum(dim=0), x)
            J = J.squeeze().cpu().detach().numpy()
            J_sym = (J + J.T) / 2
            max_eig = np.max(np.linalg.eigvalsh(J_sym))
            max_eigs.append(max_eig)
        
        max_eigs = np.array(max_eigs)
        stable_percentage = np.mean(max_eigs < 0) * 100
        
        # Trajectory stability
        unstable_trajectories = 0
        total_trajectories = 20
        
        for _ in range(total_trajectories):
            x0 = torch.randn(2, device=device) * 2
            traj = simulate_trajectory(model, x0)
            final_norm = np.linalg.norm(traj[-1])
            if final_norm > 2.0:
                unstable_trajectories += 1
        
        metrics[name] = {
            'stable_percentage': stable_percentage,
            'unstable_trajectories': unstable_trajectories,
            'mean_max_eigenvalue': max_eigs.mean(),
            'max_eigenvalue_std': max_eigs.std()
        }
        
        print(f"  Stable points: {stable_percentage:.1f}%")
        print(f"  Unstable trajectories: {unstable_trajectories}/{total_trajectories}")
        print(f"  Mean max eigenvalue: {max_eigs.mean():.4f}")
        print(f"  Eigenvalue std: {max_eigs.std():.4f}")
    
    return metrics

def sensitivity_analysis(model, device):
    """
    Analyze sensitivity to different regions of state space
    """
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Test different regions
    regions = {
        'Near Origin': torch.randn(50, 2, device=device) * 0.5,
        'Medium Range': torch.randn(50, 2, device=device) * 2.0,
        'Far Range': torch.randn(50, 2, device=device) * 5.0,
    }
    
    results = {}
    
    for region_name, points in regions.items():
        max_eigs = []
        for x in points:
            x = x.unsqueeze(0).requires_grad_(True)
            J = torch.autograd.functional.jacobian(lambda x: model(x).sum(dim=0), x)
            J = J.squeeze().cpu().detach().numpy()
            J_sym = (J + J.T) / 2
            max_eig = np.max(np.linalg.eigvalsh(J_sym))
            max_eigs.append(max_eig)
        
        max_eigs = np.array(max_eigs)
        stable_percentage = np.mean(max_eigs < 0) * 100
        
        results[region_name] = {
            'stable_percentage': stable_percentage,
            'mean_max_eigenvalue': max_eigs.mean(),
            'worst_case_eigenvalue': max_eigs.max()
        }
        
        print(f"{region_name:15}: {stable_percentage:5.1f}% stable, "
              f"mean λ_max: {max_eigs.mean():7.4f}, "
              f"worst λ_max: {max_eigs.max():7.4f}")
    
    return results

def create_comprehensive_plots(metrics_with, metrics_without, verification_results, history_with, history_without):
    """
    Create comprehensive visualization of verification results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Eigenvalue distribution comparison
    axes[0, 0].hist(verification_results['max_eigenvalues'], bins=50, alpha=0.7, 
                   color='blue', edgecolor='black')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stability Boundary')
    axes[0, 0].set_xlabel('Maximum Eigenvalue')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Maximum Eigenvalues\n(With Contraction Loss)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Training history
    axes[0, 1].plot(history_with['total_loss'], label='With Contraction Loss', color='blue')
    axes[0, 1].plot(history_without['mse_loss'], label='Without Contraction Loss', color='red')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].set_title('Training Loss Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Contraction rates
    axes[0, 2].hist(verification_results['contraction_rates'], bins=20, alpha=0.7,
                   color='orange', edgecolor='black')
    axes[0, 2].set_xlabel('Estimated Contraction Rate β')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Pairwise Contraction Rates\n(With Contraction Loss)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Stability percentage comparison
    models = ['With Contraction', 'Without Contraction']
    stable_percentages = [metrics_with['stable_percentage'], 
                         metrics_without['stable_percentage']]
    
    bars = axes[1, 0].bar(models, stable_percentages, color=['green', 'red'], alpha=0.7)
    axes[1, 0].set_ylabel('Stable Points (%)')
    axes[1, 0].set_title('Stability Percentage Comparison')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, stable_percentages):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value:.1f}%', ha='center', va='bottom')
    
    # 5. Mean maximum eigenvalues comparison
    mean_eigs = [metrics_with['mean_max_eigenvalue'], 
                metrics_without['mean_max_eigenvalue']]
    
    bars = axes[1, 1].bar(models, mean_eigs, color=['blue', 'orange'], alpha=0.7)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Stability Boundary')
    axes[1, 1].set_ylabel('Mean Maximum Eigenvalue')
    axes[1, 1].set_title('Mean Maximum Eigenvalue Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Unstable trajectories comparison
    unstable_traj = [metrics_with['unstable_trajectories'], 
                    metrics_without['unstable_trajectories']]
    
    bars = axes[1, 2].bar(models, unstable_traj, color=['red', 'darkred'], alpha=0.7)
    axes[1, 2].set_ylabel('Unstable Trajectories')
    axes[1, 2].set_title('Unstable Trajectories Comparison\n(out of 20 tested)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_trajectory_comparison(model_with, model_without, device):
    """
    Plot trajectory convergence for visual comparison
    """
    # Test initial conditions
    initial_conditions = [
        torch.tensor([2.0, 1.5], device=device),
        torch.tensor([-1.8, 1.2], device=device)
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for idx, x0 in enumerate(initial_conditions):
        # Simulate both models
        traj_with = simulate_trajectory(model_with, x0)
        traj_without = simulate_trajectory(model_without, x0)
        
        # Plot trajectories
        axes[idx].plot(traj_with[:, 0], traj_with[:, 1], 'b-', linewidth=2, 
                      label='With Contraction Loss', alpha=0.8)
        axes[idx].plot(traj_without[:, 0], traj_without[:, 1], 'r--', linewidth=2, 
                      label='Without Contraction Loss', alpha=0.8)
        
        # Mark start and end points
        axes[idx].plot(traj_with[0, 0], traj_with[0, 1], 'go', markersize=8, label='Start')
        axes[idx].plot(traj_with[-1, 0], traj_with[-1, 1], 'bo', markersize=8, label='End (With)')
        axes[idx].plot(traj_without[-1, 0], traj_without[-1, 1], 'ro', markersize=8, label='End (Without)')
        
        axes[idx].set_xlabel('x1')
        axes[idx].set_ylabel('x2')
        axes[idx].set_title(f'Trajectory Comparison - Initial {x0.cpu().numpy()}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axis('equal')
    
    plt.tight_layout()
    plt.show()

def plot_phase_portraits(model_with, model_without, device):
    """
    Plot phase portraits for both models
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Generate phase portrait grid
    x1 = np.linspace(-2, 2, 15)
    x2 = np.linspace(-2, 2, 15)
    X1, X2 = np.meshgrid(x1, x2)
    
    for idx, (model, title) in enumerate([(model_with, 'With Contraction Loss'), 
                                         (model_without, 'Without Contraction Loss')]):
        
        U = np.zeros_like(X1)
        V = np.zeros_like(X2)
        
        with torch.no_grad():
            for i in range(X1.shape[0]):
                for j in range(X1.shape[1]):
                    x = torch.tensor([[X1[i, j], X2[i, j]]], dtype=torch.float32).to(device)
                    dx = model(x).cpu().numpy()[0]
                    U[i, j] = dx[0]
                    V[i, j] = dx[1]
        
        # Plot vector field
        speed = np.sqrt(U**2 + V**2)
        axes[idx].quiver(X1, X2, U, V, speed, cmap='viridis', scale=20)
        axes[idx].set_xlabel('x1')
        axes[idx].set_ylabel('x2')
        axes[idx].set_title(f'Phase Portrait: {title}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the complete experiment"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("🚀 STARTING COMPLETE STABLE DYNAMICS LEARNING EXPERIMENT")
    print("="*70)
    
    # Step 1: Train both models
    print("\n📚 STEP 1: TRAINING MODELS")
    print("-" * 40)
    
    model_with, history_with = train_model_with_contraction()
    model_without, history_without = train_model_without_contraction()
    
    # Step 2: Comprehensive verification
    print("\n🔍 STEP 2: COMPREHENSIVE VERIFICATION")
    print("-" * 40)
    
    print("\nVerifying model WITH contraction loss...")
    results_with = comprehensive_verification(model_with, device)
    
    print("\nVerifying model WITHOUT contraction loss...")
    results_without = comprehensive_verification(model_without, device)
    
    # Step 3: Comparative analysis
    print("\n📊 STEP 3: COMPARATIVE ANALYSIS")
    print("-" * 40)
    
    metrics = comparative_analysis(model_with, model_without, device)
    
    # Step 4: Sensitivity analysis
    print("\n🎯 STEP 4: SENSITIVITY ANALYSIS")
    print("-" * 40)
    
    sensitivity_results = sensitivity_analysis(model_with, device)
    
    # Step 5: Create comprehensive plots
    print("\n📈 STEP 5: GENERATING VISUALIZATIONS")
    print("-" * 40)
    
    create_comprehensive_plots(metrics['With Contraction'], 
                             metrics['Without Contraction'], 
                             results_with, history_with, history_without)
    
    plot_trajectory_comparison(model_with, model_without, device)
    plot_phase_portraits(model_with, model_without, device)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE - SUMMARY")
    print("="*70)
    
    improvement_stability = (metrics['With Contraction']['stable_percentage'] - 
                           metrics['Without Contraction']['stable_percentage'])
    improvement_eigenvalue = (metrics['Without Contraction']['mean_max_eigenvalue'] - 
                            metrics['With Contraction']['mean_max_eigenvalue'])
    
    print(f"\n📈 KEY IMPROVEMENTS WITH CONTRACTION LOSS:")
    print(f"   • Stability percentage: +{improvement_stability:.1f}%")
    print(f"   • Mean max eigenvalue improvement: {improvement_eigenvalue:.4f}")
    print(f"   • Unstable trajectories reduction: "
          f"{metrics['Without Contraction']['unstable_trajectories'] - metrics['With Contraction']['unstable_trajectories']}")
    
    print(f"\n🎯 FINAL PERFORMANCE:")
    print(f"   • With Contraction Loss: {metrics['With Contraction']['stable_percentage']:.1f}% stable points")
    print(f"   • Without Contraction Loss: {metrics['Without Contraction']['stable_percentage']:.1f}% stable points")
    
    if metrics['With Contraction']['stable_percentage'] > 90:
        print(f"\n🎉 SUCCESS: Contraction loss effectively enforced stability!")
    else:
        print(f"\n⚠️  MODERATE SUCCESS: Contraction loss improved stability but may need tuning.")
    
    return model_with, model_without, metrics, results_with

if __name__ == "__main__":
    model_with, model_without, metrics, results = main()