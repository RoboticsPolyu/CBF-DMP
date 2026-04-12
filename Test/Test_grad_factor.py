import torch
import numpy as np
import matplotlib.pyplot as plt

# Set Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9

def exponential_triggering_schedule(t, t_start=0.3, gamma_max=1.0, alpha=2.0):
    """
    Exponential triggering schedule for time-dependent gamma(t).
    
    Args:
        t: Normalized time (1 → 0 in reverse diffusion)
        t_start: Start time for constraint activation
        gamma_max: Maximum gamma value
        alpha: Sharpness parameter for exponential increase
    
    Returns:
        gamma(t) value
    """
    # Convert to numpy for calculation
    if isinstance(t, torch.Tensor):
        t_np = t.cpu().numpy()
    else:
        t_np = t
    
    # Calculate gamma based on exponential schedule
    indicator = (t_np <= t_start).astype(float)
    gamma_values = gamma_max * (1 - t_np / t_start) ** alpha * indicator
    
    # Convert back to tensor if input was tensor
    if isinstance(t, torch.Tensor):
        return torch.from_numpy(gamma_values).to(t.device).float()
    return gamma_values

def compare_guidance_effects_corrected():
    """
    Compare different guidance scaling strategies with exponential triggering schedule.
    CORRECTED VERSION: Proper σ_t definition (noise scale).
    """
    # Diffusion process parameters
    T = 100  # Total diffusion steps
    beta_start, beta_end = 0.0001, 0.02
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    # CORRECT: σ_t = sqrt(1 - ᾱ_t) for noise scale in x_t = √ᾱ_t x_0 + σ_t ε
    # In forward diffusion (t=0→T):
    # t=0: ᾱ_0=1, σ_0=0 (no noise)
    # t=T: ᾱ_T≈0, σ_T≈1 (full noise)
    sigma_ts = torch.sqrt(1 - alpha_bars)  # σ_t: noise scale at timestep t
    
    print("DEBUG: First and last few sigma_ts values:")
    print(f"sigma_ts[0:5]: {sigma_ts[:5].numpy()}")
    print(f"sigma_ts[-5:]: {sigma_ts[-5:].numpy()}")
    
    # Guidance parameters
    gamma_max = 1.0
    grad_V_norm = 1.0  # Assume constant gradient norm for comparison
    t_start = 0.7  # Start time for constraint activation (normalized)
    alpha = 0.6  # Exponential sharpness parameter
    
    # Normalized time from 1 to 0 (reverse diffusion direction)
    # IMPORTANT: In reverse diffusion, we go from t=T to t=0
    # t_norm = 1 corresponds to diffusion step t=T (high noise)
    # t_norm = 0 corresponds to diffusion step t=0 (low noise)
    times_normalized = torch.linspace(1, 0, T)
    
    # Calculate time-dependent gamma using exponential triggering schedule
    gamma_ts = exponential_triggering_schedule(
        times_normalized, 
        t_start=t_start, 
        gamma_max=gamma_max, 
        alpha=alpha
    )
    
    # Convert normalized time to diffusion step index for analysis
    # Key insight: In reverse diffusion:
    # t_norm=1.0 → step T (full noise)
    # t_norm=0.0 → step 0 (clean data)
    
    print("\n" + "=" * 80)
    print("REVERSE DIFFUSION PROCESS (t=T→0)")
    print("=" * 80)
    print("Normalized time t_norm decreases from 1 to 0")
    print("Corresponding to diffusion step index decreasing from T to 0")
    print("\n" + "=" * 80)
    print("CORRECTED ANALYSIS OF σ_t EVOLUTION")
    print("=" * 80)
    
    # Select key timesteps for analysis (reverse order: T→0)
    t_indices = [T-1, 80, 70, 50, 30, 15, 5, 0]  # From high noise to low noise
    
    print(f"{'Step':<6} {'t_norm':<8} {'σ_t':<12} {'γ(t)':<12} "
          f"{'γ×σ×∇V':<15} {'γ/σ×∇V':<15}")
    print("-" * 80)
    
    for t_idx in t_indices:
        t_norm = times_normalized[t_idx].item()
        sigma_t = sigma_ts[t_idx].item()
        gamma_t = gamma_ts[t_idx].item()
        
        # Calculate different guidance scaling effects
        multiply_effect = gamma_t * grad_V_norm * sigma_t
        divide_effect = gamma_t * grad_V_norm / (sigma_t + 1e-8)
        
        print(f"{t_idx:<6} {t_norm:<8.3f} {sigma_t:<12.6f} {gamma_t:<12.6f} "
              f"{multiply_effect:<15.6f} {divide_effect:<15.6f}")
    
    # Calculate guidance strength series
    multiply_series = gamma_ts * sigma_ts.flip(0) * grad_V_norm
    divide_series = gamma_ts / (sigma_ts + 1e-8) * grad_V_norm
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(4.3, 3.1))
    
    # 1. Main comparison: Guidance strength over REVERSE diffusion steps
    diffusion_steps = torch.arange(T)  # 0 to T-1 (forward diffusion indices)
    reverse_steps = T - 1 - diffusion_steps  # Reverse order for visualization
    
    # Plot in reverse diffusion order (from noisy to clean)
    plt.plot(reverse_steps.numpy(), multiply_series.flip(0).numpy(), 
             'b-', linewidth=2, label=r'$\gamma (t) \cdot \sigma_t $')
    plt.plot(reverse_steps.numpy(), sigma_ts.numpy(), 
             'r-', linewidth=2, label=r'$\sigma_t$')
    plt.plot(reverse_steps.numpy(), gamma_ts.flip(0).numpy(),
             'g--', linewidth=1.5, label=r'$\gamma(t)$')
    

    # Mark constraint activation region
    # t_norm <= t_start means constraints are active
    # Convert t_start (normalized) to reverse step index
    t_start_step = int((1 - t_start) * T)
    plt.axvline(x=t_start_step, color='orange', linestyle=':', linewidth=1.5, 
                label=f'Constraint Start')
    # plt.axvspan(0, t_start_step, alpha=0.2, color='green', label='Constraints Active')
    # plt.axvspan(t_start_step, T-1, alpha=0.1, color='gray', label='Constraints Inactive')
    
    plt.xlabel('Reverse Diffusion Step (T→0)', fontsize=9)
    plt.ylabel('Effective Guidance Strength', fontsize=9)
    plt.title('Guidance Strength in Reverse Diffusion', fontsize=9)
    
    legend = plt.legend(
        fontsize=9,
        loc='upper left',
        bbox_to_anchor=(0.40, 0.98),
        frameon=False 
    )
    
    plt.grid(True, alpha=0.3)
    
    # # 2. Sigma_t evolution in FORWARD diffusion
    # ax2 = axes[0, 1]
    # ax2.plot(diffusion_steps.numpy(), sigma_ts.numpy(), 
    #          'purple', linewidth=2, label='σ_t')
    
    # # Mark corresponding points in forward diffusion
    # t_start_forward = int(t_start * T)  # Convert to forward diffusion index
    # ax2.axvline(x=t_start_forward, color='orange', linestyle=':', 
    #             linewidth=1.5, label=f't_start (forward)')
    
    # ax2.set_xlabel('Forward Diffusion Step (0→T)', fontsize=11)
    # ax2.set_ylabel('σ_t (Noise Scale)', fontsize=11)
    # ax2.set_title('Noise Scale σ_t in Forward Diffusion', fontsize=12)
    # ax2.legend(fontsize=9)
    # ax2.grid(True, alpha=0.3)
    
    # # 3. Gamma(t) schedule in NORMALIZED time
    # ax3 = axes[1, 0]
    
    # # Plot gamma(t) schedule in normalized time
    # ax3.plot(times_normalized.numpy(), gamma_ts.numpy(), 
    #          'b-', linewidth=2, label=f'γ(t), α={alpha}')
    # ax3.axvline(x=t_start, color='r', linestyle='--', alpha=0.7, 
    #             label=f't_start = {t_start}')
    
    # # Highlight regions
    # ax3.axvspan(0, t_start, alpha=0.1, color='green', label='Constraints Active')
    # ax3.axvspan(t_start, 1, alpha=0.1, color='red', label='Constraints Inactive')
    
    # ax3.set_xlabel('Normalized Time (t_norm)', fontsize=11)
    # ax3.set_ylabel('γ(t)', fontsize=11)
    # ax3.set_title('Exponential Triggering Schedule γ(t)', fontsize=12)
    # ax3.legend(fontsize=9)
    # ax3.grid(True, alpha=0.3)
    # ax3.set_xlim(1, 0)  # Reverse x-axis (1 → 0)
    # ax3.set_ylim(-0.05, gamma_max * 1.05)
    
    # # 4. Critical analysis: What happens near t=0 (final denoising steps)
    # ax4 = axes[1, 1]
    
    # # Focus on the last 20 steps of reverse diffusion
    # last_n_steps = 20
    # last_steps = torch.arange(last_n_steps)
    # last_indices = torch.arange(T-last_n_steps, T)
    
    # last_multiply = multiply_series[last_indices]
    # last_divide = divide_series[last_indices]
    # last_gamma = gamma_ts[last_indices]
    # last_sigma = sigma_ts[last_indices]
    
    # ax4.plot(last_steps.numpy(), last_multiply.numpy(), 
    #          'b-', linewidth=2, label='γ×σ×∇V (Multiply)')
    # ax4.plot(last_steps.numpy(), last_divide.numpy(), 
    #          'r-', linewidth=2, label='γ/σ×∇V (Divide)')
    # ax4.plot(last_steps.numpy(), last_gamma.numpy(),
    #          'g--', linewidth=1.5, label='γ(t)')
    # ax4.plot(last_steps.numpy(), last_sigma.numpy(),
    #          'purple', linewidth=1.5, label='σ_t')
    
    # # Mark the danger zone (σ_t → 0)
    # ax4.axvspan(18, 20, alpha=0.3, color='red', label='Danger: σ_t → 0')
    # ax4.text(19, last_divide[-3].item() * 0.7, 'EXPLOSION!\n1/σ_t → ∞', 
    #          ha='center', color='red', fontsize=10, fontweight='bold')
    
    # ax4.set_xlabel('Final Reverse Steps (t→0)', fontsize=11)
    # ax4.set_ylabel('Value', fontsize=11)
    # ax4.set_title('Final Steps: Critical Behavior Analysis', fontsize=12)
    # ax4.legend(fontsize=8)
    # ax4.grid(True, alpha=0.3)
    # ax4.set_yscale('log')  # Log scale to see explosion
    
    plt.tight_layout()
    plt.savefig('guidance_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS (CORRECTED):")
    print("=" * 80)
    print("\n1. σ_t EVOLUTION:")
    print("   - Forward diffusion (0→T): σ_t increases from 0 to ~1")
    print("   - Reverse diffusion (T→0): σ_t decreases from ~1 to 0")
    print("   - Final steps (t→0): σ_t → 0 (clean data)")
    
    print("\n2. GUIDANCE STRATEGIES in REVERSE DIFFUSION:")
    print("   a) γ(t) × σ_t × ∇V (Multiply by σ_t):")
    print("      - Guidance strength DECREASES as σ_t → 0")
    print("      - Natural damping in final steps")
    print("      - Stable convergence")
    
    print("\n   b) γ(t) / σ_t × ∇V (Divide by σ_t):")
    print("      - Guidance strength INCREASES as σ_t → 0")
    print("      - 1/σ_t → ∞ when σ_t → 0")
    print("      - HIGH RISK: Gradient explosion in final steps")
    
    print("\n3. EXPONENTIAL TRIGGERING SCHEDULE:")
    print(f"   - Constraints activate when t_norm ≤ {t_start}")
    print(f"   - In reverse diffusion: Activated in early-mid stages")
    print(f"   - Deactivated in final stages (when t_norm > {t_start})")
    
    print("\n4. RECOMMENDATION:")
    print("   ✔ USE: ε_guided = ε_pred + γ(t) × ∇V / σ_t")
    print("     This matches the paper formulation and is stable")
    print("   ✗ AVOID: ε_guided = ε_pred + γ(t) × σ_t × ∇V")
    print("     This damps guidance too much in final steps")

# Run the corrected analysis
if __name__ == "__main__":
    compare_guidance_effects_corrected()
