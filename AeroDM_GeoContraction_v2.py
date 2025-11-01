import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# --- Mock AeroDM_Base.py for Demonstration ---
try:
    from AeroDM_Base import (
        Config, DiffusionTransformer, DiffusionProcess,
        generate_aerobatic_trajectories, normalize_trajectories,
        generate_target_waypoints, generate_action_styles, 
        generate_history_segments, test_model_performance,
    )
except ImportError:
    print("Warning: AeroDM_Base.py not found. Using mock classes for demonstration.")
    
    class Config:
        state_dim = 6
        latent_dim = 128
        seq_len = 50
        history_len = 10
        target_dim = 6
        action_dim = 4
        diffusion_steps = 100
        beta_start = 1e-4
        beta_end = 0.02
        contraction_weight = 0.01

    class DiffusionTransformer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.input_proj = nn.Linear(config.state_dim, config.latent_dim)
            self.output_proj = nn.Linear(config.latent_dim, config.state_dim)
            self.transformer = nn.TransformerEncoderLayer(
                d_model=config.latent_dim, nhead=4, dim_feedforward=256, batch_first=True
            )
            self.target_embed = nn.Linear(config.target_dim, config.latent_dim)
            self.action_embed = nn.Linear(config.action_dim, config.latent_dim)
            self.time_embed = nn.Embedding(config.diffusion_steps, config.latent_dim)
            
        def forward(self, x_t, t, target, action, history=None):
            x_embed = self.input_proj(x_t)
            t_embed = self.time_embed(t).unsqueeze(1)
            # Ensure target/action are properly shaped [B, 1, D]
            cond_embed = self.target_embed(target).unsqueeze(1) + self.action_embed(action).unsqueeze(1)
            
            full_seq = x_embed + t_embed + cond_embed
            
            out = self.transformer(full_seq)
            return self.output_proj(out)

    class DiffusionProcess:
        def __init__(self, config):
            self.config = config
            self.betas = torch.linspace(config.beta_start, config.beta_end, config.diffusion_steps)
            self.alphas = 1.0 - self.betas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        def q_sample(self, x_0, t):
            alpha_bars_on_device = self.alpha_bars.to(t.device)
            batch_size = x_0.shape[0]
            
            sqrt_alpha_bar = torch.sqrt(alpha_bars_on_device[t]).view(batch_size, 1, 1)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bars_on_device[t]).view(batch_size, 1, 1)
            
            noise = torch.randn_like(x_0)
            return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise, noise

        def p_sample(self, model, x_t, t_batch, target, action, history):
            with torch.no_grad():
                pred_x0 = model(x_t, t_batch, target, action, history)
                
                if t_batch[0] == 0:
                    return pred_x0
                    
                alpha_bars_on_device = self.alpha_bars.to(x_t.device)
                
                t_prev = t_batch - 1
                alpha_bar_t = alpha_bars_on_device[t_batch].view(-1, 1, 1)
                alpha_bar_t_prev = alpha_bars_on_device[t_prev].view(-1, 1, 1)
                
                pred_noise = (x_t - torch.sqrt(alpha_bar_t) * pred_x0) / torch.sqrt(1.0 - alpha_bar_t)
                
                x_prev = (
                    torch.sqrt(alpha_bar_t_prev) * pred_x0 +
                    torch.sqrt(1.0 - alpha_bar_t_prev) * pred_noise
                )
                return x_prev

    def generate_aerobatic_trajectories(num_trajectories, seq_len):
        t = torch.linspace(0, 4 * torch.pi, seq_len).unsqueeze(0)
        x = torch.cos(t) + torch.randn(num_trajectories, seq_len) * 0.1
        y = torch.sin(t) + torch.randn(num_trajectories, seq_len) * 0.1
        z = torch.linspace(0, 5, seq_len).unsqueeze(0).repeat(num_trajectories, 1)
        return torch.stack([x, y, z, x*0.1, y*0.1, z*0.1], dim=2)

    def normalize_trajectories(trajectories):
        mean = trajectories.mean(dim=[0, 1])
        std = trajectories.std(dim=[0, 1]) + 1e-6
        return (trajectories - mean) / std, mean, std

    def generate_target_waypoints(full_traj):
        return full_traj[:, -1, :] 

    def generate_action_styles(batch_size, action_dim, device):
        return torch.randn(batch_size, action_dim, device=device)

    def generate_history_segments(full_traj, history_len, device):
        return full_traj[:, :history_len, :]

    def test_model_performance(model, trajectories_norm, mean, std):
        # Simplified for brevity in the final answer, using the original mock code.
        print("\n--- Testing Model Performance (Placeholder) ---")
        sample = trajectories_norm[0:1]
        target = generate_target_waypoints(sample).squeeze(1)
        action = generate_action_styles(1, model.config.action_dim, device=sample.device)
        history = generate_history_segments(sample, model.config.history_len, device=sample.device)
        generated_traj = model.sample(target, action, history, batch_size=1)
        print(f"Generated trajectory shape: {generated_traj.shape}")

# --- End of Mock Classes ---


class GeometricContractionRegularizer(nn.Module):
    def __init__(self, config, diffusion_process, alpha=0.1, rank_approx=10):
        super().__init__()
        self.config = config
        self.diffusion_process = diffusion_process
        self.alpha = alpha
        self.rank_approx = rank_approx
        
        self.metric_net = NeuralMetricNetwork(
            input_dim=config.state_dim,
            hidden_dim=config.latent_dim,
            output_dim=config.state_dim,
            rank=rank_approx
        )
        
    def forward(self, model, x_t, t, target, action, history=None):
        batch_size, seq_len, state_dim = x_t.shape
        
        # Stochastic sequence index selection
        seq_idx = torch.randint(0, seq_len, (batch_size,), device=x_t.device)
        
        # Corrected advanced indexing (B, 1, D)
        batch_indices = torch.arange(batch_size, device=x_t.device)
        x_current = x_t[batch_indices, seq_idx, :].unsqueeze(1) 

        x_current.requires_grad_(True)

        try:
            # 1. Compute metric tensor M_x [B, D, D]
            M_x = self.metric_net(x_current)
            
            # 2. Define the dynamics function h(x)
            def dynamics_func(x_in):
                pred_x0 = model(x_in, t, target, action, history)
                h = self.compute_dynamics(pred_x0, x_in, t)
                return h.squeeze(1)

            # 3. Compute Jacobian J_h [B, D, D]
            J_h = torch.autograd.functional.jacobian(
                dynamics_func, x_current, create_graph=True, strict=True
            )
            J_h = torch.stack([J_h[i, :, i, 0, :] for i in range(batch_size)])
            
            # 4. Compute Lie Derivative of the metric Lie_M = J_h^T M + M J_h
            # This is the core term for contraction
            Lie_M = J_h.transpose(-1, -2) @ M_x + M_x @ J_h
            
            # 5. Compute penalties
            # Use Lie_M and M_x for contraction
            contract_penalty = self.contraction_penalty(Lie_M, M_x)
            
            # Use Lie_M and M_x for smoothness (Frobenius norm)
            frob_penalty = self.geometric_frobenius_norm(Lie_M, M_x)
            
            total_regularization = contract_penalty + self.alpha * frob_penalty
            
            return total_regularization

        except Exception as e:
            print(f"Warning: Geometric regularizer failed. {e}")
            return torch.tensor(0.0, device=x_t.device)
    
    def compute_dynamics(self, pred_x0, x_t, t):
        # Probability Flow ODE drift: h = f - 0.5 * g^2 * score
        
        t_idx = t.view(-1)
        
        betas = self.diffusion_process.betas.to(x_t.device)
        alpha_bars = self.diffusion_process.alpha_bars.to(x_t.device)
        
        beta_t = betas[t_idx].view(-1, 1, 1)
        alpha_bar_t = alpha_bars[t_idx].view(-1, 1, 1)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        one_minus_alpha_bar_t = 1.0 - alpha_bar_t

        f_term = -0.5 * beta_t * x_t
        
        # Score calculation
        score_num = x_t - sqrt_alpha_bar_t * pred_x0
        score_den = one_minus_alpha_bar_t.clamp(min=1e-6)
        score = - score_num / score_den
        
        g_term = -0.5 * beta_t * score
        
        h_theta = f_term - g_term 
        return h_theta
    
    def contraction_penalty(self, Lie_M, M_x):
        """
        Compute contraction penalty using Lie Derivative and Metric.
        Penalty = ReLU(max_eigenvalue(M⁻¹ Lie_M) + margin)
        """
        batch_size = M_x.shape[0]
        state_dim = M_x.shape[1]
        penalties = []
        
        for i in range(batch_size):
            try:
                Lie_M_i = Lie_M[i]
                M_x_i = M_x[i]
                
                # --- IMPROVEMENT: Use torch.linalg.solve for M⁻¹ Lie_M ---
                # This avoids unstable explicit M_inv calculation
                # F_geo_i = M_x_i⁻¹ @ Lie_M_i
                F_geo_i = torch.linalg.solve(M_x_i, Lie_M_i)
                
                # We need the max eigenvalue of the symmetric part of F_geo_i
                F_sym = 0.5 * (F_geo_i + F_geo_i.transpose(-1, -2))
                
                # Use standard Eigendecomposition (safer than power iteration for contraction analysis)
                # Note: This is on the boundary of what's allowed in a differentiable graph. 
                # Since this is used for loss, the `create_graph=True` in jacobian allows this.
                # However, for pure numerical stability, power iteration is often preferred in practice.
                # Sticking to your original power iteration for now for stability/differentiability compromise.
                
                # Power iteration for dominant eigenvalue
                v = torch.randn(state_dim, 1, device=F_sym.device)
                for _ in range(5):
                    v = F_sym @ v
                    v = v / (torch.norm(v) + 1e-8)
                
                lambda_max = (v.t() @ F_sym @ v).squeeze()
                
                # Penalty for positive eigenvalues (non-contracting)
                # The contraction rate is usually < 0. We penalize if rate is > -0.1
                penalty = F.relu(lambda_max + 0.1)
                penalties.append(penalty)
                
            except Exception as e:
                # Catch numerical/linalg errors here
                penalties.append(torch.tensor(0.0, device=Lie_M.device))
        
        if not penalties:
            return torch.tensor(0.0, device=Lie_M.device)
        return torch.stack(penalties).mean()
    
    def geometric_frobenius_norm(self, Lie_M, M_x):
        """
        Compute Frobenius norm of the Lie Derivative using the inverse metric:
        ‖Lie_M‖_{M⁻¹}² = tr(Lie_M M⁻¹ Lie_M M⁻¹)
        We use the unweighted squared Frobenius norm: ‖Lie_M‖²_F = tr(Lie_Mᵀ Lie_M)
        """
        batch_size = Lie_M.shape[0]
        norms = []
        
        for i in range(batch_size):
            Lie_M_i = Lie_M[i]
            
            # Unweighted Frobenius norm of the Lie Derivative (Simplest smoothness regularizer)
            norm_sq = torch.trace(Lie_M_i.t() @ Lie_M_i)
            norms.append(norm_sq)

        if not norms:
            return torch.tensor(0.0, device=Lie_M.device)
        return torch.stack(norms).mean()

class NeuralMetricNetwork(nn.Module):
    # ... (No changes needed here) ...
    def __init__(self, input_dim, hidden_dim, output_dim, rank=10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, rank * output_dim)
        )
        
        self.diag_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.epsilon = 1e-6
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.reshape(batch_size * seq_len, input_dim) 
        
        factors_flat = self.network(x_flat)
        factors = factors_flat.reshape(batch_size * seq_len, self.rank, self.output_dim) 
        
        diag_flat = self.diag_net(x_flat)
        diag = F.softplus(diag_flat) + self.epsilon
        
        factors_T = factors.transpose(1, 2)
        low_rank_batch = torch.bmm(factors_T, factors) 
        
        D_batch = torch.diag_embed(diag)
        
        I_batch = torch.eye(self.output_dim, device=x.device).expand(batch_size * seq_len, -1, -1)
        
        M_flat = I_batch + low_rank_batch + D_batch
        
        M = M_flat.reshape(batch_size, seq_len, self.output_dim, self.output_dim)
        
        return M[:, -1, :, :]

# Updated AeroDM class with geometric contraction
class AeroDMWithGeometricContraction(nn.Module):
    # ... (No changes needed here) ...
    def __init__(self, config, contraction_alpha=0.1):
        super().__init__()
        self.config = config
        self.diffusion_model = DiffusionTransformer(config)
        self.diffusion_process = DiffusionProcess(config)
        
        self.geometric_regularizer = GeometricContractionRegularizer(
            config, self.diffusion_process, alpha=contraction_alpha
        )
        
        self.contraction_weight = config.contraction_weight
        
    def forward(self, x_t, t, target, action, history=None):
        return self.diffusion_model(x_t, t, target, action, history)
    
    def compute_loss(self, x_0, t, target, action, history=None):
        """Compute loss with geometric contraction regularization"""
        
        # Forward diffusion
        noisy_x, noise = self.diffusion_process.q_sample(x_0, t)
        
        # Model prediction
        pred_x0 = self.diffusion_model(noisy_x, t, target, action, history)
        
        # Reconstruction loss
        mse_loss = F.mse_loss(pred_x0, x_0)
        
        # Geometric contraction regularization
        contraction_loss = self.geometric_regularizer(
            self.diffusion_model, noisy_x, t, target, action, history
        )
        
        # Handle potential NaNs from regularization
        if torch.isnan(contraction_loss):
            print("Warning: Contraction loss is NaN. Setting to 0.")
            contraction_loss = torch.tensor(0.0, device=mse_loss.device)
            
        total_loss = mse_loss + self.contraction_weight * contraction_loss
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'contraction_loss': contraction_loss
        }
    
    def sample(self, target, action, history=None, batch_size=1):
        # ... (No changes needed here) ...
        device = next(self.parameters()).device
        
        if target.dim() == 1:
            target = target.unsqueeze(0)
        if target.size(0) != batch_size:
            target = target.repeat(batch_size, 1)
        if action.size(0) != batch_size:
            action = action.repeat(batch_size, 1)
        if history is not None and history.size(0) != batch_size:
            history = history.repeat(batch_size, 1, 1)
        
        x_t = torch.randn(batch_size, self.config.seq_len, self.config.state_dim).to(device)
        
        for t_step in reversed(range(self.config.diffusion_steps)):
            t_batch = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
            x_t = self.diffusion_process.p_sample(
                self.diffusion_model, x_t, t_batch, target, action, history
            )
        
        return x_t

# Updated training function
def train_aerodm_with_geometric_contraction():
    # ... (No changes needed here) ...
    config = Config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AeroDMWithGeometricContraction(config, contraction_alpha=0.1).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 50
    batch_size = 8
    num_trajectories = 100
    
    print("Generating trajectory data...")
    trajectories = generate_aerobatic_trajectories(
        num_trajectories=num_trajectories,
        seq_len=config.seq_len + config.history_len
    )
    
    trajectories_norm, mean, std = normalize_trajectories(trajectories)
    trajectories_norm = trajectories_norm.to(device)
    mean = mean.to(device)
    std = std.to(device)
    
    losses = {'total': [], 'mse': [], 'contraction': [], 'geometric_ratio': []}
    
    print("Starting training with Geometric Contraction Regularization...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_total_loss = 0
        epoch_mse_loss = 0
        epoch_contraction_loss = 0
        num_batches = 0
        
        indices = torch.randperm(num_trajectories)
        for i in range(0, num_trajectories, batch_size):
            actual_batch_size = min(batch_size, num_trajectories - i)
            batch_indices = indices[i:i+actual_batch_size]
            
            full_traj = trajectories_norm[batch_indices]
            target = generate_target_waypoints(full_traj).squeeze(1) 
            action = generate_action_styles(actual_batch_size, config.action_dim, device=device)
            history = generate_history_segments(full_traj, config.history_len, device=device)
            x_0 = full_traj[:, config.history_len:, :]
            
            t = torch.randint(0, config.diffusion_steps, (actual_batch_size,), device=device)
            
            loss_dict = model.compute_loss(x_0, t, target, action, history)
            
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_total_loss += loss_dict['total_loss'].item()
            epoch_mse_loss += loss_dict['mse_loss'].item()
            epoch_contraction_loss += loss_dict['contraction_loss'].item() 
            num_batches += 1
        
        if num_batches > 0:
            avg_total = epoch_total_loss / num_batches
            avg_mse = epoch_mse_loss / num_batches
            avg_contraction = epoch_contraction_loss / num_batches
            
            weighted_contraction_loss = avg_contraction * model.contraction_weight
            geometric_ratio = weighted_contraction_loss / (avg_mse + 1e-8)
            
            losses['total'].append(avg_total)
            losses['mse'].append(avg_mse)
            losses['contraction'].append(avg_contraction)
            losses['geometric_ratio'].append(geometric_ratio)
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}/{num_epochs} | Time: {time.time()-start_time:.2f}s | "
                  f"Total: {avg_total:.4f} | MSE: {avg_mse:.4f} | "
                  f"Contraction: {weighted_contraction_loss:.4f} (Weighted) | Ratio: {geometric_ratio:.4f}")
            
            if geometric_ratio < 0.01:
                model.contraction_weight = min(0.1, model.contraction_weight * 1.2)
            elif geometric_ratio > 0.1:
                model.contraction_weight = max(0.001, model.contraction_weight * 0.9)
    
    print(f"\nTraining finished in {time.time()-start_time:.2f} seconds.")
    
    plot_enhanced_training_losses(losses, model.contraction_weight)
    
    test_model_performance(model, trajectories_norm, mean, std)
    
    return model, losses, trajectories, mean, std

def plot_enhanced_training_losses(losses, final_contraction_weight):
    # ... (No changes needed here) ...
    plt.figure(figsize=(15, 10))
    epochs = range(len(losses['total']))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, losses['total'], 'b-', label='Total Loss', linewidth=2)
    plt.plot(epochs, losses['mse'], 'r--', label='MSE Loss', linewidth=2)
    plt.plot(epochs, [c * final_contraction_weight for c in losses['contraction']], 'g-.', label='Weighted Contraction Loss', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Losses')
    plt.legend(); plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, losses['geometric_ratio'], 'm-', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Weighted Contraction / MSE Ratio')
    plt.title('Geometric Regularization Ratio')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.semilogy(epochs, losses['total'], 'b-', label='Total Loss', linewidth=2)
    plt.semilogy(epochs, losses['mse'], 'r--', label='MSE Loss', linewidth=2)
    plt.semilogy(epochs, [c * final_contraction_weight for c in losses['contraction']], 'g-.', label='Weighted Contraction Loss', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Loss (log scale)'); plt.title('Training Losses (Log Scale)')
    plt.legend(); plt.grid(True)
    
    plt.subplot(2, 2, 4)
    final_mse = losses['mse'][-1]
    final_contraction_weighted = losses['contraction'][-1] * final_contraction_weight
    sizes = [final_mse, final_contraction_weighted]
    labels = [f'MSE: {final_mse:.4f}', f'Weighted Contraction: {final_contraction_weighted:.4f}']
    plt.pie(sizes, labels=labels, colors=['lightcoral', 'lightskyblue'], autopct='%1.1f%%', startangle=140)
    plt.title(f'Final Loss Composition (Weight={final_contraction_weight:.4f})')
    
    plt.tight_layout()
    plt.show()

# Analysis function for geometric properties
@torch.no_grad()
def analyze_geometric_properties(model, trajectories_norm, num_samples=5):
    # ... (No changes needed here) ...
    print("\n=== Geometric Contraction Analysis (No Grad) ===")
    device = next(model.parameters()).device
    model.eval()
    
    sample_indices = torch.randperm(trajectories_norm.shape[0])[:num_samples]
    
    for i in sample_indices:
        full_traj = trajectories_norm[i:i+1]
        target = generate_target_waypoints(full_traj).squeeze(1)
        action = generate_action_styles(1, model.config.action_dim, device=device)
        history = generate_history_segments(full_traj, model.config.history_len, device=device)
        x_0 = full_traj[:, model.config.history_len:, :]
        
        print(f"\nSample {i}:")
        for t_val in [0, model.config.diffusion_steps//2, model.config.diffusion_steps-1]:
            t = torch.full((1,), t_val, device=device, dtype=torch.long)
            noisy_x, _ = model.diffusion_process.q_sample(x_0, t)
            
            with torch.enable_grad():
                contraction_loss = model.geometric_regularizer(
                    model.diffusion_model, noisy_x, t, target, action, history
                )
            
            print(f"  t={t_val}: Contraction Loss = {contraction_loss.item():.6f}")
    
    model.train()

# Main execution
if __name__ == "__main__":
    print("Training AeroDM with Geometric Contraction Regularization...")
    trained_model, losses, trajectories, mean, std = train_aerodm_with_geometric_contraction()
    
    analyze_geometric_properties(trained_model, trajectories.to(next(trained_model.parameters()).device))
    print("Training and analysis complete!")