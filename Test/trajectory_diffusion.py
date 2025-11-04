import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters
K = 20  # Number of diffusion steps
T = 10  # Trajectory length
num_epochs = 1000
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cosine noise schedule (fixed to ensure length K)
def cosine_beta_schedule(timesteps, s=0.008):
    """Generate a cosine-based noise schedule with exactly timesteps elements."""
    steps = np.arange(timesteps + 1, dtype=np.float32) / timesteps  # Include endpoint
    f_t = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    betas = np.minimum(1 - f_t[1:] / f_t[:-1], 0.999)
    # Pad betas to ensure length equals timesteps
    if len(betas) < timesteps:
        betas = np.append(betas, betas[-1])  # Repeat last value
    betas = np.clip(betas, 0.0001, 0.9999)
    return betas

# Generate beta, alpha, and alpha_bar with correct length
beta = cosine_beta_schedule(K)
assert len(beta) == K, f"Expected beta to have length {K}, got {len(beta)}"
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)
assert len(alpha_bar) == K, f"Expected alpha_bar to have length {K}, got {len(alpha_bar)}"

# Generate synthetic trajectory data
def generate_trajectories(batch_size, T):
    """Generate batch of sine wave trajectories."""
    t = np.linspace(0, 2 * np.pi, T)
    trajectories = np.sin(t)[None, :] + np.random.normal(0, 0.1, (batch_size, T))
    return torch.tensor(np.stack(trajectories), dtype=torch.float32).to(device)

# Forward diffusion process
def forward_diffusion(trajectories, k):
    """Add noise to trajectories at step k."""
    if k >= len(alpha_bar):
        raise IndexError(f"Diffusion step {k} is out of bounds for alpha_bar with size {len(alpha_bar)}")
    noise = torch.randn_like(trajectories)
    sqrt_alpha_bar = torch.tensor(np.sqrt(alpha_bar[k]), device=device)
    sqrt_one_minus_alpha_bar = torch.tensor(np.sqrt(1 - alpha_bar[k]), device=device)
    noisy_trajs = sqrt_alpha_bar * trajectories + sqrt_one_minus_alpha_bar * noise
    return noisy_trajs, noise

# Noise prediction model
class NoiseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for time embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.time_embed = nn.Embedding(K, 1)  # Embedding for diffusion step

    def forward(self, x, k):
        batch_size = x.shape[0]
        k_tensor = torch.tensor([k] * batch_size, device=device, dtype=torch.long)
        t_embed = self.time_embed(k_tensor).view(batch_size, 1)
        x_with_t = torch.cat([x, t_embed], dim=-1)
        return self.model(x_with_t)

# Backward diffusion process
def backward_diffusion(model, noisy_traj, k):
    """Denoise trajectory at step k."""
    with torch.no_grad():
        pred_noise = model(noisy_traj, k)
        sqrt_alpha = torch.tensor(np.sqrt(alpha[k]), device=device)
        beta_k = torch.tensor(beta[k], device=device)
        sqrt_one_minus_alpha_bar = torch.tensor(np.sqrt(1 - alpha_bar[k]), device=device)
        mu_theta = (noisy_traj - (beta_k / sqrt_one_minus_alpha_bar) * pred_noise) / sqrt_alpha
        if k > 0:
            sigma_k = torch.tensor(np.sqrt(beta[k] * (1 - alpha_bar[k-1]) / (1 - alpha_bar[k])), device=device)
            noise = torch.randn_like(noisy_traj)
            denoised_traj = mu_theta + sigma_k * noise
        else:
            denoised_traj = mu_theta
    return denoised_traj

# Plotting utility
def plot_trajectories(trajectories, true_trajectory, title, filename, steps_to_plot=None):
    """Plot trajectories at specified diffusion steps."""
    plt.figure(figsize=(10, 6))
    if steps_to_plot is None:
        steps_to_plot = [0, K // 2, K - 1]
    for k, traj in enumerate(trajectories):
        if k in steps_to_plot:
            plt.plot(traj[0].cpu().numpy(), alpha=0.7, label=f'Step {k}')
    plt.plot(true_trajectory[0].cpu().numpy(), 'k--', linewidth=2, label='True Trajectory')
    plt.title(title)
    plt.xlabel('Time step')
    plt.ylabel('Position')
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# Training function
def train_model(model, trajectories, num_epochs, optimizer):
    """Train the noise prediction model."""
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for _ in range(len(trajectories) // batch_size):
            k = np.random.randint(0, K)
            batch_idx = np.random.choice(len(trajectories), batch_size)
            batch_trajs = trajectories[batch_idx]
            noisy_trajs, true_noise = forward_diffusion(batch_trajs, k)
            pred_noise = model(noisy_trajs, k)
            loss = nn.functional.mse_loss(pred_noise, true_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / (len(trajectories) // batch_size))
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")
    return losses

# Main execution
def main():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")

    # Generate data
    trajectories = generate_trajectories(batch_size * 10, T)
    true_trajectory = generate_trajectories(1, T)

    # Simulate forward diffusion
    forward_trajs = []
    for k in range(K):
        noisy_traj, _ = forward_diffusion(true_trajectory, k)
        forward_trajs.append(noisy_traj)
    plot_trajectories(forward_trajs, true_trajectory, 'Forward Diffusion Process', 'Figs/forward_diffusion.png')

    # Train model
    model = NoiseModel(input_dim=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = train_model(model, trajectories, num_epochs, optimizer)

    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('Figs/training_loss.png')
    plt.close()

    # Simulate backward diffusion
    current_traj = torch.randn(1, T, device=device)
    backward_trajs = [current_traj]
    for k in reversed(range(K)):
        current_traj = backward_diffusion(model, current_traj, k)
        backward_trajs.append(current_traj)
    plot_trajectories(backward_trajs, true_trajectory, 'Backward Diffusion Process', 'Figs/backward_diffusion.png', steps_to_plot=[K, K//2, 0])

    # Evaluate: Compute MSE between final denoised trajectory and true trajectory
    final_mse = nn.functional.mse_loss(backward_trajs[-1], true_trajectory).item()
    print(f"Final MSE between denoised and true trajectory: {final_mse:.4f}")

if __name__ == "__main__":
    main()