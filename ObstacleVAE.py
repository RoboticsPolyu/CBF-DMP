import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from sklearn.model_selection import train_test_split

from AeroDM_SafeTrj_v2_Test import (
    ObstacleEncoder, 
    generate_guaranteed_colliding_obstacles, 
    generate_random_obstacles, 
    generate_aerobatic_trajectories
    )

# Fixed Obstacle Decoder with Correct Dimension Handling
class ObstacleDecoder(nn.Module):
    """
    Obstacle Decoder that reconstructs obstacle parameters from latent embeddings.
    Generates obstacle centers (x, y, z) and radii from encoded representations.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # MLP for decoding obstacle parameters from latent space - FIXED DIMENSIONS
        self.decoder_mlp = nn.Sequential(
            nn.Linear(config.obs_latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, config.max_obstacles * config.obstacle_feat_dim)
        )
        
    def forward(self, obstacle_embeddings):
        """
        Decode obstacle parameters from latent embeddings.
        
        Args:
            obstacle_embeddings: Tensor of shape [batch_size, obs_latent_dim]
            
        Returns:
            obstacles_list: List of lists containing obstacle dictionaries
            obstacle_tensor: Tensor of shape [batch_size, max_obstacles, obstacle_feat_dim]
        """
        batch_size = obstacle_embeddings.shape[0]
        device = obstacle_embeddings.device
        
        # Generate obstacle features for all obstacles in batch - FIXED APPROACH
        obstacle_features = self.decoder_mlp(obstacle_embeddings)  # [batch_size, max_obstacles * feat_dim]
        obstacle_params = obstacle_features.view(batch_size, self.config.max_obstacles, self.config.obstacle_feat_dim)
        
        # Convert to obstacle dictionary format
        obstacles_list = []
        for batch_idx in range(batch_size):
            sample_obstacles = []
            for obs_idx in range(self.config.max_obstacles):
                # Extract parameters: [x, y, z, radius]
                params = obstacle_params[batch_idx, obs_idx]
                
                # Apply activation functions for valid ranges
                center = params[:3]  # x, y, z coordinates
                radius = F.softplus(params[3]) + 0.05  # Ensure positive radius with minimum
                
                # Only include obstacles with reasonable radius
                if radius > 0.05 and radius < 10.0:  # Reasonable bounds
                    obstacle = {
                        'center': center,
                        'radius': radius,
                        'id': obs_idx
                    }
                    sample_obstacles.append(obstacle)
            
            obstacles_list.append(sample_obstacles)
        
        return obstacles_list, obstacle_params


# Alternative: Per-Obstacle Decoder (if you want individual obstacle control)
class PerObstacleDecoder(nn.Module):
    """
    Alternative decoder that generates each obstacle individually.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # MLP to generate parameters for each obstacle
        self.obstacle_generator = nn.Sequential(
            nn.Linear(config.obs_latent_dim + config.max_obstacles, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.obstacle_feat_dim)
        )
        
        # Obstacle embedding (learnable positions for each obstacle slot)
        self.obstacle_embedding = nn.Embedding(config.max_obstacles, config.max_obstacles)
        
    def forward(self, obstacle_embeddings):
        batch_size = obstacle_embeddings.shape[0]
        device = obstacle_embeddings.device
        
        obstacle_params = []
        for obs_idx in range(self.config.max_obstacles):
            # Get obstacle position embedding
            obs_pos = self.obstacle_embedding(torch.tensor([obs_idx], device=device))
            obs_pos = obs_pos.expand(batch_size, -1)
            
            # Combine with latent code
            combined_input = torch.cat([obstacle_embeddings, obs_pos], dim=1)
            
            # Generate obstacle parameters
            params = self.obstacle_generator(combined_input)  # [batch_size, obstacle_feat_dim]
            obstacle_params.append(params)
        
        obstacle_tensor = torch.stack(obstacle_params, dim=1)  # [batch_size, max_obstacles, obstacle_feat_dim]
        
        # Convert to obstacle dictionary format
        obstacles_list = []
        for batch_idx in range(batch_size):
            sample_obstacles = []
            for obs_idx in range(self.config.max_obstacles):
                params = obstacle_tensor[batch_idx, obs_idx]
                center = params[:3]
                radius = F.softplus(params[3]) + 0.1
                
                if radius > 0.1 and radius < 10.0:
                    obstacle = {
                        'center': center,
                        'radius': radius,
                        'id': obs_idx
                    }
                    sample_obstacles.append(obstacle)
            
            obstacles_list.append(sample_obstacles)
        
        return obstacles_list, obstacle_tensor


# Simplified Variational Autoencoder for Obstacle Generation
class ObstacleVAE(nn.Module):
    """
    Variational Autoencoder for generating and reconstructing obstacle configurations.
    """
    def __init__(self, config, use_per_obstacle_decoder=False):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = ObstacleEncoder(config)
        
        # Variational layers
        self.fc_mu = nn.Linear(config.obs_latent_dim, config.obs_latent_dim)
        self.fc_logvar = nn.Linear(config.obs_latent_dim, config.obs_latent_dim)
        
        # Decoder (choose which type to use)
        if use_per_obstacle_decoder:
            self.decoder = PerObstacleDecoder(config)
        else:
            self.decoder = ObstacleDecoder(config)
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling from latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, obstacles_data):
        """
        Encode obstacles into latent distribution parameters.
        """
        # Get obstacle embeddings from encoder
        obstacle_emb = self.encoder(obstacles_data)
        
        # Compute distribution parameters
        mu = self.fc_mu(obstacle_emb)
        logvar = self.fc_logvar(obstacle_emb)
        
        return mu, logvar
    
    def decode(self, z):
        """
        Decode latent vectors back to obstacle configurations.
        """
        return self.decoder(z)
    
    def forward(self, obstacles_data):
        """
        Full VAE forward pass: encode -> sample -> decode.
        """
        # Encode obstacles
        mu, logvar = self.encode(obstacles_data)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed_obstacles, obstacle_tensor = self.decode(z)
        
        return reconstructed_obstacles, obstacle_tensor, mu, logvar


# Improved Loss Function for Obstacle VAE Training
class ObstacleVAELoss(nn.Module):
    """
    Loss function for training the Obstacle VAE.
    Combines reconstruction loss and KL divergence.
    """
    def __init__(self, config, recon_weight=1.0, kl_weight=0.01, center_weight=1.0, radius_weight=0.5):
        super().__init__()
        self.config = config
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.center_weight = center_weight
        self.radius_weight = radius_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, reconstructed_tensor, target_tensor, mu, logvar):
        """
        Compute VAE loss using tensors directly.
        
        Args:
            reconstructed_tensor: Tensor [batch_size, max_obstacles, obstacle_feat_dim]
            target_tensor: Tensor [batch_size, max_obstacles, obstacle_feat_dim]
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            total_loss: Combined reconstruction and KL loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        batch_size = reconstructed_tensor.shape[0]
        
        # Create mask for valid obstacles (non-zero centers)
        valid_mask = (target_tensor[:, :, :3].norm(dim=2) > 0.1).float()  # [batch_size, max_obstacles]
        
        # Separate center and radius losses
        center_recon = reconstructed_tensor[:, :, :3]  # x, y, z
        radius_recon = reconstructed_tensor[:, :, 3]   # radius
        center_target = target_tensor[:, :, :3]
        radius_target = target_tensor[:, :, 3]
        
        # Masked reconstruction losses
        center_diff = (center_recon - center_target).norm(dim=2)  # [batch_size, max_obstacles]
        center_loss = (center_diff * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        radius_diff = (radius_recon - radius_target).abs()
        radius_loss = (radius_diff * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        # Combined reconstruction loss
        recon_loss = self.center_weight * center_loss + self.radius_weight * radius_loss
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / batch_size  # Normalize by batch size
        
        # Total loss
        total_loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss


# Utility function to convert obstacles to tensor
def obstacles_to_tensor(obstacles_list, config, device='cpu'):
    """
    Convert list of obstacle configurations to tensor format.
    
    Args:
        obstacles_list: List of lists of obstacle dictionaries
        config: Configuration object
        device: Target device
        
    Returns:
        obstacle_tensor: Tensor of shape [batch_size, max_obstacles, obstacle_feat_dim]
    """
    batch_size = len(obstacles_list)
    obstacle_tensor = torch.zeros(batch_size, config.max_obstacles, 
                                config.obstacle_feat_dim, device=device)
    
    for batch_idx, obstacles in enumerate(obstacles_list):
        for obs_idx, obstacle in enumerate(obstacles):
            if obs_idx < config.max_obstacles:
                center = obstacle['center']
                radius = obstacle['radius']
                # Ensure center is tensor and on correct device
                if isinstance(center, torch.Tensor):
                    center = center.to(device)
                else:
                    center = torch.tensor(center, device=device, dtype=torch.float32)
                
                obstacle_tensor[batch_idx, obs_idx] = torch.cat([
                    center, 
                    torch.tensor([radius], device=device, dtype=torch.float32)
                ])
    
    return obstacle_tensor


# Obstacle Dataset Generator
class ObstacleDatasetGenerator:
    """
    Generates synthetic obstacle datasets for training the obstacle VAE.
    Creates diverse obstacle configurations with realistic distributions.
    """
    def __init__(self, config):
        self.config = config
        
    def generate_obstacle_dataset(self, num_samples, device, trajectory_bounds=None):
        """
        Generate a dataset of obstacle configurations.
        
        Args:
            num_samples: Number of obstacle configurations to generate
            trajectory_bounds: Optional bounds for obstacle placement
            
        Returns:
            obstacles_dataset: List of obstacle configurations
        """
        trajectories = generate_aerobatic_trajectories(
            num_trajectories=num_samples, 
            seq_len=65
        )

        obstacles_dataset = []
        for b in range(num_samples):
            obstacles = generate_random_obstacles(
                trajectories[b], 
                num_obstacles_range=(1, self.config.max_obstacles), 
                radius_range=(0.2, 0.6), 
                check_collision=True,
                device=device
            )
            obstacles_dataset.append(obstacles)
        
        return obstacles_dataset
    
# Safe conversion function for plotting
def safe_convert_to_numpy(tensor_data):
    """Safely convert tensor data to numpy for plotting."""
    if hasattr(tensor_data, 'detach'):
        return tensor_data.detach().cpu().numpy()
    elif hasattr(tensor_data, 'numpy'):
        return tensor_data.numpy()
    else:
        return np.array(tensor_data)

# Evaluation functions
def evaluate_obstacle_vae(model, test_obstacles, config, device):
    """
    Evaluate the trained VAE on test data and compute accuracy metrics.
    """
    model.eval()
    
    with torch.no_grad():
        # Convert test obstacles to tensor
        test_tensor = obstacles_to_tensor(test_obstacles, config, device)
        # Forward pass
        reconstructed_obstacles, reconstructed_tensor, mu, logvar = model(test_obstacles)
        
        # Compute reconstruction errors
        valid_mask = (test_tensor[:, :, :3].norm(dim=2) > 0.1).float()
        
        # Center error (Euclidean distance)
        center_errors = (reconstructed_tensor[:, :, :3] - test_tensor[:, :, :3]).norm(dim=2)
        avg_center_error = (center_errors * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        # Radius error (absolute difference)
        radius_errors = (reconstructed_tensor[:, :, 3] - test_tensor[:, :, 3]).abs()
        avg_radius_error = (radius_errors * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        # Obstacle count accuracy
        count_accuracy = 0
        for i, (true_obs, recon_obs) in enumerate(zip(test_obstacles, reconstructed_obstacles)):
            true_count = len(true_obs)
            recon_count = len(recon_obs)
            if abs(true_count - recon_count) <= 1:  # Allow 1 obstacle difference
                count_accuracy += 1
        count_accuracy = count_accuracy / len(test_obstacles)
        
        return {
            'avg_center_error': avg_center_error.item(),
            'avg_radius_error': avg_radius_error.item(),
            'count_accuracy': count_accuracy,
            'reconstructed_tensor': reconstructed_tensor,
            'test_tensor': test_tensor
        }

def plot_training_loss(losses, save_path=None):
    """
    Plot training losses over epochs.
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses['total'])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(losses['reconstruction'])
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(losses['kl'])
    plt.title('KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_test_comparisons(test_obstacles, reconstructed_obstacles, evaluation_results, num_samples=5, save_path=None):
    """
    Plot comparisons between original and reconstructed obstacles for test cases.
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(min(num_samples, len(test_obstacles))):
        # Original obstacles
        ax1 = axes[0, i]
        true_obstacles = test_obstacles[i]
        
        for obs in true_obstacles:
            # Safely convert to numpy
            center = safe_convert_to_numpy(obs['center'])
            radius = safe_convert_to_numpy(obs['radius'])
            
            circle = plt.Circle((center[0], center[1]), float(radius), fill=True, 
                              alpha=0.7, color='red', label='True' if i == 0 else "")
            ax1.add_patch(circle)
            ax1.text(center[0], center[1], f'r={float(radius):.1f}', 
                    ha='center', va='center', fontsize=8)
        
        ax1.set_xlim(-25, 25)
        ax1.set_ylim(-25, 25)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'True Obstacles\n({len(true_obstacles)} obstacles)')
        if i == 0:
            ax1.set_ylabel('Y coordinate')
        
        # Reconstructed obstacles
        ax2 = axes[1, i]
        recon_obstacles = reconstructed_obstacles[i]
        
        for obs in recon_obstacles:
            # Safely convert to numpy
            center = safe_convert_to_numpy(obs['center'])
            radius = safe_convert_to_numpy(obs['radius'])
            
            circle = plt.Circle((center[0], center[1]), float(radius), fill=True, 
                              alpha=0.7, color='blue', label='Reconstructed' if i == 0 else "")
            ax2.add_patch(circle)
            ax2.text(center[0], center[1], f'r={float(radius):.1f}', 
                    ha='center', va='center', fontsize=8)
        
        ax2.set_xlim(-25, 25)
        ax2.set_ylim(-25, 25)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'Reconstructed\n({len(recon_obstacles)} obstacles)')
        if i == 0:
            ax2.set_ylabel('Y coordinate')
        ax2.set_xlabel('X coordinate')
    
    # Add evaluation metrics to the plot
    metrics_text = (f"Evaluation Metrics:\n"
                   f"Avg Center Error: {evaluation_results['avg_center_error']:.3f}\n"
                   f"Avg Radius Error: {evaluation_results['avg_radius_error']:.3f}\n"
                   f"Count Accuracy: {evaluation_results['count_accuracy']:.3f}")
    
    fig.text(0.02, 0.02, metrics_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Enhanced Training Function for Obstacle VAE
def train_obstacle_vae(config, num_epochs=100, batch_size=32, dataset_size=1000, 
                      test_size=0.2, use_per_obstacle_decoder=False):
    """
    Train the Obstacle VAE to generate realistic obstacle configurations.
    """
    print("Initializing Obstacle VAE Training...")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and components
    obstacle_vae = ObstacleVAE(config, use_per_obstacle_decoder=use_per_obstacle_decoder).to(device)
    dataset_generator = ObstacleDatasetGenerator(config)
    criterion = ObstacleVAELoss(config)
    optimizer = torch.optim.Adam(obstacle_vae.parameters(), lr=1e-4)
    
    # Generate and split dataset
    print("Generating obstacle dataset...")
    obstacles_dataset = dataset_generator.generate_obstacle_dataset(dataset_size, device)
    
    # Split into train and test sets
    train_obstacles, test_obstacles = train_test_split(
        obstacles_dataset, test_size=test_size, random_state=42
    )
    
    print(f"Generated {len(obstacles_dataset)} obstacle configurations")
    print(f"Training set: {len(train_obstacles)} samples")
    print(f"Test set: {len(test_obstacles)} samples")
    
    # Convert training dataset to tensor format
    train_tensor = obstacles_to_tensor(train_obstacles, config, device)
    
    # Training loop
    obstacle_vae.train()
    losses = {'total': [], 'reconstruction': [], 'kl': []}
    
    for epoch in range(num_epochs):
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        num_batches = 0
        
        # Shuffle training dataset
        indices = torch.randperm(len(train_obstacles))
        
        for i in range(0, len(train_obstacles), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_obstacles = [train_obstacles[idx] for idx in batch_indices]
            batch_target = train_tensor[batch_indices]
            
            # Forward pass
            reconstructed_obstacles, reconstructed_tensor, mu, logvar = obstacle_vae(batch_obstacles)
            
            # Compute loss using tensors directly
            total_loss, recon_loss, kl_loss = criterion(
                reconstructed_tensor, batch_target, mu, logvar
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(obstacle_vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1
        
        # Calculate averages
        if num_batches > 0:
            avg_total = epoch_total_loss / num_batches
            avg_recon = epoch_recon_loss / num_batches
            avg_kl = epoch_kl_loss / num_batches
            
            losses['total'].append(avg_total)
            losses['reconstruction'].append(avg_recon)
            losses['kl'].append(avg_kl)
        
        # Print progress and evaluate on test set periodically
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            # Evaluate on test set
            test_results = evaluate_obstacle_vae(obstacle_vae, test_obstacles[:batch_size], config, device)
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Total Loss: {avg_total:.4f} | "
                  f"Recon: {avg_recon:.4f} | "
                  f"KL: {avg_kl:.4f} | "
                  f"Test Center Error: {test_results['avg_center_error']:.3f} | "
                  f"Test Radius Error: {test_results['avg_radius_error']:.3f} | "
                  f"Count Accuracy: {test_results['count_accuracy']:.3f}")
    
    print("Obstacle VAE training completed!")
    
    # Final evaluation on test set
    print("\n=== Final Evaluation on Test Set ===")
    final_results = evaluate_obstacle_vae(obstacle_vae, test_obstacles, config, device)
    print(f"Average Center Error: {final_results['avg_center_error']:.4f}")
    print(f"Average Radius Error: {final_results['avg_radius_error']:.4f}")
    print(f"Obstacle Count Accuracy: {final_results['count_accuracy']:.4f}")
    
    # Plot training losses
    plot_training_loss(losses, save_path="model/training_losses.png")
    
    # Plot test comparisons - FIXED: Use model in eval mode and detach properly
    obstacle_vae.eval()
    with torch.no_grad():
        reconstructed_obstacles, _, _, _ = obstacle_vae(test_obstacles[:5])
    
    plot_test_comparisons(
        test_obstacles[:5], 
        reconstructed_obstacles,
        final_results,
        num_samples=min(5, len(test_obstacles)),
        save_path="model/test_comparisons.png"
    )
    
    # Save trained model
    model_name = "obstacle_vae_per_obstacle.pth" if use_per_obstacle_decoder else "obstacle_vae_standard.pth"
    torch.save({
        'model_state_dict': obstacle_vae.state_dict(),
        'config': config,
        'losses': losses,
        'test_results': final_results
    }, f"model/{model_name}")
    
    return obstacle_vae, losses, final_results, test_obstacles

# Main execution
if __name__ == "__main__":
    class ObstacleVAEConfig:
        obs_latent_dim = 256
        max_obstacles = 10
        obstacle_feat_dim = 4
    
    config = ObstacleVAEConfig()
    
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    # Train with smaller dataset for testing (use standard decoder)
    print("\n=== Training Standard Obstacle VAE ===")
    obstacle_vae, losses, test_results, test_obstacles = train_obstacle_vae(
        config, 
        num_epochs=100, 
        batch_size=32, 
        dataset_size=5000,  # Reduced for faster testing
        test_size=0.05,
        use_per_obstacle_decoder=False
    )
    
    print("\nObstacle VAE training and evaluation completed successfully!")
    print(f"Final test performance:")
    print(f"  - Average center error: {test_results['avg_center_error']:.4f}")
    print(f"  - Average radius error: {test_results['avg_radius_error']:.4f}")
    print(f"  - Obstacle count accuracy: {test_results['count_accuracy']:.4f}")