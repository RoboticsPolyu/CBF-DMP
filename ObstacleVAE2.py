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

# Obstacle Decoder with Better Architecture
class ObstacleDecoder(nn.Module):
    """
    Enhanced Obstacle Decoder with progressive decoding and better initialization.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Progressive decoding with batch normalization and dropout
        self.decoder = nn.Sequential(
            nn.Linear(config.obs_latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, config.max_obstacles * config.obstacle_feat_dim)
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, obstacle_embeddings):
        """
        Decode obstacle parameters from latent embeddings.
        
        Args:
            obstacle_embeddings: Tensor of shape [batch_size, obs_latent_dim]
            
        Returns:
            obstacle_tensor: Tensor of shape [batch_size, max_obstacles, obstacle_feat_dim]
        """
        batch_size = obstacle_embeddings.shape[0]
        
        # Generate obstacle features
        obstacle_features = self.decoder(obstacle_embeddings)
        obstacle_tensor = obstacle_features.view(batch_size, self.config.max_obstacles, 
                                               self.config.obstacle_feat_dim)
        
        return obstacle_tensor
    
    def decode_with_confidence(self, z, threshold=0.1):
        """Decode with confidence scores for each obstacle"""
        batch_size = z.shape[0]
        obstacle_tensor = self.forward(z)
        
        obstacles_list = []
        confidence_scores = []
        
        for batch_idx in range(batch_size):
            sample_obstacles = []
            sample_confidences = []
            
            for obs_idx in range(self.config.max_obstacles):
                params = obstacle_tensor[batch_idx, obs_idx]
                center = params[:3]
                radius = F.softplus(params[3]) + 0.05
                
                # Calculate confidence based on radius and center magnitude
                radius_confidence = torch.sigmoid(radius - 0.1)
                center_confidence = torch.sigmoid(-center.norm() / 10.0)
                confidence = (radius_confidence + center_confidence) / 2
                
                if confidence > threshold:
                    obstacle = {
                        'center': center,
                        'radius': radius,
                        'confidence': confidence.item(),
                        'id': obs_idx
                    }
                    sample_obstacles.append(obstacle)
                    sample_confidences.append(confidence.item())
            
            obstacles_list.append(sample_obstacles)
            confidence_scores.append(sample_confidences)
        
        return obstacles_list, obstacle_tensor, confidence_scores

# Per-Obstacle Decoder
class PerObstacleDecoder(nn.Module):
    """
    Alternative decoder that generates each obstacle individually with attention.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Shared MLP for obstacle generation
        self.obstacle_generator = nn.Sequential(
            nn.Linear(config.obs_latent_dim + 16, 256),  # + positional encoding
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, config.obstacle_feat_dim)
        )
        
        # Learnable positional encodings for each obstacle slot
        self.positional_encoding = nn.Parameter(
            torch.randn(config.max_obstacles, 16)
        )
        
        # Attention mechanism for obstacle interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=config.obs_latent_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.positional_encoding)
        for layer in self.obstacle_generator:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    def forward(self, obstacle_embeddings):
        batch_size = obstacle_embeddings.shape[0]
        device = obstacle_embeddings.device
        
        # Apply self-attention to obstacle embeddings
        attended_embeddings, _ = self.attention(
            obstacle_embeddings.unsqueeze(1),
            obstacle_embeddings.unsqueeze(1),
            obstacle_embeddings.unsqueeze(1)
        )
        attended_embeddings = attended_embeddings.squeeze(1)
        
        obstacle_params = []
        for obs_idx in range(self.config.max_obstacles):
            # Get positional encoding for this obstacle slot
            pos_enc = self.positional_encoding[obs_idx].unsqueeze(0)
            pos_enc = pos_enc.expand(batch_size, -1)
            
            # Combine with attended latent code
            combined_input = torch.cat([attended_embeddings, pos_enc], dim=1)
            
            # Generate obstacle parameters
            params = self.obstacle_generator(combined_input)
            obstacle_params.append(params)
        
        obstacle_tensor = torch.stack(obstacle_params, dim=1)
        
        return obstacle_tensor

# Variational Autoencoder with Regularization
class ObstacleVAE(nn.Module):
    """
    Enhanced VAE with better training stability and latent space regularization.
    """
    def __init__(self, config, use_enhanced_decoder=True):
        super().__init__()
        self.config = config
        
        # Encoder (from original code)
        self.encoder = ObstacleEncoder(config)
        
        # Variational layers with larger capacity
        self.fc_mu = nn.Sequential(
            nn.Linear(config.obs_latent_dim, config.obs_latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.obs_latent_dim * 2, config.obs_latent_dim)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(config.obs_latent_dim, config.obs_latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.obs_latent_dim * 2, config.obs_latent_dim)
        )
        
        # Choose decoder type
        if use_enhanced_decoder:
            self.decoder = PerObstacleDecoder(config)
        else:
            self.decoder = ObstacleDecoder(config)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Proper weight initialization"""
        for module in [self.fc_mu, self.fc_logvar]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick with temperature scheduling.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, obstacles_data):
        """Encode obstacles into latent distribution parameters."""
        obstacle_emb = self.encoder(obstacles_data)
        mu = self.fc_mu(obstacle_emb)
        logvar = self.fc_logvar(obstacle_emb)
        return mu, logvar
    
    def decode(self, z, threshold=0.1):
        """Decode latent vectors back to obstacle configurations."""
        obstacle_tensor = self.decoder(z)
        
        # Convert to obstacle dictionary format
        batch_size = z.shape[0]
        obstacles_list = []
        
        for batch_idx in range(batch_size):
            sample_obstacles = []
            for obs_idx in range(self.config.max_obstacles):
                params = obstacle_tensor[batch_idx, obs_idx]
                center = params[:3]
                radius = F.softplus(params[3]) + 0.05
                
                # Simple validity check
                if radius > 0.1 and radius < 10.0 and center.norm() < 50.0:
                    obstacle = {
                        'center': center,
                        'radius': radius,
                        'id': obs_idx
                    }
                    sample_obstacles.append(obstacle)
            
            obstacles_list.append(sample_obstacles)
        
        return obstacles_list, obstacle_tensor
    
    def forward(self, obstacles_data):
        """Full VAE forward pass."""
        mu, logvar = self.encode(obstacles_data)
        z = self.reparameterize(mu, logvar)
        reconstructed_obstacles, obstacle_tensor = self.decode(z)
        
        return reconstructed_obstacles, obstacle_tensor, mu, logvar
    
    def sample(self, num_samples, device):
        """Sample new obstacle configurations from prior."""
        z = torch.randn(num_samples, self.config.obs_latent_dim, device=device)
        generated_obstacles, _ = self.decode(z)
        return generated_obstacles

# Loss Function with Beta-VAE and Free Bits
class ObstacleVAELoss(nn.Module):
    """
    Enhanced loss function with Beta-VAE, free bits, and better reconstruction.
    """
    def __init__(self, config, beta=1.0, free_bits=2.0, center_weight=2.0, radius_weight=1.0):
        super().__init__()
        self.config = config
        self.beta = beta
        self.free_bits = free_bits
        self.center_weight = center_weight
        self.radius_weight = radius_weight
        
    def forward(self, reconstructed_tensor, target_tensor, mu, logvar):
        batch_size = reconstructed_tensor.shape[0]
        
        # Create validity mask (non-zero obstacles)
        valid_mask = (target_tensor[:, :, :3].norm(dim=2, keepdim=True) > 0.1).float()
        
        # Separate center and radius losses with masking
        center_recon = reconstructed_tensor[:, :, :3]
        radius_recon = reconstructed_tensor[:, :, 3]
        center_target = target_tensor[:, :, :3]
        radius_target = target_tensor[:, :, 3]
        
        # Masked center loss (MSE)
        center_diff = (center_recon - center_target).pow(2).sum(dim=2)  # [batch_size, max_obstacles]
        center_loss = (center_diff * valid_mask.squeeze(-1)).sum() / (valid_mask.sum() + 1e-8)
        
        # Masked radius loss (MSE)
        radius_diff = (radius_recon - radius_target).pow(2)
        radius_loss = (radius_diff * valid_mask.squeeze(-1)).sum() / (valid_mask.sum() + 1e-8)
        
        # Combined reconstruction loss
        recon_loss = self.center_weight * center_loss + self.radius_weight * radius_loss
        
        # KL divergence with free bits (prevents posterior collapse)
        kl_per_dim = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_per_dim)
        
        # Apply free bits constraint
        kl_loss = torch.max(kl_loss, torch.tensor(self.free_bits * self.config.obs_latent_dim, 
                                                device=kl_loss.device))
        
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss

# Obstacle Dataset Generator with Curriculum Support
class ObstacleDatasetGenerator:
    """
    Generates synthetic obstacle datasets with curriculum learning support.
    """
    def __init__(self, config):
        self.config = config
        
    def generate_curriculum_dataset(self, num_samples, device, curriculum_stage=0):
        """
        Generate dataset with curriculum learning - progressively harder obstacles.
        """
        trajectories = generate_aerobatic_trajectories(
            num_trajectories=num_samples, 
            seq_len=65
        )

        # Curriculum stages
        stages = [
            {'max_obstacles': 3, 'radius_range': (0.3, 1.0), 'placement': 'sparse'},
            {'max_obstacles': 5, 'radius_range': (0.2, 1.5), 'placement': 'moderate'},
            {'max_obstacles': 8, 'radius_range': (0.1, 2.0), 'placement': 'dense'},
            {'max_obstacles': 10, 'radius_range': (0.1, 3.0), 'placement': 'mixed'}
        ]
        
        stage = stages[min(curriculum_stage, len(stages) - 1)]
        
        obstacles_dataset = []
        for b in range(num_samples):
            obstacles = generate_random_obstacles(
                trajectories[b], 
                num_obstacles_range=(1, stage['max_obstacles']), 
                radius_range=stage['radius_range'], 
                check_collision=True,
                device=device
            )
            obstacles_dataset.append(obstacles)
        
        print(f"Generated {len(obstacles_dataset)} samples (Stage {curriculum_stage + 1}: "
              f"max_obs={stage['max_obstacles']}, radius={stage['radius_range']})")
        
        return obstacles_dataset

# Training Function with Curriculum Learning
def train_obstacle_vae(config, num_epochs=200, batch_size=32, dataset_size=5000, 
                               test_size=0.1, use_enhanced_decoder=True):
    """
    Train the enhanced Obstacle VAE with curriculum learning and better optimization.
    """
    print("Initializing Enhanced Obstacle VAE Training...")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and components
    obstacle_vae = ObstacleVAE(config, use_enhanced_decoder=use_enhanced_decoder).to(device)
    dataset_generator = ObstacleDatasetGenerator(config)
    criterion = ObstacleVAELoss(config, beta=1.0, free_bits=2.0)
    
    # Use AdamW with weight decay
    optimizer = torch.optim.AdamW(
        obstacle_vae.parameters(), 
        lr=1e-4, 
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Curriculum learning stages
    curriculum_stages = [
        {'stage': 0, 'epochs': num_epochs // 4, 'dataset_size': dataset_size},
        {'stage': 1, 'epochs': num_epochs // 4, 'dataset_size': dataset_size},
        {'stage': 2, 'epochs': num_epochs // 4, 'dataset_size': dataset_size},
        {'stage': 3, 'epochs': num_epochs // 4, 'dataset_size': dataset_size}
    ]
    
    # Training tracking
    losses = {'total': [], 'reconstruction': [], 'kl': [], 'lr': []}
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    print(f"Training for {num_epochs} epochs with curriculum learning...")
    
    for stage_config in curriculum_stages:
        stage = stage_config['stage']
        stage_epochs = stage_config['epochs']
        stage_dataset_size = stage_config['dataset_size']
        
        print(f"\n=== Curriculum Stage {stage + 1} ===")
        
        # Generate dataset for current stage
        obstacles_dataset = dataset_generator.generate_curriculum_dataset(
            stage_dataset_size, device, curriculum_stage=stage
        )
        
        # Split into train and test
        train_obstacles, test_obstacles = train_test_split(
            obstacles_dataset, test_size=test_size, random_state=42
        )
        
        train_tensor = obstacles_to_tensor(train_obstacles, config, device)
        
        print(f"Stage {stage + 1}: {len(train_obstacles)} training samples")
        
        for epoch in range(stage_epochs):
            obstacle_vae.train()
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            num_batches = 0
            
            # Shuffle training data
            indices = torch.randperm(len(train_obstacles))
            
            for i in range(0, len(train_obstacles), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_obstacles = [train_obstacles[idx] for idx in batch_indices]
                batch_target = train_tensor[batch_indices]
                
                # Forward pass
                reconstructed_obstacles, reconstructed_tensor, mu, logvar = obstacle_vae(batch_obstacles)
                
                # Compute loss
                total_loss, recon_loss, kl_loss = criterion(
                    reconstructed_tensor, batch_target, mu, logvar
                )
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(obstacle_vae.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Accumulate losses
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                num_batches += 1
            
            # Update learning rate
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            # Calculate averages
            if num_batches > 0:
                avg_total = epoch_total_loss / num_batches
                avg_recon = epoch_recon_loss / num_batches
                avg_kl = epoch_kl_loss / num_batches
                
                losses['total'].append(avg_total)
                losses['reconstruction'].append(avg_recon)
                losses['kl'].append(avg_kl)
                losses['lr'].append(current_lr)
            
            # Early stopping check
            if avg_total < best_loss:
                best_loss = avg_total
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': obstacle_vae.state_dict(),
                    'config': config,
                    'losses': losses,
                    'epoch': epoch
                }, "model/best_obstacle_vae.pth")
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0 or epoch == stage_epochs - 1:
                # Evaluate on test set
                test_results = evaluate_obstacle_vae(
                    obstacle_vae, test_obstacles[:batch_size], config, device
                )
                
                print(f"Stage {stage + 1} | Epoch {epoch + 1}/{stage_epochs} | "
                      f"Total: {avg_total:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Test Center: {test_results['avg_center_error']:.3f} | "
                      f"Test Radius: {test_results['avg_radius_error']:.3f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    print("Enhanced Obstacle VAE training completed!")
    return obstacle_vae, losses, test_obstacles

# Comprehensive Evaluation Function
def evaluate_obstacle_vae(model, test_obstacles, config, device):
    """Comprehensive evaluation with multiple metrics"""
    model.eval()
    
    with torch.no_grad():
        # Reconstruction evaluation
        reconstructed_obstacles, reconstructed_tensor, mu, logvar = model(test_obstacles)
        test_tensor = obstacles_to_tensor(test_obstacles, config, device)
        
        metrics = {}
        
        # 1. Basic reconstruction errors
        valid_mask = (test_tensor[:, :, :3].norm(dim=2) > 0.1).float()
        
        # Center error
        center_errors = (reconstructed_tensor[:, :, :3] - test_tensor[:, :, :3]).norm(dim=2)
        metrics['avg_center_error'] = (center_errors * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        # Radius error
        radius_errors = (reconstructed_tensor[:, :, 3] - test_tensor[:, :, 3]).abs()
        metrics['avg_radius_error'] = (radius_errors * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        # 2. Obstacle count accuracy
        count_differences = []
        for true_obs, recon_obs in zip(test_obstacles, reconstructed_obstacles):
            count_differences.append(abs(len(true_obs) - len(recon_obs)))
        metrics['mean_count_diff'] = np.mean(count_differences)
        metrics['count_accuracy'] = np.mean([1 if diff <= 1 else 0 for diff in count_differences])
        
        # 3. Latent space quality
        metrics['avg_kl'] = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 4. Sampling quality
        num_samples = min(100, len(test_obstacles))
        generated_obstacles = model.sample(num_samples, device)
        
        reasonable_count = 0
        for obs_list in generated_obstacles:
            if 1 <= len(obs_list) <= config.max_obstacles:
                # Check if obstacles have reasonable parameters
                valid_obs = all(0.1 < obs['radius'] < 5.0 and obs['center'].norm() < 30.0 
                              for obs in obs_list)
                if valid_obs:
                    reasonable_count += 1
        
        metrics['sampling_quality'] = reasonable_count / num_samples
        
        # Convert to CPU for return
        for key, value in metrics.items():
            if hasattr(value, 'item'):
                metrics[key] = value.item()
        
        return metrics

# Plotting Functions
def plot_training_loss(losses, save_path=None):
    """Plot training losses with learning rate"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    ax1.plot(losses['total'])
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Reconstruction loss
    ax2.plot(losses['reconstruction'])
    ax2.set_title('Reconstruction Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # KL divergence
    ax3.plot(losses['kl'])
    ax3.set_title('KL Divergence Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)
    
    # Learning rate
    ax4.plot(losses['lr'])
    ax4.set_title('Learning Rate')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_comprehensive_comparisons(test_obstacles, reconstructed_obstacles, evaluation_results, 
                                 num_samples=5, save_path=None):
    """Enhanced comparison plotting with metrics"""
    fig = plt.figure(figsize=(6 * num_samples, 12))
    
    for i in range(min(num_samples, len(test_obstacles))):
        # Original obstacles
        ax1 = plt.subplot(2, num_samples, i + 1)
        true_obstacles = test_obstacles[i]
        
        for obs in true_obstacles:
            center = safe_convert_to_numpy(obs['center'])
            radius = safe_convert_to_numpy(obs['radius'])
            
            circle = plt.Circle((center[0], center[1]), float(radius), fill=True, 
                              alpha=0.7, color='red', label='True' if i == 0 else "")
            ax1.add_patch(circle)
            ax1.text(center[0], center[1], f'r={float(radius):.1f}', 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax1.set_xlim(-25, 25)
        ax1.set_ylim(-25, 25)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'True Obstacles\n({len(true_obstacles)} obstacles)', fontweight='bold')
        if i == 0:
            ax1.set_ylabel('Y coordinate', fontweight='bold')
        
        # Reconstructed obstacles
        ax2 = plt.subplot(2, num_samples, i + num_samples + 1)
        recon_obstacles = reconstructed_obstacles[i]
        
        for obs in recon_obstacles:
            center = safe_convert_to_numpy(obs['center'])
            radius = safe_convert_to_numpy(obs['radius'])
            
            circle = plt.Circle((center[0], center[1]), float(radius), fill=True, 
                              alpha=0.7, color='blue', label='Reconstructed' if i == 0 else "")
            ax2.add_patch(circle)
            ax2.text(center[0], center[1], f'r={float(radius):.1f}', 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax2.set_xlim(-25, 25)
        ax2.set_ylim(-25, 25)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'Reconstructed\n({len(recon_obstacles)} obstacles)', fontweight='bold')
        if i == 0:
            ax2.set_ylabel('Y coordinate', fontweight='bold')
        ax2.set_xlabel('X coordinate', fontweight='bold')
    
    # Add comprehensive metrics
    metrics_text = (f"Comprehensive Evaluation Metrics:\n"
                   f"• Avg Center Error: {evaluation_results['avg_center_error']:.3f}\n"
                   f"• Avg Radius Error: {evaluation_results['avg_radius_error']:.3f}\n"
                   f"• Mean Count Difference: {evaluation_results['mean_count_diff']:.2f}\n"
                   f"• Count Accuracy: {evaluation_results['count_accuracy']:.3f}\n"
                   f"• Avg KL Divergence: {evaluation_results['avg_kl']:.3f}\n"
                   f"• Sampling Quality: {evaluation_results['sampling_quality']:.3f}")
    
    plt.figtext(0.02, 0.02, metrics_text, fontsize=11, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Utility functions (keep from original)
def obstacles_to_tensor(obstacles_list, config, device='cpu'):
    """Convert list of obstacle configurations to tensor format."""
    batch_size = len(obstacles_list)
    obstacle_tensor = torch.zeros(batch_size, config.max_obstacles, 
                                config.obstacle_feat_dim, device=device)
    
    for batch_idx, obstacles in enumerate(obstacles_list):
        for obs_idx, obstacle in enumerate(obstacles):
            if obs_idx < config.max_obstacles:
                center = obstacle['center']
                radius = obstacle['radius']
                if isinstance(center, torch.Tensor):
                    center = center.to(device)
                else:
                    center = torch.tensor(center, device=device, dtype=torch.float32)
                
                obstacle_tensor[batch_idx, obs_idx] = torch.cat([
                    center, 
                    torch.tensor([radius], device=device, dtype=torch.float32)
                ])
    
    return obstacle_tensor

def safe_convert_to_numpy(tensor_data):
    """Safely convert tensor data to numpy for plotting."""
    if hasattr(tensor_data, 'detach'):
        return tensor_data.detach().cpu().numpy()
    elif hasattr(tensor_data, 'numpy'):
        return tensor_data.numpy()
    else:
        return np.array(tensor_data)

# Main execution
if __name__ == "__main__":
    class ObstacleVAEConfig:
        obs_latent_dim = 256
        max_obstacles = 10
        obstacle_feat_dim = 4
    
    config = ObstacleVAEConfig()
    
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    print("\n=== Training Enhanced Obstacle VAE ===")
    print("Features:")
    print("• Curriculum learning")
    print("• Beta-VAE with free bits")
    print("• Enhanced decoder architecture")
    print("• AdamW optimizer with cosine annealing")
    print("• Comprehensive evaluation metrics")
    
    obstacle_vae, losses, test_obstacles = train_obstacle_vae(
        config, 
        num_epochs=200, 
        batch_size=32, 
        dataset_size=5000,
        test_size=0.1,
        use_enhanced_decoder=True
    )
    
    # Final comprehensive evaluation
    print("\n=== Final Comprehensive Evaluation ===")
    final_results = evaluate_obstacle_vae(obstacle_vae, test_obstacles, config, 
                                                 next(obstacle_vae.parameters()).device)
    
    # Plot results
    plot_training_loss(losses, save_path="model/enhanced_training_losses.png")
    
    # Plot test comparisons
    obstacle_vae.eval()
    with torch.no_grad():
        reconstructed_obstacles, _, _, _ = obstacle_vae(test_obstacles[:5])
    
    plot_comprehensive_comparisons(
        test_obstacles[:5], 
        reconstructed_obstacles,
        final_results,
        num_samples=min(5, len(test_obstacles)),
        save_path="model/enhanced_test_comparisons.png"
    )
    
    print("\n=== Final Results ===")
    for metric, value in final_results.items():
        print(f"{metric:.<25} {value:.4f}")
    
    print("\nEnhanced Obstacle VAE training and evaluation completed successfully!")