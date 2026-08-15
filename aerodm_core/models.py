"""Neural-network building blocks for obstacle-aware trajectory diffusion."""

import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ObstacleEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # MLP for encoding individual obstacles
        self.obstacle_mlp = nn.Sequential(
            nn.Linear(config.obstacle_feat_dim, config.obs_latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.obs_latent_dim * 2, config.obs_latent_dim),
            nn.ReLU(),
            nn.Linear(config.obs_latent_dim, config.obs_latent_dim)
        )

    def forward(self, obstacles_data):
        """
        Process multiple obstacles and generate a global obstacle embedding.

        Args:
            obstacles_data: List of lists, where each inner list contains obstacle dicts for a batch sample

        Returns:
            global_features: Global obstacle embedding tensor of shape [batch_size, obs_latent_dim]
        """
        if obstacles_data is None or len(obstacles_data) == 0:
            # Return zero embedding if no obstacles
            batch_size = 1 if obstacles_data is None else len(obstacles_data)
            return torch.zeros(batch_size, self.config.obs_latent_dim, device=next(self.parameters()).device)

        device = next(self.parameters()).device
        # print(" -------- device:", device)
        batch_size = len(obstacles_data)

        # Preprocessing: Prepare a fixed number of obstacles for each sample
        batch_obstacle_tensors = []
        valid_counts = []  # Record the number of effective obstacles for each sample

        for sample_obstacles in obstacles_data:
            if not sample_obstacles:
                # Empty sample, create zero tensor
                obstacle_tensor = torch.zeros(self.config.max_obstacles, self.config.obstacle_feat_dim, device=device)
                valid_counts.append(0)
            else:
                # Extracting obstacle features
                obstacle_tensors = []
                for obstacle in sample_obstacles:
                    center = obstacle['center'].to(device)
                    radius = obstacle['radius']
                    obstacle_feat = torch.cat([
                        center,
                        torch.tensor([radius], device=device, dtype=center.dtype)
                    ])
                    obstacle_tensors.append(obstacle_feat)

               # Stack and handle quantity limits
                if obstacle_tensors:
                    obstacle_tensor = torch.stack(obstacle_tensors)
                    valid_count = len(obstacle_tensors)

                    if valid_count < self.config.max_obstacles:
                        padding = torch.zeros(self.config.max_obstacles - valid_count,
                                            self.config.obstacle_feat_dim, device=device)
                        obstacle_tensor = torch.cat([obstacle_tensor, padding], dim=0)
                    elif valid_count > self.config.max_obstacles:
                        obstacle_tensor = obstacle_tensor[:self.config.max_obstacles]
                        valid_count = self.config.max_obstacles

                    valid_counts.append(valid_count)
                else:
                    obstacle_tensor = torch.zeros(self.config.max_obstacles, self.config.obstacle_feat_dim, device=device)
                    valid_counts.append(0)

            batch_obstacle_tensors.append(obstacle_tensor)

        # Batch process all samples
        batch_obstacle_tensor = torch.stack(batch_obstacle_tensors)  # [batch_size, max_obstacles, obstacle_feat_dim]

        # Reshaping for batch processing
        batch_size, max_obs, feat_dim = batch_obstacle_tensor.shape
        flattened_obstacles = batch_obstacle_tensor.view(-1, feat_dim)  # [batch_size * max_obstacles, feat_dim]

        # Batch encode all obstacles
        encoded_obstacles = self.obstacle_mlp(flattened_obstacles)  # [batch_size * max_obstacles, obs_latent_dim]

        # Restore to original structure
        encoded_obstacles = encoded_obstacles.view(batch_size, max_obs, -1)  # [batch_size, max_obstacles, obs_latent_dim]

        # Create an effective mask (to exclude filled obstacles)
        valid_mask = torch.zeros(batch_size, max_obs, device=device)
        for i, count in enumerate(valid_counts):
            if count > 0:
                valid_mask[i, :count] = 1.0

        # Perform average pooling on valid obstacles
        masked_embeddings = encoded_obstacles * valid_mask.unsqueeze(-1)  # Apply mask
        sum_embeddings = masked_embeddings.sum(dim=1)  # Sum [batch_size, obs_latent_dim]
        valid_counts_tensor = torch.tensor(valid_counts, device=device).float().clamp(min=1.0)  # Avoid division by zero

        # Calculate the average
        global_features = sum_embeddings / valid_counts_tensor.unsqueeze(-1)  # [batch_size, obs_latent_dim]

        return global_features

class AttentionObstacleEncoder(nn.Module):
    """
    Attention-based obstacle encoder for small obstacle sets (<10 obstacles).
    Uses self-attention to model interactions between obstacles and handles
    variable numbers of obstacles naturally without padding.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Project obstacle features to latent space
        # Input: [x, y, z, radius] -> Output: [obs_latent_dim]
        self.obstacle_proj = nn.Sequential(
            nn.Linear(config.obstacle_feat_dim, config.obs_latent_dim),
            nn.ReLU(),
            nn.Linear(config.obs_latent_dim, config.obs_latent_dim)
        )

        # Multi-head attention for obstacle-obstacle interactions
        # Allows obstacles to "see" each other and understand spatial relationships
        self.attention = nn.MultiheadAttention(
            config.obs_latent_dim,
            num_heads=4,
            batch_first=True,
            dropout=config.dropout
        )

        # Learnable positional encoding for obstacles (optional but helpful)
        # Since obstacles are unordered, this helps the model distinguish them
        max_obs = config.max_obstacles
        self.pos_encoding = nn.Parameter(torch.randn(1, max_obs, config.obs_latent_dim) * 0.1)

        # Global feature extraction after attention
        self.global_proj = nn.Sequential(
            nn.Linear(config.obs_latent_dim * 2, config.obs_latent_dim),  # *2 for concat of mean+max
            nn.ReLU(),
            nn.Linear(config.obs_latent_dim, config.obs_latent_dim),
            nn.LayerNorm(config.obs_latent_dim)  # Add layer norm for stability
        )

    def forward(self, obstacles_data):
        """
        Forward pass for obstacle encoding.

        Args:
            obstacles_data: List of lists, each inner list contains obstacle dicts
                           Each dict: {'center': tensor([x,y,z]), 'radius': float}

        Returns:
            global_features: Tensor of shape [batch_size, obs_latent_dim]
                            Global obstacle embedding for each sample
        """
        batch_size = len(obstacles_data)
        device = next(self.parameters()).device

        batch_embeddings = []

        for batch_idx, sample_obs in enumerate(obstacles_data):
            num_obstacles = len(sample_obs)

            # Case 1: No obstacles in this sample
            if num_obstacles == 0:
                batch_embeddings.append(
                    torch.zeros(self.config.obs_latent_dim, device=device)
                )
                continue

            # Extract obstacle features: [center_x, center_y, center_z, radius]
            obstacle_features = []
            for obstacle in sample_obs:
                center = obstacle['center'].to(device)
                radius = obstacle['radius']
                # Ensure radius is a tensor with correct dtype
                radius_tensor = torch.tensor([radius], device=device, dtype=center.dtype)
                feat = torch.cat([center, radius_tensor])
                obstacle_features.append(feat)

            # Stack obstacles for this sample: [num_obs, obstacle_feat_dim]
            obs_tensor = torch.stack(obstacle_features)

            # Project to latent space: [num_obs, obs_latent_dim]
            obs_emb = self.obstacle_proj(obs_tensor)

            # Add positional encoding (use first num_obs positions)
            # This helps the model maintain consistency despite unordered input
            pos_enc = self.pos_encoding[:, :num_obstacles, :].to(device)
            obs_emb = obs_emb.unsqueeze(0) + pos_enc  # [1, num_obs, obs_latent_dim]

            # Apply self-attention to model obstacle interactions
            # Each obstacle attends to all others to understand spatial relationships
            attn_output, attn_weights = self.attention(
                query=obs_emb,
                key=obs_emb,
                value=obs_emb
            )  # attn_output: [1, num_obs, obs_latent_dim]

            # Remove batch dimension
            obs_emb = attn_output.squeeze(0)  # [num_obs, obs_latent_dim]

            # Global pooling: combine mean and max pooling
            # Mean pooling captures the "average" obstacle context
            mean_pool = obs_emb.mean(dim=0)  # [obs_latent_dim]
            # Max pooling captures the "most critical" obstacle
            max_pool = obs_emb.max(dim=0)[0]  # [obs_latent_dim]

            # Concatenate both pooling results
            global_feat = torch.cat([mean_pool, max_pool])  # [obs_latent_dim * 2]

            # Final projection to get global obstacle embedding
            global_feat = self.global_proj(global_feat)  # [obs_latent_dim]

            batch_embeddings.append(global_feat)

        # Stack all batch embeddings: [batch_size, obs_latent_dim]
        return torch.stack(batch_embeddings)

class ConditionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Diffusion timestep embedding
        self.t_embed = nn.Sequential(
            nn.Linear(1, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )

        # Target waypoint embedding
        self.target_embed = nn.Sequential(
            nn.Linear(config.target_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )

        # Learned null embeddings for missing conditions
        self.null_target_embed = nn.Parameter(torch.randn(config.latent_dim))
        self.null_action_embed = nn.Parameter(torch.randn(config.latent_dim))
        self.null_obstacle_embed = nn.Parameter(torch.randn(config.latent_dim))

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(config.action_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )

        # Obstacle embedding
        self.obstacle_encoder = AttentionObstacleEncoder(config)

        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.latent_dim * 4, config.latent_dim * 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim * 2, config.latent_dim)
        )

    def forward(self, t, target=None, action=None, obstacles_data=None):
        """
        Forward pass with optional None inputs for any condition.

        Args:
            t: Diffusion timestep (batch,)
            target: Target waypoint (batch, target_dim) or None
            action: Action style (batch, action_dim) or None
            obstacles_data: List of obstacle dicts or None

        Returns:
            cond_emb: Combined condition embedding (batch, latent_dim)
        """
        batch_size = t.shape[0]
        device = t.device

        # Timestep embedding (always required)
        t_emb = self.t_embed(t.unsqueeze(-1).float())

        # Target embedding (use null if None)
        if target is not None:
            target_emb = self.target_embed(target)
        else:
            target_emb = self.null_target_embed.unsqueeze(0).expand(batch_size, -1)

        # Action embedding (use null if None)
        if action is not None:
            action_emb = self.action_embed(action)
        else:
            action_emb = self.null_action_embed.unsqueeze(0).expand(batch_size, -1)

        # Obstacle embedding (use null if None or disabled)
        if obstacles_data is not None and self.config.enable_obstacle_encoding:
            obstacle_emb = self.obstacle_encoder(obstacles_data)
        else:
            obstacle_emb = self.null_obstacle_embed.unsqueeze(0).expand(batch_size, -1)

        # Combine all conditions with feature fusion
        combined_emb = torch.cat([t_emb, target_emb, action_emb, obstacle_emb], dim=-1)
        cond_emb = self.fusion_layer(combined_emb)

        return cond_emb

class ObstacleAwareDiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.state_dim, config.latent_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.latent_dim)

        # condition embedding with obstacle information
        self.cond_embed = ConditionEmbedding(config)

        # Transformer layers
        transformer_layer = nn.TransformerDecoderLayer(
            d_model=config.latent_dim,
            nhead=config.num_heads,
            dim_feedforward=config.latent_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(
            transformer_layer, num_layers=config.num_layers
        )

        # Output projection
        self.output_proj = nn.Linear(config.latent_dim, config.state_dim)

    def forward(self, x, t, target=None, action=None, history=None, obstacles_data=None):
        batch_size, seq_len, _ = x.shape

        # Initialize if None
        if target is None:
            target = torch.ones(batch_size, self.config.target_dim, device=x.device) * 1e-6  # or some default value

        if action is None:
            action = torch.zeros(batch_size, self.config.action_dim, device=x.device)  # or some default value

        # During training, randomly drop style information to improve robustness
        if self.training and action is not None:
            drop_mask = torch.rand(x.size(0), device=x.device) < self.config.drop_style_prob
            action = action.clone()
            action[drop_mask] = 0  # Zero out action for dropped samples (style information is lost)

        if self.training and target is not None:
            drop_mask = torch.rand(x.size(0), device=x.device) < self.config.drop_target_prob
            target = target.clone()
            target[drop_mask] = 1e-6  # Set to a large value to indicate missing target (position doesn't matter, but should be distinguishable from valid targets)

        # Project input to latent space (no PE yet)
        x_proj = self.input_proj(x)

        # Prepare transformer input
        if history is not None:
            if history.size(0) != batch_size:
                if history.size(0) == 1:
                    history = history.repeat(batch_size, 1, 1)
                else:
                    raise ValueError(f"History data batch size mismatch")

            history_proj = self.input_proj(history)
            # Concatenate projected history and current *before* adding PE
            transformer_input = torch.cat([history_proj, x_proj], dim=1)
            total_seq_len = history_proj.size(1) + seq_len
        else:
            transformer_input = x_proj
            total_seq_len = seq_len

        # Now add positional encoding to the *combined* input (correct absolute positions)
        transformer_input = self.pos_encoding(transformer_input.transpose(0, 1)).transpose(0, 1)

        # Generate causal mask for the total sequence
        memory_mask = self._generate_square_subsequent_mask(total_seq_len).to(x.device)

        # Get condition embedding with obstacle information
        cond_emb = self.cond_embed(t, target, action, obstacles_data)
        cond_seq = cond_emb.unsqueeze(1).expand(-1, total_seq_len, -1)

        # Add condition to transformer input
        transformer_input = transformer_input + cond_seq

        # Self-attention with causal mask
        transformer_output = self.transformer(
            tgt=transformer_input,
            memory=transformer_input,
            tgt_mask=memory_mask,
            memory_mask=memory_mask
        )

        # Extract current sequence part (exclude history data)
        if history is not None:
            current_output = transformer_output[:, -seq_len:, :]
        else:
            current_output = transformer_output

        # Final projection
        output = self.output_proj(current_output)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
