import torch
import torch.nn as nn

class ObstacleEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # MLP for encoding individual obstacles
        self.obstacle_mlp = nn.Sequential(
            nn.Linear(config.obstacle_feat_dim, config.latent_dim // 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim // 2, config.latent_dim),
            nn.ReLU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        )
        
    def forward(self, obstacles_data):
        """
        Process multiple obstacles and generate a global obstacle embedding.
        
        Args:
            obstacles_data: List of lists, where each inner list contains obstacle dicts for a batch sample
        
        Returns:
            global_features: Global obstacle embedding tensor of shape [batch_size, latent_dim]
        """
        if obstacles_data is None or len(obstacles_data) == 0:
            # Return zero embedding if no obstacles
            batch_size = 1 if obstacles_data is None else len(obstacles_data)
            return torch.zeros(batch_size, self.config.latent_dim, device=next(self.parameters()).device)
        
        device = next(self.parameters()).device
        batch_size = len(obstacles_data)
        
        # 预处理：为每个样本准备固定数量的障碍物
        batch_obstacle_tensors = []
        valid_counts = []  # 记录每个样本的有效障碍物数量
        
        for sample_obstacles in obstacles_data:
            if not sample_obstacles:
                # 空样本，创建零张量
                obstacle_tensor = torch.zeros(self.config.max_obstacles, self.config.obstacle_feat_dim, device=device)
                valid_counts.append(0)
            else:
                # 提取障碍物特征
                obstacle_tensors = []
                for obstacle in sample_obstacles:
                    center = obstacle['center']
                    radius = obstacle['radius']
                    obstacle_feat = torch.cat([
                        center,
                        torch.tensor([radius], device=device, dtype=center.dtype)
                    ])
                    obstacle_tensors.append(obstacle_feat)
                
                # 堆叠并处理数量限制
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
        
        # 批量处理所有样本
        batch_obstacle_tensor = torch.stack(batch_obstacle_tensors)  # [batch_size, max_obstacles, obstacle_feat_dim]
        
        # 重塑以便批量处理
        batch_size, max_obs, feat_dim = batch_obstacle_tensor.shape
        flattened_obstacles = batch_obstacle_tensor.view(-1, feat_dim)  # [batch_size * max_obstacles, feat_dim]
        
        # 批量编码所有障碍物
        encoded_obstacles = self.obstacle_mlp(flattened_obstacles)  # [batch_size * max_obstacles, latent_dim]
        
        # 重塑回原始结构
        encoded_obstacles = encoded_obstacles.view(batch_size, max_obs, -1)  # [batch_size, max_obstacles, latent_dim]
        
        # 创建有效掩码（排除填充的障碍物）
        valid_mask = torch.zeros(batch_size, max_obs, device=device)
        for i, count in enumerate(valid_counts):
            if count > 0:
                valid_mask[i, :count] = 1.0
        
        # 对有效障碍物进行平均池化
        masked_embeddings = encoded_obstacles * valid_mask.unsqueeze(-1)  # 应用掩码
        sum_embeddings = masked_embeddings.sum(dim=1)  # 求和 [batch_size, latent_dim]
        valid_counts_tensor = torch.tensor(valid_counts, device=device).float().clamp(min=1.0)  # 避免除零
        
        # 计算平均值
        global_features = sum_embeddings / valid_counts_tensor.unsqueeze(-1)  # [batch_size, latent_dim]
        
        return global_features


# test code
if __name__ == "__main__":
    # 模拟配置
    class Config:
        def __init__(self):
            self.obstacle_feat_dim = 4
            self.latent_dim = 256
            self.max_obstacles = 5
    
    config = Config()
    encoder = ObstacleEncoder(config)
    
    # 模拟输入数据
    obstacles_data = [
        [  # 样本1：3个障碍物
            {'center': torch.tensor([1.0, 2.0, 3.0]), 'radius': 0.5},
            {'center': torch.tensor([4.0, 5.0, 6.0]), 'radius': 0.8},
            {'center': torch.tensor([7.0, 8.0, 9.0]), 'radius': 1.2}
        ],
        [  # 样本2：没有障碍物
        ],
        [  # 样本3：6个障碍物（会截断到5个）
            {'center': torch.tensor([1.0, 1.0, 1.0]), 'radius': 0.1},
            {'center': torch.tensor([2.0, 2.0, 2.0]), 'radius': 0.2},
            {'center': torch.tensor([3.0, 3.0, 3.0]), 'radius': 0.3},
            {'center': torch.tensor([4.0, 4.0, 4.0]), 'radius': 0.4},
            {'center': torch.tensor([5.0, 5.0, 5.0]), 'radius': 0.5},
            {'center': torch.tensor([6.0, 6.0, 6.0]), 'radius': 0.6}  # 这个会被截断
        ]
    ]
    
    # 前向传播
    output = encoder(obstacles_data)
    print(f"输出形状: {output.shape}")  # 应该是 torch.Size([3, 256])
    print(f"输出范数: {output.norm(dim=1)}")