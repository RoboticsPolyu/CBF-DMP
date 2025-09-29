import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch
import torch.nn as nn

# 1. 定义参数
T = 1.0          # 总时间
num_steps = 1000 # 去噪步数
dt = T / num_steps
beta = lambda t: 30.0 + 100.0 * t**2  # 噪声调度函数
gamma = lambda t: 1.0 / (1 + np.exp(-50 * (0.7 - t)))  # 约束权重调度

# 2. 定义障碍物和屏障函数
obstacles = [np.array([-0.5, -0.5])]  # 障碍物位置
def barrier_function(x):
    """屏障函数：惩罚靠近障碍物的轨迹点"""
    penalty = 0.0
    for obs in obstacles:
        dist = np.linalg.norm(x - obs)
        penalty += 2.4 * (dist < 0.2)  # α=0.4，障碍物半径0.2
    return penalty

# 3. 定义简化的分数模型（实际应用中为U-Net）
class ScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x, t):
        # 简化：假设分数是朝向原点 (0,0) 的力
        return x  # sθ(x,t) ≈ -x (模拟向中心收敛)

score_model = ScoreModel()

# 4. 反向扩散过程（带约束）
def reverse_diffusion(x_init, num_steps=1000):
    x = x_init.clone()
    trajectory = []
    
    for k in range(num_steps):
        t = 1.0 - k / num_steps  # 时间从1→0
        
        # 计算分数和约束梯度
        score = score_model(x, t)
        x_np = x.detach().numpy()
        grad_V = np.zeros_like(x_np)
        
        # 数值计算屏障梯度 ∇V
        eps = 1e-3
        for i in range(len(x_np)):
            x_plus = x_np.copy()
            x_plus[i] += eps
            x_minus = x_np.copy()
            x_minus[i] -= eps
            grad_V[i] = (barrier_function(x_plus) - barrier_function(x_minus)) / (2 * eps)
        
        grad_V = torch.tensor(grad_V, dtype=torch.float32)
        
        # 更新x (Euler-Maruyama离散化)
        drift = -beta(t) * (x + (1 + 0.1) * (score - gamma(t) * grad_V)) * dt
        noise = np.sqrt(2 * beta(t) * dt) * torch.randn_like(x) * 0.1  # η=0.1
        
        x = x + drift + noise
        trajectory.append(x.numpy().copy())
    
    return np.array(trajectory)

# 5. 生成初始噪声并去噪
x_init = torch.tensor([[-2.8, -2.8]], dtype=torch.float32)  # 初始噪声轨迹点
traj = reverse_diffusion(x_init, num_steps=500)

# 6. 可视化
plt.figure(figsize=(10, 6))
plt.scatter(obstacles[0][0], obstacles[0][1], c='red', s=200, label='Obstacle')
plt.plot(traj[:, 0, 0], traj[:, 0, 1], 'b-', lw=2, label='Generated Trajectory')
plt.scatter(0, 0, c='green', s=100, marker='*', label='Target')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('CoDiG: Constraint-Aware Trajectory Generation')
plt.legend()
plt.grid()
plt.show()
