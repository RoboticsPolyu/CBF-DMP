# --------------------------------------------------------------
#  LidarDataEncoder + GIF + 3-D POINT-CLOUD VISUALISATION (Matplotlib)
# How to represent the hided obstacles???
# --------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio.v3 as iio
from tqdm import tqdm
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from AeroDM_SafeTrj_v2_Test import (
    generate_aerobatic_trajectories,
    generate_random_obstacles)

# ======================
#  SYNTHETIC DATA
# ======================
def generate_synthetic_point_cloud():
    np.random.seed(42)
    t = np.linspace(0, 2*np.pi, 100)

    # Moving ball (radius 3 m, height 0.5 m, sphere radius 0.3 m)
    ball_center = np.column_stack([3*np.cos(t), 3*np.sin(t), 0.5*np.ones_like(t)])
    ball_points = []
    for c in ball_center:
        phi = np.random.uniform(0, 2*np.pi, 20)
        costheta = np.random.uniform(-1, 1, 20)
        theta = np.arccos(costheta)
        r = 0.3
        x = c[0] + r * np.sin(theta) * np.cos(phi)
        y = c[1] + r * np.sin(theta) * np.sin(phi)
        z = c[2] + r * np.cos(theta)
        ball_points.extend(np.column_stack([x, y, z]))
    ball_points = np.array(ball_points)

    # Static box at (5,0,0.7), size 1.5×1.5×1.4
    box = np.array([
        [5-0.75, 0-0.75, 0.7-0.7], [5-0.75, 0+0.75, 0.7-0.7],
        [5+0.75, 0-0.75, 0.7-0.7], [5+0.75, 0+0.75, 0.7-0.7],
        [5-0.75, 0-0.75, 0.7+0.7], [5-0.75, 0+0.75, 0.7+0.7],
        [5+0.75, 0-0.75, 0.7+0.7], [5+0.75, 0+0.75, 0.7+0.7],
    ])
    return ball_points, box


# ======================
#  FIXED 3D ANIMATION with real obstacle maps
# ======================
def create_3d_animation_precomputed(ball_all, box, frame_points_list, obstacle_maps_history, 
                                  R_w_b, t_w_b, quad_pos, save_gif=True):
    """
    Create 3D animation using pre-computed obstacle maps (better performance)
    """
    fig = plt.figure(figsize=(15, 6))
    
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_map = fig.add_subplot(122)
    
    def animate(frame_idx):
        print("frame_idx: ", frame_idx)
        ax_3d.cla()
        ax_map.cla()
        
        pts = frame_points_list[frame_idx]
        ball_pts = pts[:-box.shape[0]] if pts.shape[0] > box.shape[0] else np.empty((0, 3))
        
        # 3D Plot
        ax_3d.scatter([quad_pos[0]], [quad_pos[1]], [quad_pos[2]], 
                     c='cyan', s=100, marker='o', label='Quadrotor')
        ax_3d.scatter(box[:, 0], box[:, 1], box[:, 2], 
                     c='red', s=50, marker='s', label='Static Box')
        
        if len(ball_pts) > 0:
            ax_3d.scatter(ball_pts[:, 0], ball_pts[:, 1], ball_pts[:, 2], 
                         c='lime', s=30, alpha=0.7, label='Ball')
        
        ax_3d.set_xlim(-5, 7)
        ax_3d.set_ylim(-5, 5)
        ax_3d.set_zlim(0, 2)
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title(f'3D Point Cloud - Frame {frame_idx+1}/{len(frame_points_list)}')
        ax_3d.legend()
        
        # Use pre-computed obstacle map
        obstacle_map = obstacle_maps_history[frame_idx]
        
        im = ax_map.imshow(
            obstacle_map,  
            cmap='gray', 
            vmin=0, 
            vmax=1,
            origin='lower',
            aspect='auto',
            interpolation='nearest'
        )
        ax_map.set_xlabel("Angular Bin (0° to 360°)")
        ax_map.set_ylabel("Time History")
        ax_map.set_title(f"Real Obstacle Map - Frame {frame_idx+1}")
        
        if obstacle_map.shape[0] > 1:
            time_ticks = np.linspace(0, obstacle_map.shape[0]-1, 5, dtype=int)
            ax_map.set_yticks(time_ticks)
            ax_map.set_yticklabels([f'-{t}' for t in time_ticks[::-1]])
            ax_map.set_ylabel("Time Steps Before Present")
        
        if frame_idx == 1:
            plt.colorbar(im, ax=ax_map, label="Normalized Distance (0=close, 1=far)")
        
        plt.tight_layout()
    
    anim = animation.FuncAnimation(
        fig, animate, 
        frames=len(frame_points_list), 
        interval=300,
        repeat=False
    )

    if save_gif:
        print("Saving 3D animation with pre-computed maps as GIF...")
        anim.save('Figs/3d_point_cloud_precomputed_maps.gif', writer='pillow', fps=3, dpi=100)
        print("3D animation saved as '3d_point_cloud_precomputed_maps.gif'")
    
    plt.show()
    return anim


def plot_static_3d_frames(ball_all, box, frame_points_list, quad_pos, num_frames=4):
    """
    Plot multiple static 3D frames in a grid
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Select frames to display
    frame_indices = np.linspace(0, len(frame_points_list)-1, num_frames, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        pts = frame_points_list[frame_idx]
        ball_pts = pts[:-box.shape[0]] if pts.shape[0] > box.shape[0] else np.empty((0, 3))
        
        # Plot elements
        ax.scatter([quad_pos[0]], [quad_pos[1]], [quad_pos[2]], 
                  c='cyan', s=100, marker='o', label='Quadrotor')
        ax.scatter(box[:, 0], box[:, 1], box[:, 2], 
                  c='red', s=50, marker='s', label='Static Box')
        
        if len(ball_pts) > 0:
            ax.scatter(ball_pts[:, 0], ball_pts[:, 1], ball_pts[:, 2], 
                      c='lime', s=30, alpha=0.7, label='Ball')
        
        ax.set_xlim(-5, 7)
        ax.set_ylim(-5, 5)
        ax.set_zlim(0, 2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Frame {frame_idx+1}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

# ======================
#  LIDAR ENCODER
# ======================
class LidarDataEncoder:
    def __init__(self, n_bins=36, d_max=10.0, h=1.0, history_frames=36):
        self.n_bins = n_bins
        self.d_max = d_max
        self.h = h
        self.history_frames = history_frames
        self.angle_per_bin = 2 * np.pi / n_bins
        self.history = []  # List of O(k) vectors

    def transform_to_world_frame(self, points_body, R_w_b, t_w_b):
        return (R_w_b @ points_body.T).T + t_w_b

    def encode_single_frame(self, points_world, quad_pos):
        z_q = quad_pos[2]
        z_min, z_max = z_q - self.h, z_q + self.h
        O = np.ones(self.n_bins)  # default = far away

        if len(points_world) == 0:
            return O

        rel = points_world - quad_pos
        x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
        dist = np.sqrt(x**2 + y**2 + z**2)
        mask = (dist <= self.d_max) & (z >= -self.h) & (z <= self.h) & (dist > 1e-3)

        if not np.any(mask):
            return O

        x_f, y_f, z_f = x[mask], y[mask], z[mask]
        dist_f = dist[mask]

        theta = np.mod(np.arctan2(y_f, x_f), 2 * np.pi)
        bin_idx = np.clip((theta / self.angle_per_bin).astype(int), 0, self.n_bins - 1)

        for i in range(self.n_bins):
            in_bin = (bin_idx == i)
            if np.any(in_bin):
                O[i] = np.min(dist_f[in_bin]) / self.d_max
        return O

    def update_and_get_obstacle_map(self, points_body, R_w_b, t_w_b, quad_pos):
        points_world = self.transform_to_world_frame(points_body, R_w_b, t_w_b)
        O_k = self.encode_single_frame(points_world, quad_pos)

        self.history.append(O_k)
        if len(self.history) > self.history_frames:
            self.history.pop(0)

        if len(self.history) < self.history_frames:
            padded = [np.ones(self.n_bins)] * (self.history_frames - len(self.history))
            map_stack = np.array(padded + self.history)
        else:
            map_stack = np.array(self.history)
        return map_stack

    def visualize_obstacle_map(self, obstacle_map, ax=None, title=None):
        if ax is None:
            _, ax = plt.subplots()
        
        im = ax.imshow(
            obstacle_map,  
            cmap='gray', 
            vmin=0, 
            vmax=1,
            origin='lower',  
            aspect='auto',
            interpolation='nearest'
        )
        ax.set_xlabel("Angular Bin (0° to 360°)")
        ax.set_ylabel("Time History (past → present)")
        ax.set_title(title or "Obstacle Map\nDarker = Closer Obstacle")
        
        if obstacle_map.shape[0] > 1:
            time_ticks = np.linspace(0, obstacle_map.shape[0]-1, 5, dtype=int)
            ax.set_yticks(time_ticks)
            ax.set_yticklabels([f'-{t}' for t in time_ticks[::-1]])
            ax.set_ylabel("Time Steps Before Present")
        
        plt.colorbar(im, ax=ax, label="Normalized Distance (0=close, 1=far)")
        return ax

    def get_current_frame(self):
        if len(self.history) == 0:
            return np.ones(self.n_bins)
        return self.history[-1]


def test_lidar_map():
    encoder = LidarDataEncoder(n_bins=36, d_max=10.0, h=1.0, history_frames=36)
    
    # Quadrotor (fixed at origin, z = 1 m)
    quad_pos = np.array([0.0, -1.0, 1.0])
    R_w_b = np.eye(3)
    t_w_b = np.zeros(3)

    ball_points_all, static_box = generate_synthetic_point_cloud()
    points_per_frame = 20
    total_points = ball_points_all.shape[0]

    gif_frames = []
    frame_point_clouds = []
    obstacle_maps_history = []  
    
    print("Encoding frames for GIF + 3-D viewer ...")
    for i in tqdm(range(0, total_points, points_per_frame)):
        frame_ball = ball_points_all[i:i+points_per_frame]
        frame_points = np.vstack([frame_ball, static_box])
        frame_point_clouds.append(frame_points.copy())

        obstacle_map = encoder.update_and_get_obstacle_map(
            points_body=frame_points,
            R_w_b=R_w_b,
            t_w_b=t_w_b,
            quad_pos=quad_pos
        )
        obstacle_maps_history.append(obstacle_map.copy())

        fig = plt.figure(figsize=(6, 5))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        encoder.visualize_obstacle_map(
            obstacle_map, 
            ax=ax,
            title=f"Frame {i//points_per_frame + 1}/{(total_points-1)//points_per_frame + 1}\nLatest at Top"
        )
        
        fig.tight_layout()
        canvas.draw()
        img = np.array(canvas.renderer.buffer_rgba())[:, :, :3]
        gif_frames.append(img)
        plt.close(fig)

    gif_path = "Figs/obstacle_map.gif"
    print(f"Saving obstacle animation → {gif_path}")
    iio.imwrite(gif_path, gif_frames, fps=8, loop=0)
    print(f"Fixed animation saved: {os.path.abspath(gif_path)}")


    # print("Creating 3D animation with pre-computed obstacle maps...")
    create_3d_animation_precomputed(ball_points_all, static_box, frame_point_clouds, 
                                  obstacle_maps_history, R_w_b, t_w_b, quad_pos, save_gif=True)

    # print("Showing static 3D frames...")
    # plot_static_3d_frames(ball_points_all, static_box, frame_point_clouds, quad_pos)

    # Final analysis plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    encoder.visualize_obstacle_map(obstacle_map, ax=ax1)
    ax1.set_title("Final Obstacle Map History\n(Latest Frame at Top)")

    current_frame = encoder.get_current_frame()
    current_frame_2d = current_frame.reshape(1, -1)
    im2 = ax2.imshow(current_frame_2d, cmap='gray', vmin=0, vmax=1, aspect='auto')
    ax2.set_title("Latest Single Frame Only")
    ax2.set_xlabel("Angular Bin")
    ax2.set_ylabel("Current")
    ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2, label="Normalized Distance")

    mid_idx = len(obstacle_maps_history) // 2
    if mid_idx < len(obstacle_maps_history):
        mid_map = obstacle_maps_history[mid_idx]
        encoder.visualize_obstacle_map(mid_map, ax=ax3)
        ax3.set_title(f"Mid-sequence Map (Frame {mid_idx})")

    plt.tight_layout()
    plt.show()

    print("\n=== FIXED VISUALIZATION COMPLETE ===")
    print("✓ Latest frames now appear at TOP of obstacle map")
    print("✓ Time progresses from bottom (past) to top (present)")
    print("✓ Each GIF frame shows the accumulated history up to that point")
    print(f"✓ Fixed GIF saved: {gif_path}")


def test_point_generation():
    trajectries = generate_aerobatic_trajectories(1, 60, 10, 5)
    obstacles = generate_random_obstacles(trajectory=trajectries, num_obstacles_range=(1,3), radius_range=(1,2))
    
def convert_obstacles_pointcloud(obstacles, x_range=(-20, 20), y_range=(-20, 20), z_range=(-20, 20), 
                               x_dim=64, y_dim=64, z_dim=64, points_per_obstacle=100):
    """
    将障碍物列表转换为固定维度的3D点云体素网格
    
    Args:
        obstacles: 障碍物字典列表，每个障碍物包含 'center' 和 'radius'
        x_range, y_range, z_range: 空间范围 (min, max)
        x_dim, y_dim, z_dim: 体素网格的维度
        points_per_obstacle: 每个障碍物采样的点数
    
    Returns:
        point_cloud_3d: 3D体素网格，形状为 (x_dim, y_dim, z_dim)，值为占用概率
        point_coords: 采样的3D点坐标，形状为 (N, 3)
        voxel_grid: 二值体素网格，形状为 (x_dim, y_dim, z_dim)
    """
    
    def sample_sphere_surface(center, radius, num_points):
        """在球体表面均匀采样点"""
        # 生成单位球面上的随机点
        theta = 2 * np.pi * np.random.rand(num_points)
        phi = np.arccos(2 * np.random.rand(num_points) - 1)
        
        # 转换为笛卡尔坐标
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        # 平移到障碍物中心
        points = np.column_stack([x, y, z]) + center
        return points
    
    def sample_sphere_volume(center, radius, num_points):
        """在球体体积内均匀采样点"""
        # 首先生成单位球体内的随机点
        u = np.random.rand(num_points)
        v = np.random.rand(num_points)
        w = np.random.rand(num_points)
        
        # 转换为球坐标
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        r = radius * np.cbrt(w)  # 立方根确保体积均匀
        
        # 转换为笛卡尔坐标
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # 平移到障碍物中心
        points = np.column_stack([x, y, z]) + center
        return points
    
    # 初始化点云列表
    all_points = []
    
    # 对每个障碍物进行采样
    for obstacle in obstacles:
        center = obstacle['center']
        radius = obstacle['radius']
        
        # 将tensor转换为numpy数组（如果需要）
        if hasattr(center, 'cpu'):
            center = center.cpu().numpy()
        
        # 采样点（表面或体积）
        points = sample_sphere_surface(center, radius, points_per_obstacle)
        # 或者使用体积采样：points = sample_sphere_volume(center, radius, points_per_obstacle)
        
        all_points.append(points)
    
    # 合并所有点
    if all_points:
        point_coords = np.vstack(all_points)
    else:
        point_coords = np.empty((0, 3))
    
    # 创建体素网格
    voxel_grid = np.zeros((x_dim, y_dim, z_dim), dtype=np.float32)
    point_cloud_3d = np.zeros((x_dim, y_dim, z_dim), dtype=np.float32)
    
    if len(point_coords) > 0:
        # 将点坐标映射到体素索引
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range
        
        # 过滤在范围内的点
        valid_mask = (
            (point_coords[:, 0] >= x_min) & (point_coords[:, 0] <= x_max) &
            (point_coords[:, 1] >= y_min) & (point_coords[:, 1] <= y_max) &
            (point_coords[:, 2] >= z_min) & (point_coords[:, 2] <= z_max)
        )
        valid_points = point_coords[valid_mask]
        
        if len(valid_points) > 0:
            # 计算体素索引
            x_indices = ((valid_points[:, 0] - x_min) / (x_max - x_min) * (x_dim - 1)).astype(int)
            y_indices = ((valid_points[:, 1] - y_min) / (y_max - y_min) * (y_dim - 1)).astype(int)
            z_indices = ((valid_points[:, 2] - z_min) / (z_max - z_min) * (z_dim - 1)).astype(int)
            
            # 确保索引在有效范围内
            x_indices = np.clip(x_indices, 0, x_dim - 1)
            y_indices = np.clip(y_indices, 0, y_dim - 1)
            z_indices = np.clip(z_indices, 0, z_dim - 1)
            
            # 填充体素网格
            for x, y, z in zip(x_indices, y_indices, z_indices):
                voxel_grid[x, y, z] = 1.0
                point_cloud_3d[x, y, z] += 1.0  # 累积占用次数
            
            # 归一化为概率 [0, 1]
            if point_cloud_3d.max() > 0:
                point_cloud_3d = point_cloud_3d / point_cloud_3d.max()
    
    return point_cloud_3d, point_coords, voxel_grid

def visualize_point_cloud_3d(point_cloud_3d, point_coords=None, obstacles=None, 
                           trajectory=None, show_flag=True, title="3D Obstacle Point Cloud"):
    """
    可视化3D点云体素网格
    
    Args:
        point_cloud_3d: 3D体素网格，形状为 (x_dim, y_dim, z_dim)
        point_coords: 原始点坐标，形状为 (N, 3)
        obstacles: 原始障碍物列表
        trajectory: 轨迹数据
        show_flag: 是否显示图像
        title: 图像标题
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 3D体素可视化
    ax1 = fig.add_subplot(131, projection='3d')
    
    # 提取非零体素的位置
    x_idx, y_idx, z_idx = np.where(point_cloud_3d > 0)
    
    if len(x_idx) > 0:
        # 将体素索引转换回实际坐标（近似）
        x_range = (-20, 20)  # 与convert函数中的范围一致
        y_range = (-20, 20)
        z_range = (-20, 20)
        
        x_coords = np.interp(x_idx, [0, point_cloud_3d.shape[0]-1], x_range)
        y_coords = np.interp(y_idx, [0, point_cloud_3d.shape[1]-1], y_range)
        z_coords = np.interp(z_idx, [0, point_cloud_3d.shape[2]-1], z_range)
        
        # 根据占用概率设置颜色和透明度
        values = point_cloud_3d[x_idx, y_idx, z_idx]
        scatter = ax1.scatter(x_coords, y_coords, z_coords, 
                             c=values, cmap='viridis', alpha=0.6, 
                             s=20, marker='o')
        plt.colorbar(scatter, ax=ax1, label='Occupancy Probability')
    
    # 绘制轨迹（如果提供）
    if trajectory is not None:
        traj_points = trajectory.detach().cpu().numpy() if hasattr(trajectory, 'detach') else trajectory
        if traj_points.ndim == 3:  # 如果是批量数据，取第一个
            traj_points = traj_points[0]
        if traj_points.shape[1] >= 4:  # 提取位置 (x,y,z)
            positions = traj_points[:, 1:4]
            ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    'r-', linewidth=3, label='Trajectory', alpha=0.8)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Voxel Grid')
    ax1.legend()
    
    # 2. 原始点云可视化
    ax2 = fig.add_subplot(132, projection='3d')
    
    if point_coords is not None and len(point_coords) > 0:
        ax2.scatter(point_coords[:, 0], point_coords[:, 1], point_coords[:, 2],
                   c='blue', alpha=0.5, s=10, label='Sampled Points')
    
    # 绘制原始障碍物球体
    if obstacles is not None:
        colors = plt.cm.Set3(np.linspace(0, 1, len(obstacles)))
        for i, obstacle in enumerate(obstacles):
            center = obstacle['center']
            radius = obstacle['radius']
            
            if hasattr(center, 'cpu'):
                center = center.cpu().numpy()
            
            # 绘制球体表面
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax2.plot_surface(x, y, z, alpha=0.3, color=colors[i])
            
            if i == 0:
                from matplotlib.patches import Patch
                # 为图例创建代理
                proxy = Patch(color=colors[i], alpha=0.5, label='Obstacles')
                ax2.legend(handles=[proxy])
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Original Obstacles & Points')
    
    # 3. 2D投影（XY平面）
    ax3 = fig.add_subplot(133)
    
    if point_coords is not None and len(point_coords) > 0:
        ax3.scatter(point_coords[:, 0], point_coords[:, 1],
                   c='blue', alpha=0.5, s=10, label='Points')
    
    # 绘制障碍物圆圈
    if obstacles is not None:
        for i, obstacle in enumerate(obstacles):
            center = obstacle['center']
            radius = obstacle['radius']
            
            if hasattr(center, 'cpu'):
                center = center.cpu().numpy()
            
            circle = plt.Circle((center[0], center[1]), radius, 
                              color=plt.cm.Set3(i/len(obstacles)), 
                              alpha=0.5, label=f'Obstacle {i+1}' if i < 3 else "")
            ax3.add_patch(circle)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('XY Projection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if show_flag:
        plt.show()
    else:
        filename = "Figs/obstacle_point_cloud_3d.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
        plt.close()

def test_point_generation():
    """测试点云生成功能"""
    # 生成测试数据
    trajectories = generate_aerobatic_trajectories(1, 60, 10, 5)
    obstacles = generate_random_obstacles(
        trajectory=trajectories[0], 
        num_obstacles_range=(2, 4), 
        radius_range=(0.2, 1.0),
        check_collision=True
    )
    
    print(f"Generated {len(obstacles)} obstacles")
    for i, obs in enumerate(obstacles):
        print(f"Obstacle {i}: center={obs['center']}, radius={obs['radius']:.2f}")
    
    # 转换为点云
    point_cloud_3d, point_coords, voxel_grid = convert_obstacles_pointcloud(
        obstacles, 
        x_range=(-20, 20), 
        y_range=(-20, 20), 
        z_range=(-20, 20),
        x_dim=128,  # 降低分辨率以提高性能
        y_dim=128, 
        z_dim=128,
        points_per_obstacle=200
    )
    
    print(f"Point cloud shape: {point_cloud_3d.shape}")
    print(f"Number of sampled points: {len(point_coords)}")
    print(f"Occupied voxels: {np.sum(voxel_grid > 0)}")
    
    # 可视化
    visualize_point_cloud_3d(
        point_cloud_3d, 
        point_coords, 
        obstacles, 
        trajectory=trajectories[0],
        show_flag=True,
        title="Obstacle Point Cloud Visualization"
    )
    
    return point_cloud_3d, point_coords, voxel_grid, obstacles

if __name__ == "__main__":
    # test_lidar_map()
    point_cloud, points, voxels, obstacles = test_point_generation()