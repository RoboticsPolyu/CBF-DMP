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


def example():
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

if __name__ == "__main__":
    example()