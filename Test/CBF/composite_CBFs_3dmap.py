import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
d_safe = 2.0  # Safe distance threshold
num_obstacles = 3  # Total number of obstacles
num_inside = 0  # Number of obstacles inside safe region
space_size = 10.0  # 3D space size (0 to space_size in x, y, z)
grid_resolution = 30  # Number of points per axis (reduced for speed)
num_z_slices = 30  # Number of z-slices (reduced for speed)
k = 10  # Scaling factor for log-sum-exp

# Generate z-slices
z_slices = np.linspace(0, space_size, num_z_slices)
print(f"Z-slices: {z_slices}")

# Generate obstacle positions
obstacle_pos = np.zeros((num_obstacles, 3))
temp_robot_pos = np.random.uniform(2, space_size-2, 3)
for i in range(num_inside):
    r = np.random.uniform(0, d_safe)
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, np.pi)
    obstacle_pos[i] = temp_robot_pos + [
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ]
for i in range(num_inside, num_obstacles):
    while True:
        pos = np.random.uniform(0, space_size, 3)
        dist = np.sqrt(np.sum((pos - temp_robot_pos) ** 2))
        if dist > d_safe:
            obstacle_pos[i] = pos
            break

# Create 2D grid for robot positions
x = np.linspace(0, space_size, grid_resolution)
y = np.linspace(0, space_size, grid_resolution)
X, Y = np.meshgrid(x, y)

# Compute log-sum-exp for each z-slice (vectorized)
log_sum_exp_slices = []
for z_slice in z_slices:
    Z = np.full_like(X, z_slice)
    robot_positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    # Vectorized computation
    diffs = robot_positions[:, np.newaxis, :] - obstacle_pos[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=2))
    violations = distances - d_safe
    # Numerically stable log-sum-exp
    max_violations = np.max(-k * violations, axis=1, keepdims=True)
    exp_violations = np.exp(-k * violations - max_violations)
    sum_exp = np.sum(exp_violations, axis=1)
    log_sum_exp_values = -1/k * (np.log(sum_exp) + max_violations.squeeze())
    
    log_sum_exp_slices.append(log_sum_exp_values.reshape(grid_resolution, grid_resolution))

# Create output directory
os.makedirs('figs', exist_ok=True)

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot heatmap for each z-slice using plot_surface
for z_slice, log_sum_exp_values in zip(z_slices, log_sum_exp_slices):
    Z = np.full_like(X, z_slice)
    # Normalize colors globally across all slices
    norm_values = log_sum_exp_values / np.max([np.max(slice) for slice in log_sum_exp_slices])
    surface = ax.plot_surface(
        X, Y, Z, facecolors=plt.cm.viridis(norm_values),
        alpha=0.6, rstride=2, cstride=2, zorder=1, edgecolor='none'
    )
    # Add colorbar only once
    if z_slice == z_slices[0]:
        mappable = plt.cm.ScalarMappable(
            cmap='viridis',
            norm=plt.Normalize(vmin=np.min([np.min(slice) for slice in log_sum_exp_slices]),
                             vmax=np.max([np.max(slice) for slice in log_sum_exp_slices]))
        )
        fig.colorbar(mappable, ax=ax, label='Log-sum-exp Value', shrink=0.8, aspect=20)

# Plot obstacles
ax.scatter(obstacle_pos[:, 0], obstacle_pos[:, 1], obstacle_pos[:, 2],
           c='red', marker='o', s=80, label='Obstacles', zorder=10)

# Plot safety spheres
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
U, V = np.meshgrid(u, v)
for idx, obs in enumerate(obstacle_pos):
    x_sphere = obs[0] + d_safe * np.cos(U) * np.sin(V)
    y_sphere = obs[1] + d_safe * np.sin(U) * np.sin(V)
    z_sphere = obs[2] + d_safe * np.cos(V)
    ax.plot_surface(
        x_sphere, y_sphere, z_sphere,
        color='gray', alpha=0.3, edgecolor='none', zorder=5
    )

# Set plot properties
ax.set_xlim(0, space_size)
ax.set_ylim(0, space_size)
ax.set_zlim(0, space_size)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Log-sum-exp Values with {num_z_slices} Z-Slices')
# ax.legend(['Obstacles', 'Safety Spheres'])
ax.view_init(elev=30, azim=45)

# Save plot
plt.savefig('figs/log_sum_exp_heatmap_3d_optimized.png')
plt.show()

# Print sample results
for z_idx, z_slice in enumerate(z_slices[:2]):
    print(f"\nSample robot positions and log-sum-exp values for z={z_slice:.3f}:")
    Z = np.full((grid_resolution, grid_resolution), z_slice)
    robot_positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    for i in range(min(5, len(robot_positions))):
        print(f"Robot pos {i+1}: {robot_positions[i]}, Log-sum-exp: {log_sum_exp_slices[z_idx].ravel()[i]:.3f}")
print(f"\nObstacle positions:")
for i, pos in enumerate(obstacle_pos):
    print(f"Obstacle {i+1}: {pos}")