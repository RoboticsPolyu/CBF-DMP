import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib import font_manager

# Set Times New Roman font globally
font_path = 'figs/timr45w.ttf'  # Adjust path if necessary
font_prop = font_manager.FontProperties(fname=font_path, size=10)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_obstacles = 8  # Total number of obstacles
num_inside = 0  # Number of obstacles inside safe region
space_size = 4.0  # 2D space size (0 to space_size in x and y)
grid_resolution = 200  # Number of points per axis for robot position grid

# Define individual d_safe for each obstacle (e.g., random between 0.2 and 0.5 for demonstration)
d_safe_per_obstacle = np.random.uniform(0.2, 0.5, num_obstacles)

# Generate obstacle positions (fixed for all robot positions)
obstacle_pos = np.zeros((num_obstacles, 2))
# Temporary robot position to place obstacles
temp_robot_pos = np.random.uniform(2, space_size-2, 2)
# Place num_inside obstacles (none in this case, as num_inside=0)
for i in range(num_inside):
    r = np.random.uniform(0, d_safe_per_obstacle[i])
    theta = np.random.uniform(0, 2*np.pi)
    obstacle_pos[i] = temp_robot_pos + [r * np.cos(theta), r * np.sin(theta)]
# Place remaining obstacles outside the largest d_safe
max_d_safe = np.max(d_safe_per_obstacle)
for i in range(num_inside, num_obstacles):
    while True:
        pos = np.random.uniform(0, space_size, 2)
        dist = np.sqrt(np.sum((pos - temp_robot_pos) ** 2))
        if dist > max_d_safe:
            obstacle_pos[i] = pos
            break

# Create grid of robot positions
x = np.linspace(0, space_size, grid_resolution)
y = np.linspace(0, space_size, grid_resolution)
X, Y = np.meshgrid(x, y)
robot_positions = np.vstack([X.ravel(), Y.ravel()]).T  # Shape: (grid_resolution^2, 2)

# Compute log-sum-exp for each robot position
log_sum_exp_values = np.zeros(len(robot_positions))
for idx, robot_pos in enumerate(robot_positions):
    # Calculate Euclidean distances from robot to each obstacle
    distances = np.sqrt(np.sum((obstacle_pos - robot_pos) ** 2, axis=1))
    # Use individual d_safe for each obstacle
    violations = distances - d_safe_per_obstacle
    # Compute log-sum-exp: -1/k * log(sum(exp(-k * violations)))
    k = 10
    exp_violations = np.exp(-k * violations)
    sum_exp = np.sum(exp_violations)
    log_sum_exp_values[idx] = -1/k * np.log(sum_exp)

# Reshape log_sum_exp_values for contour plotting
Z = log_sum_exp_values.reshape(grid_resolution, grid_resolution)

# Create figure with width = 9 cm (converted to inches: 9 cm / 2.54 ≈ 3.54 inches)
cm_to_inch = 1 / 2.54
fig_width = 9 * cm_to_inch
fig_height = fig_width * 0.8
plt.figure(figsize=(fig_width, fig_height))

# Scatter plot of robot positions colored by log-sum-exp value
sc = plt.scatter(robot_positions[:, 0], robot_positions[:, 1], c=log_sum_exp_values,
                 cmap='viridis', s=50, alpha=0.6)
cbar = plt.colorbar(sc)
cbar.set_label('Log-sum-exp', fontproperties=font_prop)

# Plot obstacles
plt.scatter(obstacle_pos[:, 0], obstacle_pos[:, 1], c='red', marker='o', s=100, label='Obstacles')

# Plot safety region boundary (circle of radius d_safe around each obstacle)
theta = np.linspace(0, 2*np.pi, 100)
for idx, obs in enumerate(obstacle_pos):
    x_circle = obs[0] + d_safe_per_obstacle[idx] * np.cos(theta)
    y_circle = obs[1] + d_safe_per_obstacle[idx] * np.sin(theta)
    plt.plot(x_circle, y_circle, 'k--', linewidth=1)

# Plot contour where log_sum_exp_values = 0
plt.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2, linestyles='solid')

# Set plot properties
plt.xlim(0, space_size)
plt.ylim(0, space_size)
plt.xlabel('X (m)', fontproperties=font_prop)
plt.ylabel('Y (m)', fontproperties=font_prop)
plt.title('Log-sum-exp values with variable safety distances', fontproperties=font_prop)
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

# Save plot with high resolution
plt.savefig('figs/log_sum_exp_heatmap_variable_d_safe.svg', dpi=300, bbox_inches='tight')
plt.show()

# Print sample results
print("Sample robot positions and their log-sum-exp values:")
for i in range(min(5, len(robot_positions))):
    print(f"Robot pos {i+1}: {robot_positions[i]}, Log-sum-exp: {log_sum_exp_values[i]:.3f}")
print(f"Obstacle positions and their d_safe values:")
for i, pos in enumerate(obstacle_pos):
    print(f"Obstacle {i+1}: {pos}, d_safe: {d_safe_per_obstacle[i]:.3f}")