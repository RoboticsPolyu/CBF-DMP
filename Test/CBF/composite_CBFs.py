import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
d_safe = 2.0  # Safe distance threshold
num_obstacles = 10  # Total number of obstacles
num_inside = 4  # Number of obstacles inside safe region
space_size = 10.0  # 2D space size (0 to space_size in x and y)

# Generate random robot position (x_k)
robot_pos = np.random.uniform(2, space_size-2, 2)  # Avoid edges for better visualization

# Generate obstacle positions
obstacle_pos = np.zeros((num_obstacles, 2))

# Place num_inside obstacles within d_safe
for i in range(num_inside):
    # Use polar coordinates to place obstacle within d_safe
    r = np.random.uniform(0, d_safe)  # Random radius < d_safe
    theta = np.random.uniform(0, 2*np.pi)  # Random angle
    obstacle_pos[i] = robot_pos + [r * np.cos(theta), r * np.sin(theta)]

# Place remaining obstacles outside d_safe
for i in range(num_inside, num_obstacles):
    while True:
        pos = np.random.uniform(0, space_size, 2)
        dist = np.sqrt(np.sum((pos - robot_pos) ** 2))
        if dist > d_safe:  # Ensure outside safe region
            obstacle_pos[i] = pos
            break

# Calculate Euclidean distances from robot to each obstacle
distances = np.sqrt(np.sum((obstacle_pos - robot_pos) ** 2, axis=1))

# Compute max(d_safe - distance, 0) for each obstacle
# violations = np.maximum(d_safe - distances, 0)

violations = distances - d_safe

# Compute log-sum-exp: log(sum(exp(violations)))
k = 10
exp_violations = np.exp(-k* violations)
sum_exp = np.sum(exp_violations)
result = -1/k* np.log(sum_exp)

# Create plot
plt.figure(figsize=(8, 8))

# Plot robot
plt.scatter(robot_pos[0], robot_pos[1], c='green', marker='*', s=200, label='Robot')

# Plot obstacles
plt.scatter(obstacle_pos[:, 0], obstacle_pos[:, 1], c='red', marker='o', s=50, label='Obstacles')

# Plot safe distance circle
circle = plt.Circle(robot_pos, d_safe, color='blue', fill=False, linestyle='--', label='Safe Distance')
plt.gca().add_patch(circle)

# Annotate obstacles with distance and violation
for i, (pos, dist, viol) in enumerate(zip(obstacle_pos, distances, violations)):
    plt.annotate(f'Obs{i+1}\nDist: {dist:.2f}\nViol: {viol:.2f}',
                 (pos[0], pos[1]), xytext=(5, 5), textcoords='offset points')

# Set plot properties
plt.xlim(0, space_size)
plt.ylim(0, space_size)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Safety Constraint Plot\nLog-sum-exp: {result:.3f}')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')

# Save plot
plt.savefig('figs/plot.png')
plt.show()

# Print results
print("Robot position:", robot_pos)
print("Obstacle positions:")
for i, pos in enumerate(obstacle_pos):
    print(f"Obstacle {i+1}: {pos}, Distance: {distances[i]:.3f}, Violation: {violations[i]:.3f}")
print(f"Log-sum-exp result: {result:.3f}")
