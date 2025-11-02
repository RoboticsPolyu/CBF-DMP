import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata 

# --- PARAMETERS ---
MAP_CENTER = np.array([0, 0, 0])  # The observation position (center of the sphere)
SPHERE_RADIUS = 5.0              # The radius for the visualization sphere
N_RAYS = 5000                    # Number of rays (resolution of the projection)
RANGE_IMAGE_SHAPE = (720, 360)   # (Width, Height) for the 2D range image (2:1 aspect ratio)

# --- 1. SIMULATE 3D INDOOR MAP (POINT CLOUD) ---
def generate_indoor_map_points():
    """Generates a simplified 3D point cloud representing a room."""
    points = []
    
    # Floor (y=-1)
    x_floor, z_floor = np.meshgrid(np.linspace(-10, 10, 50), np.linspace(-10, 10, 50))
    points.append(np.stack([x_floor.ravel(), -np.ones(x_floor.size), z_floor.ravel()], axis=1))
    
    # Ceiling (y=5)
    x_ceil, z_ceil = np.meshgrid(np.linspace(-10, 10, 50), np.linspace(-10, 10, 50))
    points.append(np.stack([x_ceil.ravel(), 5 * np.ones(x_ceil.size), z_ceil.ravel()], axis=1))

    # Wall 1 (x=10)
    y_wall, z_wall = np.meshgrid(np.linspace(-1, 5, 20), np.linspace(-10, 10, 50))
    points.append(np.stack([10 * np.ones(y_wall.size), y_wall.ravel(), z_wall.ravel()], axis=1))

    # Wall 2 (z=-10)
    x_wall2, y_wall2 = np.meshgrid(np.linspace(-10, 10, 50), np.linspace(-1, 5, 20))
    points.append(np.stack([x_wall2.ravel(), y_wall2.ravel(), -10 * np.ones(x_wall2.size)], axis=1))
    
    # Obstacle (Cube near center)
    box_x = np.linspace(2, 4, 10)
    box_y = np.linspace(-1, 1, 10)
    box_z = np.linspace(-2, 0, 10)
    box_points = np.array([[x, y, z] for x in box_x for y in box_y for z in box_z])
    points.append(box_points)
    
    return np.concatenate(points, axis=0)

# --- 2. RAYCASTING AND PROJECTION LOGIC ---

def project_and_raycast(map_points, origin, radius, n_rays):
    """
    Generates rays, finds the closest point in the map to each ray, 
    and records the distance from the origin to that closest point.
    """
    # Generate ray directions using Fibonacci lattice for uniform distribution
    indices = np.arange(n_rays) + 0.5
    phi = np.arccos(1 - 2 * indices / n_rays)
    theta = np.pi * (1 + 5**0.5) * indices
    
    ray_directions = np.array([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi)
    ]).T # Shape: (N_RAYS, 3)

    distances = np.full(n_rays, np.inf)
    
    # Project all map points onto the unit sphere to get "potential" ray directions
    v_to_point = map_points - origin
    norms = np.linalg.norm(v_to_point, axis=1)
    
    valid_indices = norms > 1e-6
    v_to_point_unit = v_to_point[valid_indices] / norms[valid_indices, np.newaxis]
    valid_norms = norms[valid_indices]
    
    # Find the closest *ray index* for each *map point*
    closest_ray_indices = np.argmin(cdist(v_to_point_unit, ray_directions), axis=1)

    unique_ray_indices = np.unique(closest_ray_indices)
    
    for ray_idx in unique_ray_indices:
        # Get all map points that project to this ray index
        point_indices = np.where(closest_ray_indices == ray_idx)[0]
        # Find the distance of the closest point among these
        min_distance = np.min(valid_norms[point_indices])
        
        distances[ray_idx] = min_distance
    
    finite_mask = distances < np.inf
    
    hit_directions = ray_directions[finite_mask]
    hit_distances = distances[finite_mask]
    
    projected_points = origin + radius * hit_directions
    
    return projected_points, hit_distances

# --- 3. 3D VISUALIZATION ---

def visualize_projection(map_points, projected_points, hit_distances):
    """Plots the original 3D map and the projected sphere side-by-side."""
    fig = plt.figure(figsize=(15, 7))
    limit = 12
    max_dist = 15.0  # Cap the max distance for a sensible color scale
    
    # --- Colormap setup ---
    norm_distances = np.clip(hit_distances, 0, max_dist)
    cmap = cm.get_cmap('plasma')
    colors = cmap(norm_distances / max_dist)
    
    # --- Subplot 1: Original 3D Map (Point Cloud) ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('🗺️ Original 3D Indoor Map (Point Cloud)')
    
    # Plot the point cloud
    ax1.scatter(map_points[:, 0], map_points[:, 1], map_points[:, 2], 
                c='blue', marker='.', s=1, alpha=0.3)
    
    # Plot the observation point
    ax1.scatter(MAP_CENTER[0], MAP_CENTER[1], MAP_CENTER[2], 
                color='red', marker='*', s=300, label='Observation Position')
    
    # Set limits and aspect
    ax1.set_xlim([-limit, limit])
    ax1.set_ylim([-limit, limit])
    ax1.set_zlim([-3, 7])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # --- Subplot 2: Color-Coded Spherical Projection ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('🌐 Spherical Projection (Omnidirectional Range)')
    
    # Plot the projected points
    ax2.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2], 
               c=colors, marker='o', s=50, alpha=0.8)

    # Plot the origin (Observation Position)
    ax2.scatter(MAP_CENTER[0], MAP_CENTER[1], MAP_CENTER[2], 
               color='red', marker='*', s=300, label='Observation Position')

    # Add a wireframe sphere to show the projection surface
    u, v_sphere = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x_s = SPHERE_RADIUS * np.cos(u) * np.sin(v_sphere)
    y_s = SPHERE_RADIUS * np.sin(u) * np.sin(v_sphere)
    z_s = SPHERE_RADIUS * np.cos(v_sphere)
    ax2.plot_wireframe(x_s, y_s, z_s, color="grey", alpha=0.1)
    
    # Set limits for the sphere plot
    lim_s = SPHERE_RADIUS * 1.2
    ax2.set_xlim([-lim_s, lim_s])
    ax2.set_ylim([-lim_s, lim_s])
    ax2.set_zlim([-lim_s, lim_s])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # --- Color Bar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_dist))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, shrink=0.6, pad=0.1)
    cbar.set_label('Distance from Origin to Nearest Map Point')

    plt.tight_layout()

# --- 4. SPHERICAL-TO-PLANAR PROJECTION (Range Image) ---

def create_range_image(projected_points, hit_distances, image_shape):
    """
    Converts the 3D spherical projection hits into a 2D equirectangular range image.
    
    Returns: The 2D range image.
    """
    width, height = image_shape
    
    # 1. Get Unit Vectors
    vectors = projected_points - MAP_CENTER 
    norms = np.linalg.norm(vectors, axis=1)
    unit_vectors = vectors / norms[:, np.newaxis]
    
    X, Y, Z = unit_vectors.T
    
    # 2. Convert to Spherical Coordinates (Y is up/down)
    theta = np.arctan2(X, Z) # Azimuth [-pi, pi]
    theta[theta < 0] += 2 * np.pi # Convert to [0, 2*pi]
    
    phi = np.arccos(Y) # Polar Angle [0, pi] (0=Ceiling, pi=Floor)
    
    # 3. Map Angles to Image Pixel Coordinates
    u = (theta / (2 * np.pi)) * width # Horizontal index
    u = np.clip(u.astype(int), 0, width - 1)
    
    v = (phi / np.pi) * height # Vertical index
    v = np.clip(v.astype(int), 0, height - 1)
    
    # 4. Create the Raw Range Image
    max_dist = 20.0 
    range_image = np.full( (height, width), max_dist, dtype=float) # Shape (H, W)
    
    # Populate the range image with hit distances (taking the min distance for duplicates)
    for i in range(len(hit_distances)):
        current_v = v[i]
        current_u = u[i]
        
        range_image[current_v, current_u] = min(range_image[current_v, current_u], hit_distances[i])
        
    return range_image

# --- 5. INTERPOLATION FUNCTION ---

def post_process_range_image_with_interpolation(range_image, method='cubic', fill_value=np.nan):
    """
    Interpolates missing (max_dist) values in the range image using scipy.interpolate.griddata.
    """
    height, width = range_image.shape
    
    # Identify known data points and their values (anything not close to the max initialization value)
    max_dist_threshold = 19.9 
    known_mask = range_image < max_dist_threshold
    
    # Coordinates of known data points
    known_coords_v, known_coords_u = np.where(known_mask)
    known_points = np.array([known_coords_v, known_coords_u]).T # (N, 2)
    known_values = range_image[known_mask]
    
    # Coordinates of all target points (the entire image grid)
    grid_u, grid_v = np.meshgrid(np.arange(width), np.arange(height))
    target_points = np.array([grid_v.ravel(), grid_u.ravel()]).T # (H*W, 2)
    
    # Perform interpolation
    interpolated_values = griddata(known_points, known_values, target_points, 
                                   method=method, fill_value=fill_value)
    
    interpolated_image = interpolated_values.reshape(height, width)
    
    return interpolated_image


# --- 6. 2D VISUALIZATION ---

def visualize_range_images_with_interpolation(raw_range_image, interpolated_range_image):
    """Plots the raw and interpolated 2D range images side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    max_dist_cap = 15.0 # Max distance for color scale

    # --- Raw Range Image ---
    ax1 = axes[0]
    ax1.set_title('🖼️ Raw 2D Equirectangular Range Image')
    
    c1 = ax1.imshow(raw_range_image, cmap='hot_r', vmin=0, vmax=max_dist_cap, aspect='auto')
    
    # Set tick labels
    height, width = raw_range_image.shape
    ax1.set_xticks(np.linspace(0, width, 5))
    ax1.set_xticklabels(['0°', '90°', '180°', '270°', '360°'])
    ax1.set_yticks(np.linspace(0, height, 3))
    ax1.set_yticklabels(['0° (Ceiling)', '90° (Horizon)', '180° (Floor)'])
    ax1.set_xlabel('Azimuthal Angle (Longitude)')
    ax1.set_ylabel('Polar Angle (Elevation/Latitude)')
    fig.colorbar(c1, ax=ax1, label='Distance from Origin (m)')

    # --- Interpolated Range Image ---
    ax2 = axes[1]
    ax2.set_title('✨ Interpolated 2D Range Image (Cubic)')
    
    # Replace NaN values (from interpolation fill_value) with the max visualization distance
    interpolated_range_image_display = np.nan_to_num(interpolated_range_image, nan=max_dist_cap)

    c2 = ax2.imshow(interpolated_range_image_display, cmap='hot_r', vmin=0, vmax=max_dist_cap, aspect='auto')
    
    # Set tick labels
    ax2.set_xticks(np.linspace(0, width, 5))
    ax2.set_xticklabels(['0°', '90°', '180°', '270°', '360°'])
    ax2.set_yticks(np.linspace(0, height, 3))
    ax2.set_yticklabels(['0° (Ceiling)', '90° (Horizon)', '180° (Floor)'])
    ax2.set_xlabel('Azimuthal Angle (Longitude)')
    ax2.set_ylabel('Polar Angle (Elevation/Latitude)')
    fig.colorbar(c2, ax=ax2, label='Distance from Origin (m)')
    
    plt.tight_layout()
    plt.show()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    print("--- 3D Raycasting and Projection Simulation Start ---")
    
    # 1. Generate map
    map_points = generate_indoor_map_points()
    print(f"Generated {len(map_points)} map points.")
    
    # 2. Perform projection and raycasting
    projected_points, hit_distances = project_and_raycast(
        map_points, 
        MAP_CENTER, 
        SPHERE_RADIUS, 
        N_RAYS
    )
    print(f"Projected {len(projected_points)} rays with hits (Resolution: {N_RAYS}).")
    
    # 3. Visualize the 3D plots (Map and Spherical Projection)
    visualize_projection(map_points, projected_points, hit_distances)
    
    # 4. Create the 2D Range Image
    raw_range_image = create_range_image(
        projected_points, 
        hit_distances, 
        image_shape=RANGE_IMAGE_SHAPE
    )
    print(f"Created Raw 2D Range Image of shape {raw_range_image.shape} (H, W).")

    # 5. Interpolate the Range Image (Cubic method for smoothness)
    interpolated_range_image = post_process_range_image_with_interpolation(
        raw_range_image, 
        method='cubic', 
        fill_value=np.nan # Use NaN for areas outside the convex hull, handled in visualization
    )
    print("Interpolated Range Image using 'cubic' method.")

    # 6. Visualize both raw and interpolated range images
    visualize_range_images_with_interpolation(raw_range_image, interpolated_range_image)
    
    print("--- Simulation Complete ---")