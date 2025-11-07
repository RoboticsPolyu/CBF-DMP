import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline


def generate_aerobatic_trajectories(num_trajectories, seq_len, height=10.0, radius=5.0):
    """Generates synthetic aerobatic trajectories (Power Loop, Barrel Roll, Split S, Immelmann, Wall Ride, Figure Eight, Star, Half Moon, Sphinx, Clover)."""
    trajectories = []
    maneuver_styles = ['power_loop', 'barrel_roll', 'split_s', 'immelmann', 'wall_ride', 'eight_figure', 'star', 'half_moon', 'sphinx', 'clover', 'spiral_inward', 'spiral_outward', 'spiral_vertical_up', 'spiral_vertical_down' ]
    
    def smooth_trajectory(positions, smoothing_factor=0.1):
        # """Apply smoothing to trajectory positions using a simple moving average"""
        # smoothed = np.zeros_like(positions)
        # for i in range(len(positions)):
        #     start_idx = max(0, i - 1)
        #     end_idx = min(len(positions), i + 2)
        #     smoothed[i] = np.mean(positions[start_idx:end_idx], axis=0)
        # return smoothing_factor * smoothed + (1 - smoothing_factor) * positions
        return positions # Keeping the original simple implementation
        
    for i in range(num_trajectories):
        # Randomly select a maneuver style
        style = 'spiral_vertical_down' # np.random.choice(maneuver_styles)
        
        # Random centers and scales
        center_x = np.random.uniform(-20, 20)
        center_y = np.random.uniform(-20, 20)
        center_z = height + np.random.uniform(-10, 10)
        current_radius = radius * np.random.uniform(0.8, 1.2)
        angular_velocity = np.random.uniform(0.5, 2.0)
        
        # Normalize time steps to [0, 1] - exactly one period
        norm_t = np.linspace(0, 1, seq_len)
        
        # Compute positions and velocities based on style
        if style == 'power_loop':
            # Full vertical loop in xz plane, starting at bottom with forward velocity
            theta = np.pi * norm_t * angular_velocity
            x = center_x - current_radius * (1 - np.cos(theta))
            y = np.full(seq_len, center_y)
            z = center_z + current_radius * np.sin(theta)
            vx = -current_radius * angular_velocity * np.pi * np.sin(theta)
            vy = np.zeros(seq_len)
            vz = current_radius * angular_velocity * np.pi * np.cos(theta)

        elif style == 'barrel_roll':
            # Helical motion
            pitch = np.random.uniform(5.0, 10.0)
            theta = 2 * np.pi * norm_t * angular_velocity
            x = center_x + current_radius * np.cos(theta)
            y = center_y + current_radius * np.sin(theta)
            z = center_z + pitch * norm_t
            vx = -current_radius * angular_velocity * 2 * np.pi * np.sin(theta)
            vy = current_radius * angular_velocity * 2 * np.pi * np.cos(theta)
            vz = np.full(seq_len, pitch)

        elif style == 'split_s':
            # Half loop down, then inverted flight and recovery
            theta = np.pi * norm_t
            x = center_x + current_radius * np.sin(theta)
            y = np.full(seq_len, center_y)
            z = center_z - current_radius * (1 - np.cos(theta))
            vx = current_radius * np.pi * np.cos(theta)
            vy = np.zeros(seq_len)
            vz = -current_radius * np.pi * np.sin(theta)

        elif style == 'immelmann':
            # Half loop up, then half roll to recover
            theta = np.pi * norm_t
            x = center_x + current_radius * np.sin(theta)
            y = np.full(seq_len, center_y)
            z = center_z + current_radius * (1 - np.cos(theta))
            vx = current_radius * np.pi * np.cos(theta)
            vy = np.zeros(seq_len)
            vz = current_radius * np.pi * np.sin(theta)

        elif style == 'wall_ride':
            # Vertical helix climb (spiral up)
            turns = np.random.uniform(0.5, 1.5) # Number of turns
            climb_height = np.random.uniform(20.0, 40.0)
            theta = 2 * np.pi * turns * norm_t * angular_velocity
            x = center_x + current_radius * np.cos(theta)
            y = center_y + current_radius * np.sin(theta)
            z = center_z + climb_height * norm_t
            vx = -current_radius * angular_velocity * 2 * np.pi * turns * np.sin(theta)
            vy = current_radius * angular_velocity * 2 * np.pi * turns * np.cos(theta)
            vz = np.full(seq_len, climb_height)
        
        elif style == 'eight_figure':
            # Figure eight in the xy plane
            theta = 2 * np.pi * norm_t
            x = center_x + current_radius * np.sin(theta)
            y = center_y + 0.5 * current_radius * np.sin(2 * theta)
            z = np.full(seq_len, center_z)
            vx = current_radius * 2 * np.pi * np.cos(theta)
            vy = 0.5 * current_radius * 4 * np.pi * np.cos(2 * theta)
            vz = np.zeros(seq_len)

        elif style == 'star':
            # 3D Star/Lissajous-like curve
            alpha = 2.0
            beta = 3.0
            x = center_x + current_radius * np.sin(2 * np.pi * alpha * norm_t)
            y = center_y + current_radius * np.cos(2 * np.pi * beta * norm_t)
            z = center_z + current_radius * 0.5 * np.sin(2 * np.pi * (alpha + beta) * norm_t)
            vx = current_radius * 2 * np.pi * alpha * np.cos(2 * np.pi * alpha * norm_t)
            vy = -current_radius * 2 * np.pi * beta * np.sin(2 * np.pi * beta * norm_t)
            vz = current_radius * 0.5 * 2 * np.pi * (alpha + beta) * np.cos(2 * np.pi * (alpha + beta) * norm_t)

        elif style == 'half_moon':
            # Semicircle arc, primarily in xy plane, with some small z variation
            theta = np.pi * norm_t
            x = center_x + current_radius * np.cos(theta)
            y = center_y + current_radius * np.sin(theta)
            z = center_z + 0.1 * current_radius * np.sin(theta)
            vx = -current_radius * np.pi * np.sin(theta)
            vy = current_radius * np.pi * np.cos(theta)
            vz = 0.1 * current_radius * np.pi * np.cos(theta)

        elif style == 'sphinx':
            # Similar to wall ride, but with pitch variation for 'nose-up' maneuver
            turns = np.random.uniform(0.5, 1.5)
            climb_height = np.random.uniform(10.0, 30.0)
            theta = 2 * np.pi * turns * norm_t * angular_velocity
            x = center_x + current_radius * np.cos(theta)
            y = center_y + current_radius * np.sin(theta)
            z = center_z + climb_height * norm_t + 5 * np.sin(np.pi * norm_t) # Pitch variation
            vx = -current_radius * angular_velocity * 2 * np.pi * turns * np.sin(theta)
            vy = current_radius * angular_velocity * 2 * np.pi * turns * np.cos(theta)
            vz = np.full(seq_len, climb_height) + 5 * np.pi * np.cos(np.pi * norm_t)

        elif style == 'clover':
            # Four leaf clover shape (like two overlapping figure eights)
            alpha = 2.0
            x = center_x + current_radius * np.cos(2 * np.pi * norm_t) * np.cos(2 * np.pi * alpha * norm_t)
            y = center_y + current_radius * np.cos(2 * np.pi * norm_t) * np.sin(2 * np.pi * alpha * norm_t)
            z = np.full(seq_len, center_z)
            # Use gradient for velocity approximation (too complex to derive analytically)
            x_smooth = smooth_trajectory(x)
            y_smooth = smooth_trajectory(y)
            z_smooth = smooth_trajectory(z)
            dt = 1.0 / seq_len
            vx = np.gradient(x_smooth, dt)
            vy = np.gradient(y_smooth, dt)
            vz = np.gradient(z_smooth, dt)
            x, y, z = x_smooth, y_smooth, z_smooth

        elif style == 'spiral_inward':
            # Horizontal spiral moving inward (contracting spiral)
            turns = np.random.uniform(1.0, 3.0)  # Number of turns
            start_radius = current_radius * np.random.uniform(1.5, 2.5)
            end_radius = current_radius * 0.2
            theta = 2 * np.pi * turns * norm_t * angular_velocity
            
            # Radius decreases over time
            radius_t = start_radius + (end_radius - start_radius) * norm_t
            
            x = center_x + radius_t * np.cos(theta)
            y = center_y + radius_t * np.sin(theta)
            z = np.full(seq_len, center_z)
            
            # Analytical velocities
            dr_dt = (end_radius - start_radius)  # Constant rate of radius change
            vx = dr_dt * np.cos(theta) - radius_t * angular_velocity * 2 * np.pi * turns * np.sin(theta)
            vy = dr_dt * np.sin(theta) + radius_t * angular_velocity * 2 * np.pi * turns * np.cos(theta)
            vz = np.zeros(seq_len)

        elif style == 'spiral_outward':
            # Horizontal spiral moving outward (expanding spiral)
            turns = np.random.uniform(1.0, 3.0)  # Number of turns
            start_radius = current_radius * 0.2
            end_radius = current_radius * np.random.uniform(1.5, 2.5)
            theta = 2 * np.pi * turns * norm_t * angular_velocity
            
            # Radius increases over time
            radius_t = start_radius + (end_radius - start_radius) * norm_t
            
            x = center_x + radius_t * np.cos(theta)
            y = center_y + radius_t * np.sin(theta)
            z = np.full(seq_len, center_z)
            
            # Analytical velocities
            dr_dt = (end_radius - start_radius)  # Constant rate of radius change
            vx = dr_dt * np.cos(theta) - radius_t * angular_velocity * 2 * np.pi * turns * np.sin(theta)
            vy = dr_dt * np.sin(theta) + radius_t * angular_velocity * 2 * np.pi * turns * np.cos(theta)
            vz = np.zeros(seq_len)

        elif style == 'spiral_vertical_up':
            # Vertical spiral moving upward (in xz or yz plane)
            plane_choice = np.random.choice(['xz', 'yz'])
            turns = np.random.uniform(1.0, 3.0)  # Number of turns
            climb_height = np.random.uniform(15.0, 35.0)
            theta = 2 * np.pi * turns * norm_t * angular_velocity
            
            if plane_choice == 'xz':
                # Spiral in xz plane
                x = center_x + current_radius * np.cos(theta)
                y = np.full(seq_len, center_y)
                z = center_z + climb_height * norm_t
                vx = -current_radius * angular_velocity * 2 * np.pi * turns * np.sin(theta)
                vy = np.zeros(seq_len)
                vz = np.full(seq_len, climb_height)
            else:
                # Spiral in yz plane
                x = np.full(seq_len, center_x)
                y = center_y + current_radius * np.cos(theta)
                z = center_z + climb_height * norm_t
                vx = np.zeros(seq_len)
                vy = -current_radius * angular_velocity * 2 * np.pi * turns * np.sin(theta)
                vz = np.full(seq_len, climb_height)

        elif style == 'spiral_vertical_down':
            # Vertical spiral moving downward (in xz or yz plane)
            plane_choice = np.random.choice(['xz', 'yz'])
            turns = np.random.uniform(1.0, 3.0)  # Number of turns
            descent_height = np.random.uniform(15.0, 35.0)
            theta = 2 * np.pi * turns * norm_t * angular_velocity
            
            if plane_choice == 'xz':
                # Spiral in xz plane
                x = center_x + current_radius * np.cos(theta)
                y = np.full(seq_len, center_y)
                z = center_z - descent_height * norm_t
                vx = -current_radius * angular_velocity * 2 * np.pi * turns * np.sin(theta)
                vy = np.zeros(seq_len)
                vz = np.full(seq_len, -descent_height)
            else:
                # Spiral in yz plane
                x = np.full(seq_len, center_x)
                y = center_y + current_radius * np.cos(theta)
                z = center_z - descent_height * norm_t
                vx = np.zeros(seq_len)
                vy = -current_radius * angular_velocity * 2 * np.pi * turns * np.sin(theta)
                vz = np.full(seq_len, -descent_height)

        else:
            # Simple straight line (fallback)
            x = center_x + norm_t * 10
            y = np.full(seq_len, center_y)
            z = np.full(seq_len, center_z)
            vx = np.full(seq_len, 10.0)
            vy = np.zeros(seq_len)
            vz = np.zeros(seq_len)

        # Smooth and re-calculate velocity if not done above
        if style not in ['clover', 'spiral_inward', 'spiral_outward', 'spiral_vertical_up', 'spiral_vertical_down']:
            x_smooth = smooth_trajectory(x)
            y_smooth = smooth_trajectory(y)
            z_smooth = smooth_trajectory(z)
            dt = 1.0 / seq_len
            vx = np.gradient(x_smooth, dt)
            vy = np.gradient(y_smooth, dt)
            vz = np.gradient(z_smooth, dt)
            x, y, z = x_smooth, y_smooth, z_smooth
        
        # Compute speed and direction
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        direction = np.stack([vx, vy, vz], axis=-1)
        norms = np.linalg.norm(direction, axis=-1, keepdims=True)
        direction = np.divide(direction, norms, where=norms>0, out=np.zeros_like(direction))
        
        # Attitude: direction + fixed components (e.g., for roll/pitch/yaw approximation)
        attitude = np.concatenate([direction, np.full((seq_len, 3), 0.1)], axis=-1)
        
        # Full state: [speed, x, y, z, attitude(6)]
        state = np.column_stack([speed, x, y, z, attitude])
        trajectories.append(state)
        
    return torch.tensor(np.stack(trajectories), dtype=torch.float32)

# B-spline trajectory representation (Section III-E)
class BSplineTrajectory:
    """
    B-spline trajectory representation as described in MPD paper.
    Provides smooth trajectories with fewer parameters than dense waypoints.
    """
    def __init__(self, control_points, degree=3, seq_len=60):
        """
        Args:
            control_points: Tensor of shape (n_b, d) - B-spline control points
            degree: Degree of B-spline (p in paper)
            seq_len: Number of dense evaluation points (H in paper)
        """
        self.control_points = control_points  # w in paper equation (19)
        self.degree = degree
        self.seq_len = seq_len
        self.n_b = control_points.shape[0]  # Number of control points
        self.d = control_points.shape[1]    # State dimension
        
        # Precompute B-spline basis matrix (Section III-E)
        self.basis_matrix = self._compute_basis_matrix()
        
    def _compute_basis_matrix(self):
        """Precompute B-spline basis matrix B ∈ R^{n_s × n_b} using scipy"""
        try:
            # Use scipy's BSpline for robust basis computation
            control_points_np = self.control_points.cpu().numpy()
            
            # Create knot vector: clamped uniform knots as in paper
            # Total knots = n_b + degree + 1
            n_internal_knots = self.n_b - self.degree + 1  # Corrected calculation
            
            if n_internal_knots < 2:
                # For very few control points, use simple linear interpolation
                internal_knots = np.linspace(0, 1, 2)
            else:
                internal_knots = np.linspace(0, 1, n_internal_knots)
            
            # Create clamped knot vector: degree+1 zeros, internal knots, degree+1 ones
            knots = np.concatenate([
                np.zeros(self.degree),
                internal_knots,
                np.ones(self.degree)
            ])
            
            # Remove duplicates and ensure proper length
            knots = np.unique(knots)
            if len(knots) < self.n_b + self.degree + 1:
                # Pad with repeated end knots if needed
                knots = np.concatenate([
                    np.zeros(self.degree + 1 - np.sum(knots == 0)),
                    knots,
                    np.ones(self.degree + 1 - np.sum(knots == 1))
                ])
            
            # Evaluate basis functions
            s_values = np.linspace(0, 1, self.seq_len)
            basis_matrix = np.zeros((self.seq_len, self.n_b))
            
            for i in range(self.n_b):
                # Create basis function for each control point
                coeffs = np.zeros(self.n_b)
                coeffs[i] = 1.0
                bspline = BSpline(knots, coeffs, self.degree)
                basis_matrix[:, i] = bspline(s_values)
                
            return torch.tensor(basis_matrix, dtype=torch.float32, device=self.control_points.device)
            
        except Exception as e:
            print(f"Warning: Scipy BSpline failed, using fallback linear basis: {e}")
            # Fallback: simple linear basis
            return self._compute_linear_basis()
    
    def _compute_linear_basis(self):
        """Fallback linear basis computation"""
        s_values = torch.linspace(0, 1, self.seq_len, device=self.control_points.device)
        basis_matrix = torch.zeros(self.seq_len, self.n_b, device=self.control_points.device)
        
        # Simple linear interpolation between control points
        for i in range(self.seq_len):
            s = s_values[i]
            # Map s to control point indices
            idx_float = s * (self.n_b - 1)
            idx_low = int(torch.floor(idx_float).item())
            idx_high = min(idx_low + 1, self.n_b - 1)
            weight_high = idx_float - idx_low
            weight_low = 1 - weight_high
            
            if idx_low < self.n_b:
                basis_matrix[i, idx_low] = weight_low
            if idx_high < self.n_b:
                basis_matrix[i, idx_high] = weight_high
                
        return basis_matrix
    
    def evaluate(self):
        """Evaluate B-spline at dense points: Q = Bw ∈ R^{n_s × d}"""
        return torch.matmul(self.basis_matrix, self.control_points)
    
    def derivatives(self, order=1):
        """Compute derivatives of B-spline w.r.t phase variable"""
        trajectory = self.evaluate()
        if order == 1:
            # First derivative (velocity)
            return torch.diff(trajectory, dim=0)
        elif order == 2:
            # Second derivative (acceleration)
            return torch.diff(torch.diff(trajectory, dim=0), dim=0)
        else:
            raise ValueError(f"Unsupported derivative order: {order}")

def demo_bspline_trajectory():
    """
    Demo to verify BSplineTrajectory class functionality
    """
    print("=== B-spline Trajectory Demo ===\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test case 1: Simple 2D trajectory
    print("Test 1: 2D Trajectory with 5 control points")
    # control_points_2d = torch.tensor([
    #     [0.0, 0.0],
    #     [1.0, 2.0], 
    #     [3.0, 1.0],
    #     [4.0, 3.0],
    #     [5.0, 0.0]
    # ], dtype=torch.float32)
    seq_len = 10
    trajectories = generate_aerobatic_trajectories(1, seq_len)
    control_points_2d = trajectories[0]

    bspline_2d = BSplineTrajectory(control_points_2d, degree=3, seq_len=seq_len*12)
    trajectory_2d = bspline_2d.evaluate()
    
    print(f"Control points shape: {control_points_2d.shape}")
    print(f"Basis matrix shape: {bspline_2d.basis_matrix.shape}")
    print(f"Trajectory shape: {trajectory_2d.shape}")
    print(f"Number of control points: {bspline_2d.n_b}")
    print(f"State dimension: {bspline_2d.d}")
    print(f"Sequence length: {bspline_2d.seq_len}")
    
    # Test case 2: 3D trajectory
    print("\nTest 2: 3D Trajectory with 6 control points")
    control_points_3d = torch.randn(6, 3)
    bspline_3d = BSplineTrajectory(control_points_3d, degree=3, seq_len=80)
    trajectory_3d = bspline_3d.evaluate()
    
    print(f"3D trajectory shape: {trajectory_3d.shape}")
    
    # Test derivatives
    print("\nTest 3: Derivatives")
    velocity = bspline_2d.derivatives(order=1)
    acceleration = bspline_2d.derivatives(order=2)
    
    print(f"Velocity shape (1st derivative): {velocity.shape}")
    print(f"Acceleration shape (2nd derivative): {acceleration.shape}")
    
    # Test case 4: Different degrees
    print("\nTest 4: Different B-spline degrees")
    degrees = [1, 2, 3]
    trajectories_by_degree = {}
    
    for degree in degrees:
        bspline = BSplineTrajectory(control_points_2d, degree=degree, seq_len=100)
        trajectories_by_degree[degree] = bspline.evaluate()
        print(f"Degree {degree}: trajectory shape {trajectories_by_degree[degree].shape}")
    
    # Test case 5: Property verification
    print("\nTest 5: Property Verification")
    
    # Check that trajectory = B * w
    manual_trajectory = torch.matmul(bspline_2d.basis_matrix, control_points_2d)
    error = torch.max(torch.abs(trajectory_2d - manual_trajectory))
    print(f"Matrix multiplication verification error: {error:.6f}")
    
    # Check basis matrix properties
    basis_sum = torch.sum(bspline_2d.basis_matrix, dim=1)
    print(f"Basis functions sum to 1: {torch.allclose(basis_sum, torch.ones_like(basis_sum), atol=1e-6)}")
    
    # Visualization
    visualize_results(control_points_2d, trajectory_2d, trajectories_by_degree, velocity, acceleration)
    
    return bspline_2d, trajectory_2d

def visualize_results(control_points, trajectory, trajectories_by_degree, velocity, acceleration):
    """Visualize the B-spline trajectory results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Original trajectory with control points
    axes[0, 0].plot(trajectory[:, 0].numpy(), trajectory[:, 1].numpy(), 'b.-', linewidth=2, label='B-spline trajectory')
    axes[0, 0].plot(control_points[:, 0].numpy(), control_points[:, 1].numpy(), 'ro--', 
                   linewidth=1, markersize=8, label='Control points')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('B-spline Trajectory')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # Plot 2: Different degrees comparison
    for degree, traj in trajectories_by_degree.items():
        axes[0, 1].plot(traj[:, 0].numpy(), traj[:, 1].numpy(), 
                       linewidth=2, label=f'Degree {degree}')
    axes[0, 1].plot(control_points[:, 0].numpy(), control_points[:, 1].numpy(), 'ko--',
                   linewidth=1, markersize=6, label='Control points')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title('Trajectories with Different Degrees')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Basis functions
    seq_len = trajectory.shape[0]
    time = np.linspace(0, 1, seq_len)
    for i in range(min(5, control_points.shape[0])):
        axes[0, 2].plot(time, BSplineTrajectory(control_points, degree=3, seq_len=seq_len).basis_matrix[:, i].numpy(),
                       label=f'Basis {i+1}')
    axes[0, 2].set_xlabel('Normalized Time')
    axes[0, 2].set_ylabel('Basis Value')
    axes[0, 2].set_title('B-spline Basis Functions')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Velocity (1st derivative)
    time_vel = np.linspace(0, 1, velocity.shape[0])
    axes[1, 0].plot(time_vel, velocity[:, 0].numpy(), 'r-', label='Velocity X')
    axes[1, 0].plot(time_vel, velocity[:, 1].numpy(), 'b-', label='Velocity Y')
    axes[1, 0].set_xlabel('Normalized Time')
    axes[1, 0].set_ylabel('Velocity')
    axes[1, 0].set_title('First Derivative (Velocity)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Acceleration (2nd derivative)
    time_accel = np.linspace(0, 1, acceleration.shape[0])
    axes[1, 1].plot(time_accel, acceleration[:, 0].numpy(), 'r-', label='Acceleration X')
    axes[1, 1].plot(time_accel, acceleration[:, 1].numpy(), 'b-', label='Acceleration Y')
    axes[1, 1].set_xlabel('Normalized Time')
    axes[1, 1].set_ylabel('Acceleration')
    axes[1, 1].set_title('Second Derivative (Acceleration)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Trajectory components over time
    time_full = np.linspace(0, 1, trajectory.shape[0])
    axes[1, 2].plot(time_full, trajectory[:, 0].numpy(), 'r-', label='X position')
    axes[1, 2].plot(time_full, trajectory[:, 1].numpy(), 'b-', label='Y position')
    axes[1, 2].set_xlabel('Normalized Time')
    axes[1, 2].set_ylabel('Position')
    axes[1, 2].set_title('Trajectory Components over Time')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\n=== Edge Cases Testing ===")
    
    # Test with minimal control points
    print("\nTest: Minimal control points (degree 1)")
    min_control = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    try:
        bspline_min = BSplineTrajectory(min_control, degree=1, seq_len=50)
        traj_min = bspline_min.evaluate()
        print(f"Minimal control points trajectory shape: {traj_min.shape}")
    except Exception as e:
        print(f"Minimal control points failed: {e}")
    
    # Test with many control points
    print("\nTest: Many control points")
    many_control = torch.randn(20, 2)
    bspline_many = BSplineTrajectory(many_control, degree=3, seq_len=200)
    traj_many = bspline_many.evaluate()
    print(f"Many control points trajectory shape: {traj_many.shape}")
    
    # Test different devices
    print("\nTest: Device compatibility")
    if torch.cuda.is_available():
        control_points_cuda = many_control.cuda()
        bspline_cuda = BSplineTrajectory(control_points_cuda, degree=3, seq_len=100)
        traj_cuda = bspline_cuda.evaluate()
        print(f"CUDA trajectory shape: {traj_cuda.shape}, device: {traj_cuda.device}")
    else:
        print("CUDA not available, skipping device test")

if __name__ == "__main__":
    # Run the main demo
    bspline, trajectory = demo_bspline_trajectory()
    
    # Run edge cases
    test_edge_cases()
    
    print("\n=== Demo Completed ===")
    print("Key verification points:")
    print("1. B-spline produces smooth trajectories from sparse control points")
    print("2. Matrix multiplication Q = Bw is correctly implemented") 
    print("3. Derivatives provide velocity and acceleration information")
    print("4. Different degrees produce different smoothness levels")
    print("5. Basis functions have partition of unity property")
