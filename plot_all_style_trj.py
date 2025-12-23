import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams

# Set Times New Roman font
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 10
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8

def generate_style_demonstration_figure():
    """Generate examples for each flight trajectory style and create a summary PDF figure"""
    
    # Define all flight trajectory styles
    maneuver_styles = [
        'power_loop', 'barrel_roll', 'split_s', 'immelmann', 'wall_ride', 
        'eight_figure', 'star', 'half_moon', 'sphinx', 'clover',
        'spiral_inward', 'spiral_outward', 'spiral_vertical_up', 'spiral_vertical_down'
    ]
    
    # Create main figure with 3 rows
    fig = plt.figure(figsize=(16, 12))  # Adjusted for 3 rows
    
    # Generate trajectory for each style
    for idx, style in enumerate(maneuver_styles):
        # Generate single trajectory
        trajectory = generate_single_style_trajectory(style, seq_len=60)
        
        # Calculate subplot position (3 rows x 5 columns, last row has 4 plots)
        row = idx // 5
        col = idx % 5
        ax = fig.add_subplot(3, 5, idx + 1, projection='3d')
        
        # Plot trajectory
        plot_single_style_trajectory(ax, trajectory, style, idx)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Add overall title
    fig.suptitle('Aerobatic Maneuver Styles Demonstration', fontsize=14, fontweight='bold', y=0.98)
    
    # Save as PDF
    plt.savefig('maneuver_styles_demonstration.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('maneuver_styles_demonstration.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("All flight trajectory style examples saved as 'maneuver_styles_demonstration.pdf'")

from AeroDM_SafeTrj_v2_Test import generate_single_style_trajectory

def plot_single_style_trajectory(ax, trajectory, style_name, style_idx):
    """Plot a single style trajectory on 3D axis"""
    
    # Color scheme for different styles
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    color = colors[style_idx % len(colors)]
    
    # Plot trajectory line
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            color=color, linewidth=2.0, alpha=0.9)
    
    # Mark start and end points
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
              color='green', s=40, marker='o', alpha=0.8)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
              color='red', s=40, marker='s', alpha=0.8)
    
    # Mark trajectory points every 10 steps
    ax.scatter(trajectory[::10, 0], trajectory[::10, 1], trajectory[::10, 2], 
              color=color, s=15, alpha=0.6, marker='.')
    
    # Set title and labels
    title = style_name.replace('_', ' ').title()
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('X', labelpad=3)
    ax.set_ylabel('Y', labelpad=3)
    ax.set_zlabel('Z', labelpad=3)
    
    # Set consistent axis ranges
    ax.set_xlim([-12, 12])
    ax.set_ylim([-12, 12])
    ax.set_zlim([-5, 35])
    
    # Grid and appearance
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Set consistent view angle
    ax.view_init(elev=20, azim=45)

def create_style_comparison_table():
    """Create a comparison table for flight trajectory styles"""
    styles_info = {
        'power_loop': 'Complete vertical loop maneuver',
        'barrel_roll': 'Helical rolling maneuver', 
        'split_s': 'Split-S maneuver (half loop descent)',
        'immelmann': 'Immelmann turn (half loop ascent)',
        'wall_ride': 'Vertical spiral climb',
        'eight_figure': 'Figure-eight horizontal maneuver',
        'star': '3D star/Lissajous curve',
        'half_moon': 'Semicircular arc maneuver',
        'sphinx': 'Spiral climb with pitch variation',
        'clover': 'Four-leaf clover horizontal pattern',
        'spiral_inward': 'Horizontal inward contracting spiral',
        'spiral_outward': 'Horizontal outward expanding spiral', 
        'spiral_vertical_up': 'Vertical upward spiral',
        'spiral_vertical_down': 'Vertical downward spiral'
    }
    
    # Create description table figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for style, description in styles_info.items():
        english_name = style.replace('_', ' ').title()
        table_data.append([english_name, description])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Maneuver Style', 'Description'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.3, 0.7])
    
    # Set table style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4C72B0')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(len(table_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i+1, j)].set_facecolor('#F0F0F0')
    
    plt.title('Aerobatic Maneuver Styles Description', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('maneuver_styles_description.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('maneuver_styles_description.svg', format='svg', bbox_inches='tight')
    plt.show()

def create_compact_demonstration():
    """Create a more compact version with better spacing"""
    
    maneuver_styles = [
        'power_loop', 'barrel_roll', 'split_s', 'immelmann', 'wall_ride', 
        'eight_figure', 'star', 'half_moon', 'sphinx', 'clover',
        'spiral_inward', 'spiral_outward', 'spiral_vertical_up', 'spiral_vertical_down'
    ]
    
    # Create figure with 3 rows and 5 columns
    fig = plt.figure(figsize=(18, 10))
    
    for idx, style in enumerate(maneuver_styles):
        trajectory = generate_single_style_trajectory(style, seq_len=60)
        
        # 3 rows x 5 columns layout
        ax = fig.add_subplot(3, 5, idx + 1, projection='3d')
        
        # Color scheme
        colors = plt.cm.Set3(np.linspace(0, 1, len(maneuver_styles)))
        color = colors[idx]
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                color=color, linewidth=2.0, alpha=0.9)
        
        # Mark points
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                  color='green', s=30, marker='o', alpha=0.8)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                  color='red', s=30, marker='s', alpha=0.8)
        
        # Title and labels
        title = style.replace('_', ' ').title()
        ax.set_title(title, fontsize=9, fontweight='bold', pad=5)
        ax.set_xlabel('X', labelpad=2, fontsize=7)
        ax.set_ylabel('Y', labelpad=2, fontsize=7)
        ax.set_zlabel('Z', labelpad=2, fontsize=7)
        
        # Consistent ranges
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([0, 30])
        
        # Grid and view
        ax.grid(True, alpha=0.2)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.view_init(elev=20, azim=45)
        
        # Adjust tick labels for better readability
        ax.tick_params(axis='both', which='major', labelsize=6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle('Aerobatic Maneuver Styles - 3D Trajectory Demonstration', 
                 fontsize=12, fontweight='bold', y=0.98)
    
    plt.savefig('maneuver_styles_compact.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('maneuver_styles_compact.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.show()

# Execute the code
if __name__ == "__main__":
    print("Generating flight trajectory style demonstration figures...")
    
    # Generate main demonstration figure (3 rows)
    generate_style_demonstration_figure()
    
    # Generate compact version
    create_compact_demonstration()
    
    # Generate description table
    # create_style_comparison_table()
    
    print("\nAll figures generated successfully!")
    print("Generated files:")
    print("1. maneuver_styles_demonstration.pdf - Main 3-row demonstration figure")
    print("2. maneuver_styles_compact.pdf - Compact version")
    print("3. maneuver_styles_description.pdf - Style description table")
    print("4. SVG versions of all figures")