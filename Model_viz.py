import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import numpy as np
import sys
import os

# Add the directory containing AeroDM.py to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from AeroDM import Config, AeroDM
except ImportError:
    print("Warning: Could not import AeroDM. Creating mock model for visualization.")
    # Mock classes for visualization if AeroDM is not available
    class Config:
        latent_dim = 256
        num_layers = 4
        num_heads = 4
        dropout = 0.1
        seq_len = 60
        state_dim = 10
        target_dim = 3
        action_dim = 5
        history_len = 5
        diffusion_steps = 30
    
    class AeroDM(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

def create_detailed_architecture_diagram(config):
    """
    Create a comprehensive architecture diagram without Graphviz
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Main architecture diagram
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    create_main_architecture(ax1, config)
    
    # Condition embedding details
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    create_condition_embedding_diagram(ax2, config)
    
    # Transformer details
    ax3 = plt.subplot2grid((2, 3), (1, 2))
    create_transformer_details(ax3, config)
    
    plt.tight_layout()
    plt.savefig('Figs/aero_dm_detailed_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_main_architecture(ax, config):
    """Create the main architecture diagram"""
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 15)
    
    # Define component positions
    components = {
        'Input\nNoisy Trajectory': (1, 12, 2, 1),
        'History\nContext': (1, 9, 2, 1),
        'Input Projection\nLinear(10→256)': (4, 12, 3, 1),
        'Positional\nEncoding': (4, 9, 3, 1),
        'Condition\nEmbedding': (8, 12, 3, 2),
        'Feature Fusion\n(Concatenation + Addition)': (12, 10.5, 3, 1),
        'Transformer\nDecoder\n(4 Layers)': (12, 7, 3, 2),
        'Output Projection\nLinear(256→10)': (16, 10.5, 2, 1),
        'Output\nDenoised Trajectory': (19, 10.5, 2, 1)
    }
    
    # Draw components
    for name, (x, y, w, h) in components.items():
        ax.add_patch(Rectangle((x, y), w, h, 
                             facecolor='lightblue', 
                             edgecolor='navy',
                             linewidth=2,
                             alpha=0.8))
        ax.text(x + w/2, y + h/2, name, 
               ha='center', va='center', 
               fontsize=8, fontweight='bold',
               wrap=True)
    
    # Draw data flow arrows
    arrows = [
        # Main flow
        ((3, 12.5), (4, 12.5), 'Input data'),
        ((7, 12.5), (8, 12.5), 'Projected features'),
        ((11, 12.5), (12, 10.5), 'Conditioned features'),
        ((15, 10.5), (16, 10.5), 'Transformer output'),
        ((18, 10.5), (19, 10.5), 'Final output'),
        
        # History flow
        ((3, 9.5), (4, 9.5), 'History context'),
        ((7, 9.5), (12, 9.5), 'Positional info'),
        
        # Condition inputs
        ((8, 14), (8, 13), 'Timestep t'),
        ((8, 8), (8, 9), 'Target + Action'),
    ]
    
    for (x1, y1), (x2, y2), label in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', 
                                 color='red', 
                                 lw=2,
                                 alpha=0.7))
        # Add label near arrow
        label_x = (x1 + x2) / 2
        label_y = (y1 + y2) / 2
        ax.text(label_x, label_y + 0.2, label, 
               ha='center', va='bottom', 
               fontsize=6, color='darkred',
               bbox=dict(boxstyle="round,pad=0.1", facecolor="yellow", alpha=0.7))
    
    ax.set_title('AeroDM: Complete Architecture Overview\n'
                'Diffusion Transformer for Aerobatic Trajectory Generation', 
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

def create_condition_embedding_diagram(ax, config):
    """Create detailed condition embedding diagram"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    
    # Condition embedding components
    components = {
        'Timestep t\n(B,)': (1, 6, 2, 0.5),
        'Target Waypoint\n(B, 3)': (1, 4, 2, 0.5),
        'Action Style\n(B, 5)': (1, 2, 2, 0.5),
        'Time Embedding\nMLP': (4, 6, 2, 0.5),
        'Target Embedding\nMLP': (4, 4, 2, 0.5),
        'Action Embedding\nMLP': (4, 2, 2, 0.5),
        'Feature Fusion\n(Element-wise Sum)': (7, 4, 2, 1),
        'Condition Output\n(B, 256)': (9, 4, 1, 0.5)
    }
    
    for name, (x, y, w, h) in components.items():
        color = 'lightgreen' if 'Input' in name else 'lightyellow' if 'Embedding' in name else 'lightcoral'
        ax.add_patch(Rectangle((x, y), w, h, 
                             facecolor=color, 
                             edgecolor='black',
                             linewidth=1))
        ax.text(x + w/2, y + h/2, name, 
               ha='center', va='center', 
               fontsize=6, wrap=True)
    
    # Draw connections
    for i, input_comp in enumerate(['Timestep', 'Target', 'Action']):
        y_pos = 6 - i * 2
        ax.annotate('', xy=(4, y_pos + 0.25), xytext=(3, y_pos + 0.25),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1))
        ax.annotate('', xy=(7, 4.5), xytext=(6, y_pos + 0.25),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1))
    
    ax.annotate('', xy=(9, 4.25), xytext=(9, 4.25),
               arrowprops=dict(arrowstyle='->', color='blue', lw=1))
    
    ax.set_title('Condition Embedding Module', fontsize=10, fontweight='bold')
    ax.axis('off')

def create_transformer_details(ax, config):
    """Create transformer layer details"""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    
    # Single transformer layer components
    layer_components = {
        'Input\n(B, L, 256)': (1, 8, 2, 0.5),
        'Layer Normalization': (4, 8, 2, 0.5),
        'Multi-Head\nAttention\n(4 heads)': (4, 6, 2, 1),
        'Add & Normalize': (7, 7, 1, 0.5),
        'Feed Forward\n(256→1024→256)': (4, 4, 2, 1),
        'Add & Normalize': (7, 4.5, 1, 0.5),
        'Output\n(B, L, 256)': (9, 6, 2, 0.5)
    }
    
    for name, (x, y, w, h) in layer_components.items():
        color = 'lightblue'
        if 'Attention' in name:
            color = 'lightcoral'
        elif 'Feed' in name:
            color = 'lightgreen'
        elif 'Normalize' in name:
            color = 'lightyellow'
            
        ax.add_patch(Rectangle((x, y), w, h, 
                             facecolor=color, 
                             edgecolor='black',
                             linewidth=1))
        ax.text(x + w/2, y + h/2, name, 
               ha='center', va='center', 
               fontsize=6, wrap=True)
    
    # Draw connections
    connections = [
        ((3, 8.25), (4, 8.25)),
        ((6, 8.25), (7, 7.25)),
        ((7, 6.75), (4, 6.5), 'Residual'),
        ((6, 6.5), (4, 4.5)),
        ((6, 4.5), (7, 4.5)),
        ((7, 4), (4, 3.5), 'Residual'),
        ((6, 3.5), (9, 6.25))
    ]
    
    for i, connection in enumerate(connections):
        if len(connection) == 2:
            (x1, y1), (x2, y2) = connection
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1))
        else:
            (x1, y1), (x2, y2), label = connection
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='green', lw=1))
            ax.text((x1+x2)/2, (y1+y2)/2, label, 
                   ha='center', va='center', fontsize=5, color='darkgreen')
    
    ax.text(1, 2, f'× {config.num_layers} Layers', 
           ha='left', va='center', fontsize=8, fontweight='bold')
    
    ax.set_title('Transformer Decoder Layer', fontsize=10, fontweight='bold')
    ax.axis('off')

def create_model_parameters_chart(config):
    """Create a bar chart showing model parameter distribution"""
    # Estimated parameters based on configuration
    components = {
        'Input Projection': config.state_dim * config.latent_dim + config.latent_dim,
        'Output Projection': config.latent_dim * config.state_dim + config.state_dim,
        'Time Embedding': (1 + config.latent_dim + config.latent_dim) * config.latent_dim,
        'Target Embedding': (config.target_dim + config.latent_dim + config.latent_dim) * config.latent_dim,
        'Action Embedding': (config.action_dim + config.latent_dim + config.latent_dim) * config.latent_dim,
        'Transformer Layers': config.num_layers * (
            # Self-attention
            3 * config.latent_dim * config.latent_dim + config.latent_dim * config.latent_dim +
            # Feed-forward
            2 * config.latent_dim * config.latent_dim * 4 + config.latent_dim * 4 + config.latent_dim
        )
    }
    
    total_params = sum(components.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart
    names = list(components.keys())
    values = list(components.values())
    
    bars = ax1.barh(names, values, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Number of Parameters')
    ax1.set_title('Model Parameters by Component')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax1.text(width + max(values)*0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:,}', ha='left', va='center', fontweight='bold')
    
    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
    ax2.pie(values, labels=names, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title(f'Parameter Distribution\nTotal: {total_params:,} parameters')
    
    plt.tight_layout()
    plt.savefig('Figs/model_parameters_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return total_params

def print_model_summary(config):
    """Print detailed model configuration summary"""
    print("\n" + "="*70)
    print("AeroDM Model Configuration Summary")
    print("="*70)
    
    print(f"\nModel Dimensions:")
    print(f"  Latent dimension: {config.latent_dim}")
    print(f"  Number of layers: {config.num_layers}")
    print(f"  Attention heads: {config.num_heads}")
    print(f"  Dropout rate: {config.dropout}")
    
    print(f"\nSequence Parameters:")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  State dimension: {config.state_dim}")
    print(f"  History length: {config.history_len}")
    
    print(f"\nCondition Dimensions:")
    print(f"  Target dimension: {config.target_dim}")
    print(f"  Action dimension: {config.action_dim}")
    
    print(f"\nDiffusion Process:")
    print(f"  Diffusion steps: {config.diffusion_steps}")
    print(f"  CBF guidance: {getattr(config, 'enable_cbf_guidance', 'Not specified')}")
    
    print("\n" + "="*70)

def main():
    """Main function to run all visualizations"""
    print("AeroDM Model Visualization Tool")
    print("No Graphviz required - Using matplotlib for visualization")
    print("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    try:
        # Print model summary
        print_model_summary(config)
        
        # Create detailed architecture diagram
        print("\nCreating detailed architecture diagram...")
        create_detailed_architecture_diagram(config)
        
        # Create parameter analysis
        print("\nAnalyzing model parameters...")
        total_params = create_model_parameters_chart(config)
        print(f"Estimated total parameters: {total_params:,}")
        
        print("\n" + "="*60)
        print("Visualization completed successfully!")
        print("Generated files:")
        print("  - aero_dm_detailed_architecture.png")
        print("  - model_parameters_analysis.png")
        print("="*60)
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()