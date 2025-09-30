# CBF-AeroDM: Diffusion-based Motion Planning with Control Barrier Function Guided Sampling for Urban Air Vehicles

A PyTorch implementation of a diffusion-based trajectory generation model for aerobatic maneuvers, enhanced with Control Barrier Function (CBF) guidance for safe obstacle avoidance.

## Overview

This project implements a conditional diffusion transformer model that generates diverse aerobatic trajectories while incorporating safety constraints through CBF guidance. The model can generate five distinct maneuver styles while avoiding obstacles in 3D space.

![1759221920289](images/README/Framework.png)

## Key Features

- **Diffusion-based Trajectory Generation**: Uses a transformer-based architecture to generate smooth aerobatic trajectories
- **Multiple Maneuver Styles**: Supports five aerobatic maneuvers:
  - Power Loop
  - Barrel Roll
  - Split-S
  - Immelmann Turn
  - Wall Ride
- **CBF Safety Guidance**: Integrates Control Barrier Functions for obstacle avoidance during inference
- **Conditional Generation**: Generates trajectories based on target waypoints and maneuver style actions
- **Historical Context**: Incorporates 5-frame historical observations for context-aware generation

## Model Architecture

### Core Components

1. **DiffusionTransformer**: Transformer-based denoising network

   - Positional encoding for temporal information
   - Multi-head self-attention with causal masking
   - Conditional embedding for target waypoints and maneuver styles
2. **Diffusion Process**: Implements DDPM with linear noise schedule

   - 30 diffusion steps
   - Configurable beta schedule
3. **CBF Guidance**: Quadratic barrier function for obstacle avoidance

   - Spherical obstacle representation
   - Gradient-based guidance during reverse diffusion
   - Configurable safety radius and guidance strength

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd AeroDM-CBF

# Install dependencies
pip install torch matplotlib numpy
```

## Usage

### Training

```python
from AeroDM_CBF import train_improved_aerodm

# Train the model
model, losses, trajectories, mean, std = train_improved_aerodm()
```

### Inference

```python
# Generate trajectories with CBF guidance
sampled_trajectory = model.sample(
    target=target_waypoints,
    action=maneuver_style, 
    history=historical_observations,
    enable_guidance=True,
    guidance_gamma=1.0
)
```

### Configuration

Key parameters in `Config` class:

- `latent_dim`: 256 (transformer hidden dimension)
- `diffusion_steps`: 30
- `seq_len`: 60 (trajectory length)
- `state_dim`: 10 (speed + position + attitude)
- `enable_cbf_guidance`: Toggle for obstacle avoidance
- `obstacle_radius`: 5.0 (safety distance)

## Data Generation

The model includes synthetic data generation for five aerobatic maneuvers:

```python
trajectories = generate_enhanced_circular_trajectories(
    num_trajectories=100,
    seq_len=60,
    radius=10.0
)
```

## Visualization

Comprehensive visualization tools included:

- 3D trajectory plots with obstacle visualization
- Multi-view projections (XY, XZ, YZ)
- Temporal analysis of position components
- Error analysis and statistics
- Z-axis performance monitoring

## Safety Features

### Control Barrier Function

```python
def compute_barrier_and_grad(x, config):
    # Quadratic barrier for spherical obstacles
    # V = sum_τ max(0, r - ||pos_τ - center||)^2
    # Returns barrier value V and gradient ∇V
```

The CBF guidance:

- Computes safety constraints during reverse diffusion
- Adjusts generated trajectories to avoid obstacles
- Maintains smoothness while ensuring safety

## Performance Monitoring

- **Balanced Loss Function**: Special weighting for Z-axis learning
- **Multi-component Loss**: Position, velocity, and attitude losses
- **Error Analysis**: Comprehensive trajectory error metrics
- **Training Visualization**: Loss progression plots

## File Structure

```
AeroDM_CBF.py
├── Config Class (Model parameters)
├── PositionalEncoding (Transformer positional encoding)
├── ConditionEmbedding (Target and action conditioning)
├── DiffusionTransformer (Main denoising network)
├── DiffusionProcess (DDPM implementation)
├── CBF Functions (Safety guidance)
├── AeroDM Class (Complete model wrapper)
├── Loss Functions (Improved training objectives)
├── Data Generation (Synthetic trajectory creation)
└── Visualization Tools (Plotting and analysis)
```

## Applications

- Autonomous drone navigation
- Aerobatic trajectory planning
- Safe motion generation in constrained environments
- Robotics and autonomous systems

## References

Based on diffusion models and Control Barrier Function guidance principles from recent literature on safe trajectory generation.

## License

[Add appropriate license information]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add appropriate citation format]
```
