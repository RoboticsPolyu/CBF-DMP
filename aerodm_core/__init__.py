"""Reusable obstacle-aware AeroDM package."""

from .config import Config
from .data import (
    add_target_noise,
    denormalize_obstacle,
    denormalize_target,
    denormalize_trajectories,
    generate_random_obstacles,
    generate_target_waypoints,
    normalize_obstacle,
    normalize_target,
    normalize_trajectories,
    project_target_outside_obstacles,
)
from .diffusion import ObstacleAwareDiffusionProcess
from .guidance import compute_barrier_and_grad, compute_barrier_and_grad_logistic
from .losses import AeroDMLoss
from .model import AeroDM
from .metrics import (compute_collision_rate, compute_success_rates,
                      compute_trajectory_errors, get_collision_mask)
from .models import (
    AttentionObstacleEncoder,
    ConditionEmbedding,
    ObstacleAwareDiffusionTransformer,
    ObstacleEncoder,
    PositionalEncoding,
)


def run_experiment(*args, **kwargs):
    """Lazily import and run the end-to-end experiment workflow."""
    from .experiment import run_experiment as _run_experiment
    return _run_experiment(*args, **kwargs)


def test_model_performance_cb_eva(*args, **kwargs):
    """Lazily import the quantitative evaluation workflow."""
    from .evaluation import test_model_performance_cb_eva as _evaluate
    return _evaluate(*args, **kwargs)


def plot_combined_trajectories(*args, **kwargs):
    """Lazily import the combined-results visualization."""
    from .visualization import plot_combined_trajectories as _plot
    return _plot(*args, **kwargs)


def plot_test_results(*args, **kwargs):
    """Lazily import the detailed-results visualization."""
    from .visualization import plot_test_results as _plot
    return _plot(*args, **kwargs)

__all__ = [
    "Config",
    "AeroDM",
    "AeroDMLoss",
    "AttentionObstacleEncoder",
    "ConditionEmbedding",
    "ObstacleAwareDiffusionProcess",
    "ObstacleAwareDiffusionTransformer",
    "ObstacleEncoder",
    "PositionalEncoding",
    "add_target_noise",
    "compute_barrier_and_grad",
    "compute_barrier_and_grad_logistic",
    "compute_collision_rate",
    "compute_success_rates",
    "compute_trajectory_errors",
    "denormalize_obstacle",
    "denormalize_target",
    "denormalize_trajectories",
    "generate_random_obstacles",
    "generate_target_waypoints",
    "get_collision_mask",
    "normalize_obstacle",
    "normalize_target",
    "normalize_trajectories",
    "project_target_outside_obstacles",
    "plot_combined_trajectories",
    "plot_test_results",
    "run_experiment",
    "test_model_performance_cb_eva",
]
