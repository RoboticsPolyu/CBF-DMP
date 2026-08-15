"""Public visualization API."""

from .results import plot_combined_trajectories, plot_test_results
from .styles import (
    plot_action_comparison,
    plot_connection_info,
    plot_style_information,
    plot_style_statistics,
    plot_trajectory_statistics,
)
from .trajectory import (
    plot_2d_obstacles,
    plot_2d_projection,
    plot_3d_obstacles,
    plot_3d_trajectory,
    plot_collision_markers,
    plot_error_analysis,
    plot_position_time,
    plot_speed_comparison,
    plot_trajectories_demo,
)

__all__ = [name for name in globals() if name.startswith("plot_")]
