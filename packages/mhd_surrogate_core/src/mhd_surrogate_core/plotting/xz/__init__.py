"""
The 2D (XZ) plotting package for the MHD Surrogate Core project.
"""

from .video import generate_video_from_npz
from .processed import (
    plot_xz_snapshot,
    plot_x_time_evolution,
    plot_z_time_evolution,
)
from .evaluation import (
    plot_r2_performance,
    plot_prediction_rollout_error,
    plot_prediction_dashboard,
)

__all__ = [
    # Video
    "generate_video_from_npz",
    # Processed (static)
    "plot_xz_snapshot",
    "plot_x_time_evolution",
    "plot_z_time_evolution",
    # Evaluation
    "plot_r2_performance",
    "plot_prediction_rollout_error",
    "plot_prediction_dashboard",
]

