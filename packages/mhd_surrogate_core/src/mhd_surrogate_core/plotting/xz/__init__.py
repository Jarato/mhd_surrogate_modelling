"""
The 2D (XZ) plotting package for the MHD Surrogate Core project.
"""

from .video import generate_video_from_npz
from .processed import (
    plot_xz_snapshot,
    plot_x_time_evolution,
    plot_z_time_evolution,
)

__all__ = [
    "generate_video_from_npz",
    "plot_xz_snapshot",
    "plot_x_time_evolution",
    "plot_z_time_evolution",
]

