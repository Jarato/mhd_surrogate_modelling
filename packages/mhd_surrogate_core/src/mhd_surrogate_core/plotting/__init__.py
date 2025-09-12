"""
The plotting package for the MHD Surrogate Core project.

This package provides a set of functions for visualizing simulation data
and model performance metrics.
"""

from .evaluation import (
    plot_snapshot_comparison,
    plot_prediction_comparison,
    plot_prediction_rollout_error,
    plot_latent_rollout_error,
    plot_latent_trajectories,
    plot_koopman_mode_evolution,
    plot_r2_performance,
    plot_reconstruction_error_over_time,
    plot_prediction_dashboard,
)

from .processed import (
    plot_z_time_evolution,
    plot_x_time_evolution,
    plot_y_time_evolution,
    plot_xz_slice,
    plot_xy_slice,
    plot_yz_slice,
)

from .raw import (
    plot_interpolated_xz_slice,
    plot_interpolated_xy_slice,
    plot_interpolated_yz_slice,
    plot_interpolated_z_time_evolution,
    plot_interpolated_y_time_evolution,
    plot_interpolated_x_time_evolution,
)

from .video import (
    generate_slice_video,
    generate_slice_video_from_npz,
)

__all__ = [
    # Evaluation functions
    "plot_snapshot_comparison",
    "plot_prediction_comparison",
    "plot_prediction_rollout_error",
    "plot_latent_rollout_error",
    "plot_latent_trajectories",
    "plot_koopman_mode_evolution",
    "plot_r2_performance",
    "plot_reconstruction_error_over_time",
    "plot_prediction_dashboard",
    # Processed data functions
    "plot_z_time_evolution",
    "plot_x_time_evolution",
    "plot_y_time_evolution",
    "plot_xz_slice",
    "plot_xy_slice",
    "plot_yz_slice",
    # Raw data functions
    "plot_interpolated_xz_slice",
    "plot_interpolated_xy_slice",
    "plot_interpolated_yz_slice",
    "plot_interpolated_z_time_evolution",
    "plot_interpolated_y_time_evolution",
    "plot_interpolated_x_time_evolution",
    # Video generation
    "generate_slice_video",
    "generate_slice_video_from_npz",
]

