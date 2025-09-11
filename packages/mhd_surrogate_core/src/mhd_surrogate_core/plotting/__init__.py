# mhd_surrogate_core/src/mhd_surrogate_core/plotting/__init__.py

"""
A collection of plotting utilities for the MHD surrogate model project.
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
)

__all__ = [
    # Evaluation plots
    "plot_snapshot_comparison",
    "plot_prediction_comparison",
    "plot_prediction_rollout_error",
    "plot_latent_rollout_error",
    "plot_latent_trajectories",
    "plot_koopman_mode_evolution",
    "plot_r2_performance",
    "plot_reconstruction_error_over_time",
    "plot_prediction_dashboard",
    # Processed data plots
    "plot_z_time_evolution",
    "plot_x_time_evolution",
    "plot_y_time_evolution",
    "plot_xz_slice",
    "plot_xy_slice",
    "plot_yz_slice",
    # Raw data plots
    "plot_interpolated_xz_slice",
    "plot_interpolated_xy_slice",
    "plot_interpolated_yz_slice",
]
