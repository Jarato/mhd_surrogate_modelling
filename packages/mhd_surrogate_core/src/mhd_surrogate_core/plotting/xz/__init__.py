"""
The 2D (XZ) plotting package for the MHD Surrogate Core project.
"""

from .video import generate_video_from_npz, generate_multi_sample_video
from .processed import (
    plot_xz_snapshot,
    plot_x_time_evolution,
    plot_z_time_evolution,
    plot_velocity_quiver,
)
from .evaluation import (
    plot_r2_performance,
    plot_prediction_rollout_error,
    plot_prediction_dashboard,
    plot_latent_rollout_error,
    plot_latent_trajectories,
    plot_koopman_eigenvector_evolution,
    plot_koopman_eigenvalues,
    plot_koopman_mode_spectrum,
)
from .comparison import generate_comparison_video_from_npz

__all__ = [
    # Video
    "generate_video_from_npz",
    "generate_multi_sample_video",
    # Processed (static)
    "plot_xz_snapshot",
    "plot_x_time_evolution",
    "plot_z_time_evolution",
    "plot_velocity_quiver",
    # Evaluation
    "plot_r2_performance",
    "plot_prediction_rollout_error",
    "plot_prediction_dashboard",
    "plot_latent_rollout_error",
    "plot_latent_trajectories",
    "plot_koopman_eigenvector_evolution",
    "plot_koopman_eigenvalues",
    "plot_koopman_mode_spectrum",
    # Comparison Video
    "generate_comparison_video_from_npz",
]
