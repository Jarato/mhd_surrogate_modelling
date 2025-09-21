# -*- coding: utf-8 -*-
# mhd_surrogate_core/plotting/xz/evaluation.py

"""
Functions for plotting model evaluation metrics for 2D (X-Z) data.
"""

import logging
from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt

def plot_prediction_rollout_error(stats_path: Path | str):
    """
    Loads and plots the per-channel and total error from a prediction rollout stats file.
    """
    stats_path = Path(stats_path)
    if not stats_path.exists():
        logging.error(f"Statistics file not found at: {stats_path}")
        return
    
    logging.info(f"Loading prediction rollout error from {stats_path}...")
    with np.load(stats_path, allow_pickle=True) as data:
        per_step_channel_error = data['per_step_channel_error']
        channel_names = data['channel_names']
        
    num_timesteps, num_channels = per_step_channel_error.shape
    timesteps = range(1, num_timesteps + 1)
    total_per_step_error = per_step_channel_error.mean(axis=1)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    cols = min(4, num_channels)
    channel_rows = math.ceil(num_channels / cols)
    total_rows = 1 + channel_rows
    fig_height = 3.5 * total_rows
    fig = plt.figure(figsize=(16, fig_height))
    gs = fig.add_gridspec(total_rows, cols, hspace=0.7, wspace=0.3)
    
    ax_total = fig.add_subplot(gs[0, :])
    ax_total.plot(timesteps, total_per_step_error, 'o-', color='black', label='Total Average MSE')
    ax_total.set_title('Total Prediction Rollout Error Over Time', fontsize=16, weight='bold')
    ax_total.set_ylabel("MSE")
    ax_total.set_yscale('log')
    ax_total.grid(True, which="both", ls="--")
    ax_total.legend()

    for i in range(num_channels):
        row, col = (i // cols) + 1, i % cols
        ax = fig.add_subplot(gs[row, col])
        ax.plot(timesteps, per_step_channel_error[:, i], 'o-', color='crimson', markersize=3, alpha=0.8)
        ax.set_title(f"Channel: {channel_names[i]}")
        ax.set_ylabel("MSE")
        ax.set_xlabel("Timestep")
        ax.grid(True)
        ax.set_yscale('log')

    fig.suptitle('Per-Channel Prediction Rollout Error', fontsize=20, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_r2_performance(stats_path: Path | str, eval_type: str = "Prediction"):
    """
    Loads and plots the per-channel R-squared scores from a statistics file.
    """
    stats_path = Path(stats_path)
    if not stats_path.exists():
        logging.error(f"Evaluation stats file not found at: {stats_path}")
        return

    logging.info(f"Loading R² performance from {stats_path}...")
    with np.load(stats_path, allow_pickle=True) as data:
        r_squared_per_channel = data['r_squared_per_channel']
        channel_names = data['channel_names']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, max(4, len(channel_names) * 0.5)))
    
    y_pos = np.arange(len(channel_names))
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in r_squared_per_channel]
    
    ax.barh(y_pos, r_squared_per_channel, align='center', color=colors)
    ax.set_yticks(y_pos, labels=channel_names)
    ax.invert_yaxis()
    ax.set_xlabel('R-squared (R²) Score')
    ax.set_title(f'Per-Channel {eval_type} Performance')
    
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    for i, v in enumerate(r_squared_per_channel):
        ax.text(v + 0.01, i, f'{v:.3f}', color='black', va='center')
        
    plt.tight_layout()
    plt.show()

def _plot_single_xz_slice(ax, data, coords, title, cmap='viridis', vmin=None, vmax=None):
    """Helper function to plot a single x-z slice."""
    im = ax.pcolormesh(
        coords['x'], coords['z'], data.T,
        shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Z Coordinate")
    return im

def _plot_single_z_time_evolution(ax, data, coords, title, cmap='viridis', vmin=None, vmax=None):
    """Helper function to plot a single z-time evolution."""
    im = ax.pcolormesh(
        range(data.shape[0]), coords['z'], data.T,
        shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Z Coordinate")
    return im

def plot_prediction_dashboard(
    ground_truth_timeseries: np.ndarray,
    predicted_timeseries: np.ndarray,
    difference_timeseries: np.ndarray,
    coords: dict,
    channel: str,
    x_index: int,
    time_index: int,
):
    """
    Creates a full dashboard visualizing ground truth, prediction, and error
    for both spatial slices and time evolution for 2D data.
    """
    try:
        channel_idx = coords['labels'].index(channel)
    except ValueError:
        logging.error(f"Channel '{channel}' not found.")
        return

    # --- Prepare Data Slices ---
    gt_slice = ground_truth_timeseries[time_index, :, :, channel_idx]
    pred_slice = predicted_timeseries[time_index, :, :, channel_idx]
    diff_slice = difference_timeseries[time_index, :, :, channel_idx]
    
    gt_time = ground_truth_timeseries[:, x_index, :, channel_idx]
    pred_time = predicted_timeseries[:, x_index, :, channel_idx]
    diff_time = difference_timeseries[:, x_index, :, channel_idx]

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    fig.suptitle(
        f"Comprehensive Analysis for Channel '{channel}'\n"
        f"Time Index: {time_index}, X Index: {x_index}",
        fontsize=16, y=0.98
    )

    # --- Row 1: Spatial Slices (X-Z) ---
    vmin_val = min(gt_slice.min(), pred_slice.min())
    vmax_val = max(gt_slice.max(), pred_slice.max())
    
    im1 = _plot_single_xz_slice(axes[0, 0], gt_slice, coords, "Ground Truth", vmin=vmin_val, vmax=vmax_val)
    _plot_single_xz_slice(axes[0, 1], pred_slice, coords, "Prediction", vmin=vmin_val, vmax=vmax_val)
    im3 = _plot_single_xz_slice(axes[0, 2], diff_slice, coords, "Difference (Error)", cmap='inferno')
    
    fig.colorbar(im1, ax=axes[0, :2], fraction=0.046, pad=0.04, label="Value")
    fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04, label="Abs. Error")

    # --- Row 2: Time Evolution (Time-Z) ---
    vmin_time = min(gt_time.min(), pred_time.min())
    vmax_time = max(gt_time.max(), pred_time.max())

    im4 = _plot_single_z_time_evolution(axes[1, 0], gt_time, coords, "Ground Truth", vmin=vmin_time, vmax=vmax_time)
    _plot_single_z_time_evolution(axes[1, 1], pred_time, coords, "Prediction", vmin=vmin_time, vmax=vmax_time)
    im6 = _plot_single_z_time_evolution(axes[1, 2], diff_time, coords, "Difference (Error)", cmap='inferno')

    fig.colorbar(im4, ax=axes[1, :2], fraction=0.046, pad=0.04, label="Value")
    fig.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04, label="Abs. Error")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()



# ==============================================================================
# LATENT SPACE VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_latent_rollout_error(eval_path: Path | str):
    """
    Loads and plots the per-timestep latent space error from an evaluation rollout.
    """
    eval_path = Path(eval_path)
    if not eval_path.exists():
        logging.error(f"Latent evaluation file not found at: {eval_path}")
        return
    
    logging.info(f"Loading latent rollout error from {eval_path}...")
    with np.load(eval_path) as data:
        if 'per_step_latent_error' in data:
            per_step_latent_error = data['per_step_latent_error']
        else:
            logging.error("Could not find 'per_step_latent_error' in the file.")
            return
            
    timesteps = range(1, len(per_step_latent_error) + 1)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    
    plt.plot(timesteps, per_step_latent_error, 'o-', label='Per-Step Latent MSE', color='purple')
    
    plt.title('Latent Space Autoregressive Rollout Error', fontsize=16)
    plt.xlabel('Prediction Timestep', fontsize=12)
    plt.ylabel('Latent Space MSE', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_latent_trajectories(eval_path: Path | str, num_dims_to_plot: int = 16):
    """
    Loads and plots the predicted vs. true latent space trajectories.
    """
    eval_path = Path(eval_path)
    if not eval_path.exists():
        logging.error(f"Latent evaluation file not found at: {eval_path}")
        return

    logging.info(f"Loading latent trajectories from {eval_path}...")
    with np.load(eval_path) as data:
        true_traj = data['true_latent_trajectory']
        pred_traj = data['predicted_latent_trajectory']

    num_timesteps, latent_dim = true_traj.shape
    timesteps = range(num_timesteps)
    dims_to_plot = min(latent_dim, num_dims_to_plot)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    cols = 4
    rows = math.ceil(dims_to_plot / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True)
    axes = axes.flatten()

    for i in range(dims_to_plot):
        axes[i].plot(timesteps, true_traj[:, i], '-', color='royalblue', label='Ground Truth')
        axes[i].plot(timesteps, pred_traj[:, i], '--', color='darkorange', label='Prediction')
        axes[i].set_title(f"Latent Dimension {i}")
        axes[i].set_ylabel("Value")
        axes[i].grid(True, which="both", ls="--")

    for j in range(dims_to_plot, len(axes)):
        axes[j].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=12)
    fig.suptitle('Latent Space Trajectory Rollout', fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_koopman_eigenvector_evolution(
    eval_path: Path | str, 
    num_eigenvectors_to_plot: int = 16,
    sort_by: str = 'amplitude'
):
    """
    Plots the time evolution of the system projected onto the Koopman eigenvectors.

    Args:
        eval_path (Path | str): Path to the latent space analysis .npz file.
        num_eigenvectors_to_plot (int): The number of eigenvectors to display.
        sort_by (str): The criterion for sorting eigenvectors. 
                         'amplitude' sorts by initial projection amplitude (descending).
                         'eigenvalue_magnitude' sorts by eigenvalue magnitude (descending).
    """
    eval_path = Path(eval_path)
    if not eval_path.exists():
        logging.error(f"Latent evaluation file not found at: {eval_path}")
        return

    logging.info(f"Loading Koopman eigenvector data from {eval_path}...")
    with np.load(eval_path, allow_pickle=True) as data:
        eigenvalues = data['eigenvalues']
        true_proj = data['true_projected_trajectory']
        pred_proj = data['pred_projected_trajectory']
        initial_amplitudes = data.get('initial_mode_amplitudes')
    
    if true_proj.size == 0 or pred_proj.size == 0 or initial_amplitudes is None:
        logging.error("Required data for eigenvector evolution plot not found in file (or eigenvector matrix was singular).")
        return

    # --- Sorting Logic ---
    if sort_by == 'amplitude':
        sort_indices = np.argsort(initial_amplitudes)[::-1]
        sort_title_str = "Sorted by Importance (Initial Amplitude)"
        plot_title_suffix = "(Sorted by Amp.)"
    elif sort_by == 'eigenvalue_magnitude':
        sort_indices = np.argsort(np.abs(eigenvalues))[::-1]
        sort_title_str = "Sorted by Eigenvalue Magnitude"
        plot_title_suffix = "(Sorted by |λ|)"
    else:
        logging.error(f"Invalid sort_by value: '{sort_by}'. Choose 'amplitude' or 'eigenvalue_magnitude'.")
        return

    sorted_eigenvalues = eigenvalues[sort_indices]
    sorted_true_proj = true_proj[:, sort_indices]
    sorted_pred_proj = pred_proj[:, sort_indices]
    sorted_amplitudes = initial_amplitudes[sort_indices]

    num_timesteps, latent_dim = true_proj.shape
    timesteps = range(num_timesteps)
    eigenvectors_to_plot = min(latent_dim, num_eigenvectors_to_plot)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    cols = 4
    rows = math.ceil(eigenvectors_to_plot / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5), sharex=True)
    axes = axes.flatten()

    for i in range(eigenvectors_to_plot):
        ax = axes[i]
        ax.plot(timesteps, np.abs(sorted_true_proj[:, i]), '-', color='royalblue', label='Ground Truth')
        ax.plot(timesteps, np.abs(sorted_pred_proj[:, i]), '--', color='darkorange', label='Prediction')
        
        eig_val = sorted_eigenvalues[i]
        amp = sorted_amplitudes[i]
        title = (f"Eigenvector {i+1} {plot_title_suffix}\n"
                 f"λ = {eig_val.real:.3f} + {eig_val.imag:.3f}i | |λ| = {np.abs(eig_val):.4f}\n"
                 f"Initial Projection Amplitude: {amp:.3f}")
        ax.set_title(title)
        ax.set_ylabel("Projection Amplitude")
        ax.grid(True, which="both", ls="--")

    for j in range(eigenvectors_to_plot, len(axes)):
        axes[j].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=12)
    fig.suptitle(f'Evolution of Koopman Eigenvector Projections\n({sort_title_str})', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

