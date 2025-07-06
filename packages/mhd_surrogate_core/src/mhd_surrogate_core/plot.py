# -*- coding: utf-8 -*-
# mhd_surrogate_core/src/mhd_surrogate_core/plot.py

import logging
from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def plot_snapshot_comparison(true_data, recon_data, channel_name, timestep):
    """
    Creates a 3-panel plot comparing a true data slice, its reconstruction,
    and the absolute error.
    """
    slice_idx = true_data.shape[1] // 2
    true_slice = true_data[:, slice_idx, :]
    recon_slice = recon_data[:, slice_idx, :]
    error_slice = np.abs(true_slice - recon_slice)

    vmin = min(true_slice.min(), recon_slice.min())
    vmax = max(true_slice.max(), recon_slice.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Reconstruction Comparison for Channel '{channel_name}' at Timestep {timestep}", fontsize=16)

    im1 = axes[0].imshow(true_slice.T, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth"); axes[0].set_xlabel("X-axis"); axes[0].set_ylabel("Z-axis")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(recon_slice.T, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title("Reconstruction"); axes[1].set_xlabel("X-axis"); axes[1].set_yticklabels([])

    im3 = axes[2].imshow(error_slice.T, origin='lower', cmap='inferno')
    axes[2].set_title("Absolute Error"); axes[2].set_xlabel("X-axis"); axes[2].set_yticklabels([])
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95]); return fig


def plot_prediction_comparison(true_data, pred_data, channel_name, timestep):
    """
    Creates a 3-panel plot comparing a true data slice, its prediction,
    and the absolute error.
    """
    slice_idx = true_data.shape[1] // 2
    true_slice = true_data[:, slice_idx, :]
    pred_slice = pred_data[:, slice_idx, :]
    error_slice = np.abs(true_slice - pred_slice)

    vmin = min(true_slice.min(), pred_slice.min())
    vmax = max(true_slice.max(), pred_slice.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Prediction Comparison for Channel '{channel_name}' at Timestep {timestep}", fontsize=16)

    im1 = axes[0].imshow(true_slice.T, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth"); axes[0].set_xlabel("X-axis"); axes[0].set_ylabel("Z-axis")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(pred_slice.T, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction"); axes[1].set_xlabel("X-axis"); axes[1].set_yticklabels([])

    im3 = axes[2].imshow(error_slice.T, origin='lower', cmap='inferno')
    axes[2].set_title("Absolute Error"); axes[2].set_xlabel("X-axis"); axes[2].set_yticklabels([])
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95]); return fig


def plot_prediction_rollout_error(error_path: Path | str):
    """
    Loads and plots the per-channel and total error from a prediction rollout.
    """
    error_path = Path(error_path)
    if not error_path.exists():
        logging.error(f"Rollout error file not found at: {error_path}")
        return
    
    logging.info(f"Loading prediction rollout error from {error_path}...")
    with np.load(error_path, allow_pickle=True) as data:
        per_step_channel_error = data['per_step_channel_error']
        channel_names = data['channel_names']
        
    num_timesteps, num_channels = per_step_channel_error.shape
    timesteps = range(1, num_timesteps + 1)
    total_per_step_error = per_step_channel_error.mean(axis=1)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    cols = 4
    channel_rows = math.ceil(num_channels / cols)
    total_rows = 1 + channel_rows
    fig_height = 3 * total_rows
    fig = plt.figure(figsize=(16, fig_height))
    gs = fig.add_gridspec(total_rows, cols, hspace=0.6, wspace=0.3)
    
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

    fig.suptitle('Per-Channel Prediction Rollout Error', fontsize=20, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def plot_latent_rollout_error(error_path: Path | str):
    """
    Loads and plots the per-timestep latent space error from an evaluation rollout.

    Args:
        error_path (Path | str): Path to the latent_space_analysis.npz file.
    """
    error_path = Path(error_path)
    if not error_path.exists():
        logging.error(f"Latent evaluation file not found at: {error_path}")
        return
    
    logging.info(f"Loading latent rollout error from {error_path}...")
    with np.load(error_path) as data:
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


def plot_koopman_mode_evolution(eval_path: Path | str, num_modes_to_plot: int = 16):
    """
    Plots the time evolution of the system projected onto the Koopman eigenvectors.
    """
    eval_path = Path(eval_path)
    if not eval_path.exists():
        logging.error(f"Latent evaluation file not found at: {eval_path}")
        return

    logging.info(f"Loading Koopman mode data from {eval_path}...")
    with np.load(eval_path, allow_pickle=True) as data:
        eigenvalues = data['eigenvalues']
        true_proj = data['true_projected_trajectory']
        pred_proj = data['pred_projected_trajectory']
        initial_amplitudes = data.get('initial_mode_amplitudes')
    
    if true_proj.size == 0 or pred_proj.size == 0 or initial_amplitudes is None:
        logging.error("Required data for mode evolution plot not found in file.")
        return

    sort_indices = np.argsort(initial_amplitudes)[::-1]
    sorted_eigenvalues = eigenvalues[sort_indices]
    sorted_true_proj = true_proj[:, sort_indices]
    sorted_pred_proj = pred_proj[:, sort_indices]
    sorted_amplitudes = initial_amplitudes[sort_indices]

    num_timesteps, latent_dim = true_proj.shape
    timesteps = range(num_timesteps)
    modes_to_plot = min(latent_dim, num_modes_to_plot)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    cols = 4
    rows = math.ceil(modes_to_plot / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5), sharex=True)
    axes = axes.flatten()

    for i in range(modes_to_plot):
        ax = axes[i]
        ax.plot(timesteps, np.abs(sorted_true_proj[:, i]), '-', color='royalblue', label='Ground Truth')
        ax.plot(timesteps, np.abs(sorted_pred_proj[:, i]), '--', color='darkorange', label='Prediction')
        
        eig_val = sorted_eigenvalues[i]
        amp = sorted_amplitudes[i]
        title = (f"Mode {i+1} (Sorted by Amp.)\n"
                 f"λ = {eig_val.real:.3f} + {eig_val.imag:.3f}i | |λ| = {np.abs(eig_val):.4f}\n"
                 f"Initial Amplitude: {amp:.3f}")
        ax.set_title(title)
        ax.set_ylabel("Mode Amplitude")
        ax.grid(True, which="both", ls="--")

    for j in range(modes_to_plot, len(axes)):
        axes[j].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=12)
    fig.suptitle('Evolution of Koopman Modes (Sorted by Importance)', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


def plot_r2_performance(eval_path: Path | str, eval_type: str = "Reconstruction"):
    """
    Loads and plots the per-channel R-squared scores.

    Args:
        eval_path (Path | str): Path to the evaluation analysis .npz file.
        eval_type (str): The type of evaluation (e.g., "Reconstruction", "Prediction")
                         to use in the plot title.
    """
    eval_path = Path(eval_path)
    if not eval_path.exists():
        logging.error(f"Evaluation analysis file not found at: {eval_path}")
        return

    logging.info(f"Loading R² performance from {eval_path}...")
    with np.load(eval_path, allow_pickle=True) as data:
        r_squared_per_channel = data['r_squared_per_channel']
        channel_names = data['channel_names']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(channel_names))
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in r_squared_per_channel]
    
    ax.barh(y_pos, r_squared_per_channel, align='center', color=colors)
    ax.set_yticks(y_pos, labels=channel_names)
    ax.invert_yaxis()
    ax.set_xlabel('R-squared (R²) Score')
    ax.set_title(f'Per-Channel {eval_type} Performance')
    
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_reconstruction_error_over_time(error_path: Path | str):
    """
    Loads and plots the per-channel and total reconstruction error over time.
    """
    error_path = Path(error_path)
    if not error_path.exists():
        logging.error(f"Reconstruction error file not found at: {error_path}")
        return
    
    logging.info(f"Loading reconstruction error from {error_path}...")
    with np.load(error_path, allow_pickle=True) as data:
        per_step_channel_error = data['per_step_channel_recon_error']
        channel_names = data['channel_names']
        
    num_timesteps, num_channels = per_step_channel_error.shape
    timesteps = range(num_timesteps)
    total_per_step_error = per_step_channel_error.mean(axis=1)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    cols = 4
    channel_rows = math.ceil(num_channels / cols)
    total_rows = 1 + channel_rows
    
    fig_height = 3 * total_rows
    fig = plt.figure(figsize=(16, fig_height))
    
    gs = fig.add_gridspec(total_rows, cols, hspace=0.6, wspace=0.3)
    
    ax_total = fig.add_subplot(gs[0, :])
    ax_total.plot(timesteps, total_per_step_error, 'o-', color='black', label='Total Average MSE')
    ax_total.set_title('Total Reconstruction Error Over Time', fontsize=16, weight='bold')
    ax_total.set_ylabel("MSE")
    ax_total.set_yscale('log')
    ax_total.grid(True, which="both", ls="--")
    ax_total.legend()

    for i in range(num_channels):
        row, col = (i // cols) + 1, i % cols
        ax = fig.add_subplot(gs[row, col])
        ax.plot(timesteps, per_step_channel_error[:, i], 'o-', color='darkcyan', markersize=3, alpha=0.8)
        ax.set_title(f"Channel: {channel_names[i]}")
        ax.set_ylabel("MSE")
        ax.set_xlabel("Timestep")
        ax.grid(True)
        ax.set_yscale('log')

    fig.suptitle('Per-Channel Reconstruction Error', fontsize=20, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
