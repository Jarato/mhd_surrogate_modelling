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
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D

def plot_prediction_rollout_error(stats_path: Path | str, show_title: bool = True, font_size: int = 12):
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
    if show_title:
        ax_total.set_title('Total Prediction Rollout Error Over Time', fontsize=font_size + 4, weight='bold')
    ax_total.set_ylabel("MSE", fontsize=font_size)
    ax_total.set_yscale('log')
    ax_total.tick_params(axis='both', which='major', labelsize=font_size-2)
    ax_total.grid(True, which="both", ls="--")
    ax_total.legend(fontsize=font_size)

    for i in range(num_channels):
        row, col = (i // cols) + 1, i % cols
        ax = fig.add_subplot(gs[row, col])
        ax.plot(timesteps, per_step_channel_error[:, i], 'o-', color='crimson', markersize=3, alpha=0.8)
        if show_title:
            ax.set_title(f"Channel: {channel_names[i]}", fontsize=font_size)
        ax.set_ylabel("MSE", fontsize=font_size)
        ax.set_xlabel("Timestep", fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size-2)
        ax.grid(True)
        ax.set_yscale('log')

    if show_title:
        fig.suptitle('Per-Channel Prediction Rollout Error', fontsize=font_size + 8, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_r2_performance(stats_path: Path | str, eval_type: str = "Prediction", show_title: bool = True, font_size: int = 12):
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
    ax.set_xlabel('R-squared (R²) Score', fontsize=font_size)
    if show_title:
        ax.set_title(f'Per-Channel {eval_type} Performance', fontsize=font_size + 4)
    
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    for i, v in enumerate(r_squared_per_channel):
        ax.text(v + 0.01, i, f'{v:.3f}', color='black', va='center', fontsize=font_size - 2)
    
    ax.tick_params(axis='both', which='major', labelsize=font_size-2)
    plt.tight_layout()
    plt.show()

def _plot_single_xz_slice(ax, data, coords, title, cmap='viridis', vmin=None, vmax=None, show_title=True, font_size=12):
    """Helper function to plot a single x-z slice."""
    im = ax.pcolormesh(
        coords['x'], coords['z'], data.T,
        shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax,
    )
    if show_title:
        ax.set_title(title, fontsize=font_size)
    ax.set_xlabel("X Coordinate", fontsize=font_size)
    ax.set_ylabel("Z Coordinate", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size-2)
    return im

def _plot_single_z_time_evolution(ax, data, coords, title, cmap='viridis', vmin=None, vmax=None, show_title=True, font_size=12):
    """Helper function to plot a single z-time evolution."""
    im = ax.pcolormesh(
        range(data.shape[0]), coords['z'], data.T,
        shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax,
    )
    if show_title:
        ax.set_title(title, fontsize=font_size)
    ax.set_xlabel("Time Index", fontsize=font_size)
    ax.set_ylabel("Z Coordinate", fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size-2)
    return im

def plot_prediction_dashboard(
    ground_truth_timeseries: np.ndarray,
    predicted_timeseries: np.ndarray,
    difference_timeseries: np.ndarray,
    coords: dict,
    channel: str,
    x_index: int,
    time_index: int,
    show_title: bool = True,
    font_size: int = 12,
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
    
    if show_title:
        fig.suptitle(
            f"Comprehensive Analysis for Channel '{channel}'\n"
            f"Time Index: {time_index}, X Index: {x_index}",
            fontsize=font_size + 4, y=0.98
        )

    # --- Row 1: Spatial Slices (X-Z) ---
    vmin_val = min(gt_slice.min(), pred_slice.min())
    vmax_val = max(gt_slice.max(), pred_slice.max())
    
    im1 = _plot_single_xz_slice(axes[0, 0], gt_slice, coords, "Ground Truth", vmin=vmin_val, vmax=vmax_val, show_title=show_title, font_size=font_size)
    _plot_single_xz_slice(axes[0, 1], pred_slice, coords, "Prediction", vmin=vmin_val, vmax=vmax_val, show_title=show_title, font_size=font_size)
    im3 = _plot_single_xz_slice(axes[0, 2], diff_slice, coords, "Difference (Error)", cmap='inferno', show_title=show_title, font_size=font_size)
    
    cbar1 = fig.colorbar(im1, ax=axes[0, :2], fraction=0.046, pad=0.04)
    cbar1.set_label("Value", size=font_size)
    cbar1.ax.tick_params(labelsize=font_size-2)
    cbar3 = fig.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar3.set_label("Abs. Error", size=font_size)
    cbar3.ax.tick_params(labelsize=font_size-2)

    # --- Row 2: Time Evolution (Time-Z) ---
    vmin_time = min(gt_time.min(), pred_time.min())
    vmax_time = max(gt_time.max(), pred_time.max())

    im4 = _plot_single_z_time_evolution(axes[1, 0], gt_time, coords, "Ground Truth", vmin=vmin_time, vmax=vmax_time, show_title=show_title, font_size=font_size)
    _plot_single_z_time_evolution(axes[1, 1], pred_time, coords, "Prediction", vmin=vmin_time, vmax=vmax_time, show_title=show_title, font_size=font_size)
    im6 = _plot_single_z_time_evolution(axes[1, 2], diff_time, coords, "Difference (Error)", cmap='inferno', show_title=show_title, font_size=font_size)

    cbar4 = fig.colorbar(im4, ax=axes[1, :2], fraction=0.046, pad=0.04)
    cbar4.set_label("Value", size=font_size)
    cbar4.ax.tick_params(labelsize=font_size-2)
    cbar6 = fig.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar6.set_label("Abs. Error", size=font_size)
    cbar6.ax.tick_params(labelsize=font_size-2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


# ==============================================================================
# LATENT SPACE VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_latent_rollout_error(eval_path: Path | str, show_title: bool = True, font_size: int = 12):
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
    
    if show_title:
        plt.title('Latent Space Autoregressive Rollout Error', fontsize=font_size + 4)
    plt.xlabel('Prediction Timestep', fontsize=font_size)
    plt.ylabel('Latent Space MSE', fontsize=font_size)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=font_size-2)
    plt.legend(fontsize=font_size)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()


def plot_latent_trajectories(eval_path: Path | str, num_dims_to_plot: int = 16, show_title: bool = True, font_size: int = 12):
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
        if show_title:
            axes[i].set_title(f"Latent Dimension {i}", fontsize=font_size)
        axes[i].set_ylabel("Value", fontsize=font_size)
        axes[i].tick_params(axis='both', which='major', labelsize=font_size-2)
        axes[i].grid(True, which="both", ls="--")

    # Add x-label to the bottom row of plots
    for i in range(cols * (rows - 1), cols * rows):
        if i < dims_to_plot:
            axes[i].set_xlabel("Timestep", fontsize=font_size)

    for j in range(dims_to_plot, len(axes)):
        axes[j].set_visible(False)
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=font_size)
    if show_title:
        fig.suptitle('Latent Space Trajectory Rollout', fontsize=font_size + 4, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_koopman_eigenvector_evolution(
    eval_path: Path | str,
    num_eigenvectors_to_plot: int = 16,
    sort_by: str = 'amplitude',
    show_title: bool = True,
    font_size: int = 12,
):
    """
    Plots the time evolution of the system projected onto the Koopman eigenvectors.
    """
    eval_path = Path(eval_path)
    if not eval_path.exists():
        logging.error(f"Latent evaluation file not found at: {eval_path}")
        return

    logging.info(f"Loading Koopman eigenvector data from {eval_path}...")
    with np.load(eval_path, allow_pickle=True) as data:
        eigenvalues = data['eigenvalues']
        true_proj = data.get('true_projected_trajectory')
        pred_proj = data.get('pred_projected_trajectory')
        initial_amplitudes = data.get('initial_mode_amplitudes')
        koopman_mode_magnitudes = data.get('koopman_mode_magnitudes')

    if true_proj is None or pred_proj is None or initial_amplitudes is None:
        logging.error("Required projected trajectory data not found in file. Was the eigenvector matrix singular?")
        return

    # --- Sorting Logic ---
    if sort_by == 'amplitude':
        sort_indices = np.argsort(initial_amplitudes)[::-1]
        plot_title_str = "Eigenvector {i} (Sorted by Amplitude)"
    elif sort_by == 'eigenvalue_magnitude':
        sort_indices = np.argsort(np.abs(eigenvalues))[::-1]
        plot_title_str = "Eigenvector {i} (Sorted by |λ|)"
    elif sort_by == 'koopman_mode_magnitude':
        if koopman_mode_magnitudes is None:
            logging.error("'koopman_mode_magnitudes' not found in file. Cannot sort by this criterion.")
            return
        sort_indices = np.argsort(koopman_mode_magnitudes)[::-1]
        plot_title_str = "Eigenvector {i} (Sorted by Mode Magnitude)"
    else:
        logging.error(f"Unknown sort_by criterion: '{sort_by}'")
        return

    sorted_eigenvalues = eigenvalues[sort_indices]
    sorted_true_proj = true_proj[:, sort_indices]
    sorted_pred_proj = pred_proj[:, sort_indices]
    sorted_amplitudes = initial_amplitudes[sort_indices]

    num_timesteps, latent_dim = true_proj.shape
    timesteps = range(num_timesteps)
    modes_to_plot = min(latent_dim, num_eigenvectors_to_plot)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    cols = 4
    rows = math.ceil(modes_to_plot / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5), sharex=True)
    axes = axes.flatten()

    for i in range(modes_to_plot):
        ax = axes[i]
        ax.plot(timesteps, np.abs(sorted_true_proj[:, i]), '-', color='royalblue', label='Ground Truth')
        ax.plot(timesteps, np.abs(sorted_pred_proj[:, i]), '--', color='darkorange', label='Prediction')
        
        original_index = sort_indices[i]
        eig_val = sorted_eigenvalues[i]
        amp = sorted_amplitudes[i]
        
        if show_title:
            title = (f"{plot_title_str.format(i=original_index)}\n"
                     f"λ = {eig_val.real:.3f} + {eig_val.imag:.3f}i | |λ| = {np.abs(eig_val):.4f}\n"
                     f"Initial Projection Amplitude: {amp:.3f}")
            ax.set_title(title, fontsize=font_size)
        ax.set_ylabel("Projection Amplitude", fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size-2)
        ax.grid(True, which="both", ls="--")

    for i in range(cols * (rows - 1), cols * rows):
        if i < modes_to_plot:
            axes[i].set_xlabel("Timestep", fontsize=font_size)

    for j in range(modes_to_plot, len(axes)):
        axes[j].set_visible(False)
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=font_size)
    if show_title:
        fig.suptitle('Evolution of Koopman Eigenvector Projections', fontsize=font_size + 4, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

def plot_koopman_eigenvalues(
    eval_path: Path | str,
    highlight_threshold: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_title: bool = True,
    font_size: int = 12,
    show_threshold_legend: bool = True,
    show_unit_circle_legend: bool = False,
):
    """
    Plots the eigenvalues of the Koopman operator in the complex plane.
    """
    eval_path = Path(eval_path)
    if not eval_path.exists():
        logging.error(f"Latent evaluation file not found at: {eval_path}")
        return

    logging.info(f"Loading Koopman eigenvalue data from {eval_path}...")
    with np.load(eval_path, allow_pickle=True) as data:
        eigenvalues = data['eigenvalues']
        koopman_mode_magnitudes = data.get('koopman_mode_magnitudes')

    if koopman_mode_magnitudes is None:
        logging.error("'koopman_mode_magnitudes' not found in the file. Cannot color eigenvalues.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the unit circle for reference
    unit_circle = Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1.5, zorder=5)
    ax.add_artist(unit_circle)

    # Create the scatter plot, colored by mode magnitude
    scatter = ax.scatter(
        eigenvalues.real,
        eigenvalues.imag,
        c=koopman_mode_magnitudes,
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        zorder=10
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Koopman Mode Magnitude (Energy Norm)", size=font_size)
    cbar.ax.tick_params(labelsize=font_size-2)

    legend_handles = []

    # Highlight eigenvalues corresponding to high-energy modes
    if highlight_threshold is not None:
        highlight_indices = np.where(koopman_mode_magnitudes > highlight_threshold)[0]
        highlight_scatter = ax.scatter(
            eigenvalues[highlight_indices].real,
            eigenvalues[highlight_indices].imag,
            facecolors='none',
            edgecolors='r',
            s=80, # Make circles larger to be visible
            linewidths=1.5,
            label=f'Energy > {highlight_threshold}'
        )
        if show_threshold_legend:
            legend_handles.append(highlight_scatter)

    # Add unit circle legend if requested
    if show_unit_circle_legend:
        # Use a Patch for a rectangular legend handle instead of a line
        unit_circle_handle = Patch(facecolor='none', edgecolor='black', linestyle='--', linewidth=1.5, label='Unit Circle')
        legend_handles.append(unit_circle_handle)

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=font_size)

    if show_title:
        ax.set_title('Koopman Eigenvalue Spectrum', fontsize=font_size + 4)
    ax.set_xlabel('Real Part (Re)', fontsize=font_size)
    ax.set_ylabel('Imaginary Part (Im)', fontsize=font_size)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=font_size-2)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

def plot_koopman_mode_spectrum(
    eval_path: Path | str,
    dt: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    show_title: bool = True,
    font_size: int = 12
):
    """
    Plots the energy (magnitude) of Koopman modes against their frequency.
    This plot is inspired by Figure 2(b) of Rowley et al. (2009), J. Fluid Mech.
    """
    eval_path = Path(eval_path)
    if not eval_path.exists():
        logging.error(f"Latent evaluation file not found at: {eval_path}")
        return

    logging.info(f"Loading Koopman mode spectrum data from {eval_path}...")
    with np.load(eval_path, allow_pickle=True) as data:
        eigenvalues = data['eigenvalues']
        koopman_mode_magnitudes = data.get('koopman_mode_magnitudes')

    if koopman_mode_magnitudes is None:
        logging.error("'koopman_mode_magnitudes' not found in the file.")
        return

    # Calculate cyclical frequencies (f) from eigenvalues (lambda)
    # omega = angle(lambda) / dt  => f = angle(lambda) / (2 * pi * dt)
    frequencies = np.angle(eigenvalues) / (2 * np.pi * dt)

    # We are only interested in positive frequencies. The mean flow mode (freq=0) is usually excluded.
    positive_freq_indices = np.where(frequencies > 1e-6)[0]

    freqs_to_plot = frequencies[positive_freq_indices]
    mags_to_plot = koopman_mode_magnitudes[positive_freq_indices]
    
    # Sort by frequency for a clean plot
    sort_indices = np.argsort(freqs_to_plot)
    freqs_to_plot = freqs_to_plot[sort_indices]
    mags_to_plot = mags_to_plot[sort_indices]
    
    # Normalize magnitudes for coloring, using a shared vmin/vmax if provided
    norm_vmin = vmin if vmin is not None else mags_to_plot.min() if len(mags_to_plot) > 0 else 0
    norm_vmax = vmax if vmax is not None else mags_to_plot.max() if len(mags_to_plot) > 0 else 1
    norm = plt.Normalize(norm_vmin, norm_vmax)
    colors = plt.cm.viridis(norm(mags_to_plot))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a stem plot. We color each stem individually.
    for freq, mag, color in zip(freqs_to_plot, mags_to_plot, colors):
        ax.plot([freq, freq], [0, mag], color=color, linewidth=1.5)
        ax.plot(freq, mag, 'o', color=color, markersize=5)
    
    ax.axhline(0, color='black', linewidth=0.8) # Baseline

    if show_title:
        ax.set_title('Koopman Mode Energy Spectrum', fontsize=font_size + 4)
    ax.set_xlabel('Frequency', fontsize=font_size)
    ax.set_ylabel('Koopman Mode Magnitude (Energy)', fontsize=font_size)
    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=font_size-2)
    
    # Set y-limit to give some space at the top
    if len(mags_to_plot) > 0:
        ax.set_ylim(0, mags_to_plot.max() * 1.1)
    
    plt.tight_layout()
    plt.show()

