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


def plot_rollout_error(error_path: Path | str):
    """
    Loads and plots the per-channel and total error from an evaluation rollout.

    Args:
        error_path (Path | str): Path to the rollout_error.npz file.
    """
    error_path = Path(error_path)
    if not error_path.exists():
        logging.error(f"Rollout error file not found at: {error_path}")
        return
    
    logging.info(f"Loading rollout error from {error_path}...")
    with np.load(error_path, allow_pickle=True) as data:
        per_step_channel_error = data['per_step_channel_error']
        channel_names = data['channel_names']
        
    num_timesteps = per_step_channel_error.shape[0]
    num_channels = per_step_channel_error.shape[1]
    timesteps = range(1, num_timesteps + 1)
    
    # Calculate the total MSE at each timestep by averaging across channels
    total_per_step_error = per_step_channel_error.mean(axis=1)
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- THE FIX IS HERE ---
    # Dynamically determine the figure layout to give subplots more space
    cols = 4
    # Calculate how many rows are needed for the channel plots
    channel_rows = math.ceil(num_channels / cols)
    # Total rows = 1 for the main plot + rows for channel plots
    total_rows = 1 + channel_rows
    
    # Adjust figure height based on the number of rows
    fig_height = 3 * total_rows
    fig = plt.figure(figsize=(16, fig_height))
    
    # Create a gridspec for the entire figure layout
    gs = fig.add_gridspec(total_rows, cols, hspace=0.6, wspace=0.3)
    
    # Main plot for total error spans the top row
    ax_total = fig.add_subplot(gs[0, :])
    ax_total.plot(timesteps, total_per_step_error, 'o-', color='black', label='Total Average MSE')
    ax_total.set_title('Total Rollout Error Over Time', fontsize=16, weight='bold')
    ax_total.set_ylabel("MSE")
    ax_total.set_yscale('log')
    ax_total.grid(True, which="both", ls="--")
    ax_total.legend()

    # Create subplots for individual channels in the remaining space
    for i in range(num_channels):
        row = (i // cols) + 1 # Start from the second row of the main grid
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        ax.plot(timesteps, per_step_channel_error[:, i], 'o-', color='crimson', markersize=3, alpha=0.8)
        ax.set_title(f"Channel: {channel_names[i]}")
        ax.set_ylabel("MSE")
        ax.set_xlabel("Timestep") # Add x-label to each subplot
        ax.grid(True)
        ax.set_yscale('log')

    fig.suptitle('Per-Channel Autoregressive Rollout Error', fontsize=20, y=0.99)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
