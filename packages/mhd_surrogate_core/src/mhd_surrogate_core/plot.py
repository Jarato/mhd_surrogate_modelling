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
    Loads and plots the per-channel, per-timestep error from an evaluation rollout.

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
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Determine grid size for subplots
    cols = 4
    rows = math.ceil(num_channels / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True)
    axes = axes.flatten() # Flatten to make looping easier

    for i in range(num_channels):
        ax = axes[i]
        ax.plot(timesteps, per_step_channel_error[:, i], 'o-', color='crimson', markersize=4)
        ax.set_title(f"Channel: {channel_names[i]}")
        ax.set_ylabel("MSE")
        ax.grid(True)
        ax.set_yscale('log') # Use a log scale to better see variations

    # Hide any unused subplots
    for j in range(num_channels, len(axes)):
        axes[j].set_visible(False)

    # Add a common x-label
    fig.text(0.5, 0.02, 'Prediction Timestep', ha='center', va='center', fontsize=12)
    fig.suptitle('Per-Channel Autoregressive Rollout Error', fontsize=16, y=0.99)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
