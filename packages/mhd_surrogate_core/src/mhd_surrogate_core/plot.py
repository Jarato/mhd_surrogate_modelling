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
    ax_total.set_title('Total Rollout Error Over Time', fontsize=16, weight='bold')
    ax_total.set_ylabel("MSE")
    ax_total.set_yscale('log')
    ax_total.grid(True, which="both", ls="--")
    ax_total.legend()

    gs_channels = gs[1:, :].subgridspec(channel_rows, cols, hspace=0.5, wspace=0.3)

    for i in range(num_channels):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs_channels[row, col])
        ax.plot(timesteps, per_step_channel_error[:, i], 'o-', color='crimson', markersize=3, alpha=0.8)
        ax.set_title(f"Channel: {channel_names[i]}")
        ax.set_ylabel("MSE")
        ax.set_xlabel("Timestep")
        ax.grid(True)
        ax.set_yscale('log')

    fig.suptitle('Per-Channel Autoregressive Rollout Error', fontsize=20, y=0.99)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def plot_latent_rollout_error(error_path: Path | str):
    """
    Loads and plots the per-timestep latent space error from an evaluation rollout.

    Args:
        error_path (Path | str): Path to the latent_rollout_error.npz file.
    """
    error_path = Path(error_path)
    if not error_path.exists():
        logging.error(f"Latent rollout error file not found at: {error_path}")
        return
    
    logging.info(f"Loading latent rollout error from {error_path}...")
    with np.load(error_path) as data:
        per_step_latent_error = data['per_step_latent_error']
        
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
