# -*- coding: utf-8 -*-
# mhd_surrogate_core/src/mhd_surrogate_core/plot.py

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def plot_rollout_error(error_path: Path | str):
    """
    Loads and plots the per-timestep error from an evaluation rollout.

    Args:
        error_path (Path | str): Path to the rollout_error.npz file.
    """
    error_path = Path(error_path)
    if not error_path.exists():
        logging.error(f"Rollout error file not found at: {error_path}")
        return
    
    logging.info(f"Loading rollout error from {error_path}...")
    with np.load(error_path) as data:
        per_step_error = data['per_step_error']
        
    timesteps = range(1, len(per_step_error) + 1)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    
    plt.plot(timesteps, per_step_error, 'o-', label='Per-Step MSE', color='crimson')
    
    plt.title('Autoregressive Rollout Error Over Time', fontsize=16)
    plt.xlabel('Prediction Timestep', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

