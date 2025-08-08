# -*- coding: utf-8 -*-
# experiments/study_2_q2d_model/run_sweep.py

import subprocess
import itertools
from pathlib import Path
import logging

# --- Configuration ---
# This script should be run from the experiment directory, e.g.:
# cd experiments/study_2_q2d_model
# python run_sweep.py
# For logging use:
# python run_sweep.py | tee output/sweep_log.txt

# --- Script and Data Paths ---
TRAIN_SCRIPT = "train.py"
DATA_PATH = "/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/interp/prep2/train_val_set.npz"
NORM_STATS_PATH = "/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/interp/prep2/normalization_stats.npz"
BASE_OUTPUT_DIR = Path("output")

# --- Hyperparameter Grid ---
# Define the parameter space for the grid search.
param_grid = {
    'lr': [1e-4],
    # Coupled parameters are defined as a list of tuples.
    # Each tuple is a complete set of (latent_dim, bottleneck_dim, batch_size).
    'model_params': [
        (1024, 4096, 226),
        (2048, 8192, 226),
        # (4096, 16384, 226),
    ],
    'w_recon': [1.0],
    'w_pred': [1.0],
    'w_lin': [1.0, 10.0],
    'w_eig': [0.1],
}

# --- Fixed Training Arguments ---
# These arguments will be the same for all runs.
fixed_args = {
    "epochs": 2,
    "patience": 40,
    "lr_patience": 10,
    "clip_grad_value": 0.2,
    "lr_factor": 0.1,
}


def main():
    """Main function to run the hyperparameter sweep."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting hyperparameter sweep...")

    # Create a list of all hyperparameter combinations
    keys, values = zip(*param_grid.items())
    
    run_configs = []
    for v in itertools.product(*values):
        config = dict(zip(keys, v))
        # Unpack the coupled parameters
        config['latent_dim'], config['bottleneck_dim'], config['batch_size'] = config.pop('model_params')
        run_configs.append(config)

    logging.info(f"Generated {len(run_configs)} unique hyperparameter configurations.")

    for i, config in enumerate(run_configs):
        # Create a run name that includes all key swept parameters
        run_name_parts = [
            f"ld{config['latent_dim']}",
            f"bd{config['bottleneck_dim']}",
            f"wr{config['w_recon']}",
            f"wp{config['w_pred']}",
            f"wl{config['w_lin']}",
            f"we{config['w_eig']}",
        ]
        run_name = "_".join(run_name_parts)
        
        output_dir = BASE_OUTPUT_DIR / run_name

        if output_dir.exists():
            logging.info(f"--- SKIPPING Run {i+1}/{len(run_configs)}: {run_name} (directory exists) ---")
            continue

        logging.info(f"--- STARTING Run {i+1}/{len(run_configs)}: {run_name} ---")
        
        # Construct the command line arguments
        cmd = [
            "python",
            TRAIN_SCRIPT,
            "--data-path", DATA_PATH,
            "--norm-stats-path", NORM_STATS_PATH,
            "--output-dir", str(output_dir),
        ]

        # Add hyperparameters from the config
        for key, value in config.items():
            # --- THE FIX IS HERE ---
            # Convert python_style variable names to command-line-style arguments
            cmd.append(f"--{key.replace('_', '-')}")
            cmd.append(str(value))
            
        # Add fixed arguments
        for key, value in fixed_args.items():
            cmd.append(f"--{key.replace('_', '-')}")
            cmd.append(str(value))

        # Execute the training script
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"!!!!!! Run {run_name} failed with exit code {e.returncode} !!!!!!")

    logging.info("Hyperparameter sweep finished.")


if __name__ == "__main__":
    main()
