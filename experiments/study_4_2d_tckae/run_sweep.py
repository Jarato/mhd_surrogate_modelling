# -*- coding: utf-8 -*-
# experiments/study_3_q2d_tckae/run_sweep.py

import subprocess
import itertools
from pathlib import Path
import logging
import argparse

# --- Configuration ---
# This script should be run from its own directory:
# cd experiments/study_3_q2d_tckae
# python run_sweep.py | tee output/sweep_log.txt

# --- Script and Data Paths ---
TRAIN_SCRIPT = "train.py"
DATA_PATH = "/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1492_x1151_y1_z127_c2/preprocessed/train_val_set.npz"
NORM_STATS_PATH = "/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1492_x1151_y1_z127_c2/preprocessed/normalization_stats.npz"
# DATA_PATH = "/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/interp/prep2/train_val_set.npz"
# NORM_STATS_PATH = "/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/interp/prep2/normalization_stats.npz"

# Define the mandatory persistent directory and optional scratch directory
# These will be the top-level directories for all sweep runs.
BASE_PERSISTENT_DIR = Path("/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/prediction/mhd_surrogate_modelling/experiments/study_4_2d_tckae/output/test/")

# BASE_SCRATCH_DIR = Path("/raid/skowronek/mhd_surrogate_modelling/experiments/study_3_q2d_tckae/output/test/")
BASE_SCRATCH_DIR = None

# --- Hyperparameter Grid ---
# Define the parameter space for the grid search.
param_grid = {
    'lr': [1e-4],
    'latent_dim': [256],
    'bottleneck_dim': [4096],
    'use_bottleneck': [True],
    'batch_size': [16],
    'validation_batch_size': [4],
    'sequence_length': [8], # This is M - Reduced from 4 to 2 to lower memory usage
    'steps': [8],              # This is K for forward dynamics
    'steps_back': [0],         # K for backward dynamics
    'steps_tc': [8],
    'epoch-trans': [0],
    'gamma_identity': [1.0],
    'gamma_fwd': [1.0],
    'gamma_tc': [1.0],
    'gamma_bwd': [0],         # Used only if backward=True
    'gamma_con': [0],        # Used only if backward=True
    'backward': [False],  # Sweep between tcKAE and tcKAE+cKAE
}

# --- Fixed Training Arguments ---
# These arguments will be the same for all runs.
fixed_args = {
    "epochs": 2,
    "patience": 40,
    "lr_patience": 10,
    "clip_grad_value": 128,
    "lr_factor": 0.1,
    "num_workers": 8,
    "validation_num_workers": 0,
    "checkpoint_save_freq": 32,
    "persistent_save_freq": 1024,
    "validation_rollout_steps": 8,
    "val-split": 0.1,
}


def main():
    """Main function to run the hyperparameter sweep."""
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep for tcKAE.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, resume incomplete runs from the latest checkpoint instead of skipping them."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting tcKAE hyperparameter sweep...")
    
    # Create a list of all hyperparameter combinations
    keys, values = zip(*param_grid.items())
    run_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    logging.info(f"Generated {len(run_configs)} unique hyperparameter configurations.")

    for i, config in enumerate(run_configs):
        # Create a descriptive run name
        run_name_parts = [
            f"ld{config['latent_dim']}",
            f"M{config['sequence_length']}",
            f"K{config['steps']}",
            f"Ktc{config['steps_tc']}",
            f"gtc{config['gamma_tc']}",
        ]
        if config['backward']:
            run_name_parts.append(f"bwd_True_gb{config['gamma_bwd']}_gc{config['gamma_con']}")
        else:
            run_name_parts.append("bwd_False")
        run_name = "_".join(run_name_parts)
        
        persistent_dir = BASE_PERSISTENT_DIR / run_name
        scratch_dir = BASE_SCRATCH_DIR / run_name if BASE_SCRATCH_DIR else None
        
        should_resume = False
        if persistent_dir.exists():
            if args.resume:
                logging.info(f"--- ATTEMPTING TO RESUME Run {i+1}/{len(run_configs)}: {run_name} ---")
                should_resume = True
            else:
                logging.info(f"--- SKIPPING Run {i+1}/{len(run_configs)}: {run_name} (directory exists) ---")
                continue
        else:
            logging.info(f"--- STARTING Run {i+1}/{len(run_configs)}: {run_name} ---")

        # Construct the command line arguments
        cmd = [
            "python",
            TRAIN_SCRIPT,
            "--data-path", DATA_PATH,
            "--norm-stats-path", NORM_STATS_PATH,
            "--persistent-dir", str(persistent_dir),
        ]
        
        if scratch_dir:
            cmd.extend(["--scratch-dir", str(scratch_dir)])
            
        if should_resume:
            cmd.append("--resume")

        # Add hyperparameters from the config
        for key, value in config.items():
            if key in ['use_bottleneck', 'backward']:
                if value:
                    cmd.append(f'--{key.replace("_", "-")}')
            else:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
                
        for key, value in fixed_args.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        # Execute the training script
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"!!!!!! Run {run_name} failed with exit code {e.returncode} !!!!!!")

    logging.info("Hyperparameter sweep finished.")

if __name__ == "__main__":
    main()

