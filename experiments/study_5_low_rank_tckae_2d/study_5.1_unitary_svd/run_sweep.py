# -*- coding: utf-8 -*-
# experiments/study_5_low_rank_tckae_2d/study_5.1_unitary_svd/run_sweep.py
# --- MODIFIED to run with train.py and its new dynamic architecture flags ---
# --- MODIFIED to group all runs into a single, timestamped sweep directory ---

import subprocess
import itertools
from pathlib import Path
import logging
import argparse
import shutil
from datetime import datetime # <-- Added import

# --- Configuration ---
# This script should be run from its own directory:
# cd experiments/study_3_q2d_tckae
# python run_sweep.py | tee output/sweep_log.txt

# --- NEW: Sweep Naming ---
SWEEP_NAME = "svd_dynamic_sweep"
NO_TIMESTAMPS = True # If True, uses SWEEP_NAME only. If False, appends timestamp.

# --- Script and Data Paths ---
TRAIN_SCRIPT = "train.py"
DATA_PATH = "/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1492_x1151_y1_z127_c2/preprocessed/train_val_set.npz"
NORM_STATS_PATH = "/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1492_x1151_y1_z127_c2/preprocessed/normalization_stats.npz"
# DATA_PATH = "/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/interp/prep2/train_val_set.npz"
# NORM_STATS_PATH = "/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/interp/prep2/normalization_stats.npz"

# Define the mandatory persistent directory and optional scratch directory
# These will be the top-level directories for all sweep runs.
BASE_PERSISTENT_DIR = Path("/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/prediction/mhd_surrogate_modelling/experiments/study_5_low_rank_tckae_2d/study_5.1_unitary_svd/output/test/")

# BASE_SCRATCH_DIR = Path("/raid/skowronek/mhd_surrogate_modelling/experiments/study_5_low_rank_tckae_2d/study_5.1_unitary_svd/output/test/")
BASE_SCRATCH_DIR = None

# --- Hyperparameter Grid (MODIFIED for dynamic architecture) ---
# Define the parameter space for the grid search.
param_grid = {
    'lr': [1e-4],
    'batch_size': [64],
    'validation_batch_size': [16],
    'sequence_length': [1],
    'steps': [1],
    'steps_back': [0],
    'steps_tc': [0],
    'epoch-trans': [0],
    'gamma_identity': [1.0],
    'gamma_fwd': [1.0],
    'gamma_tc': [0],
    'gamma_bwd': [0],
    'gamma_svd_ortho': [1e-5], # Weight for orthonormality loss
    'gamma_svd_sigma': [1e-5], # Weight for singular value loss
    'backward': [False],
    
    # --- Architectural Sweep ---
    # These flags control the model's structure
    'use_flattened_for_koopman': [True],
    'use_bottleneck': [False],
    'latent_dim': [None],            # 'd' (Required if not flat)
    'bottleneck_dim': [None],       # 'b' (Required if not flat and use_bottleneck)
    'koopman_rank': [16],     # 'r' (None defaults to latent_dim in bottleneck modes)
}

# --- Fixed Training Arguments ---
# These arguments will be the same for all runs.
fixed_args = {
    "epochs": 1,
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
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep for tcKAE (SVD Version).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, resume incomplete runs from the latest checkpoint instead of skipping them."
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="If set, delete all created run directories after the entire sweep is finished."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting tcKAE (SVD) hyperparameter sweep...")
    
    # --- NEW: Create a unique, timestamped directory for this specific sweep ---
    if NO_TIMESTAMPS:
        sweep_dir_name = f"{SWEEP_NAME}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir_name = f"{SWEEP_NAME}_{timestamp}"

    SWEEP_BASE_PERSISTENT_DIR = BASE_PERSISTENT_DIR / sweep_dir_name
    SWEEP_BASE_SCRATCH_DIR = BASE_SCRATCH_DIR / sweep_dir_name if BASE_SCRATCH_DIR else None

    # Create the directories for this sweep
    SWEEP_BASE_PERSISTENT_DIR.mkdir(parents=True, exist_ok=True)
    if SWEEP_BASE_SCRATCH_DIR:
        SWEEP_BASE_SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving all runs for this sweep in: {SWEEP_BASE_PERSISTENT_DIR}")
    # --- END NEW ---

    # Create a list of all hyperparameter combinations
    keys, values = zip(*param_grid.items())
    run_configs_all = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    logging.info(f"Generated {len(run_configs_all)} total parameter combinations. Validating...")
    
    run_configs = []
    # --- Validation Block ---
    # Filter out invalid architectural combinations
    for config in run_configs_all:
        is_flat = config['use_flattened_for_koopman']
        is_bottle = config['use_bottleneck']
        ld = config['latent_dim']
        bd = config['bottleneck_dim']
        kr = config['koopman_rank']

        if is_flat:
            # STATE 1: Koopman on flattened space.
            # ld, bd must be None. kr must be set.
            if ld is None and bd is None and kr is not None:
                # To reduce redundant runs, only run this when use_bottleneck is False
                if not is_bottle:
                    run_configs.append(config)
        
        else:
            # STATE 2 or 3: Koopman on latent space.
            # ld must NOT be None.
            if ld is None:
                continue

            if is_bottle:
                # STATE 2: Bottleneck path F -> b -> d
                # bd must NOT be None.
                if bd is not None:
                    run_configs.append(config)
            else:
                # STATE 3: Direct path F -> d
                # bd must BE None.
                if bd is None:
                    run_configs.append(config)
    # --- End Validation ---

    logging.info(f"Found {len(run_configs)} valid hyperparameter configurations to run.")

    created_dirs_for_cleanup = []

    for i, config in enumerate(run_configs):
        
        # --- Create a descriptive run name (MODIFIED) ---
        run_name_parts = []
        kr_str = config['koopman_rank'] if config['koopman_rank'] is not None else 'auto'
        
        if config['use_flattened_for_koopman']:
            run_name_parts.append(f"flat_kr{kr_str}")
        else:
            run_name_parts.append(f"ld{config['latent_dim']}_kr{kr_str}")
            if config['use_bottleneck']:
                run_name_parts.append(f"bn{config['bottleneck_dim']}")
            else:
                run_name_parts.append("direct")

        run_name_parts.extend([
            f"M{config['sequence_length']}",
            f"K{config['steps']}",
            f"Ktc{config['steps_tc']}",
            f"gtc{config['gamma_tc']}",
            f"gi{config['gamma_identity']}", # <-- ADDED
            f"gfwd{config['gamma_fwd']}",   # <-- ADDED
            f"g_svdo{config['gamma_svd_ortho']}",
            f"g_svds{config['gamma_svd_sigma']}",
        ])
        
        if config['backward']:
            run_name_parts.append(f"bwd_True_gb{config['gamma_bwd']}")
        else:
            run_name_parts.append("bwd_False")
        
        run_name = "_".join(run_name_parts)
        # --- End of run name modification ---
        
        # --- MODIFIED: Use sweep-specific base directories ---
        persistent_dir = SWEEP_BASE_PERSISTENT_DIR / run_name
        scratch_dir = SWEEP_BASE_SCRATCH_DIR / run_name if SWEEP_BASE_SCRATCH_DIR else None
        
        # Store directory paths for potential cleanup later
        created_dirs_for_cleanup.append(persistent_dir)
        if scratch_dir:
            created_dirs_for_cleanup.append(scratch_dir)

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

        # --- MODIFIED: Add hyperparameters from the config ---
        for key, value in config.items():
            # Handle boolean flags
            if key in ['use_bottleneck', 'backward', 'use_flattened_for_koopman']:
                if value:
                    cmd.append(f'--{key.replace("_", "-")}')
            # Handle None values (don't add them to cmd)
            elif value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        # --- END MODIFICATION ---
                
        for key, value in fixed_args.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        # Execute the training script
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"!!!!!! Run {run_name} failed with exit code {e.returncode} !!!!!!")

    logging.info("Hyperparameter sweep finished.")

    # --- Cleanup Phase ---
    if args.cleanup:
        logging.info("--- Starting Cleanup: Deleting run directories as --cleanup flag was set. ---")
        for dir_path in created_dirs_for_cleanup:
            try:
                if dir_path.exists() and dir_path.is_dir():
                    shutil.rmtree(dir_path)
                    logging.info(f"Successfully deleted directory: {dir_path}")
                else:
                    # This case handles if a run failed before creating its directory
                    logging.warning(f"Directory not found for cleanup, skipping: {dir_path}")
            except OSError as e:
                logging.error(f"Error deleting directory {dir_path}: {e}")
        logging.info("--- Cleanup Finished. ---")


if __name__ == "__main__":
    main()

