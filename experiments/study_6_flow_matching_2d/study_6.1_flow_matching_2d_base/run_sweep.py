# -*- coding: utf-8 -*-
# experiments/study_6_flow_matching_2d/study_6.1_flow_matching_2d_base/run_sweep.py
#
# --- MODIFIED ---
# This script is adapted from the "study_5.2_stable_svd" sweep script.
# It is configured to run the NEW Flow Matching model by:
# 1. Calling "train.py" for study 6.1.
# 2. Using the new model-specific hyperparameters (features, time-embed-dim).
# 3. Pointing to new output directories for "study_6.1".
# 4. Removing all tcKAE-specific logic (param grid, validation, run naming).
# 5. Configured for a single test run as requested.
# --- END MODIFICATION ---

import subprocess
import itertools
from pathlib import Path
import logging
import argparse
import shutil
from datetime import datetime

# --- Configuration ---
# This script should be run from its own directory:
# cd experiments/study_6_flow_matching_2d/study_6.1_flow_matching_2d_base
# python run_sweep.py 2>&1 | tee output/sweep_log.txt

# --- Sweep Naming ---
SWEEP_NAME = "fm_base_sweep" # <-- MODIFIED
NO_TIMESTAMPS = True # If True, uses SWEEP_NAME only. If False, appends timestamp.

# --- Script and Data Paths (MODIFIED) ---
# Assumes this script is in .../study_6.1_flow_matching_2d_base/
# and train.py is in the same directory.
TRAIN_SCRIPT = "train.py" 
DATA_PATH = "/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1492_x1151_y1_z127_c2/preprocessed/train_val_set.npz"
NORM_STATS_PATH = "/raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1492_x1151_y1_z127_c2/preprocessed/normalization_stats.npz"

# Define the mandatory persistent directory and optional scratch directory
BASE_PERSISTENT_DIR = Path("/cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/prediction/mhd_surrogate_modelling/experiments/study_6_flow_matching_2d/study_6.1_flow_matching_2d_base/output/test/")
BASE_SCRATCH_DIR = Path("/raid/skowronek/mhd_surrogate_modelling/experiments/study_6_flow_matching_2d/study_6.1_flow_matching_2d_base/output/test/")

# --- Hyperparameter Grid (MODIFIED for Flow Matching) ---
# Configured for a single test run.
param_grid = {
    'lr': [1e-4],
    'batch_size': [32],
    'validation_batch_size': [128],
    
    # --- NEW Model Architecture ---
    'features': [[64, 128, 256]], # Note: The value is a list
    'time_embed_dim': [64],
}

# --- Fixed Training Arguments (MODIFIED) ---
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
    "validation_ode_steps": 8,      # <-- NEW
    "validation_solver": "midpoint"  # <-- NEW
    # "save_best_to_scratch" has been removed.
}


def main():
    """Main function to run the hyperparameter sweep."""
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep for Flow Matching.") # <-- MODIFIED
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
    logging.info("Starting Flow Matching hyperparameter sweep...") # <-- MODIFIED
    
    # --- Create a unique, timestamped directory for this specific sweep ---
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

    # Create a list of all hyperparameter combinations
    keys, values = zip(*param_grid.items())
    run_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # --- MODIFICATION: Removed tcKAE validation block ---
    # The new model's parameters (features, time_embed_dim) do not
    # have invalid combinations, so the validation block is removed.
    
    logging.info(f"Generated {len(run_configs)} valid hyperparameter configurations to run.")

    created_dirs_for_cleanup = []

    for i, config in enumerate(run_configs):
        
        # --- Create a descriptive run name (MODIFIED) ---
        run_name_parts = []
        
        # Convert feature list [64, 128] to string "f64-128"
        feat_str = "f" + "-".join(map(str, config['features']))
        
        run_name_parts.extend([
            feat_str,
            f"t{config['time_embed_dim']}",
        ])
        
        run_name = "_".join(run_name_parts)
        # --- End of run name modification ---
        
        # --- Use sweep-specific base directories ---
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

        # --- MODIFICATION: Handle new parameter types ---
        for key, value in config.items():
            
            # --- Handle 'features' list ---
            if key == 'features':
                cmd.append(f"--{key.replace('_', '-')}")
                # Add each feature value as a separate argument
                cmd.extend([str(v) for v in value])

            # --- Handle None values (don't add them to cmd) ---
            elif value is None:
                continue # Skip None values

            # --- Handle all other key-value pairs (lr, batch_size, etc.) ---
            else:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        # --- END MODIFICATION ---
            
        # --- MODIFICATION: Handle fixed args (removed booleans) ---
        for key, value in fixed_args.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        # --- END MODIFICATION ---

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
                    logging.warning(f"Directory not found for cleanup, skipping: {dir_path}")
            except OSError as e:
                logging.error(f"Error deleting directory {dir_path}: {e}")
        logging.info("--- Cleanup Finished. ---")


if __name__ == "__main__":
    main()