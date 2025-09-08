# -*- coding: utf-8 -*-
# scripts/compute_stats_and_split.py

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

# This script is self-contained and does not need to import from the model packages.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics and create a temporal train/val split for sequence data."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the pre-split train_val_set.npz data file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/normalization_stats.npz",
        help="Path to save the computed statistics and indices.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data from the *end* of the timeseries to use for validation.",
    )
    # The sequence and steps arguments are needed to correctly calculate valid indices
    parser.add_argument(
        "--sequence-length",
        type=int,
        required=True,
        help="Length of a sequence block (M from the paper)."
    )
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Number of prediction steps in the future (K from the paper)."
    )
    return parser.parse_args()


def main():
    """Main function to compute and save normalization statistics."""
    args = parse_args()

    # --- Setup ---
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # --- Data Loading ---
    logging.info(f"Loading dataset from {args.data_path}...")
    with np.load(args.data_path, allow_pickle=True) as data:
        timeseries = data["timeseries"]
        labels = list(data["labels"])
    
    total_timesteps = timeseries.shape[0]
    logging.info(f"Loaded timeseries with {total_timesteps} total timesteps.")

    # --- 1. Perform a Temporal Train/Validation Split ---
    val_size = int(total_timesteps * args.val_split)
    train_size = total_timesteps - val_size
    
    train_data = timeseries[:train_size]
    # val_data is not explicitly needed, we just need the time range
    
    logging.info(f"Temporal split created: {train_size} timesteps for training, {val_size} for validation.")
    
    if train_size < 1:
        raise ValueError("The training set is empty. Cannot compute statistics.")

    # --- 2. Compute Statistics on Training Set ONLY ---
    logging.info("Calculating min/max statistics per channel on the training set...")
    # Calculate stats across all axes except the channel axis
    min_vals = np.min(train_data, axis=(0, 1, 2, 3))
    max_vals = np.max(train_data, axis=(0, 1, 2, 3))

    logging.info("Min/Max calculation complete.")
    logging.info("--- Per-Channel Statistics (from Training Set) ---")
    for i, name in enumerate(labels):
        logging.info(f"Channel '{name}': Min = {min_vals[i]:.6f}, Max = {max_vals[i]:.6f}")
    logging.info("-------------------------------------------------")
    
    # --- 3. Generate Valid Starting Indices for Blocks ---
    # A block is a sequence of M sequences, each of length (steps + 1)
    # The total length of one block in time is (M + steps)
    block_length = args.sequence_length + args.steps

    # A starting index 'i' is valid if the entire block starting at 'i' fits within the time range.
    train_indices = np.arange(0, train_size - block_length + 1)
    val_indices = np.arange(train_size, total_timesteps - block_length + 1)
    
    logging.info(f"Generated {len(train_indices)} valid training start indices.")
    logging.info(f"Generated {len(val_indices)} valid validation start indices.")

    # --- 4. Save the Statistics and Indices ---
    np.savez(
        output_path,
        min_vals=min_vals,
        max_vals=max_vals,
        train_indices=train_indices,
        val_indices=val_indices,
    )
    logging.info(f"Normalization stats and indices saved to {output_path}")


if __name__ == "__main__":
    main()
