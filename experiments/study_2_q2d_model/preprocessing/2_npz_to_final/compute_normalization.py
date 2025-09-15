# -*- coding: utf-8 -*-
# scripts/compute_normalization.py

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

# We need to make sure the packages are importable.
# This script should be run from the root of the project, e.g.:
# python scripts/compute_normalization.py --data-path ...
from mhd_q2d_kae.data import MHDDataset

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute and save normalization statistics and train/val indices."
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
        help="Fraction of data used for validation (to identify the training set).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the split to match the training script.",
    )
    return parser.parse_args()


def main():
    """Main function to compute and save normalization statistics."""
    args = parse_args()

    # --- Setup ---
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)

    # --- Data Loading and Splitting ---
    logging.info(f"Loading dataset from {args.data_path} to identify training split...")
    train_val_dataset = MHDDataset(file_path=args.data_path)

    val_size = int(len(train_val_dataset) * args.val_split)
    train_size = len(train_val_dataset) - val_size

    if train_size < 1:
        raise ValueError("The training set is empty. Cannot compute statistics.")

    train_dataset, val_dataset = random_split(
        train_val_dataset, [train_size, val_size], generator=generator
    )
    logging.info(f"Identified {len(train_dataset)} samples for the training set.")

    # --- Compute Statistics on Training Set ONLY ---
    logging.info("Calculating min/max statistics per channel on the training set...")
    
    num_channels = train_dataset[0][0].shape[-1]
    min_vals = np.full(num_channels, float('inf'))
    max_vals = np.full(num_channels, float('-inf'))

    for x_t, x_t_plus_1 in train_dataset:
        for snapshot in (x_t, x_t_plus_1):
            current_min = np.min(snapshot.numpy(), axis=(0, 1, 2))
            current_max = np.max(snapshot.numpy(), axis=(0, 1, 2))
            min_vals = np.minimum(min_vals, current_min)
            max_vals = np.maximum(max_vals, current_max)

    logging.info("Min/Max calculation complete.")
    
    # Log the computed statistics for each channel by name
    logging.info("--- Per-Channel Statistics ---")
    # The original dataset object holds the channel names
    channel_names = train_val_dataset.channel_names
    for i, name in enumerate(channel_names):
        logging.info(f"Channel '{name}': Min = {min_vals[i]:.6f}, Max = {max_vals[i]:.6f}")
    logging.info("-----------------------------")


    # --- Save the Statistics and Indices ---
    np.savez(
        output_path,
        min_vals=min_vals,
        max_vals=max_vals,
        train_indices=train_dataset.indices,
        val_indices=val_dataset.indices,
    )
    logging.info(f"Normalization stats and indices saved to {output_path}")


if __name__ == "__main__":
    main()
