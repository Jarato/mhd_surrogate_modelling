# -*- coding: utf-8 -*-
# scripts/compute_normalization_stats.py

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute and save normalization statistics from the training split of a dataset."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the train_val_set.npz data file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/normalization_stats.npz",
        help="Path to save the computed statistics.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data from the end of the timeseries used for validation, to identify the training set.",
    )
    return parser.parse_args()


def main():
    """Main function to compute and save normalization statistics."""
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading dataset from {args.data_path} to identify training split...")
    with np.load(args.data_path, allow_pickle=True) as data:
        timeseries = data["timeseries"]
        labels = list(data["labels"])

    # Perform a temporal split to isolate the training data
    total_timesteps = timeseries.shape[0]
    val_size = int(total_timesteps * args.val_split)
    train_size = total_timesteps - val_size
    
    train_data = timeseries[:train_size]
    logging.info(f"Temporal split identified: Using first {train_size} of {total_timemsteps} timesteps for stats calculation.")

    if train_size < 1:
        raise ValueError("The training set is empty. Cannot compute statistics.")

    # Compute statistics on the training set ONLY
    logging.info("Calculating min/max statistics per channel on the training set...")
    
    # KEY CHANGE: Dynamically determine the axes for reduction. This works for both
    # 4D (T, X, Z, C) and 5D (T, X, Y, Z, C) data by selecting all axes except
    # the last one (the channel axis).
    stat_axes = tuple(range(train_data.ndim - 1))
    logging.info(f"Input data has {train_data.ndim} dimensions. Calculating stats over axes: {stat_axes}")

    min_vals = np.min(train_data, axis=stat_axes)
    max_vals = np.max(train_data, axis=stat_axes)

    logging.info("Min/Max calculation complete.")
    logging.info("--- Per-Channel Statistics (from Training Set) ---")
    for i, name in enumerate(labels):
        logging.info(f"Channel '{name}': Min = {min_vals[i]:.6f}, Max = {max_vals[i]:.6f}")
    logging.info("-------------------------------------------------")

    # Save only the computed statistics
    np.savez(
        output_path,
        min_vals=min_vals,
        max_vals=max_vals,
    )
    logging.info(f"Normalization stats saved to {output_path}")


if __name__ == "__main__":
    main()
