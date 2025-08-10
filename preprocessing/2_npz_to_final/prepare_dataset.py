# -*- coding: utf-8 -*-
# scripts/prepare_dataset.py

import argparse
import logging
from pathlib import Path

import numpy as np

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split raw MHD data into a train/val set and a contiguous test set."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the raw .npz data file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save the processed dataset files.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Fraction of data to hold out for the final, contiguous test set.",
    )
    return parser.parse_args()


def main():
    """Main function to perform the data splitting."""
    args = parse_args()

    # --- Setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Data Loading ---
    logging.info(f"Loading full dataset from {args.data_path}...")
    with np.load(args.data_path, allow_pickle=True) as raw_data:
        full_timeseries = raw_data["timeseries"]
        full_labels = raw_data["labels"]
        # --- THE FIX IS HERE (Part 1: Load coordinates) ---
        x_coords = raw_data["x_coords"]
        y_coords = raw_data["y_coords"]
        z_coords = raw_data["z_coords"]
    
    num_timesteps = full_timeseries.shape[0]
    logging.info(f"Full dataset loaded with {num_timesteps} snapshots.")

    # --- Perform Chronological Split ---
    test_start_index = int(num_timesteps * (1 - args.test_split))
    
    train_val_timeseries = full_timeseries[:test_start_index]
    test_timeseries = full_timeseries[test_start_index:]

    if len(train_val_timeseries) < 2 or len(test_timeseries) < 2:
         raise ValueError(
            "The dataset is too small to create a non-empty train/val and test split. "
            "Please use a larger dataset or adjust the split ratio."
        )

    logging.info(
        f"Splitting data chronologically: "
        f"{len(train_val_timeseries)} snapshots for train/val, "
        f"{len(test_timeseries)} snapshots for test."
    )

    # --- Save the Datasets as contiguous NPZ files ---
    train_val_path = output_dir / "train_val_set.npz"
    test_path = output_dir / "test_set.npz"

    np.savez(
        train_val_path,
        timeseries=train_val_timeseries,
        labels=full_labels,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
    )
    logging.info(f"Train/Validation set saved to {train_val_path}")

    np.savez(
        test_path,
        timeseries=test_timeseries,
        labels=full_labels,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
    )
    logging.info(f"Contiguous test set saved to {test_path}")

    logging.info("Data preparation complete.")


if __name__ == "__main__":
    main()
