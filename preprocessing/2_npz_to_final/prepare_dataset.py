# -*- coding: utf-8 -*-
# scripts/prepare_dataset.py
"""
Prepares the raw MHD simulation data for model training and evaluation.

This script performs a crucial preprocessing step: splitting a single, large
time-series dataset into two distinct sets:
1.  A training/validation set: This portion of the data will be used for
    training the model and for validation during hyperparameter tuning. The
    samples within this set can be shuffled during training.
2.  A contiguous test set: This is a hold-out set, taken from the end of the
    original time-series. It is kept contiguous (unshuffled) to allow for
    realistic, long-term autoregressive rollout evaluation of the trained model.

The split is performed chronologically to prevent data leakage from the future
(test set) into the past (training set).

Example:
    python scripts/prepare_dataset.py \\
        --data-path /path/to/raw_simulation.npz \\
        --output-dir data/processed \\
        --test-split 0.15
"""

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
        description=(
            "Splits a raw time-series dataset into a train/val set and a "
            "contiguous test set for autoregressive model evaluation."
        )
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the raw .npz data file containing 'timeseries' and optional 'labels' arrays.",
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
    # We load the raw numpy arrays first to perform the chronological split
    with np.load(args.data_path, allow_pickle=True) as raw_data:
        if "timeseries" not in raw_data:
            raise KeyError("Input .npz file must contain a 'timeseries' array.")
        full_timeseries = raw_data["timeseries"]
        # Safely get labels, which are assumed to be static channel names.
        full_labels = raw_data.get("labels")

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

    # Build dictionaries of data to save, conditionally including static labels
    train_val_data = {"timeseries": train_val_timeseries}
    if full_labels is not None:
        train_val_data["labels"] = full_labels

    test_data = {"timeseries": test_timeseries}
    if full_labels is not None:
        test_data["labels"] = full_labels

    np.savez(train_val_path, **train_val_data)
    logging.info(f"Train/Validation set saved to {train_val_path}")

    np.savez(test_path, **test_data)
    logging.info(f"Contiguous test set saved to {test_path}")

    logging.info("Data preparation complete.")


if __name__ == "__main__":
    main()
