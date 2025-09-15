# -*- coding: utf-8 -*-
# scripts/subsample_spatial_data.py

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
        description="Subsample spatial dimensions of a large timeseries dataset."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the large, original .npz data file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the new, subsampled .npz data file.",
    )
    parser.add_argument(
        "--factor-x",
        type=int,
        default=1,
        help="The integer factor by which to subsample the X dimension.",
    )
    parser.add_argument(
        "--factor-z",
        type=int,
        default=1,
        help="The integer factor by which to subsample the Z dimension.",
    )
    return parser.parse_args()


def main():
    """Main function to load, subsample, and save the dataset."""
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.factor_x < 1 or args.factor_z < 1:
        raise ValueError("Subsample factors must be 1 or greater.")

    logging.info(f"Loading dataset from {input_path}...")
    with np.load(input_path, allow_pickle=True) as data:
        timeseries = data["timeseries"]
        labels = data["labels"]
    
    original_shape = timeseries.shape
    logging.info(f"Original timeseries shape: {original_shape}")

    # --- Generate slices based on the specified logic ---

    # For X-axis (axis 1), use a simple stride starting from 0. No divisibility check is needed.
    x_slice = slice(None, None, args.factor_x)
    
    # For Z-axis (axis 3), enforce uniform striding by checking divisibility.
    z_size = original_shape[3]
    if (z_size - 1) % args.factor_z != 0:
        compatible_z = np.floor((z_size - 1) / args.factor_z) * args.factor_z + 1
        raise ValueError(
            f"Z-dimension size ({z_size}) is not compatible with a uniform stride of {args.factor_z}. "
            f"For uniform striding, (size - 1) must be divisible by the factor. "
            f"Please interpolate your data to a compatible size, e.g., {int(compatible_z)}."
        )
    z_slice = slice(None, None, args.factor_z)

    logging.info(f"Subsampling X-axis (1) with factor {args.factor_x} and Z-axis (3) with factor {args.factor_z}...")
    
    # Apply the uniform subsampling using simple slices
    subsampled_timeseries = timeseries[:, x_slice, :, z_slice, :]
    
    logging.info(f"Subsampled timeseries shape: {subsampled_timeseries.shape}")

    # Save the new, smaller dataset using the same keys
    np.savez(
        output_path,
        timeseries=subsampled_timeseries,
        labels=labels,
    )
    logging.info(f"Subsampled dataset saved to {output_path}")


if __name__ == "__main__":
    main()

