import numpy as np
from pathlib import Path
import argparse
from typing import Tuple

def analyze_npz_file(npz_file: Path):
    """
    Loads a preprocessed .npz file and prints detailed information about its contents.
    """
    if not npz_file.exists():
        print(f"ERROR: Input file not found at '{npz_file}'")
        return

    print(f"\n--- Analyzing Preprocessed File: {npz_file.name} ---")

    with np.load(npz_file) as data:
        print(f"\n--- Archive Contents ---")
        print(f"  Keys found in file: {list(data.keys())}")

        # --- Extract Data ---
        try:
            timeseries_data = data['timeseries']
            labels = data['labels']
            x_coords = data['x_coords']
            y_coords = data['y_coords']
            z_coords = data['z_coords']
        except KeyError as e:
            print(f"ERROR: The .npz file is missing an expected key: {e}")
            return

        # --- Print Timeseries Info ---
        print("\n--- Timeseries Data ---")
        print(f"  Shape: {timeseries_data.shape} (Time, X, Y, Z, Channels)")
        print(f"  Data Type: {timeseries_data.dtype}")
        print(f"  Channel Labels: {list(labels)}")

        # --- Spacing Analysis Function ---
        def print_spacing_stats(axis_name: str, coords: np.ndarray):
            if len(coords) < 2:
                print(f"  Spacing analysis requires at least 2 points.")
                return
            
            spacing = np.diff(coords)
            is_uniform = np.allclose(spacing, spacing[0])

            print(f"  --- Spacing Analysis ---")
            print(f"  Uniform spacing:  {'Yes' if is_uniform else 'No'}")
            print(f"  Min spacing:      {np.min(spacing):.6f}")
            print(f"  Max spacing:      {np.max(spacing):.6f}")
            print(f"  Mean spacing:     {np.mean(spacing):.6f}")

        # --- Print Coordinate Statistics ---
        print("\n--- X-Axis Coordinate Statistics ---")
        print(f"  Number of points: {len(x_coords)}")
        print(f"  Min value:        {np.min(x_coords):.4f}")
        print(f"  Max value:        {np.max(x_coords):.4f}")
        print_spacing_stats("X-Axis", x_coords)

        print("\n--- Y-Axis Coordinate Statistics ---")
        print(f"  Number of points: {len(y_coords)}")
        print(f"  Min value:        {np.min(y_coords):.4f}")
        print(f"  Max value:        {np.max(y_coords):.4f}")
        print_spacing_stats("Y-Axis", y_coords)

        print("\n--- Z-Axis Coordinate Statistics ---")
        print(f"  Number of points: {len(z_coords)}")
        print(f"  Min value:        {np.min(z_coords):.4f}")
        print(f"  Max value:        {np.max(z_coords):.4f}")
        print_spacing_stats("Z-Axis", z_coords)
    
    print("\n--- Analysis Complete ---")


def main():
    """
    Main function to parse arguments and run the analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze and print information about a preprocessed .npz data file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Required Arguments ---
    parser.add_argument(
        '--input-file', 
        type=Path, 
        required=True, 
        help="Path to the preprocessed .npz file to analyze."
    )

    args = parser.parse_args()
    analyze_npz_file(args.input_file)


if __name__ == "__main__":
    main()
