import numpy as np
from pathlib import Path
import argparse
from typing import Tuple

def analyze_npz_file(npz_file: Path):
    """
    Loads a preprocessed .npz file and prints detailed information about its contents.
    Handles both 4D (2D spatial) and 5D (3D spatial) data.
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
            
            # Conditionally load coordinates that exist in the file
            coords = {}
            if 'x_coords' in data: coords['X-Axis'] = data['x_coords']
            if 'y_coords' in data: coords['Y-Axis'] = data['y_coords']
            if 'z_coords' in data: coords['Z-Axis'] = data['z_coords']

        except KeyError as e:
            print(f"ERROR: The .npz file is missing an expected key: {e}")
            return

        # --- Print Timeseries Info ---
        print("\n--- Timeseries Data ---")
        # Dynamically create the shape description
        shape_desc = "(Time, X, Z, Channels)" if 'Y-Axis' not in coords else "(Time, X, Y, Z, Channels)"
        print(f"  Shape: {timeseries_data.shape} {shape_desc}")
        print(f"  Data Type: {timeseries_data.dtype}")
        print(f"  Channel Labels: {list(labels)}")

        # --- Spacing Analysis Function ---
        def print_spacing_stats(coords_array: np.ndarray):
            if len(coords_array) < 2:
                print(f"  Spacing analysis requires at least 2 points.")
                return
            
            spacing = np.diff(coords_array)
            is_uniform = np.allclose(spacing, spacing[0])

            print(f"  --- Spacing Analysis ---")
            print(f"  Uniform spacing:  {'Yes' if is_uniform else 'No'}")
            print(f"  Min spacing:      {np.min(spacing):.6f}")
            print(f"  Max spacing:      {np.max(spacing):.6f}")
            print(f"  Mean spacing:     {np.mean(spacing):.6f}")

        # --- Print Coordinate Statistics ---
        for axis_name, coord_data in coords.items():
            print(f"\n--- {axis_name} Coordinate Statistics ---")
            print(f"  Number of points: {len(coord_data)}")
            print(f"  Min value:        {np.min(coord_data):.4f}")
            print(f"  Max value:        {np.max(coord_data):.4f}")
            print_spacing_stats(coord_data)
    
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
