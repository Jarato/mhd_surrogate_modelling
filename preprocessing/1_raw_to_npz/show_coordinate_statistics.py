import numpy as np
import re
from pathlib import Path
import argparse
from typing import Tuple

def analyze_coordinates(
    snapshot_file: Path,
    nx: int,
    ny: int,
    nz: int,
):
    """
    Reads coordinate data from a single snapshot file and prints statistics,
    including an analysis of the grid spacing.
    """
    if not snapshot_file.exists():
        print(f"ERROR: Snapshot file not found at '{snapshot_file}'")
        return

    print(f"\n--- Analyzing Coordinates in: {snapshot_file.name} ---")

    input_dtype = np.float64

    with open(snapshot_file, 'rb') as f:
        # Read exactly the number of points for each coordinate axis
        x_coords = np.fromfile(f, dtype=input_dtype, count=nx)
        y_coords = np.fromfile(f, dtype=input_dtype, count=ny)
        z_coords = np.fromfile(f, dtype=input_dtype, count=nz)

    # --- Spacing Analysis Function ---
    def print_spacing_stats(
        axis_name: str,
        coords: np.ndarray,
    ):
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


    # --- Print Statistics ---
    print("\n--- X-Axis Coordinate Statistics ---")
    print(f"  Number of points: {len(x_coords)}")
    print(f"  Min value:        {np.min(x_coords):.4f}")
    print(f"  Max value:        {np.max(x_coords):.4f}")
    print(f"  Mean value:       {np.mean(x_coords):.4f}")
    print(f"  Std. Deviation:   {np.std(x_coords):.4f}")
    print_spacing_stats("X-Axis", x_coords)

    print("\n--- Y-Axis Coordinate Statistics ---")
    print(f"  Number of points: {len(y_coords)}")
    print(f"  Min value:        {np.min(y_coords):.4f}")
    print(f"  Max value:        {np.max(y_coords):.4f}")
    print(f"  Mean value:       {np.mean(y_coords):.4f}")
    print(f"  Std. Deviation:   {np.std(y_coords):.4f}")
    print_spacing_stats("Y-Axis", y_coords)

    print("\n--- Z-Axis Coordinate Statistics ---")
    print(f"  Number of points: {len(z_coords)}")
    print(f"  Min value:        {np.min(z_coords):.4f}")
    print(f"  Max value:        {np.max(z_coords):.4f}")
    print(f"  Mean value:       {np.mean(z_coords):.4f}")
    print(f"  Std. Deviation:   {np.std(z_coords):.4f}")
    print_spacing_stats("Z-Axis", z_coords)
    
    print("\n--- Analysis Complete ---")


def main():
    """
    Main function to parse arguments and run the analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze and print statistics for the coordinate data in a single raw snapshot file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Required Arguments ---
    parser.add_argument('snapshot_file', type=Path, help="Path to the single raw binary snapshot file to analyze.")
    parser.add_argument('--nx', type=int, required=True, help="Number of grid POINTS (not spaces) on the x-axis.")
    parser.add_argument('--ny', type=int, required=True, help="Number of grid POINTS (not spaces) on the y-axis.")
    parser.add_argument('--nz', type=int, required=True, help="Number of grid POINTS (not spaces) on the z-axis.")

    args = parser.parse_args()

    # --- Use provided dimensions ---
    print(f"Using provided dimensions: Nx={args.nx}, Ny={args.ny}, Nz={args.nz}")
    analyze_coordinates(
        args.snapshot_file,
        args.nx,
        args.ny,
        args.nz,
    )


if __name__ == "__main__":
    main()
