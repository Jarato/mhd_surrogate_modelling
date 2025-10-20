import numpy as np
from pathlib import Path
import argparse
from typing import List

def check_boundary_condition(boundary_data: np.ndarray) -> str:
    """
    Analyzes a boundary slice of timeseries data to determine its BC type.

    Args:
        boundary_data: A NumPy array representing the data on one boundary
                       over time. Shape is (Time, ...Spatial..., Channels).

    Returns:
        A string describing the boundary condition type.
    """
    # Check for Type 2: Constant in time, but potentially variable in space.
    # We calculate the standard deviation along the time axis (axis=0).
    # If std is close to zero for all spatial points and channels, it means no change over time.
    temporal_std = np.std(boundary_data, axis=0)
    is_constant_in_time = np.allclose(temporal_std, 0)

    if is_constant_in_time:
        # If it's constant in time, now check for Type 1: Constant in space as well.
        # We can just check the first time step, since all time steps are the same.
        first_timestep_data = boundary_data[0]
        
        # Check if all values within that single time step are close to the first value.
        is_constant_in_space = np.allclose(first_timestep_data, first_timestep_data.flat[0])

        if is_constant_in_space:
            const_value = first_timestep_data.flat[0]
            return f"Constant in Space and Time (Value: {const_value:.4g})"
        else:
            return "Constant in Time, Variable in Space"
    else:
        return "No constant BC detected"

def analyze_boundaries(timeseries_data: np.ndarray, coord_keys: List[str]):
    """
    Analyzes the boundaries of the timeseries data for constant conditions.
    
    Args:
        timeseries_data: The full timeseries data array.
        coord_keys: A list of coordinate keys like ['X-Axis', 'Y-Axis', 'Z-Axis']
    """
    print("\n--- Boundary Condition Analysis ---")
    
    # Map user-friendly axis names to array indices.
    # The timeseries_data shape is (Time, X, Y, Z, Channels) or (Time, X, Z, Channels).
    # The spatial dimensions start at index 1 of the array.
    axis_map = {key: i + 1 for i, key in enumerate(coord_keys)}

    for axis_name, axis_index in axis_map.items():
        print(f"\n  Boundary along {axis_name}:")

        # --- Analyze the 'Min' side of the boundary (e.g., X=0) ---
        # Create a slicer tuple. It's slice(None) for all dims except the one we're slicing.
        slicer_min = [slice(None)] * timeseries_data.ndim
        slicer_min[axis_index] = 0
        min_boundary_data = timeseries_data[tuple(slicer_min)]
        min_bc_type = check_boundary_condition(min_boundary_data)
        print(f"    Min side: {min_bc_type}")

        # --- Analyze the 'Max' side of the boundary (e.g., X=-1) ---
        slicer_max = [slice(None)] * timeseries_data.ndim
        slicer_max[axis_index] = -1
        max_boundary_data = timeseries_data[tuple(slicer_max)]
        max_bc_type = check_boundary_condition(max_boundary_data)
        print(f"    Max side: {max_bc_type}")


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
            spatial_coord_keys = [] # To preserve the correct X, Y, Z order
            if 'x_coords' in data: 
                coords['X-Axis'] = data['x_coords']
                spatial_coord_keys.append('X-Axis')
            if 'y_coords' in data: 
                coords['Y-Axis'] = data['y_coords']
                spatial_coord_keys.append('Y-Axis')
            if 'z_coords' in data: 
                coords['Z-Axis'] = data['z_coords']
                spatial_coord_keys.append('Z-Axis')

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
        
        # --- NEW: Perform Boundary Condition Analysis ---
        analyze_boundaries(timeseries_data, spatial_coord_keys)

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
