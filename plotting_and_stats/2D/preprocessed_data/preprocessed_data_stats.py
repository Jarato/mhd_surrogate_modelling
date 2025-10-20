import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple

def check_boundary_condition(boundary_data: np.ndarray, labels: List[str]) -> str:
    """
    Analyzes a boundary slice of timeseries data to determine its BC type.

    Args:
        boundary_data: A NumPy array representing the data on one boundary
                       over time. Shape is (Time, ...Spatial..., Channels).
        labels: A list of strings for the channel labels.

    Returns:
        A string describing the boundary condition type.
    """
    # If the boundary slice is empty (e.g., a 1x1 interior), return immediately.
    if boundary_data.size == 0 or boundary_data.shape[0] < 2:
        return "Not enough data for analysis"

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
        # Not constant in time. Find the max std for each component/channel.
        # temporal_std has shape (...Spatial..., Channels).
        num_spatial_dims = temporal_std.ndim - 1
        
        # Define the spatial axes to find the maximum over.
        spatial_axes = tuple(range(num_spatial_dims)) if num_spatial_dims > 0 else None

        # Calculate the max std dev, keeping the channel dimension.
        max_std_per_channel = np.max(temporal_std, axis=spatial_axes)

        # Build a descriptive string with the results for each channel.
        report_parts = []
        for i, label in enumerate(labels):
            report_parts.append(f"{label}: {max_std_per_channel[i]:.4g}")
        
        details_str = ", ".join(report_parts)
        return f"Not constant in time (Max temporal stds -> {details_str})"


def analyze_boundaries(timeseries_data: np.ndarray, coord_keys: List[str], labels: List[str]):
    """
    Analyzes the boundaries of the timeseries data for constant conditions.
    
    Args:
        timeseries_data: The full timeseries data array.
        coord_keys: A list of coordinate keys like ['X-Axis', 'Y-Axis', 'Z-Axis'].
        labels: A list of strings for the channel labels.
    """
    print("\n--- Boundary Condition Analysis ---")
    
    # Map user-friendly axis names to array indices.
    # The timeseries_data shape is (Time, X, Y, Z, Channels) or (Time, X, Z, Channels).
    # The spatial dimensions start at index 1 of the array.
    axis_map = {key: i + 1 for i, key in enumerate(coord_keys)}

    for axis_name, axis_index in axis_map.items():
        print(f"\n  Boundary along {axis_name}:")

        for side_name, side_index in [("Min", 0), ("Max", -1)]:
            # --- 1. Get the full boundary slice ---
            full_slicer = [slice(None)] * timeseries_data.ndim
            full_slicer[axis_index] = side_index
            full_boundary_data = timeseries_data[tuple(full_slicer)]
            full_bc_type = check_boundary_condition(full_boundary_data, labels)
            print(f"    {side_name} side (full face): {full_bc_type}")
            
            # --- 2. Get the interior slice of that boundary ---
            # Create a slicer for the *interior* of the full boundary slice.
            # This means slicing from 1 to -1 on all *other* spatial axes.
            interior_slicer = [slice(None)] * full_boundary_data.ndim
            for other_axis_index in range(1, len(coord_keys) + 1):
                if other_axis_index != axis_index:
                    # Map the original data axis index to the boundary slice's axis index
                    # The boundary slice has one fewer spatial dimension.
                    boundary_slice_axis_index = other_axis_index
                    if other_axis_index > axis_index:
                        boundary_slice_axis_index -=1
                    
                    interior_slicer[boundary_slice_axis_index] = slice(1, -1)
            
            interior_boundary_data = full_boundary_data[tuple(interior_slicer)]
            interior_bc_type = check_boundary_condition(interior_boundary_data, labels)
            print(f"    {side_name} side (interior):  {interior_bc_type}")


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
        analyze_boundaries(timeseries_data, spatial_coord_keys, list(labels))

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

