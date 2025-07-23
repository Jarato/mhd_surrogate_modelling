import numpy as np
import re
from pathlib import Path
import argparse

# --- Helper Functions ---

def get_parameters_from_run_file(meta_filepath: Path):
    """
    Reads the runParameters.txt file to extract grid dimensions.

    Returns:
        - nx (int): Number of points on the x-axis.
        - ny (int): Number of points on the y-axis.
        - nz (int): Number of points on the z-axis.
    """
    print(f"--- Reading parameters from '{meta_filepath}' ---")
    params = {}
    try:
        with open(meta_filepath, 'r') as f:
            for line in f:
                match_dim = re.match(r'^\s*(nX|nY|nZ)\s*:\s*(\d+)', line)
                if match_dim:
                    key, value = match_dim.groups()
                    params[key] = int(value) + 1
    except FileNotFoundError:
        print(f"Error: Metadata file not found at '{meta_filepath}'")
        raise

    if 'nX' not in params or 'nY' not in params or 'nZ' not in params:
        raise ValueError("Could not find nX, nY, or nZ in the metadata file.")

    print(f"Dimensions found (points): Nx={params['nX']}, Ny={params['nY']}, Nz={params['nZ']}")
    return params['nX'], params['nY'], params['nZ']


def generate_mock_data(
    dir_path: Path,
    meta_filepath: Path,
    prefix: str,
    time_indices: range,
    nx_points: int,
    ny_points: int,
    nz_points: int,
    source_labels: list,
):
    """
    Generates a mock dataset that mimics the Fortran binary output.
    It creates time-invariant data on the Y and Z boundaries to test verification.
    """
    # --- SAFETY CHECK ---
    if dir_path.exists() and any(dir_path.iterdir()):
        error_message = (
            f"\n--- SAFETY ABORT ---\n"
            f"Error: Mock data generation target directory '{dir_path}' is not empty.\n"
            f"To prevent overwriting real data, please clear this directory or set --generate-mock-data to False in the script.\n"
        )
        print(error_message)
        raise SystemExit() # Stop the script entirely

    print(f"--- Generating Mock Data in '{dir_path}' with channels: {source_labels} ---")
    dir_path.mkdir(parents=True, exist_ok=True)

    with open(meta_filepath, 'w') as f:
        f.write(f"nX: {nx_points - 1}\n")
        f.write(f"nY: {ny_points - 1}\n")
        f.write(f"nZ: {nz_points - 1}\n")

    for t in time_indices:
        filename = dir_path / f"{prefix}{t:06d}"
        print(f"Creating mock file: {filename}")
        with open(filename, 'wb') as f:
            x_coords = np.linspace(0, 1, nx_points, dtype=np.float64)
            y_coords = np.linspace(0, 1, ny_points, dtype=np.float64)
            z_coords = np.linspace(0, 1, nz_points, dtype=np.float64)
            x_coords.tofile(f)
            y_coords.tofile(f)
            z_coords.tofile(f)
            
            for z in range(nz_points):
                for i, channel in enumerate(source_labels):
                    channel_val = float(i + 1)
                    time_dependent_val = channel_val + t * 0.1 + z * 0.01
                    data_slice = np.full((nx_points, ny_points), time_dependent_val, dtype=np.float64)
                    time_independent_val = channel_val + z * 0.01
                    if z == 0 or z == nz_points - 1:
                        data_slice.fill(time_independent_val)
                    
                    data_slice[:, 0] = time_independent_val
                    data_slice[:, -1] = time_independent_val
                    
                    f.write(data_slice.tobytes(order='F'))

    print("--- Mock data generation complete. ---\n")


def process_and_subsample(
    input_dir: Path,
    output_file: Path,
    prefix: str,
    time_indices: range,
    nx: int,
    ny: int,
    nz: int,
    num_x_samples: int,
    y_indices: list,
    channel_indices_to_keep: list,
    num_input_channels: int,
    num_output_channels: int,
    final_channel_labels: list,
):
    """
    The core function to read, subsample, and save the data using a
    memory-mapped file to avoid RAM bottlenecks and converting to float32.
    """
    print("--- Starting Subsampling Process (Memory-Safe, float32 output) ---")
    
    x_subsample_indices = np.linspace(0, nx - 1, num_x_samples, dtype=int)
    
    input_dtype = np.float64
    output_dtype = np.float32
    
    itemsize = np.dtype(input_dtype).itemsize
    coord_offset_count = nx + ny + nz

    print("--- Reading and subsampling coordinates ---")
    first_file_path = input_dir / f"{prefix}{time_indices[0]:06d}"
    with open(first_file_path, 'rb') as f:
        x_coords_full = np.fromfile(f, dtype=input_dtype, count=nx)
        y_coords_full = np.fromfile(f, dtype=input_dtype, count=ny)
        z_coords_full = np.fromfile(f, dtype=input_dtype, count=nz)

    x_coords_sub = x_coords_full[x_subsample_indices].astype(output_dtype)
    y_coords_sub = y_coords_full[y_indices].astype(output_dtype)
    z_coords_sub = z_coords_full.astype(output_dtype)

    print(f"Subsampled coordinate shapes: x={x_coords_sub.shape}, y={y_coords_sub.shape}, z={z_coords_sub.shape}")
    
    final_shape = (len(time_indices), num_x_samples, len(y_indices), nz, num_output_channels)
    
    temp_filename = output_file.with_suffix(output_file.suffix + ".tmp")
    
    print(f"Creating memory-mapped file at '{temp_filename}' with shape {final_shape} and dtype {output_dtype}")
    memmap_array = np.memmap(temp_filename, dtype=output_dtype, mode='w+', shape=final_shape)
    
    for i, t in enumerate(time_indices):
        filename = input_dir / f"{prefix}{t:06d}"
        print(f"Processing file: {filename} -> Writing to timestep index {i}")

        with open(filename, 'rb') as f:
            offset_bytes = coord_offset_count * itemsize
            channel_data_1d = np.fromfile(f, dtype=input_dtype, offset=offset_bytes)

            try:
                data_4d_physical = channel_data_1d.reshape((nz, num_input_channels, ny, nx))
                data_4d_logical = data_4d_physical.transpose(1, 0, 2, 3)
            except ValueError as e:
                print(f"Error reshaping data for {filename}. Check dimensions and source channel list.")
                expected_elements = num_input_channels * nz * ny * nx
                print(f"Expected {expected_elements} elements, but read {len(channel_data_1d)}.")
                del memmap_array
                temp_filename.unlink(missing_ok=True)
                raise e
            
            selected_channels_data = data_4d_logical[channel_indices_to_keep, :, :, :]
            subsampled_timestep = selected_channels_data[:, :, y_indices, :][:, :, :, x_subsample_indices]
            transposed_timestep = subsampled_timestep.transpose(3, 2, 1, 0)
            
            memmap_array[i] = transposed_timestep.astype(output_dtype)
            
    # --- FIX ---
    # Instead of deleting and reloading the memmap file, we pass the
    # memmap object directly to np.savez_compressed. This avoids the
    # allow_pickle error and is more efficient.
    
    print(f"\nPackaging final data into {output_file}...")
    np.savez_compressed(
        output_file,
        timeseries=memmap_array, # Pass the memmap object directly
        labels=np.array(final_channel_labels),
        x_coords=x_coords_sub,
        y_coords=y_coords_sub,
        z_coords=z_coords_sub,
    )
    
    # Now that the final .npz file is saved, we can clean up the temporary file.
    del memmap_array
    temp_filename.unlink()
    
    print("--- Subsampling Process Complete ---")
    # We need to load the final file to get its shape for the printout
    with np.load(output_file) as data:
        print(f"Final array shape: {data['timeseries'].shape}")
        print(f"Final array dtype: {data['timeseries'].dtype}")


def verify_output(output_file: Path):
    """
    Loads the final .npz file and prints its contents to confirm success.
    Includes a check for time-invariant boundary conditions.
    """
    print(f"\n--- Verifying Output File: {output_file} ---")
    if not output_file.exists():
        print("Error: Output file not found!")
        return

    with np.load(output_file) as data:
        print(f"Successfully loaded archive.")
        print(f"Keys found in file: {list(data.keys())}")

        timeseries_data = data['timeseries']
        labels = data['labels']
        x_coords = data['x_coords']
        y_coords = data['y_coords']
        z_coords = data['z_coords']

        print(f"\nShape of 'timeseries' array: {timeseries_data.shape}")
        print(f"Data type of 'timeseries' array: {timeseries_data.dtype}")
        print(f"Content of 'labels' array: {labels}")
        print(f"Shape of 'x_coords' array: {x_coords.shape}")
        print(f"Shape of 'y_coords' array: {y_coords.shape}")
        print(f"Shape of 'z_coords' array: {z_coords.shape}")

        print("\nSample data (t=0, x=0, y=0, z=0):")
        print(timeseries_data[0, 0, 0, 0, :])
        
        print("\n--- Verifying Boundary Conditions ---")
        all_boundaries_ok = True
        num_timesteps = timeseries_data.shape[0]

        if num_timesteps > 1:
            for z_boundary_idx in [0, -1]:
                first_step_slice = timeseries_data[0, :, :, z_boundary_idx, :]
                for t in range(1, num_timesteps):
                    current_step_slice = timeseries_data[t, :, :, z_boundary_idx, :]
                    if not np.allclose(first_step_slice, current_step_slice):
                        print(f"FAILURE: Z-boundary at index {z_boundary_idx} is NOT constant across timesteps.")
                        all_boundaries_ok = False
                        break
                if not all_boundaries_ok: break

            if all_boundaries_ok:
                print("SUCCESS: Z boundaries are constant across all timesteps.")
        else:
            print("SKIPPED: Boundary verification requires more than one timestep.")

    print("--- Verification Complete ---")


def main():
    """
    Main function to parse arguments and run the subsampling process.
    """
    parser = argparse.ArgumentParser(
        description="Subsample 3D timeseries data from Fortran-style binary files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- I/O Arguments ---
    parser.add_argument('--input-dir', type=Path, default=Path("./output/timeseries_data/"), help="Directory containing the input timeseries files.")
    parser.add_argument('--output-file', type=Path, default=Path("./output/subsampled_timeseries.npz"), help="Path for the final compressed .npz output file.")
    parser.add_argument('--meta-file-path', type=Path, default=Path("./output/timeseries_data/runParameters.txt"), help="Full path to the runParameters.txt metadata file.")
    parser.add_argument('--file-prefix', type=str, default="patt3d_vx3d_", help="The common prefix for the timeseries files.")

    # --- Mock Data Arguments ---
    parser.add_argument('--generate-mock-data', action='store_true', help="If set, generate a local test dataset before running.")

    # --- Channel Arguments ---
    parser.add_argument('--source-channel-labels', type=str, nargs='+', default=['vx', 'vy', 'vz', 'T'], help="Space-separated list of ALL channel labels in the source data order.")
    parser.add_argument('--channel-indices-to-keep', type=int, nargs='+', default=[0, 1, 2, 3], help="Space-separated list of channel indices to keep in the final output.")

    # --- Timeseries Arguments ---
    parser.add_argument('--time-start', type=int, default=0, help="Starting time index to process.")
    parser.add_argument('--time-end', type=int, default=10, help="Ending time index to process (exclusive).")

    # --- Subsampling Arguments ---
    parser.add_argument('--num-x-samples', type=int, default=32, help="Number of evenly spaced points to select along the x-axis.")
    parser.add_argument('--y-indices-to-keep', type=int, nargs='+', default=[3, 10, 17], help="Space-separated list of specific y-indices to keep.")

    args = parser.parse_args()

    # --- Execution Logic ---
    if args.generate_mock_data:
        print("GENERATE_MOCK_DATA is True. Generating mock data for testing.")
        mock_nx, mock_ny, mock_nz = 2301, 481, 121
        generate_mock_data(
            args.input_dir,
            args.meta_file_path,
            args.file_prefix,
            range(args.time_start, args.time_end),
            mock_nx,
            mock_ny,
            mock_nz,
            source_labels=args.source_channel_labels,
        )

    Nx, Ny, Nz = get_parameters_from_run_file(args.meta_file_path)

    final_channel_labels = [args.source_channel_labels[i] for i in args.channel_indices_to_keep]
    num_input_channels = len(args.source_channel_labels)
    num_output_channels = len(args.channel_indices_to_keep)
    
    print(f"Channels selected for output: {final_channel_labels}")

    process_and_subsample(
        input_dir=args.input_dir,
        output_file=args.output_file,
        prefix=args.file_prefix,
        time_indices=range(args.time_start, args.time_end),
        nx=Nx,
        ny=Ny,
        nz=Nz,
        num_x_samples=args.num_x_samples,
        y_indices=args.y_indices_to_keep,
        channel_indices_to_keep=args.channel_indices_to_keep,
        num_input_channels=num_input_channels,
        num_output_channels=num_output_channels,
        final_channel_labels=final_channel_labels,
    )

    verify_output(args.output_file)


if __name__ == "__main__":
    main()
