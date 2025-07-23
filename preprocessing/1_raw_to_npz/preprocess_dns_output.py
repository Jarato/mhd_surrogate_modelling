import numpy as np
import re
from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing
from functools import partial

# --- Global variable for the memory-mapped array ---
# This will be inherited by each worker process
memmap_array = None

# --- Helper Functions ---

def format_bytes(size_bytes: int) -> str:
    """Converts a size in bytes to a human-readable string (e.g., KB, MB, GB)."""
    if size_bytes <= 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


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
    """
    if dir_path.exists() and any(dir_path.iterdir()):
        error_message = (
            f"\n--- SAFETY ABORT ---\n"
            f"Error: Mock data generation target directory '{dir_path}' is not empty.\n"
            f"To prevent overwriting real data, please clear this directory or set --generate-mock-data to False in the script.\n"
        )
        print(error_message)
        raise SystemExit()

    print(f"--- Generating Mock Data in '{dir_path}' with channels: {source_labels} ---")
    dir_path.mkdir(parents=True, exist_ok=True)

    with open(meta_filepath, 'w') as f:
        f.write(f"nX: {nx_points - 1}\n")
        f.write(f"nY: {ny_points - 1}\n")
        f.write(f"nZ: {nz_points - 1}\n")

    for t in tqdm(time_indices, desc="Generating mock files"):
        filename = dir_path / f"{prefix}{t:06d}"
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


def init_worker(temp_filename, shape, dtype):
    """
    Initializer for each worker process. Opens the memory-mapped file.
    """
    global memmap_array
    memmap_array = np.memmap(temp_filename, dtype=dtype, mode='r+', shape=shape)


def process_single_file(
    time_index_tuple,
    # Static arguments are passed via functools.partial
    input_dir, prefix, nx, ny, nz,
    num_input_channels, itemsize, coord_offset_count,
    input_dtype, output_dtype,
    channel_indices_to_keep, y_indices, x_subsample_indices,
    total_expected_bytes
):
    """
    Worker function that processes a single timestep file.
    """
    i, t = time_index_tuple
    filename = input_dir / f"{prefix}{t:06d}"

    # --- File Size Verification ---
    actual_bytes = filename.stat().st_size
    if actual_bytes != total_expected_bytes:
        # Use tqdm.write to print without breaking the progress bar
        tqdm.write(f"\n--- WARNING: FILE SIZE MISMATCH ---")
        tqdm.write(f"File: {filename}")
        tqdm.write(f"Expected size: {format_bytes(total_expected_bytes)}, Actual size: {format_bytes(actual_bytes)}")
        tqdm.write(f"Continuing, but the output may be incorrect.")

    with open(filename, 'rb') as f:
        offset_bytes = coord_offset_count * itemsize
        channel_data_1d = np.fromfile(f, dtype=input_dtype, offset=offset_bytes)

    try:
        data_4d_physical = channel_data_1d.reshape((nz, num_input_channels, ny, nx))
        data_4d_logical = data_4d_physical.transpose(1, 0, 2, 3)
    except ValueError as e:
        tqdm.write(f"Error reshaping data for {filename}. Check dimensions and source channel list.")
        expected_elements = num_input_channels * nz * ny * nx
        tqdm.write(f"Expected {expected_elements} elements, but read {len(channel_data_1d)}.")
        raise e
    
    selected_channels_data = data_4d_logical[channel_indices_to_keep, :, :, :]
    subsampled_timestep = selected_channels_data[:, :, y_indices, :][:, :, :, x_subsample_indices]
    transposed_timestep = subsampled_timestep.transpose(3, 2, 1, 0)
    
    # Write directly to the shared memory-mapped array
    memmap_array[i] = transposed_timestep.astype(output_dtype)


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
    num_workers: int,
):
    """
    The core function to read, subsample, and save the data in parallel.
    """
    print("--- Starting Subsampling Process (Parallel, Memory-Safe, float32 output) ---")
    
    x_subsample_indices = np.linspace(0, nx - 1, num_x_samples, dtype=int)
    
    input_dtype = np.float64
    output_dtype = np.float32
    
    itemsize = np.dtype(input_dtype).itemsize
    coord_offset_count = nx + ny + nz

    expected_coord_bytes = coord_offset_count * itemsize
    expected_channel_bytes = nz * num_input_channels * ny * nx * itemsize
    total_expected_bytes = expected_coord_bytes + expected_channel_bytes
    print(f"Calculated expected file size per timestep: {format_bytes(total_expected_bytes)}")

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
    # Create the file on disk but don't hold it open in the main process
    fp = np.memmap(temp_filename, dtype=output_dtype, mode='w+', shape=final_shape)
    del fp # Close the file handle, the file remains on disk

    # --- Parallel Processing ---
    worker_func = partial(
        process_single_file,
        input_dir=input_dir, prefix=prefix, nx=nx, ny=ny, nz=nz,
        num_input_channels=num_input_channels, itemsize=itemsize, coord_offset_count=coord_offset_count,
        input_dtype=input_dtype, output_dtype=output_dtype,
        channel_indices_to_keep=channel_indices_to_keep, y_indices=y_indices, x_subsample_indices=x_subsample_indices,
        total_expected_bytes=total_expected_bytes
    )

    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(temp_filename, final_shape, output_dtype)) as pool:
        # Use imap_unordered for efficiency, wrapped in tqdm for progress bar
        list(tqdm(pool.imap_unordered(worker_func, enumerate(time_indices)), total=len(time_indices), desc="Processing files"))

    print(f"\nPackaging final data into {output_file}...")
    # Re-open the now-populated memory-mapped file for saving
    final_timeseries = np.memmap(temp_filename, dtype=output_dtype, mode='r', shape=final_shape)
    np.savez_compressed(
        output_file,
        timeseries=final_timeseries,
        labels=np.array(final_channel_labels),
        x_coords=x_coords_sub,
        y_coords=y_coords_sub,
        z_coords=z_coords_sub,
    )
    
    del final_timeseries
    temp_filename.unlink()
    
    final_size_bytes = output_file.stat().st_size
    total_original_size_bytes = total_expected_bytes * len(time_indices)
    reduction_factor = total_original_size_bytes / final_size_bytes if final_size_bytes > 0 else float('inf')

    print("\n--- Subsampling Process Complete ---")
    print(f"Total original data processed: {format_bytes(total_original_size_bytes)}")
    print(f"Final compressed output size:  {format_bytes(final_size_bytes)}")
    print(f"Data size reduction factor:    {reduction_factor:.2f}x")
    
    with np.load(output_file) as data:
        print(f"Final array shape: {data['timeseries'].shape}")
        print(f"Final array dtype: {data['timeseries'].dtype}")


def verify_output(output_file: Path):
    """
    Loads the final .npz file and prints its contents to confirm success.
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

    parser.add_argument('--input-dir', type=Path, default=Path("./output/timeseries_data/"), help="Directory containing the input timeseries files.")
    parser.add_argument('--output-file', type=Path, default=Path("./output/subsampled_timeseries.npz"), help="Path for the final compressed .npz output file.")
    parser.add_argument('--meta-file-path', type=Path, default=Path("./output/timeseries_data/runParameters.txt"), help="Full path to the runParameters.txt metadata file.")
    parser.add_argument('--file-prefix', type=str, default="patt3d_vx3d_", help="The common prefix for the timeseries files.")
    parser.add_argument('--generate-mock-data', action='store_true', help="If set, generate a local test dataset before running.")
    parser.add_argument('--source-channel-labels', type=str, nargs='+', default=['vx', 'vy', 'vz', 'T'], help="Space-separated list of ALL channel labels in the source data order.")
    parser.add_argument('--channel-indices-to-keep', type=int, nargs='+', default=[0, 1, 2, 3], help="Space-separated list of channel indices to keep in the final output.")
    parser.add_argument('--time-start', type=int, default=0, help="Starting time index to process.")
    parser.add_argument('--time-end', type=int, default=4, help="Ending time index to process (exclusive).")
    parser.add_argument('--num-x-samples', type=int, default=32, help="Number of evenly spaced points to select along the x-axis.")
    parser.add_argument('--y-indices-to-keep', type=int, nargs='+', default=[3, 10, 17], help="Space-separated list of specific y-indices to keep.")
    parser.add_argument('--num-workers', type=int, default=multiprocessing.cpu_count(), help="Number of parallel worker processes to use.")

    args = parser.parse_args()

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
        num_workers=args.num_workers,
    )

    verify_output(args.output_file)


if __name__ == "__main__":
    main()
