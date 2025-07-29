import numpy as np
import re
from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing
from functools import partial
import logging
import sys
from typing import List, Tuple
from scipy.interpolate import make_interp_spline

# --- Global variable for the memory-mapped array ---
# This will be inherited by each worker process
memmap_array = None

# --- Helper Functions ---

def setup_logging():
    """Configures the logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def format_bytes(size_bytes: int) -> str:
    """Converts a size in bytes to a human-readable string (e.g., KB, MB, GB)."""
    if size_bytes <= 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def generate_mock_data(
    dir_path: Path,
    prefix: str,
    time_indices: range,
    nx_points: int,
    ny_points: int,
    nz_points: int,
    source_labels: List[str],
) -> None:
    """
    Generates a mock dataset that mimics the Fortran binary output.
    """
    if dir_path.exists() and any(dir_path.iterdir()):
        logging.critical(
            f"SAFETY ABORT: Mock data generation target directory '{dir_path}' is not empty. "
            f"To prevent overwriting real data, please clear this directory or set --generate-mock-data to False."
        )
        raise SystemExit()

    logging.info(f"Generating mock data in '{dir_path}' with channels: {source_labels}")
    dir_path.mkdir(parents=True, exist_ok=True)

    for t in tqdm(time_indices, desc="Generating mock files"):
        filename = dir_path / f"{prefix}{t:06d}"
        with open(filename, 'wb') as f:
            x_coords = np.linspace(0, 1, nx_points, dtype=np.float64)
            y_coords = np.linspace(0, 1, ny_points, dtype=np.float64)
            # Create a non-uniform z-grid for a better interpolation test case
            z_coords = np.geomspace(1, 10, nz_points, dtype=np.float64) - 1
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
    logging.info("Mock data generation complete.")


def init_worker(temp_filename: Path, shape: Tuple, dtype: np.dtype) -> None:
    """
    Initializer for each worker process. Opens the memory-mapped file.
    """
    global memmap_array
    memmap_array = np.memmap(temp_filename, dtype=dtype, mode='r+', shape=shape)


def process_single_file(
    time_index_tuple: Tuple[int, int],
    # Static arguments are passed via functools.partial
    input_dir: Path, prefix: str, nx: int, ny: int, nz: int,
    num_input_channels: int, itemsize: int, coord_offset_count: int,
    input_dtype: np.dtype, output_dtype: np.dtype,
    channel_indices_to_keep: List[int], y_indices: List[int], x_subsample_indices: np.ndarray,
    total_expected_bytes: int,
    z_coords_full: np.ndarray,
    new_z_coords: np.ndarray,
) -> None:
    """
    Worker function that processes a single timestep file.
    """
    i, t = time_index_tuple
    filename = input_dir / f"{prefix}{t:06d}"

    actual_bytes = filename.stat().st_size
    if actual_bytes != total_expected_bytes:
        tqdm.write(
            f"WARNING: FILE SIZE MISMATCH for {filename}. "
            f"Expected: {format_bytes(total_expected_bytes)}, Got: {format_bytes(actual_bytes)}"
        )

    with open(filename, 'rb') as f:
        offset_bytes = coord_offset_count * itemsize
        channel_data_1d = np.fromfile(f, dtype=input_dtype, offset=offset_bytes)

    try:
        data_4d_physical = channel_data_1d.reshape((nz, num_input_channels, ny, nx))
        data_4d_logical = data_4d_physical.transpose(1, 0, 2, 3)
        selected_channels_data = data_4d_logical[channel_indices_to_keep, :, :, :]
        subsampled_spatial_data = selected_channels_data[:, :, y_indices, :][:, :, :, x_subsample_indices]

        # --- Z-Interpolation Step ---
        if new_z_coords is not None:
            # The data to be interpolated has shape (channels, z, y, x)
            # We need to interpolate along the z-axis (axis=1)
            spline = make_interp_spline(z_coords_full, subsampled_spatial_data, k=1, axis=1)
            interpolated_data = spline(new_z_coords)
        else:
            interpolated_data = subsampled_spatial_data

        # Final transpose to get C-style order: (x, y, z, channel)
        transposed_timestep = interpolated_data.transpose(3, 2, 1, 0)

    except ValueError as e:
        tqdm.write(f"ERROR: Failed to reshape or process data for {filename}. Check dimensions and source channel list.")
        raise e
    
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
    y_indices: List[int],
    channel_indices_to_keep: List[int],
    num_input_channels: int,
    num_output_channels: int,
    final_channel_labels: List[str],
    num_workers: int,
    interpolate_z: bool,
    num_z_samples: int,
) -> None:
    """
    The core function to read, subsample, and save the data in parallel.
    """
    logging.info("Starting subsampling process (Parallel, Memory-Safe, float32 output)...")
    
    x_subsample_indices = np.linspace(0, nx - 1, num_x_samples, dtype=int)
    
    input_dtype = np.float64
    output_dtype = np.float32
    
    itemsize = np.dtype(input_dtype).itemsize
    coord_offset_count = nx + ny + nz

    expected_coord_bytes = coord_offset_count * itemsize
    expected_channel_bytes = nz * num_input_channels * ny * nx * itemsize
    total_expected_bytes = expected_coord_bytes + expected_channel_bytes
    logging.info(f"Calculated expected file size per timestep: {format_bytes(total_expected_bytes)}")

    logging.info("Reading coordinates from first file...")
    first_file_path = input_dir / f"{prefix}{time_indices[0]:06d}"
    with open(first_file_path, 'rb') as f:
        x_coords_full = np.fromfile(f, dtype=input_dtype, count=nx)
        y_coords_full = np.fromfile(f, dtype=input_dtype, count=ny)
        z_coords_full = np.fromfile(f, dtype=input_dtype, count=nz)

    x_coords_sub = x_coords_full[x_subsample_indices].astype(output_dtype)
    y_coords_sub = y_coords_full[y_indices].astype(output_dtype)
    
    # --- Handle Z-coordinates based on interpolation flag ---
    if interpolate_z:
        logging.info(f"Z-axis interpolation enabled. Creating uniform grid with {num_z_samples} points.")
        z_coords_final = np.linspace(z_coords_full.min(), z_coords_full.max(), num_z_samples).astype(output_dtype)
        final_nz = num_z_samples
    else:
        logging.info("Z-axis interpolation disabled. Using original Z-grid.")
        z_coords_final = z_coords_full.astype(output_dtype)
        final_nz = nz

    logging.info(f"Final coordinate shapes: x={x_coords_sub.shape}, y={y_coords_sub.shape}, z={z_coords_final.shape}")
    
    final_shape = (len(time_indices), num_x_samples, len(y_indices), final_nz, num_output_channels)
    
    temp_filename = output_file.with_suffix(output_file.suffix + ".tmp")
    
    logging.info(f"Creating memory-mapped file at '{temp_filename}' with shape {final_shape} and dtype {output_dtype}")
    fp = np.memmap(temp_filename, dtype=output_dtype, mode='w+', shape=final_shape)
    del fp

    worker_func = partial(
        process_single_file,
        input_dir=input_dir, prefix=prefix, nx=nx, ny=ny, nz=nz,
        num_input_channels=num_input_channels, itemsize=itemsize, coord_offset_count=coord_offset_count,
        input_dtype=input_dtype, output_dtype=output_dtype,
        channel_indices_to_keep=channel_indices_to_keep, y_indices=y_indices, x_subsample_indices=x_subsample_indices,
        total_expected_bytes=total_expected_bytes,
        z_coords_full=z_coords_full,
        new_z_coords=z_coords_final if interpolate_z else None,
    )

    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=(temp_filename, final_shape, output_dtype)) as pool:
        list(tqdm(pool.imap_unordered(worker_func, enumerate(time_indices)), total=len(time_indices), desc="Processing files"))

    logging.info(f"Packaging final data into {output_file}...")
    final_timeseries = np.memmap(temp_filename, dtype=output_dtype, mode='r', shape=final_shape)
    np.savez_compressed(
        output_file,
        timeseries=final_timeseries,
        labels=np.array(final_channel_labels),
        x_coords=x_coords_sub,
        y_coords=y_coords_sub,
        z_coords=z_coords_final,
    )
    
    del final_timeseries
    temp_filename.unlink()
    
    final_size_bytes = output_file.stat().st_size
    total_original_size_bytes = total_expected_bytes * len(time_indices)
    reduction_factor = total_original_size_bytes / final_size_bytes if final_size_bytes > 0 else float('inf')

    logging.info("Subsampling process complete.")
    logging.info(f"Total original data processed: {format_bytes(total_original_size_bytes)}")
    logging.info(f"Final compressed output size:  {format_bytes(final_size_bytes)}")
    logging.info(f"Data size reduction factor:    {reduction_factor:.2f}x")
    
    with np.load(output_file) as data:
        logging.info(f"Final array shape: {data['timeseries'].shape}")
        logging.info(f"Final array dtype: {data['timeseries'].dtype}")


def verify_output(output_file: Path):
    """
    Loads the final .npz file and prints its contents to confirm success.
    """
    logging.info(f"Verifying output file: {output_file}...")
    if not output_file.exists():
        logging.error("Output file not found!")
        return

    with np.load(output_file) as data:
        logging.info("Successfully loaded archive.")
        logging.info(f"Keys found in file: {list(data.keys())}")

        timeseries_data = data['timeseries']
        labels = data['labels']
        x_coords = data['x_coords']
        y_coords = data['y_coords']
        z_coords = data['z_coords']

        logging.info(f"Shape of 'timeseries' array: {timeseries_data.shape}")
        logging.info(f"Data type of 'timeseries' array: {timeseries_data.dtype}")
        logging.info(f"Content of 'labels' array: {labels}")
        logging.info(f"Shape of 'x_coords' array: {x_coords.shape}")
        logging.info(f"Shape of 'y_coords' array: {y_coords.shape}")
        logging.info(f"Shape of 'z_coords' array: {z_coords.shape}")

        logging.info("Verifying boundary conditions...")
        all_boundaries_ok = True
        num_timesteps = timeseries_data.shape[0]

        if num_timesteps > 1:
            for z_boundary_idx in [0, -1]:
                first_step_slice = timeseries_data[0, :, :, z_boundary_idx, :]
                for t in range(1, num_timesteps):
                    current_step_slice = timeseries_data[t, :, :, z_boundary_idx, :]
                    if not np.allclose(first_step_slice, current_step_slice):
                        logging.error(f"FAILURE: Z-boundary at index {z_boundary_idx} is NOT constant across timesteps.")
                        all_boundaries_ok = False
                        break
                if not all_boundaries_ok: break

            if all_boundaries_ok:
                logging.info("SUCCESS: Z boundaries are constant across all timesteps.")
        else:
            logging.warning("SKIPPED: Boundary verification requires more than one timestep.")

    logging.info("Verification complete.")


def main():
    """
    Main function to parse arguments and run the subsampling process.
    """
    parser = argparse.ArgumentParser(
        description="Subsample 3D timeseries data and optionally interpolate the Z-axis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Dimension Arguments ---
    parser.add_argument('--nx', type=int, required=True, help="Number of grid POINTS (not spaces) on the x-axis.")
    parser.add_argument('--ny', type=int, required=True, help="Number of grid POINTS (not spaces) on the y-axis.")
    parser.add_argument('--nz', type=int, required=True, help="Number of grid POINTS (not spaces) on the z-axis.")

    # --- I/O Arguments ---
    parser.add_argument('--input-dir', type=Path, default=Path("./output/timeseries_data/"), help="Directory containing the input timeseries files.")
    parser.add_argument('--output-file', type=Path, default=Path("./output/subsampled_timeseries.npz"), help="Path for the final compressed .npz output file.")
    parser.add_argument('--file-prefix', type=str, default="patt3d_vx3d_", help="The common prefix for the timeseries files.")

    # --- Mock Data Arguments ---
    parser.add_argument('--generate-mock-data', action='store_true', help="If set, generate a local test dataset before running.")

    # --- Channel Arguments ---
    parser.add_argument('--source-channel-labels', type=str, nargs='+', default=['vx', 'vy', 'vz', 'T'], help="Space-separated list of ALL channel labels in the source data order.")
    parser.add_argument('--channel-indices-to-keep', type=int, nargs='+', default=[0, 1, 2, 3], help="Space-separated list of channel indices to keep in the final output.")

    # --- Timeseries Arguments ---
    parser.add_argument('--time-start', type=int, default=0, help="Starting time index to process.")
    parser.add_argument('--time-end', type=int, default=4, help="Ending time index to process (exclusive).")

    # --- Subsampling Arguments ---
    parser.add_argument('--num-x-samples', type=int, default=32, help="Number of evenly spaced points to select along the x-axis.")
    parser.add_argument('--y-indices-to-keep', type=int, nargs='+', default=[3, 10, 17], help="Space-separated list of specific y-indices to keep.")
    parser.add_argument('--num-workers', type=int, default=1, help="Number of parallel worker processes to use. Set to -1 to use all available cores.")
    
    # --- Interpolation Arguments ---
    parser.add_argument('--interpolate-z', action='store_true', help="If set, interpolate the Z-axis to a uniform grid.")
    parser.add_argument('--num-z-samples', type=int, default=128, help="Number of points for the new uniform Z-grid (used only if --interpolate-z is set).")

    args = parser.parse_args()

    setup_logging()

    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()
        logging.info(f"Using all available cores: {num_workers}")

    if args.generate_mock_data:
        logging.info("GENERATE_MOCK_DATA is True. Generating mock data for testing.")
        generate_mock_data(
            args.input_dir,
            args.file_prefix,
            range(args.time_start, args.time_end),
            args.nx,
            args.ny,
            args.nz,
            source_labels=args.source_channel_labels,
        )

    Nx, Ny, Nz = args.nx, args.ny, args.nz
    logging.info(f"Using provided dimensions: Nx={Nx}, Ny={Ny}, Nz={Nz}")

    final_channel_labels = [args.source_channel_labels[i] for i in args.channel_indices_to_keep]
    num_input_channels = len(args.source_channel_labels)
    num_output_channels = len(args.channel_indices_to_keep)
    
    logging.info(f"Channels selected for output: {final_channel_labels}")

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
        num_workers=num_workers,
        interpolate_z=args.interpolate_z,
        num_z_samples=args.num_z_samples,
    )

    verify_output(args.output_file)


if __name__ == "__main__":
    main()
