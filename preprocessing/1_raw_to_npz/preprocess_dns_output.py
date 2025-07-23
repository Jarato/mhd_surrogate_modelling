import numpy as np
import re
from pathlib import Path

# --- Configuration ---
# --- STEP 1: CONFIGURE YOUR PARAMETERS HERE ---

# --- Master Switch for Mock Data ---
# Set this to True to generate a local test dataset.
# Set this to False when running on the cluster with real data.
GENERATE_MOCK_DATA = True

# --- Channel Settings ---
# Define ALL possible channels present in the source data, in their physical order.
# The script will use this list as the ground truth for reading the files.
SOURCE_CHANNEL_LABELS = ['vx', 'vy', 'vz', 'T'] 

# Define the indices of the channels you want to KEEP in the final output.
# For example, to keep only 'vx' (index 0) and 'T' (index 3):
CHANNEL_INDICES_TO_KEEP = [0, 1, 2, 3]

# --- I/O Settings (using pathlib) ---
INPUT_DIR = Path("./output/timeseries_data/")
OUTPUT_FILE = Path("./output/subsampled_timeseries.npz")
FILE_PREFIX = "patt3d_vx3d_"
# The full path to the metadata file.
META_FILE_PATH = Path("./output/timeseries_data/runParameters.txt")

# --- Data Dimensions & Channels (These will be set dynamically) ---
Nx, Ny, Nz = None, None, None
FINAL_CHANNEL_LABELS = []
NUM_INPUT_CHANNELS = len(SOURCE_CHANNEL_LABELS)
NUM_OUTPUT_CHANNELS = len(CHANNEL_INDICES_TO_KEEP)


# --- Timeseries Settings ---
TIME_INDICES = range(10)

# --- Subsampling Settings ---
NUM_X_SAMPLES = 32
# Define the interior Y-indices you want to keep.
Y_INDICES_TO_KEEP = [10, 20, 30, 40, 55, 80]


# --- Helper Functions ---

def get_parameters_from_run_file(meta_filepath: Path):
    """
    Reads the runParameters.txt file to extract grid dimensions.
    The coldFluid flag is read for completeness but no longer used for logic.

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
            f"To prevent overwriting real data, please clear this directory or set GENERATE_MOCK_DATA to False in the script.\n"
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
                    
                    # Create a base slice that is time-dependent
                    time_dependent_val = channel_val + t * 0.1 + z * 0.01
                    data_slice = np.full((nx_points, ny_points), time_dependent_val, dtype=np.float64)

                    # Overwrite boundaries with time-INdependent values for verification
                    time_independent_val = channel_val + z * 0.01
                    if z == 0 or z == nz_points - 1:
                        # For the first and last Z-slices, the whole slice is time-independent
                        data_slice.fill(time_independent_val)
                    
                    # For all other Z-slices, only the Y-boundaries are time-independent
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
    
    final_shape = (len(time_indices), num_x_samples, len(y_indices), nz, NUM_OUTPUT_CHANNELS)
    
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
                data_4d_physical = channel_data_1d.reshape((nz, NUM_INPUT_CHANNELS, ny, nx))
                data_4d_logical = data_4d_physical.transpose(1, 0, 2, 3)
            except ValueError as e:
                print(f"Error reshaping data for {filename}. Check dimensions and source channel list.")
                expected_elements = NUM_INPUT_CHANNELS * nz * ny * nx
                print(f"Expected {expected_elements} elements, but read {len(channel_data_1d)}.")
                del memmap_array
                temp_filename.unlink(missing_ok=True)
                raise e
            
            selected_channels_data = data_4d_logical[channel_indices_to_keep, :, :, :]
            subsampled_timestep = selected_channels_data[:, :, y_indices, :][:, :, :, x_subsample_indices]
            transposed_timestep = subsampled_timestep.transpose(3, 2, 1, 0)
            
            memmap_array[i] = transposed_timestep.astype(output_dtype)
            
    del memmap_array
    
    print(f"\nPackaging final data into {output_file}...")
    final_timeseries = np.load(temp_filename)
    np.savez_compressed(
        output_file,
        timeseries=final_timeseries,
        labels=np.array(FINAL_CHANNEL_LABELS),
        x_coords=x_coords_sub,
        y_coords=y_coords_sub,
        z_coords=z_coords_sub,
    )
    
    temp_filename.unlink()
    
    print("--- Subsampling Process Complete ---")
    print(f"Final array shape: {final_timeseries.shape}")
    print(f"Final array dtype: {final_timeseries.dtype}")


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
        
        # --- Boundary Value Verification ---
        print("\n--- Verifying Boundary Conditions ---")
        all_boundaries_ok = True
        num_timesteps = timeseries_data.shape[0]

        if num_timesteps > 1:
            # Check Z boundaries (first and last slice)
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


if __name__ == "__main__":
    if GENERATE_MOCK_DATA:
        print("GENERATE_MOCK_DATA is True. Generating mock data for testing.")
        mock_nx, mock_ny, mock_nz = 2301, 481, 121
        generate_mock_data(
            INPUT_DIR,
            META_FILE_PATH,
            FILE_PREFIX,
            TIME_INDICES,
            mock_nx,
            mock_ny,
            mock_nz,
            source_labels=SOURCE_CHANNEL_LABELS,
        )

    Nx, Ny, Nz = get_parameters_from_run_file(META_FILE_PATH)

    FINAL_CHANNEL_LABELS = [SOURCE_CHANNEL_LABELS[i] for i in CHANNEL_INDICES_TO_KEEP]
    print(f"Channels selected for output: {FINAL_CHANNEL_LABELS}")

    process_and_subsample(
        input_dir=INPUT_DIR,
        output_file=OUTPUT_FILE,
        prefix=FILE_PREFIX,
        time_indices=TIME_INDICES,
        nx=Nx,
        ny=Ny,
        nz=Nz,
        num_x_samples=NUM_X_SAMPLES,
        y_indices=Y_INDICES_TO_KEEP,
        channel_indices_to_keep=CHANNEL_INDICES_TO_KEEP,
    )

    verify_output(OUTPUT_FILE)
