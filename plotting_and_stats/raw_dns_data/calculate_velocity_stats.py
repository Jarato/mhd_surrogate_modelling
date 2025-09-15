import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import multiprocessing

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def _process_snapshot_for_stats(args):
    """
    Worker function to load one snapshot and calculate statistics (mean profile,
    min, max) for all specified velocity components.
    """
    f_path, nx, ny, nz, num_channels, channel_indices = args
    try:
        input_dtype = np.float64
        with open(f_path, 'rb') as f:
            # Seek past the coordinate data
            f.seek((nx + ny + nz) * np.dtype(input_dtype).itemsize)
            channel_data_1d = np.fromfile(f, dtype=input_dtype)
        
        data_4d_physical = channel_data_1d.reshape((nz, num_channels, ny, nx))
        snapshot_3d = data_4d_physical.transpose(3, 2, 0, 1) # -> (x, y, z, chan)
        
        snapshot_stats = {}
        for channel_name, chan_idx in channel_indices.items():
            # Isolate data for the current channel
            channel_data = snapshot_3d[:, :, :, chan_idx]
            
            # Calculate the mean profile over the y and z axes
            mean_profile = channel_data.mean(axis=(1, 2))
            
            # Find the min and max for this snapshot
            snapshot_min = channel_data.min()
            snapshot_max = channel_data.max()
            
            snapshot_stats[channel_name] = (mean_profile, snapshot_min, snapshot_max)
            
        return snapshot_stats
    except Exception as e:
        logging.error(f"Worker failed on file {f_path.name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Calculate and print statistics (mean flow, vmin, vmax) for all velocity components.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- I/O Arguments ---
    parser.add_argument("--snapshot-dir", type=Path, required=True, help="Directory containing the raw snapshot files.")
    parser.add_argument("--file-prefix", type=str, default="patt3d_vx3d_", help="The common prefix for the snapshot files.")
    
    # --- Data and Grid Arguments ---
    parser.add_argument("--nx", type=int, required=True, help="Number of grid points on the x-axis.")
    parser.add_argument("--ny", type=int, required=True, help="Number of grid points on the y-axis.")
    parser.add_argument("--nz", type=int, required=True, help="Number of grid points on the z-axis.")
    parser.add_argument("--time-start", type=int, required=True, help="Starting time index to process.")
    parser.add_argument("--time-end", type=int, required=True, help="Ending time index to process (exclusive).")
    parser.add_argument("--source-channels", type=str, nargs='+', default=['vx', 'vy', 'vz', 'T'], help="List of ALL channel labels in the order they appear in the file.")
    
    # --- Parallelization ---
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel worker processes. Set to -1 to use all available CPU cores.")

    args = parser.parse_args()
    
    # --- Main Logic ---
    time_indices = range(args.time_start, args.time_end)
    snapshot_files = [args.snapshot_dir / f"{args.file_prefix}{i:06d}" for i in time_indices]

    if not all(f.exists() for f in snapshot_files):
        logging.error("Not all snapshot files were found. Aborting.")
        return

    velocity_channels_to_process = ['vx', 'vy', 'vz']
    channel_indices = {}
    for vc in velocity_channels_to_process:
        try:
            channel_indices[vc] = args.source_channels.index(vc)
        except ValueError:
            logging.warning(f"Velocity channel '{vc}' not found in source channels. Skipping.")
    
    if not channel_indices:
        logging.error("No velocity channels found to process. Aborting.")
        return

    # Parallel data extraction
    workers = args.num_workers if args.num_workers != -1 else multiprocessing.cpu_count()
    logging.info(f"Extracting stats in parallel using {workers} workers for channels: {list(channel_indices.keys())}...")
    
    tasks = [(f, args.nx, args.ny, args.nz, len(args.source_channels), channel_indices) for f in snapshot_files]
    
    # Initialize dictionaries to hold results for each channel
    all_profiles = {chan: [] for chan in channel_indices}
    all_mins = {chan: [] for chan in channel_indices}
    all_maxs = {chan: [] for chan in channel_indices}
    
    with multiprocessing.Pool(processes=workers) as pool:
        results = list(tqdm(pool.imap(_process_snapshot_for_stats, tasks), total=len(tasks), desc="Processing snapshots"))

    # Unpack the results from the workers
    successful_runs = 0
    for res_dict in results:
        if res_dict is not None:
            successful_runs += 1
            for channel_name, (profile, s_min, s_max) in res_dict.items():
                all_profiles[channel_name].append(profile)
                all_mins[channel_name].append(s_min)
                all_maxs[channel_name].append(s_max)

    if successful_runs != len(snapshot_files):
        raise RuntimeError("One or more snapshot files failed to load. Check logs.")

    # Calculate and print the final statistics for each component
    logging.info("Calculating final statistics...")
    
    print("\n--------------------------------------------------")
    for channel_name in channel_indices.keys():
        # Calculate stats for the current channel
        time_averaged_profile = np.mean(all_profiles[channel_name], axis=0)
        overall_mean_flow = np.mean(time_averaged_profile)
        global_vmin = min(all_mins[channel_name])
        global_vmax = max(all_maxs[channel_name])
        
        # Print the final results to the console
        print(f"Statistics for {channel_name}:")
        print(f"  Overall Mean Flow: {overall_mean_flow}")
        print(f"  Global Minimum (vmin):  {global_vmin}")
        print(f"  Global Maximum (vmax):  {global_vmax}")
        print("--------------------------------------------------")

    logging.info("Calculation complete.")


if __name__ == "__main__":
    main()

"""
#### How to Use It

1.  Place this script inside your `scripts` directory.
2.  From your terminal, navigate **inside the `scripts` directory** and run a command like this:

```bash
python calculate_velocity_stats.py \
    --snapshot-dir /raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/raw/ \
    --nx 2301 --ny 481 --nz 121 \
    --time-start 609 --time-end 737 \
    --num-workers 64
```
"""