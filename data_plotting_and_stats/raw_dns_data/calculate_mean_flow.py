import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import multiprocessing
import matplotlib.pyplot as plt

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def _process_snapshot_for_mean_flow(args):
    """
    Worker function to load one snapshot, calculate the y-z mean of the vx
    channel, and return the profile along with the snapshot's min and max values.
    """
    f_path, nx, ny, nz, num_channels, vx_channel_idx = args
    try:
        input_dtype = np.float64
        with open(f_path, 'rb') as f:
            # Seek past the coordinate data
            f.seek((nx + ny + nz) * np.dtype(input_dtype).itemsize)
            channel_data_1d = np.fromfile(f, dtype=input_dtype)
        
        data_4d_physical = channel_data_1d.reshape((nz, num_channels, ny, nx))
        snapshot_3d = data_4d_physical.transpose(3, 2, 0, 1) # -> (x, y, z, chan)
        
        # Isolate vx data
        vx_data = snapshot_3d[:, :, :, vx_channel_idx]
        
        # Calculate the mean over the y and z axes
        mean_vx_profile = vx_data.mean(axis=(1, 2))
        
        # Find the min and max for this specific snapshot
        snapshot_min = vx_data.min()
        snapshot_max = vx_data.max()
            
        return (mean_vx_profile, snapshot_min, snapshot_max)
    except Exception as e:
        logging.error(f"Worker failed on file {f_path.name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Calculate and print the mean flow, global vmin, and global vmax for the x-direction (vx).",
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

    try:
        vx_channel_idx = args.source_channels.index('vx')
    except ValueError:
        logging.error("Channel 'vx' not found in --source-channels. Aborting.")
        return

    # Parallel data extraction
    workers = args.num_workers if args.num_workers != -1 else multiprocessing.cpu_count()
    logging.info(f"Extracting profiles and stats in parallel using {workers} workers...")
    
    tasks = [(f, args.nx, args.ny, args.nz, len(args.source_channels), vx_channel_idx) for f in snapshot_files]
    
    all_profiles = []
    all_mins = []
    all_maxs = []
    
    with multiprocessing.Pool(processes=workers) as pool:
        results = list(tqdm(pool.imap(_process_snapshot_for_mean_flow, tasks), total=len(tasks), desc="Processing snapshots"))

    # Unpack the results from the workers
    for res in results:
        if res is not None:
            profile, s_min, s_max = res
            all_profiles.append(profile)
            all_mins.append(s_min)
            all_maxs.append(s_max)

    if len(all_profiles) != len(snapshot_files):
        raise RuntimeError("One or more snapshot files failed to load. Check logs.")

    # Calculate the final statistics
    logging.info("Calculating final statistics...")
    time_averaged_profile = np.mean(all_profiles, axis=0)
    overall_mean_flow = np.mean(time_averaged_profile)
    global_vmin = min(all_mins)
    global_vmax = max(all_maxs)

    logging.info("Calculation complete.")
    
    # Print the final results to the console
    print("\n--------------------------------------------------")
    print(f"Overall Mean Flow (vx): {overall_mean_flow}")
    print(f"Global Minimum (vmin):  {global_vmin}")
    print(f"Global Maximum (vmax):  {global_vmax}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()

"""
#### How to Use It

1.  Place this script inside your `scripts` directory.
2.  From your terminal, navigate **inside the `scripts` directory** and run a command like this:

```bash
python calculate_mean_flow.py \
    --snapshot-dir /raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/raw/ \
    --nx 2301 --ny 481 --nz 121 \
    --time-start 609 --time-end 737 \
    --num-workers 64
```
"""