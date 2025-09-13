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
    channel, and return the resulting 1D array.
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
        
        # Isolate vx data and calculate the mean over the y and z axes
        vx_data = snapshot_3d[:, :, :, vx_channel_idx]
        mean_vx_profile = vx_data.mean(axis=(1, 2))
            
        return mean_vx_profile
    except Exception as e:
        logging.error(f"Worker failed on file {f_path.name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Calculate the time-averaged mean flow in the x-direction (vx) from raw DNS snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- I/O Arguments ---
    parser.add_argument("--snapshot-dir", type=Path, required=True, help="Directory containing the raw snapshot files.")
    parser.add_argument("--output-npz", type=Path, required=True, help="Path to save the output .npz file with the mean profile.")
    parser.add_argument("--output-plot", type=Path, required=True, help="Path to save the output .png plot of the mean profile.")
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
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    
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
    logging.info(f"Extracting mean profiles in parallel using {workers} workers...")
    
    tasks = [(f, args.nx, args.ny, args.nz, len(args.source_channels), vx_channel_idx) for f in snapshot_files]
    
    all_profiles = []
    with multiprocessing.Pool(processes=workers) as pool:
        results = list(tqdm(pool.imap(_process_snapshot_for_mean_flow, tasks), total=len(tasks), desc="Processing snapshots"))

    all_profiles = [res for res in results if res is not None]

    if len(all_profiles) != len(snapshot_files):
        raise RuntimeError("One or more snapshot files failed to load. Check logs.")

    # Calculate the final time-averaged profile
    logging.info("Calculating final time-averaged profile...")
    final_mean_profile = np.mean(all_profiles, axis=0)

    # Load x-coordinates for saving
    with open(snapshot_files[0], 'rb') as f:
        x_coords = np.fromfile(f, dtype=np.float64, count=args.nx)

    # Save the data
    logging.info(f"Saving data to {args.output_npz}...")
    np.savez_compressed(args.output_npz, x_coords=x_coords, mean_vx=final_mean_profile)

    # Create and save the plot
    logging.info(f"Saving plot to {args.output_plot}...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_coords, final_mean_profile, 'o-', label='Time-Averaged Mean Flow')
    ax.set_title(f'Mean X-Velocity Profile (Averaged over Time {args.time_start}-{args.time_end-1})')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Mean Velocity (u)')
    ax.grid(True, which='both', linestyle='--')
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300)
    plt.close(fig)

    logging.info("Mean flow calculation complete.")


if __name__ == "__main__":
    main()

"""
#### How to Use It

1.  Place this script inside your `scripts` directory.
2.  From your terminal, navigate **inside the `scripts` directory** and run a command like this:

```bash
python calculate_mean_flow.py \
    --snapshot-dir /raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/raw/ \
    --output-npz output/mean_flow/mean_flow_profile.npz \
    --output-plot output/mean_flow/mean_flow_profile.png \
    --nx 2301 --ny 481 --nz 121 \
    --time-start 609 --time-end 737 \
    --num-workers 64
```
"""