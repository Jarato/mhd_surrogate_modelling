import argparse
from pathlib import Path
import multiprocessing
import logging
from mhd_surrogate_core.plotting.xyz import generate_3d_video

def main():
    parser = argparse.ArgumentParser(
        description="Generate a 3D Volumetric video from raw DNS snapshot data using PyVista.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- I/O Arguments ---
    parser.add_argument("--snapshot-dir", type=Path, required=True, help="Directory containing the raw snapshot files.")
    parser.add_argument("--output-path", type=Path, required=True, help="Path to save the output .mp4 video file.")
    parser.add_argument("--file-prefix", type=str, default="patt3d_vx3d_", help="The common prefix for the snapshot files.")
    
    # --- Data and Grid Arguments ---
    parser.add_argument("--nx", type=int, required=True, help="Number of grid points on the x-axis.")
    parser.add_argument("--ny", type=int, required=True, help="Number of grid points on the y-axis.")
    parser.add_argument("--nz", type=int, required=True, help="Number of grid points on the z-axis.")
    parser.add_argument("--time-start", type=int, required=True, help="Starting time index to process.")
    parser.add_argument("--time-end", type=int, required=True, help="Ending time index to process (exclusive).")
    parser.add_argument("--source-channels", type=str, nargs='+', default=['vx', 'vy', 'vz', 'T'], help="List of ALL channel labels in the order they appear in the file.")
    
    # --- Visualization Arguments ---
    parser.add_argument("--channel", type=str, required=True, help="The channel to render (e.g., 'vx').")
    parser.add_argument("--channel-alias", type=str, default=None, help="Display name for the channel in titles.")
    parser.add_argument("--unit-label", type=str, default="", help="Unit label to display on the color bar.")
    parser.add_argument("--cmap", type=str, default="viridis", help="The colormap to use (e.g., coolwarm, plasma).")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the output video.")
    
    # --- Parallelization Argument ---
    # Defaulting to fewer workers for 3D as it is more memory intensive than 2D
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel worker processes. Be careful with VRAM usage.")

    args = parser.parse_args()
    
    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()

    # --- Run the video generation ---
    generate_3d_video(
        snapshot_dir=args.snapshot_dir,
        file_prefix=args.file_prefix,
        time_indices=range(args.time_start, args.time_end),
        nx=args.nx, ny=args.ny, nz=args.nz,
        source_channel_labels=args.source_channels,
        output_path=args.output_path,
        channel=args.channel,
        fps=args.fps,
        channel_alias=args.channel_alias,
        unit_label=args.unit_label,
        num_workers=num_workers,
        cmap=args.cmap,
    )

if __name__ == "__main__":
    main()
