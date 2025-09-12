import argparse
from pathlib import Path
import multiprocessing
from mhd_surrogate_core.plotting import generate_slice_video

def main():
    parser = argparse.ArgumentParser(
        description="Generate a 2D video of a slice from raw DNS snapshot data.",
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
    
    # --- Slicing Arguments ---
    parser.add_argument("--slice-orientation", type=str, required=True, choices=['xy', 'xz', 'yz'], help="The orientation of the 2D slice.")
    parser.add_argument("--slice-index", type=int, required=True, help="The integer index of the axis held constant (e.g., z-index for an xy-slice).")
    parser.add_argument("--channel", type=str, required=True, help="The channel to plot (e.g., 'vx').")

    # --- Video and Plotting Arguments ---
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the output video.")
    parser.add_argument("--interp-y", type=int, default=512, help="Number of interpolation points for the y-axis.")
    parser.add_argument("--interp-z", type=int, default=512, help="Number of interpolation points for the z-axis.")
    parser.add_argument("--channel-alias", type=str, default=None, help="Display name for the channel in titles (e.g., 'u' for 'vx').")
    parser.add_argument("--unit-label", type=str, default="", help="Unit label to display on the color bar (e.g., 'm/s').")
    parser.add_argument("--vmin", type=float, default=None, help="Override for the minimum value of the color scale.")
    parser.add_argument("--vmax", type=float, default=None, help="Override for the maximum value of the color scale.")
    
    # --- Parallelization Argument ---
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel worker processes for frame generation. Set to -1 to use all available CPU cores.")

    args = parser.parse_args()
    
    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()

    # --- Run the video generation ---
    generate_slice_video(
        snapshot_dir=args.snapshot_dir,
        file_prefix=args.file_prefix,
        time_indices=range(args.time_start, args.time_end),
        nx=args.nx, ny=args.ny, nz=args.nz,
        source_channel_labels=args.source_channels,
        output_path=args.output_path,
        slice_orientation=args.slice_orientation,
        slice_index=args.slice_index,
        channel=args.channel,
        fps=args.fps,
        num_interp_points_y=args.interp_y,
        num_interp_points_z=args.interp_z,
        channel_alias=args.channel_alias,
        unit_label=args.unit_label,
        vmin_override=args.vmin,
        vmax_override=args.vmax,
        num_workers=num_workers,
    )

if __name__ == "__main__":
    main()

"""
#### How to Use It

1.  **Installation:** Before running, make sure you have the necessary library installed. Open your terminal and run:
    ```bash
    pip install imageio[ffmpeg]
    ```
2.  Place this script inside a `scripts` directory at the root of your project.
3.  From your terminal, navigate **inside the `scripts` directory** and run a command like this:

```bash
python create_video.py \
    --snapshot-dir /raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/raw/ \
    --output-path output/xz_slice_u.mp4 \
    --nx 2301 --ny 481 --nz 121 \
    --time-start 609 --time-end 737 \
    --slice-orientation xz \
    --slice-index 240 \
    --channel vx \
    --channel-alias u \
    --unit-label "" \
    --fps 16 \
    --interp-y 1024 \
    --interp-z 1024 \
    --num-workers 64 \
    --vmin -5 \
    --vmax 6
```
"""

