# -*- coding: utf-8 -*-
# scripts/create_video_from_npz.py

import argparse
from pathlib import Path
import multiprocessing
import logging

# KEY CHANGE: Import the new 2D video generation function.
# We assume this module and function will be created.
from mhd_surrogate_core.plotting.xz import generate_video_from_npz

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    """Main function to generate a 2D video from a .npz file."""
    parser = argparse.ArgumentParser(
        description="Generate a 2D video from a preprocessed .npz file containing 2D data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- I/O Arguments ---
    parser.add_argument("--input-npz", type=Path, required=True, help="Path to the input .npz file (must contain 'timeseries' and 'labels' arrays).")
    parser.add_argument("--output-path", type=Path, required=True, help="Path to save the output .mp4 video file or a directory to save it in.")
    
    # --- Data Selection Arguments ---
    parser.add_argument("--channel", type=str, required=True, help="The channel to plot (e.g., 'vx').")
    parser.add_argument("--time-start", type=int, default=None, help="Optional: Starting time index to include in the video (inclusive).")
    parser.add_argument("--time-end", type=int, default=None, help="Optional: Ending time index to include in the video (exclusive).")

    # --- Video and Plotting Arguments ---
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the output video.")
    parser.add_argument("--channel-alias", type=str, default=None, help="Display name for the channel in titles (e.g., 'u' for 'vx').")
    parser.add_argument("--unit-label", type=str, default="", help="Unit label to display on the color bar (e.g., 'm/s').")
    parser.add_argument("--vmins", type=str, nargs='+', default=None, help="Per-channel minimum values for the color scale. Format: 'vx:-5' 'vy:-1'")
    parser.add_argument("--vmaxs", type=str, nargs='+', default=None, help="Per-channel maximum values for the color scale. Format: 'vx:6' 'vy:1'")
    parser.add_argument("--vcenters", type=str, nargs='+', default=None, help="Per-channel center values for diverging colormaps. Format: 'vx:0.84' 'vy:0.0'")
    parser.add_argument("--cmap", type=str, default="viridis", help="The colormap to use for the plot (e.g., coolwarm, plasma, seismic).")
    parser.add_argument("--base-size", type=float, default=8.0, help="The base size (in inches) for the longest dimension of the plot.")
    parser.add_argument("--min-size", type=float, default=3.0, help="The minimum size (in inches) for the shortest dimension of the plot.")
    
    # --- Parallelization Argument ---
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel worker processes for frame generation. Set to -1 to use all available CPU cores.")

    args = parser.parse_args()
    
    # --- Main Logic ---
    output_path = args.output_path

    # If the provided path is a directory, create a default filename.
    if output_path.is_dir():
        logging.info(f"Output path '{output_path}' is a directory. Creating a default filename.")
        channel_name = args.channel_alias if args.channel_alias else args.channel
        default_filename = f"video_{channel_name}.mp4"
        output_path = output_path / default_filename
        logging.info(f"Resolved output path to: {output_path}")

    # Ensure the parent directory for the output file exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Helper function to parse channel-value arguments (e.g., 'vx:0.5')
    def parse_channel_value_arg(arg_list):
        if not arg_list:
            return {}
        value_map = {}
        for item in arg_list:
            try:
                key, value = item.split(':')
                value_map[key] = float(value)
            except ValueError:
                logging.error(f"Invalid format for argument: '{item}'. Please use 'channel:value'.")
        return value_map

    vcenter_map = parse_channel_value_arg(args.vcenters)
    vmin_map = parse_channel_value_arg(args.vmins)
    vmax_map = parse_channel_value_arg(args.vmaxs)
    
    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()

    # --- Run the video generation ---
    generate_video_from_npz(
        npz_path=args.input_npz,
        output_path=output_path, # Use the resolved file path
        channel=args.channel,
        time_start=args.time_start,
        time_end=args.time_end,
        fps=args.fps,
        channel_alias=args.channel_alias,
        unit_label=args.unit_label,
        vmins_override=vmin_map,
        vmaxs_override=vmax_map,
        vcenters=vcenter_map,
        num_workers=num_workers,
        cmap=args.cmap,
        base_size=args.base_size,
        min_size=args.min_size,
    )

if __name__ == "__main__":
    main()

"""
#### How to Use It

1.  Place this script inside your `scripts` directory.
2.  From your terminal, navigate **inside the `scripts` directory** and run a command like this.
    The new `--vmins`, `--vmaxs`, and `--vcenters` flags allow per-component control.

```bash
python create_video_from_npz.py \
    --input-npz /cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/data/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1492_x1151_y1_z127_c2/T1492_x1151_y1_z127_c2.npz \
    --output-path output/videos/ \
    --channel vx \
    --channel-alias u \
    --unit-label "" \
    --fps 16 \
    --num-workers 31 \
    --vmins vx:-2.16 vz:-3 \
    --vmaxs vx:3.84 vz:3 \
    --vcenters vx:0.84 vz:0.0 \
    --cmap seismic \
    --base-size 25.0 \
    --min-size 4.0
```
"""

