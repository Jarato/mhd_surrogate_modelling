# -*- coding: utf-8 -*-
# scripts/create_video_from_npz_2d.py

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
    parser.add_argument("--output-path", type=Path, required=True, help="Path to save the output .mp4 video file.")
    
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
    
    # --- Parallelization Argument ---
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel worker processes for frame generation. Set to -1 to use all available CPU cores.")

    args = parser.parse_args()
    
    # --- Main Logic ---
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
    # KEY CHANGE: Call the 2D video function without slice arguments.
    generate_video_from_npz(
        npz_path=args.input_npz,
        output_path=args.output_path,
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
    --input-npz /cephfs/users/skowronek/Documents/PhD/nuclear_fusion_cooling/prediction/mhd_surrogate_modelling/experiments/study_4_2d_tckae/output/ld512_M8_K8_Ktc8_gtc1.0_bwd_False/eval/predicted_timeseries.npz \
    --output-path output/videos/preprocessed_xz_slice_u.mp4 \
    --channel vx \
    --channel-alias u \
    --unit-label "" \
    --fps 16 \
    --num-workers 16 \
    --vmins vx:-2.16 vz:-3 \
    --vmaxs vx:3.84 vz:3 \
    --vcenters vx:0.84 vz:0.0 \
    --cmap seismic
```
"""

