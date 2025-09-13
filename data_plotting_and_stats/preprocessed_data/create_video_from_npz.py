import argparse
from pathlib import Path
import multiprocessing
import logging
from mhd_surrogate_core.plotting import generate_slice_video_from_npz

def main():
    parser = argparse.ArgumentParser(
        description="Generate a 2D video of a slice from a preprocessed .npz file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- I/O Arguments ---
    parser.add_argument("--input-npz", type=Path, required=True, help="Path to the input .npz file.")
    parser.add_argument("--output-path", type=Path, required=True, help="Path to save the output .mp4 video file.")
    
    # --- Slicing Arguments ---
    parser.add_argument("--slice-orientation", type=str, required=True, choices=['xy', 'xz', 'yz'], help="The orientation of the 2D slice.")
    parser.add_argument("--slice-index", type=int, required=True, help="The integer index of the axis held constant (e.g., z-index for an xy-slice).")
    parser.add_argument("--channel", type=str, required=True, help="The channel to plot (e.g., 'vx').")
    parser.add_argument("--time-start", type=int, default=None, help="Optional: Starting time index to include in the video (inclusive).")
    parser.add_argument("--time-end", type=int, default=None, help="Optional: Ending time index to include in the video (exclusive).")

    # --- Video and Plotting Arguments ---
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the output video.")
    parser.add_argument("--channel-alias", type=str, default=None, help="Display name for the channel in titles (e.g., 'u' for 'vx').")
    parser.add_argument("--unit-label", type=str, default="", help="Unit label to display on the color bar (e.g., 'm/s').")
    parser.add_argument("--vmins", type=str, nargs='+', default=None, help="Per-channel minimum values for the color scale. Format: vx:-5 vy:-1")
    parser.add_argument("--vmaxs", type=str, nargs='+', default=None, help="Per-channel maximum values for the color scale. Format: vx:6 vy:1")
    parser.add_argument("--vcenters", type=str, nargs='+', default=None, help="Per-channel center values for diverging colormaps. Format: vx:0.84 vy:0.0")
    parser.add_argument("--cmap", type=str, default="viridis", help="The colormap to use for the plot (e.g., coolwarm, plasma).")
    
    # --- Parallelization Argument ---
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel worker processes for frame generation. Set to -1 to use all available CPU cores.")

    args = parser.parse_args()
    
    # --- Main Logic ---
    # Helper function to parse channel-value arguments
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
    generate_slice_video_from_npz(
        npz_path=args.input_npz,
        output_path=args.output_path,
        slice_orientation=args.slice_orientation,
        slice_index=args.slice_index,
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
    --input-npz /raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1220_x1151_y5_z127_c3/T1220_x1151_y5_z127_c3.npz \
    --output-path output/videos/preprocessed_xz_slice_u.mp4 \
    --slice-orientation xz \
    --slice-index 2 \
    --channel vx \
    --channel-alias u \
    --unit-label "" \
    --fps 16 \
    --num-workers 64 \
    --vmins vx:-2.16 vy:-3 vz:-3 \
    --vmaxs vx:3.84 vy:3 vz:3 \
    --vcenters vx:0.84 vy:0.0 vz:0.0 \
    --cmap seismic
```
"""

