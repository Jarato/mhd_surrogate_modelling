import argparse
from pathlib import Path
import multiprocessing
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
    parser.add_argument("--vmin", type=float, default=None, help="Override for the minimum value of the color scale.")
    parser.add_argument("--vmax", type=float, default=None, help="Override for the maximum value of the color scale.")
    
    # --- Parallelization Argument ---
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel worker processes for frame generation. Set to -1 to use all available CPU cores.")

    args = parser.parse_args()
    
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
        vmin_override=args.vmin,
        vmax_override=args.vmax,
        num_workers=num_workers,
    )

if __name__ == "__main__":
    main()

"""
#### How to Use It

1.  Place this script inside your `scripts` directory.
2.  From your terminal, navigate **inside the `scripts` directory** and run a command like this.
    The new `--time-start` and `--time-end` flags are optional.

```bash
python create_video_from_npz.py \
    --input-npz /raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/T1220_x1151_y5_z127_c3/T1220_x1151_y5_z127_c3.npz \
    --output-path output/preprocessed_xz_slice_u.mp4 \
    --slice-orientation xz \
    --slice-index 2 \
    --channel vx \
    --channel-alias u \
    --unit-label "m/s" \
    --fps 1 \
    --num-workers 1 \
    --vmin -3 \
    --vmax 5 \
    --time-start 0 \
    --time-end 4
```
"""

