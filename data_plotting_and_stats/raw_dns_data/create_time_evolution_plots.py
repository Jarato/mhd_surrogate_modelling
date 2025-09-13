import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from mhd_surrogate_core.plotting import (
    plot_interpolated_z_time_evolution,
    plot_interpolated_y_time_evolution,
    plot_interpolated_x_time_evolution,
)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D time evolution plots from a series of raw DNS snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- I/O Arguments ---
    parser.add_argument("--snapshot-dir", type=Path, required=True, help="Directory containing the raw snapshot files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save the output .png plot files.")
    parser.add_argument("--file-prefix", type=str, default="patt3d_vx3d_", help="The common prefix for the snapshot files.")
    
    # --- Data and Grid Arguments ---
    parser.add_argument("--nx", type=int, required=True, help="Number of grid points on the x-axis.")
    parser.add_argument("--ny", type=int, required=True, help="Number of grid points on the y-axis.")
    parser.add_argument("--nz", type=int, required=True, help="Number of grid points on the z-axis.")
    parser.add_argument("--time-start", type=int, required=True, help="Starting time index to process.")
    parser.add_argument("--time-end", type=int, required=True, help="Ending time index to process (exclusive).")
    parser.add_argument("--source-channels", type=str, nargs='+', default=['vx', 'vy', 'vz', 'T'], help="List of ALL channel labels in the order they appear in the file.")
    
    # --- Slicing Arguments ---
    parser.add_argument("--plot-type", type=str, required=True, choices=['time-z', 'time-y', 'time-x'], help="The type of time evolution plot to generate.")
    parser.add_argument("--slice-indices", type=int, nargs=2, required=True, help="Two integer indices for the axes held constant (e.g., for 'time-z', provide x_index and y_index).")
    parser.add_argument("--channel", type=str, required=True, help="The channel to plot (e.g., 'vx').")

    # --- Plotting Arguments ---
    parser.add_argument("--channel-alias", type=str, default=None, help="Display name for the channel in titles (e.g., 'u' for 'vx').")
    parser.add_argument("--unit-label", type=str, default="", help="Unit label to display on the color bar (e.g., 'm/s').")
    parser.add_argument("--vmin", type=float, default=None, help="Override for the minimum value of the color scale.")
    parser.add_argument("--vmax", type=float, default=None, help="Override for the maximum value of the color scale.")
    parser.add_argument("--interp-y", type=int, default=1024, help="Number of interpolation points for the y-axis.")
    parser.add_argument("--interp-z", type=int, default=1024, help="Number of interpolation points for the z-axis.")

    args = parser.parse_args()
    
    # --- Main Logic ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    time_indices = range(args.time_start, args.time_end)
    snapshot_files = [args.snapshot_dir / f"{args.file_prefix}{i:06d}" for i in time_indices]

    if not all(f.exists() for f in snapshot_files):
        logging.error("Not all snapshot files were found. Aborting.")
        return

    # Load coordinates from the first file
    input_dtype = np.float64
    with open(snapshot_files[0], 'rb') as f:
        raw_coords = {
            'x': np.fromfile(f, dtype=input_dtype, count=args.nx),
            'y': np.fromfile(f, dtype=input_dtype, count=args.ny),
            'z': np.fromfile(f, dtype=input_dtype, count=args.nz),
            'labels': args.source_channels
        }

    try:
        channel_idx = raw_coords['labels'].index(args.channel)
    except ValueError:
        logging.error(f"Channel '{args.channel}' not found in source labels. Aborting.")
        return

    # Memory-efficient data extraction
    logging.info(f"Extracting 1D data lines for plot type '{args.plot_type}'...")
    all_lines = []
    for f_path in tqdm(snapshot_files, desc="Processing snapshots"):
        with open(f_path, 'rb') as f:
            f.seek((args.nx + args.ny + args.nz) * np.dtype(input_dtype).itemsize)
            channel_data_1d = np.fromfile(f, dtype=input_dtype)
        
        data_4d_physical = channel_data_1d.reshape((args.nz, len(args.source_channels), args.ny, args.nx))
        snapshot_3d = data_4d_physical.transpose(3, 2, 0, 1) # -> (x, y, z, chan)

        if args.plot_type == 'time-z':
            x_idx, y_idx = args.slice_indices
            line_1d = snapshot_3d[x_idx, y_idx, :, channel_idx]
        elif args.plot_type == 'time-y':
            x_idx, z_idx = args.slice_indices
            line_1d = snapshot_3d[x_idx, :, z_idx, channel_idx]
        elif args.plot_type == 'time-x':
            y_idx, z_idx = args.slice_indices
            line_1d = snapshot_3d[:, y_idx, z_idx, channel_idx]
        all_lines.append(line_1d)

    time_evolution_data = np.stack(all_lines, axis=0).astype(np.float32)

    # Determine color scale
    vmin, vmax = args.vmin, args.vmax
    if vmin is None or vmax is None:
        vmin, vmax = time_evolution_data.min(), time_evolution_data.max()
        logging.info(f"Auto-calculated color scale: [{vmin:.3f}, {vmax:.3f}]")

    # Generate the plot
    logging.info("Generating final plot...")
    display_name = args.channel_alias if args.channel_alias else args.channel
    output_filename = f"{args.plot_type}_{display_name}.png"
    
    if args.plot_type == 'time-z':
        x_idx, y_idx = args.slice_indices
        plot_interpolated_z_time_evolution(
            time_evolution_data=time_evolution_data, raw_coords=raw_coords,
            channel=args.channel, x_index=x_idx, y_index=y_idx,
            num_interp_points_z=args.interp_z, channel_alias=args.channel_alias,
            vmin=vmin, vmax=vmax, unit_label=args.unit_label,
            output_path=args.output_dir / output_filename
        )
    elif args.plot_type == 'time-y':
        x_idx, z_idx = args.slice_indices
        plot_interpolated_y_time_evolution(
            time_evolution_data=time_evolution_data, raw_coords=raw_coords,
            channel=args.channel, x_index=x_idx, z_index=z_idx,
            num_interp_points_y=args.interp_y, channel_alias=args.channel_alias,
            vmin=vmin, vmax=vmax, unit_label=args.unit_label,
            output_path=args.output_dir / output_filename
        )
    elif args.plot_type == 'time-x':
        y_idx, z_idx = args.slice_indices
        plot_interpolated_x_time_evolution(
            time_evolution_data=time_evolution_data, raw_coords=raw_coords,
            channel=args.channel, y_index=y_idx, z_index=z_idx,
            channel_alias=args.channel_alias,
            vmin=vmin, vmax=vmax, unit_label=args.unit_label,
            output_path=args.output_dir / output_filename
        )
        
    logging.info(f"Plot saved to {args.output_dir / output_filename}")


if __name__ == "__main__":
    main()

"""
#### How to Use It

1.  Place this script inside your `scripts` directory.
2.  From your terminal, navigate **inside the `scripts` directory** and run a command.
    Note that `--slice-indices` expects two values corresponding to the axes that are held constant.

**Example for a Time-Z plot:**
(Hold X and Y constant, plot Time vs. Z)

```bash
python create_time_evolution_plots.py \
    --snapshot-dir /path/to/your/raw/snapshots/ \
    --output-dir output/time_evolution_plots/ \
    --nx 2301 --ny 481 --nz 121 \
    --time-start 609 --time-end 737 \
    --plot-type time-z \
    --slice-indices 1150 240 \
    --channel vx \
    --channel-alias u \
    --vmin -5 --vmax 6
```

**Example for a Time-X plot:**
(Hold Y and Z constant, plot Time vs. X)

```bash
python create_time_evolution_plots.py \
    --snapshot-dir /raid/skowronek/preprocessed_dns_output/01-Cold_Runs/01-Re16K_Ha325/raw/ \
    --output-dir output/figures/ \
    --nx 2301 --ny 481 --nz 121 \
    --time-start 609 --time-end 625 \
    --plot-type time-x \
    --slice-indices 240 60 \
    --channel vy \
    --channel-alias v \
    --vmin -5 --vmax 6
```
"""

