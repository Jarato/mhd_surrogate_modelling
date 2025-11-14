# -*- coding: utf-8 -*-
# experiments/study_6_flow_matching_2d/study_6.1_flow_matching_2d_base/evaluation/prediction/create_comparison_video.py

import argparse
from pathlib import Path
import multiprocessing
import logging

# Assuming mhd_surrogate_core is available in your environment
from mhd_surrogate_core.plotting.xz import generate_comparison_video_from_npz, generate_multi_sample_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    """Main function to generate a 3-panel comparison video."""
    parser = argparse.ArgumentParser(
        description="Generate a 3-panel comparison video (Ground Truth, Prediction, Difference) from .npz files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- I/O Arguments ---
    parser.add_argument("--ground-truth-npz", type=Path, required=True, help="Path to the ground truth timeseries .npz file.")
    parser.add_argument("--predicted-npz", type=Path, required=False, help="Path to the predicted timeseries .npz file (for 'individual' mode).")
    parser.add_argument("--difference-npz", type=Path, required=False, help="Path to the difference timeseries .npz file (for 'individual' mode).")
    parser.add_argument("--output-path", type=Path, required=True, help="Path to save the output .mp4 video file.")
    
    # --- Mode Selection ---
    parser.add_argument("--mode", type=str, default="individual", choices=["individual", "combined"], help="Video generation mode.")
    parser.add_argument("--base-pred-dir", type=Path, default=None, help="Base directory containing 'sample_XX' subfolders (for 'combined' mode).")

    # --- Data Selection ---
    parser.add_argument("--channel", type=str, required=True, help="The channel to plot (e.g., 'vx').")
    parser.add_argument("--time-start", type=int, default=None, help="Optional: Starting time index.")
    parser.add_argument("--time-end", type=int, default=None, help="Optional: Ending time index.")

    # --- Video & Plotting ---
    parser.add_argument("--fps", type=int, default=15, help="Frames per second.")
    parser.add_argument("--channel-alias", type=str, default=None, help="Display name for the channel.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers. -1 to use all cores.")
    parser.add_argument("--base-size", type=float, default=8.0, help="Base size (in inches) for the plot's width.")
    parser.add_argument("--min-size", type=float, default=3.0, help="Minimum size (in inches) for a single plot's height.")

    # --- Color Scales for Main Plots ---
    main_map = parser.add_argument_group('Color Scale (Main Plots)')
    main_map.add_argument("--cmap", type=str, default="seismic", help="Colormap for Ground Truth and Prediction.")
    main_map.add_argument("--vmins", type=str, nargs='+', default=None, help="Min values. Format: 'vx:-5' 'vz:-1'")
    main_map.add_argument("--vmaxs", type=str, nargs='+', default=None, help="Max values. Format: 'vx:6' 'vz:1'")
    main_map.add_argument("--vcenters", type=str, nargs='+', default=None, help="Center values. Format: 'vx:0.84'")
    
    # --- Color Scales for Difference Plot ---
    diff_map = parser.add_argument_group('Color Scale (Difference Plot)')
    diff_map.add_argument("--cmap-diff", type=str, default="bwr", help="Colormap for the Difference plot.")
    diff_map.add_argument("--vmins-diff", type=str, nargs='+', default=None, help="Min error values.")
    diff_map.add_argument("--vmaxs-diff", type=str, nargs='+', default=None, help="Max error values.")
    diff_map.add_argument("--vcenters-diff", type=str, nargs='+', default=None, help="Center error values (usually 0).")

    args = parser.parse_args()
    
    # --- Main Logic ---
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    def parse_channel_value_arg(arg_list):
        return {item.split(':')[0]: float(item.split(':')[1]) for item in arg_list} if arg_list else {}

    vcenter_map = parse_channel_value_arg(args.vcenters)
    vmin_map = parse_channel_value_arg(args.vmins)
    vmax_map = parse_channel_value_arg(args.vmaxs)
    
    vcenter_diff_map = parse_channel_value_arg(args.vcenters_diff)
    vmin_diff_map = parse_channel_value_arg(args.vmins_diff)
    vmax_diff_map = parse_channel_value_arg(args.vmaxs_diff)
    
    num_workers = multiprocessing.cpu_count() if args.num_workers == -1 else args.num_workers

    if args.mode == "individual":
        logging.info("Running in 'individual' mode (3-panel video).")
        if not args.predicted_npz or not args.difference_npz:
            parser.error("--predicted-npz and --difference-npz are required for 'individual' mode.")
        
        generate_comparison_video_from_npz(
            gt_npz_path=args.ground_truth_npz,
            pred_npz_path=args.predicted_npz,
            diff_npz_path=args.difference_npz,
            output_path=args.output_path,
            channel=args.channel,
            time_start=args.time_start,
            time_end=args.time_end,
            fps=args.fps,
            channel_alias=args.channel_alias,
            vmins_override=vmin_map,
            vmaxs_override=vmax_map,
            vcenters=vcenter_map,
            vmins_diff_override=vmin_diff_map,
            vmaxs_diff_override=vmax_diff_map,
            vcenters_diff=vcenter_diff_map,
            num_workers=num_workers,
            cmap=args.cmap,
            cmap_diff=args.cmap_diff,
            base_size=args.base_size,
            min_size=args.min_size,
        )
    
    elif args.mode == "combined":
        logging.info("Running in 'combined' mode (multi-sample video).")
        if not args.base_pred_dir:
            parser.error("--base-pred-dir is required for 'combined' mode.")
        
        # Find all sample prediction files
        pred_npz_paths = sorted(list(args.base_pred_dir.glob("sample_*/predicted_timeseries.npz")))
        if not pred_npz_paths:
            logging.error(f"No 'sample_*/predicted_timeseries.npz' files found under {args.base_pred_dir}. Aborting.")
            return

        logging.info(f"Found {len(pred_npz_paths)} prediction samples.")

        generate_multi_sample_video(
            gt_npz_path=args.ground_truth_npz,
            pred_npz_paths=pred_npz_paths,
            output_path=args.output_path,
            channel=args.channel,
            time_start=args.time_start,
            time_end=args.time_end,
            fps=args.fps,
            channel_alias=args.channel_alias,
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