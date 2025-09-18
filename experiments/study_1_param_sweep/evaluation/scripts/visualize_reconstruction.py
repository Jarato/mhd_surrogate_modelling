# -*- coding: utf-8 -*-
# experiments/study_1_param_sweep/visualize_reconstruction.py

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from mhd_canonical_kae.model import KoopmanAutoencoder
from mhd_surrogate_core.plotting.xyz import plot_snapshot_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize model reconstruction for specific snapshots.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model checkpoint (.pth file).")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the test_set.npz or train_val_set.npz file.")
    parser.add_argument("--norm-stats-path", type=str, required=True, help="Path to the normalization_stats.npz file.")
    # --- THE FIX IS HERE (Part 1: Update help text) ---
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the output plot. Defaults to a 'visualize_reconstruction' subfolder in the model's 'eval' directory.")
    parser.add_argument("--timesteps", type=int, nargs='+', default=None, help="List of specific timesteps to visualize (e.g., 0 42 84).")
    parser.add_argument("--all-timesteps", action="store_true", help="Flag to visualize all timesteps in the data file.")
    parser.add_argument("--channel", type=str, required=True, help="Name of the channel to visualize (e.g., 'vx').")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.timesteps and not args.all_timesteps:
        raise ValueError("You must specify either --timesteps or --all-timesteps.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model_path = Path(args.model_path)
    
    # --- THE FIX IS HERE (Part 2: Update default path logic) ---
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default to a more specific subdirectory
        output_dir = model_path.parent / "eval" / "visualize_reconstruction" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Model, Data, and Stats ---
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config']
    model = KoopmanAutoencoder(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    stats = np.load(args.norm_stats_path)
    min_vals = torch.from_numpy(stats['min_vals']).float().to(device)
    max_vals = torch.from_numpy(stats['max_vals']).float().to(device)
    data_range = max_vals - min_vals + 1e-8
    def normalize(x): return (x.to(device) - min_vals) / data_range * 2.0 - 1.0
    def denormalize(x_norm): return (x_norm + 1.0) / 2.0 * data_range + min_vals

    data_handle = np.load(args.data_path, allow_pickle=True)
    full_timeseries = data_handle["timeseries"]
    all_channel_names = list(data_handle["labels"])

    if args.all_timesteps:
        timesteps_to_visualize = range(full_timeseries.shape[0])
        logging.info("Processing all timesteps...")
    else:
        timesteps_to_visualize = args.timesteps
        logging.info(f"Processing specified timesteps: {timesteps_to_visualize}")


    # --- Generate and Save Plots ---
    with torch.no_grad():
        for t in timesteps_to_visualize:
            logging.info(f"Processing timestep {t} for channel '{args.channel}'...")
            
            try:
                channel_idx_in_all = all_channel_names.index(args.channel)
            except ValueError:
                logging.error(f"Channel '{args.channel}' not found in dataset. Available: {all_channel_names}")
                continue

            snapshot_original = torch.from_numpy(full_timeseries[t]).float()
            
            snapshot_norm = normalize(snapshot_original)
            snapshot_norm = snapshot_norm.unsqueeze(0).permute(0, 4, 1, 2, 3)
            
            recon_norm = model.decode(model.encode(snapshot_norm))
            recon_denorm = denormalize(recon_norm.permute(0, 2, 3, 4, 1)).squeeze(0)

            true_slice = snapshot_original[..., channel_idx_in_all].cpu().numpy()
            recon_slice = recon_denorm[..., channel_idx_in_all].cpu().numpy()
            
            fig = plot_snapshot_comparison(true_slice, recon_slice, args.channel, t)
            
            plot_path = output_dir / f"reconstruction_t{t}_channel_{args.channel}.png"
            fig.savefig(plot_path, dpi=150)
            logging.info(f"Saved plot to {plot_path}")
            plt.close(fig)

if __name__ == "__main__":
    main()
