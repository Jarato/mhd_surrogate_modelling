# -*- coding: utf-8 -*-
# experiments/study_1_param_sweep/visualize_prediction.py

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from mhd_canonical_kae.model import KoopmanAutoencoder
from mhd_surrogate_core.plot import plot_prediction_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize model predictions for specific snapshots during a rollout.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model checkpoint (.pth file).")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the test_set.npz or train_val_set.npz file.")
    parser.add_argument("--norm-stats-path", type=str, required=True, help="Path to the normalization_stats.npz file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the output plot. Defaults to a 'visualize_prediction' subfolder in the model's 'eval' directory.")
    parser.add_argument("--timesteps", type=int, nargs='+', default=None, help="List of specific timesteps to visualize (e.g., 1 10 20).")
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
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_path.parent / "eval" / "visualize_prediction" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Model, Data, and Stats ---
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config']
    channels_used = checkpoint.get('channels_used')
    model = KoopmanAutoencoder(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    stats = np.load(args.norm_stats_path)
    all_min_vals = torch.from_numpy(stats['min_vals']).float().to(device)
    all_max_vals = torch.from_numpy(stats['max_vals']).float().to(device)
    
    data_handle = np.load(args.data_path, allow_pickle=True)
    full_timeseries = data_handle["timeseries"]
    all_channel_names = list(data_handle["labels"])

    if channels_used:
        channel_indices = [all_channel_names.index(name) for name in channels_used]
        min_vals = all_min_vals[channel_indices]
        max_vals = all_max_vals[channel_indices]
    else:
        # --- THE FIX IS HERE ---
        # Assign both min_vals and max_vals in the else branch
        min_vals = all_min_vals
        max_vals = all_max_vals
    
    data_range = max_vals - min_vals + 1e-8
    def normalize(x): return (x.to(device) - min_vals) / data_range * 2.0 - 1.0
    def denormalize(x_norm): return (x_norm + 1.0) / 2.0 * data_range + min_vals

    # --- Determine which timesteps to visualize ---
    if args.all_timesteps:
        timesteps_to_visualize = set(range(1, full_timeseries.shape[0]))
    else:
        timesteps_to_visualize = set(args.timesteps)

    # --- Perform Autoregressive Rollout and Visualize ---
    with torch.no_grad():
        # Get initial condition
        initial_state_original = torch.from_numpy(full_timeseries[0]).float()
        if channels_used:
            initial_state_original = initial_state_original[..., channel_indices]
        
        current_state_norm = normalize(initial_state_original)
        current_state_norm = current_state_norm.unsqueeze(0).permute(0, 4, 1, 2, 3)
        z_t = model.encode(current_state_norm)

        logging.info(f"Starting autoregressive rollout...")
        max_t = max(timesteps_to_visualize)
        for t in range(1, max_t + 1):
            z_t = model.koopman_step(z_t)

            if t in timesteps_to_visualize:
                logging.info(f"Visualizing prediction for timestep {t}...")
                predicted_state_norm = model.decode(z_t)
                predicted_state_denorm = denormalize(predicted_state_norm.permute(0, 2, 3, 4, 1)).squeeze(0)

                # Get ground truth for comparison
                true_state_original = torch.from_numpy(full_timeseries[t]).float()
                if channels_used:
                    true_state_original = true_state_original[..., channel_indices]

                # Select the specific channel for plotting
                try:
                    channel_idx_in_subset = channels_used.index(args.channel) if channels_used else all_channel_names.index(args.channel)
                except ValueError:
                    logging.error(f"Channel '{args.channel}' not found. Aborting visualization.")
                    return

                true_slice = true_state_original[..., channel_idx_in_subset].cpu().numpy()
                pred_slice = predicted_state_denorm[..., channel_idx_in_subset].cpu().numpy()
                
                fig = plot_prediction_comparison(true_slice, pred_slice, args.channel, t)
                
                plot_path = output_dir / f"prediction_t{t}_channel_{args.channel}.png"
                fig.savefig(plot_path, dpi=150)
                logging.info(f"Saved plot to {plot_path}")
                plt.close(fig)

if __name__ == "__main__":
    main()
