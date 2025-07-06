# -*- coding: utf-8 -*-
# experiments/study_1_param_sweep/evaluate_reconstruction.py

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from mhd_canonical_kae.model import KoopmanAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model's reconstruction performance.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model checkpoint (.pth file).")
    parser.add_argument("--test-data-path", type=str, required=True, help="Path to the contiguous test_set.npz file.")
    parser.add_argument("--norm-stats-path", type=str, required=True, help="Path to the normalization_stats.npz file.")
    parser.add_argument("--output-path", type=str, default=None, help="Full path to save reconstruction results (.npz file).")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model_path = Path(args.model_path)
    
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = model_path.parent / "eval" / "reconstruction_analysis.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load Model from Checkpoint ---
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config']
    channels_used = checkpoint.get('channels_used')
    
    logging.info(f"Re-creating model with saved config: {model_config}")
    model = KoopmanAutoencoder(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- Load Data and Stats ---
    stats = np.load(args.norm_stats_path)
    # --- THE FIX IS HERE ---
    # Move normalization stats to the correct device
    all_min_vals = torch.from_numpy(stats['min_vals']).float().to(device)
    all_max_vals = torch.from_numpy(stats['max_vals']).float().to(device)
    
    data_handle = np.load(args.test_data_path, allow_pickle=True)
    full_timeseries = data_handle["timeseries"]
    all_channel_names = list(data_handle["labels"])

    if channels_used:
        channel_indices = [all_channel_names.index(name) for name in channels_used]
        min_vals = all_min_vals[channel_indices]
        max_vals = all_max_vals[channel_indices]
        channel_names = channels_used
    else:
        min_vals = all_min_vals
        max_vals = all_max_vals
        channel_names = all_channel_names

    data_range = max_vals - min_vals + 1e-8
    def normalize(x): return (x.to(device) - min_vals) / data_range * 2.0 - 1.0
    def denormalize(x_norm): return (x_norm + 1.0) / 2.0 * data_range + min_vals

    # --- Evaluate Reconstruction Error ---
    num_timesteps, num_channels = full_timeseries.shape[0], len(channel_names)
    per_channel_recon_error = np.zeros(num_channels)
    loss_fn = nn.MSELoss(reduction='none')
    
    logging.info(f"Evaluating reconstruction for {num_timesteps} snapshots...")
    with torch.no_grad():
        for i in tqdm(range(num_timesteps), desc="Evaluating Reconstruction"):
            snapshot = full_timeseries[i]
            if channels_used:
                snapshot = snapshot[..., channel_indices]
            
            snapshot_tensor = torch.from_numpy(snapshot).float()
            snapshot_norm = normalize(snapshot_tensor)
            snapshot_norm = snapshot_norm.unsqueeze(0).permute(0, 4, 1, 2, 3) # Already on device
            
            latent_vec = model.encode(snapshot_norm)
            recon_norm = model.decode(latent_vec)
            
            recon_denorm = denormalize(recon_norm.permute(0, 2, 3, 4, 1)).squeeze(0)
            
            error_tensor = loss_fn(recon_denorm, snapshot_tensor.to(device))
            per_channel_mse = error_tensor.mean(dim=(0, 1, 2)).cpu().numpy()
            per_channel_recon_error += per_channel_mse

    avg_per_channel_recon_error = per_channel_recon_error / num_timesteps
    
    # --- Calculate R-squared Scores ---
    test_tensor_cpu = torch.from_numpy(full_timeseries[..., channel_indices] if channels_used else full_timeseries).float()
    r_squared_per_channel = np.zeros(num_channels)
    for i in range(num_channels):
        variance_channel = test_tensor_cpu[..., i].var().item()
        r_squared_per_channel[i] = 1 - (avg_per_channel_recon_error[i] / variance_channel) if variance_channel > 0 else 0.0

    avg_total_recon_error = avg_per_channel_recon_error.mean()
    total_variance = test_tensor_cpu.var().item()
    r_squared_total = 1 - (avg_total_recon_error / total_variance) if total_variance > 0 else 0.0

    # --- Save and Log Results ---
    np.savez(
        output_path,
        r_squared_per_channel=r_squared_per_channel,
        r_squared_total=r_squared_total,
        channel_names=channel_names,
    )
    logging.info(f"Reconstruction analysis results saved to {output_path}")

    logging.info("--- RECONSTRUCTION EVALUATION COMPLETE ---")
    logging.info(f"Overall Reconstruction R² Score: {r_squared_total:.4f}\n")
    for i, name in enumerate(channel_names):
        logging.info(f"Channel '{name}':\t R² = {r_squared_per_channel[i]:.4f}")

if __name__ == "__main__":
    main()
