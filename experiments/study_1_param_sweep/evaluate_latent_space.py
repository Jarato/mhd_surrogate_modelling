# -*- coding: utf-8 -*-
# experiments/study_1_param_sweep/evaluate_latent_space.py

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from mhd_canonical_kae.model import KoopmanAutoencoder
# We no longer need to import MHDDataset for this script

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model's latent space dynamics.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model checkpoint (.pth file).")
    parser.add_argument("--test-data-path", type=str, required=True, help="Path to the contiguous test_set.npz file.")
    parser.add_argument("--norm-stats-path", type=str, required=True, help="Path to the normalization_stats.npz file.")
    parser.add_argument("--output-path", type=str, default=None, help="Full path to save latent space evaluation results (.npz file).")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model_path = Path(args.model_path)
    
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = model_path.parent / "eval" / "latent_rollout_error.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load Model from Checkpoint ---
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config']
    channels_used = checkpoint.get('channels_used')
    
    logging.info(f"Re-creating model with saved config: {model_config}")
    model = KoopmanAutoencoder(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- THE FIX IS HERE (Memory-Efficient Data Handling) ---

    # --- Load Normalization Stats and Data Handles ---
    stats = np.load(args.norm_stats_path)
    all_min_vals = torch.from_numpy(stats['min_vals']).float()
    all_max_vals = torch.from_numpy(stats['max_vals']).float()

    data_handle = np.load(args.test_data_path, allow_pickle=True)
    full_timeseries = data_handle["timeseries"]
    all_channel_names = list(data_handle["labels"])

    # --- Select Channels and Stats without loading full data into RAM ---
    if channels_used:
        channel_indices = [all_channel_names.index(name) for name in channels_used]
        min_vals = all_min_vals[channel_indices].to(device)
        max_vals = all_max_vals[channel_indices].to(device)
        logging.info(f"Evaluating on selected channels: {channels_used}")
    else:
        channel_indices = None
        min_vals = all_min_vals.to(device)
        max_vals = all_max_vals.to(device)

    data_range = max_vals - min_vals + 1e-8
    def normalize(x): return (x.to(device) - min_vals) / data_range * 2.0 - 1.0

    # --- Manually encode the entire test set to get the ground truth latent trajectory ---
    true_latent_trajectory = []
    with torch.no_grad():
        for i in tqdm(range(full_timeseries.shape[0]), desc="Encoding Ground Truth"):
            snapshot = full_timeseries[i]
            if channel_indices:
                snapshot = snapshot[..., channel_indices]
            
            snapshot_tensor = torch.from_numpy(snapshot).float()
            snapshot_norm = normalize(snapshot_tensor)
            snapshot_norm = snapshot_norm.unsqueeze(0).permute(0, 4, 1, 2, 3) # Add batch and permute
            
            latent_vector = model.encode(snapshot_norm)
            true_latent_trajectory.append(latent_vector.squeeze(0).cpu())
    
    true_latent_trajectory = torch.stack(true_latent_trajectory)
    logging.info(f"Encoded ground truth trajectory. Latent shape: {true_latent_trajectory.shape}")


    # --- Autoregressive Rollout in Latent Space ---
    num_timesteps = true_latent_trajectory.shape[0]
    predicted_latent_trajectory = []
    
    z_t = true_latent_trajectory[0].unsqueeze(0).to(device)
    
    logging.info(f"Starting latent space rollout for {num_timesteps - 1} steps...")
    with torch.no_grad():
        predicted_latent_trajectory.append(z_t.squeeze(0).cpu())

        for _ in tqdm(range(num_timesteps - 1), desc="Latent Rollout"):
            z_t = model.koopman_step(z_t)
            predicted_latent_trajectory.append(z_t.squeeze(0).cpu())

    predicted_latent_trajectory = torch.stack(predicted_latent_trajectory)
    logging.info(f"Rollout complete. Predicted latent shape: {predicted_latent_trajectory.shape}")

    # --- Calculate Per-Timestep Error ---
    loss_fn = nn.MSELoss(reduction='none')
    per_step_latent_error = []
    for t in range(1, num_timesteps):
        error = loss_fn(predicted_latent_trajectory[t], true_latent_trajectory[t]).mean().item()
        per_step_latent_error.append(error)
    
    np.savez(output_path, per_step_latent_error=np.array(per_step_latent_error))
    logging.info(f"Per-timestep latent space error saved to {output_path}")

    # --- Calculate Final Average Error and R-squared Scores ---
    logging.info("--- LATENT SPACE EVALUATION COMPLETE ---")
    avg_rollout_mse = np.mean(per_step_latent_error)
    variance_latent = true_latent_trajectory.var().item()
    r_squared_latent = 1 - (avg_rollout_mse / variance_latent) if variance_latent > 0 else 0.0
    
    logging.info(f"Average Latent Space Rollout MSE: {avg_rollout_mse:.6f}")
    logging.info(f"Latent Space R-squared (R²) Score: {r_squared_latent:.4f}")


if __name__ == "__main__":
    main()
