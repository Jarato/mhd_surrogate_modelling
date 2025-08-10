# -*- coding: utf-8 -*-
# experiments/study_2_q2d_model/evaluate_latent_space.py

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# --- THE FIX IS HERE (Part 1: Import the correct model) ---
from mhd_q2d_kae.model import KoopmanAutoencoderQ2D

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Q2D model's latent space dynamics.")
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
        output_path = model_path.parent / "eval" / "latent_space_analysis.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load Model from Checkpoint ---
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config']
    channels_used = checkpoint.get('channels_used')
    
    logging.info(f"Re-creating model with saved config: {model_config}")
    # --- THE FIX IS HERE (Part 2: Instantiate the correct model) ---
    model = KoopmanAutoencoderQ2D(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- Load Data and Stats ---
    stats = np.load(args.norm_stats_path)
    all_min_vals = torch.from_numpy(stats['min_vals']).float()
    all_max_vals = torch.from_numpy(stats['max_vals']).float()
    
    data_handle = np.load(args.test_data_path, allow_pickle=True)
    full_timeseries = data_handle["timeseries"]
    all_channel_names = list(data_handle["labels"])

    if channels_used:
        channel_indices = [all_channel_names.index(name) for name in channels_used]
        min_vals = all_min_vals[channel_indices]
        max_vals = all_max_vals[channel_indices]
    else:
        min_vals = all_min_vals
        max_vals = all_max_vals

    data_range = max_vals - min_vals + 1e-8
    def normalize(x): return (x - min_vals) / data_range * 2.0 - 1.0

    # --- Encode Ground Truth Trajectory ---
    true_latent_trajectory = []
    with torch.no_grad():
        for i in tqdm(range(full_timeseries.shape[0]), desc="Encoding Ground Truth"):
            snapshot = full_timeseries[i]
            if channels_used:
                snapshot = snapshot[..., channel_indices]
            
            snapshot_tensor = torch.from_numpy(snapshot).float()
            snapshot_norm = normalize(snapshot_tensor)
            # The Q2D model expects (B, C, X, Y, Z), so we permute from (D, H, W, C) -> (C, D, H, W)
            snapshot_norm = snapshot_norm.unsqueeze(0).permute(0, 4, 1, 2, 3).to(device)
            
            latent_vector = model.encode(snapshot_norm)
            true_latent_trajectory.append(latent_vector.squeeze(0).cpu())
    
    true_latent_trajectory = torch.stack(true_latent_trajectory)
    logging.info(f"Encoded ground truth trajectory. Latent shape: {true_latent_trajectory.shape}")

    # --- Autoregressive Rollout in Latent Space ---
    num_timesteps = true_latent_trajectory.shape[0]
    predicted_latent_trajectory = []
    
    z_t = true_latent_trajectory[0].unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_latent_trajectory.append(z_t.squeeze(0).cpu())
        for _ in tqdm(range(num_timesteps - 1), desc="Latent Rollout"):
            z_t = model.koopman_step(z_t)
            predicted_latent_trajectory.append(z_t.squeeze(0).cpu())
    predicted_latent_trajectory = torch.stack(predicted_latent_trajectory)

    # --- Eigendecomposition and Projection ---
    logging.info("Performing eigendecomposition of the Koopman matrix...")
    with torch.no_grad():
        K = model.koopman_operator.weight.cpu()
        eigenvalues, eigenvectors = torch.linalg.eig(K)
        
        try:
            W_inv = torch.linalg.inv(eigenvectors)
            true_projected_traj = (W_inv @ true_latent_trajectory.cfloat().T).T
            pred_projected_traj = (W_inv @ predicted_latent_trajectory.cfloat().T).T
            initial_mode_amplitudes = np.abs(true_projected_traj[0].numpy())
        except torch.linalg.LinAlgError:
            logging.error("Eigenvector matrix is singular; cannot perform projection.")
            true_projected_traj, pred_projected_traj, initial_mode_amplitudes = None, None, None

    # --- Calculate Per-Timestep Error ---
    loss_fn = nn.MSELoss(reduction='none')
    per_step_latent_error = []
    for t in range(1, num_timesteps):
        error = loss_fn(predicted_latent_trajectory[t], true_latent_trajectory[t]).mean().item()
        per_step_latent_error.append(error)

    # --- Calculate Final Metrics ---
    avg_rollout_mse = np.mean(per_step_latent_error)
    variance_latent = true_latent_trajectory.var().item()
    r_squared_latent = 1 - (avg_rollout_mse / variance_latent) if variance_latent > 0 else 0.0
    
    # --- Save All Results ---
    np.savez(
        output_path, 
        per_step_latent_error=np.array(per_step_latent_error),
        true_latent_trajectory=true_latent_trajectory.numpy(),
        predicted_latent_trajectory=predicted_latent_trajectory.numpy(),
        eigenvalues=eigenvalues.numpy(),
        true_projected_trajectory=true_projected_traj.numpy() if true_projected_traj is not None else None,
        pred_projected_trajectory=pred_projected_traj.numpy() if pred_projected_traj is not None else None,
        initial_mode_amplitudes=initial_mode_amplitudes if initial_mode_amplitudes is not None else None,
        avg_rollout_mse=avg_rollout_mse,
        r_squared_latent=r_squared_latent,
    )
    logging.info(f"Latent space analysis results saved to {output_path}")

    # --- Log Final Metrics ---
    logging.info("--- LATENT SPACE EVALUATION COMPLETE ---")
    logging.info(f"Average Latent Space Rollout MSE: {avg_rollout_mse:.6f}")
    logging.info(f"Latent Space R-squared (R²) Score: {r_squared_latent:.4f}")


if __name__ == "__main__":
    main()
