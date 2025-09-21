# -*- coding: utf-8 -*-
# experiments/study_4_2d_tckae/gen_latent_prediction.py

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from mhd_2d_tckae.model import tcKoopmanAutoencoder2D

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained 2D tcKAE's latent space dynamics.")
    
    # --- Input Paths ---
    io_group = parser.add_argument_group("Input Paths")
    io_group.add_argument("--model-path", type=str, required=True, help="Path to the saved best_model.pth file.")
    io_group.add_argument("--test-data-path", type=str, required=True, help="Path to the contiguous test_set.npz file.")
    io_group.add_argument("--norm-stats-path", type=str, required=True, help="Path to the normalization_stats.npz file.")

    # --- Output Path ---
    out_group = parser.add_argument_group("Output Path")
    out_group.add_argument("--output-path", type=str, default=None, help="Full path to save latent space evaluation results (.npz file). Defaults to 'eval/latent_space_analysis.npz' in the model's directory.")
    
    return parser.parse_args()

def main():
    """Main function to generate latent space predictions and analyze them."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model_path = Path(args.model_path)
    
    # --- Setup Output Path ---
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = model_path.parent / "eval" / "latent_space_analysis.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load Model from Checkpoint ---
    logging.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config']
    channels_used = checkpoint.get('channels_used')
    
    logging.info(f"Re-creating model with saved config: {model_config}")
    model = tcKoopmanAutoencoder2D(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- Load Data and Stats ---
    logging.info(f"Loading normalization stats from {args.norm_stats_path}")
    stats = np.load(args.norm_stats_path)
    all_min_vals = torch.from_numpy(stats['min_vals']).float().to(device)
    all_max_vals = torch.from_numpy(stats['max_vals']).float().to(device)
    
    logging.info(f"Loading test data from {args.test_data_path}")
    data_handle = np.load(args.test_data_path, allow_pickle=True)
    full_timeseries = data_handle["timeseries"]
    all_channel_names = list(data_handle["labels"])

    if full_timeseries.ndim == 5 and full_timeseries.shape[2] == 1:
        full_timeseries = np.squeeze(full_timeseries, axis=2)
        logging.info(f"Squeezed test data to 4D. New shape: {full_timeseries.shape}")

    if channels_used:
        channel_indices = [all_channel_names.index(name) for name in channels_used]
        min_vals = all_min_vals[channel_indices]
        max_vals = all_max_vals[channel_indices]
        test_data_np = full_timeseries[..., channel_indices]
    else:
        min_vals = all_min_vals
        max_vals = all_max_vals
        test_data_np = full_timeseries

    min_vals, max_vals = min_vals.view(1, -1, 1, 1), max_vals.view(1, -1, 1, 1)
    data_range = max_vals - min_vals + 1e-8
    def normalize(x): return (x - min_vals) / data_range * 2.0 - 1.0

    # --- Encode the Entire Ground Truth Trajectory ---
    true_latent_trajectory = []
    with torch.no_grad():
        for i in tqdm(range(test_data_np.shape[0]), desc="Encoding Ground Truth"):
            snapshot_np = test_data_np[i]
            snapshot_tensor = torch.from_numpy(snapshot_np).float().to(device)
            # Permute from (X, Z, C) -> (C, X, Z) for the model
            snapshot_permuted = snapshot_tensor.permute(2, 0, 1)
            
            # Normalize and add batch dimension
            snapshot_norm = normalize(snapshot_permuted.unsqueeze(0))
            
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
        
        # Calculate the magnitude of each eigenvector
        eigenvector_magnitudes = torch.linalg.norm(eigenvectors, ord=2, dim=0).numpy()
        
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
        eigenvector_magnitudes=eigenvector_magnitudes,
        true_projected_trajectory=true_projected_traj.numpy() if true_projected_traj is not None else None,
        pred_projected_trajectory=pred_projected_traj.numpy() if pred_projected_traj is not None else None,
        initial_mode_amplitudes=initial_mode_amplitudes if initial_mode_amplitudes is not None else None,
        avg_rollout_mse=avg_rollout_mse,
        r_squared_latent=r_squared_latent,
    )
    logging.info(f"Latent space analysis results saved to {output_path}")

    # --- Log Final Metrics ---
    logging.info("--- LATENT SPACE EVALUATION COMPLETE ---")
    logging.info(f"Average Latent Space Rollout MSE: {avg_rollout_mse:.6e}")
    logging.info(f"Latent Space R-squared (R²) Score: {r_squared_latent:.4f}")

if __name__ == "__main__":
    main()

