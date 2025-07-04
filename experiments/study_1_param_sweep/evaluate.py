# -*- coding: utf-8 -*-
# experiments/study_1_param_sweep/evaluate.py

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mhd_canonical_kae.model import KoopmanAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Koopman Autoencoder.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model checkpoint (.pth file).")
    parser.add_argument("--test-data-path", type=str, required=True, help="Path to the contiguous test_set.npz file.")
    parser.add_argument("--norm-stats-path", type=str, required=True, help="Path to the normalization_stats.npz file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation results. Defaults to a new 'eval' folder in the model's directory.")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model_path = Path(args.model_path)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)


    # --- Load Normalization Stats ---
    stats = np.load(args.norm_stats_path)
    min_vals = torch.from_numpy(stats['min_vals']).float().to(device)
    max_vals = torch.from_numpy(stats['max_vals']).float().to(device)
    data_range = max_vals - min_vals + 1e-8

    def normalize(x):
        return (x.to(device) - min_vals) / data_range * 2.0 - 1.0

    def denormalize(x_norm):
        return (x_norm + 1.0) / 2.0 * data_range + min_vals

    # --- Load Test Data ---
    with np.load(args.test_data_path) as raw_data:
        test_timeseries = raw_data["timeseries"]
    
    test_tensor_cpu = torch.from_numpy(test_timeseries).float()
    logging.info(f"Test data loaded. Shape: {test_tensor_cpu.shape}")

    # --- Load Model from Checkpoint ---
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config']
    
    logging.info(f"Re-creating model with saved config: {model_config}")
    model = KoopmanAutoencoder(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- Autoregressive Rollout ---
    num_timesteps = test_tensor_cpu.shape[0]
    predictions_denorm_cpu = []
    
    initial_state_original = test_tensor_cpu[0].unsqueeze(0)
    current_state_norm = normalize(initial_state_original)
    current_state_norm = current_state_norm.permute(0, 4, 1, 2, 3)

    logging.info(f"Starting autoregressive rollout for {num_timesteps - 1} steps...")
    with torch.no_grad():
        predictions_denorm_cpu.append(initial_state_original.squeeze(0))
        z_t = model.encode(current_state_norm)

        for _ in range(num_timesteps - 1):
            z_t = model.koopman_step(z_t)
            predicted_state_norm = model.decode(z_t)
            predicted_state_denorm = denormalize(predicted_state_norm.permute(0, 2, 3, 4, 1))
            predictions_denorm_cpu.append(predicted_state_denorm.squeeze(0).cpu())
            current_state_norm = predicted_state_norm

    predicted_timeseries = torch.stack(predictions_denorm_cpu)
    logging.info(f"Rollout complete. Predicted timeseries shape: {predicted_timeseries.shape}")

    # --- Calculate Per-Timestep Error ---
    loss_fn = nn.MSELoss(reduction='none') 
    per_step_error = []
    for t in range(1, num_timesteps):
        error = loss_fn(predicted_timeseries[t], test_tensor_cpu[t]).mean().item()
        per_step_error.append(error)
    
    rollout_error_path = output_dir / "rollout_error.npz"
    np.savez(rollout_error_path, per_step_error=np.array(per_step_error))
    logging.info(f"Per-timestep rollout error saved to {rollout_error_path}")

    # --- Calculate Final Average Error and R-squared Score ---
    avg_rollout_mse = np.mean(per_step_error)
    
    # Calculate the variance of the ground truth test data
    # We calculate it on the CPU to avoid moving the whole tensor to the GPU if not needed
    variance_of_data = torch.var(test_tensor_cpu).item()
    
    # Calculate R-squared score: 1 - (MSE / Variance)
    # This score is meaningful only if variance is not zero
    r_squared_score = 1 - (avg_rollout_mse / variance_of_data) if variance_of_data > 0 else 0.0


    logging.info("--- EVALUATION COMPLETE ---")
    logging.info(f"Average Autoregressive Rollout MSE: {avg_rollout_mse:.6f}")
    logging.info(f"Variance of Test Data: {variance_of_data:.6f}")
    logging.info(f"R-squared (R²) Score: {r_squared_score:.4f}")

if __name__ == "__main__":
    main()
