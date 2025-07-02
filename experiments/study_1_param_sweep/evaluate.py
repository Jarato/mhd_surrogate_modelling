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
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

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
    
    # Keep the ground truth on the CPU
    test_tensor_cpu = torch.from_numpy(test_timeseries).float()
    logging.info(f"Test data loaded. Shape: {test_tensor_cpu.shape}")

    # --- Load Model from Checkpoint ---
    checkpoint = torch.load(args.model_path, map_location=device)
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
        # The first "prediction" is the ground truth initial state (already on CPU)
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

    # --- Calculate Final Error ---
    # Compare the CPU prediction tensor with the CPU ground truth tensor
    loss_fn = nn.MSELoss()
    rollout_error = loss_fn(predicted_timeseries, test_tensor_cpu)

    logging.info("--- EVALUATION COMPLETE ---")
    logging.info(f"Autoregressive Rollout MSE: {rollout_error.item():.6f}")

if __name__ == "__main__":
    main()
