# -*- coding: utf-8 -*-
# experiments/study_1_param_sweep/evaluate_prediction.py

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
    parser = argparse.ArgumentParser(description="Evaluate a trained Koopman Autoencoder's prediction performance.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model checkpoint (.pth file).")
    parser.add_argument("--test-data-path", type=str, required=True, help="Path to the contiguous test_set.npz file.")
    parser.add_argument("--norm-stats-path", type=str, required=True, help="Path to the normalization_stats.npz file.")
    parser.add_argument("--output-path", type=str, default=None, help="Full path to save evaluation results (.npz file). Defaults to 'eval/prediction_rollout_error.npz' in the model's directory.")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model_path = Path(args.model_path)
    
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = model_path.parent / "eval" / "prediction_rollout_error.npz"
    
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

    test_tensor_cpu = torch.from_numpy(full_timeseries[..., channel_indices] if channels_used else full_timeseries).float()
    logging.info(f"Test data loaded. Shape: {test_tensor_cpu.shape}")

    # --- Autoregressive Rollout (Memory-Efficient) ---
    num_timesteps, num_channels = test_tensor_cpu.shape[0], test_tensor_cpu.shape[-1]
    per_step_channel_error = np.zeros((num_timesteps - 1, num_channels))
    loss_fn = nn.MSELoss(reduction='none')
    
    initial_state_original = test_tensor_cpu[0].unsqueeze(0)
    current_state_norm = normalize(initial_state_original).permute(0, 4, 1, 2, 3)

    logging.info(f"Starting autoregressive rollout for {num_timesteps - 1} steps...")
    with torch.no_grad():
        z_t = model.encode(current_state_norm)
        for t in tqdm(range(num_timesteps - 1), desc="Evaluating Rollout"):
            z_t = model.koopman_step(z_t)
            predicted_state_norm = model.decode(z_t)
            
            predicted_state_denorm = denormalize(predicted_state_norm.permute(0, 2, 3, 4, 1))
            
            ground_truth_state = test_tensor_cpu[t+1].unsqueeze(0).to(device)
            
            error_tensor = loss_fn(predicted_state_denorm, ground_truth_state)
            per_channel_mse = error_tensor.mean(dim=(0, 1, 2, 3)).cpu().numpy()
            per_step_channel_error[t, :] = per_channel_mse
            
            current_state_norm = predicted_state_norm
    
    logging.info("Rollout complete.")

    # --- Calculate Final Metrics ---
    avg_rollout_mse_total = per_step_channel_error.mean()
    variance_total = test_tensor_cpu.var().item()
    r_squared_total = 1 - (avg_rollout_mse_total / variance_total) if variance_total > 0 else 0.0
    
    avg_mse_per_channel = per_step_channel_error.mean(axis=0)
    r_squared_per_channel = np.zeros(num_channels)
    for i in range(num_channels):
        variance_channel = test_tensor_cpu[..., i].var().item()
        r_squared_per_channel[i] = 1 - (avg_mse_per_channel[i] / variance_channel) if variance_channel > 0 else 0.0

    # --- Save All Results ---
    np.savez(
        output_path,
        per_step_channel_error=per_step_channel_error,
        channel_names=channel_names,
        avg_rollout_mse_total=avg_rollout_mse_total,
        r_squared_total=r_squared_total,
        avg_mse_per_channel=avg_mse_per_channel,
        r_squared_per_channel=r_squared_per_channel
    )
    logging.info(f"All evaluation results saved to {output_path}")

    # --- Log Final Metrics ---
    logging.info("--- PREDICTION EVALUATION COMPLETE ---")
    logging.info(f"Overall Average Rollout MSE: {avg_rollout_mse_total:.6f}")
    logging.info(f"Overall R-squared (R²) Score: {r_squared_total:.4f}\n")
    logging.info("--- Per-Channel Metrics ---")
    for i, name in enumerate(channel_names):
        logging.info(f"Channel '{name}':\t Avg MSE = {avg_mse_per_channel[i]:.6f},\t R² = {r_squared_per_channel[i]:.4f}")

if __name__ == "__main__":
    main()
