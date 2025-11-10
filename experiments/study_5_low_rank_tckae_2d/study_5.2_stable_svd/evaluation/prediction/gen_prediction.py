# -*- coding: utf-8 -*-
# experiments/study_5_low_rank_tckae_2d/study_5.2_stable_svd/evaluation/prediction/gen_prediction.py
#
# This script is an updated version of gen_prediction_2d_separated.py
#
# --- MODIFICATIONS ---
# 1. Imports the new tcKoopmanAutoencoder2D from the correct SVD model package.
# 2. Changed normalization logic from Min/Max to Mean/Std to match the
#    training script (train.py) and data loader (data_2d.py).
#    It now loads 'mean_vals' and 'std_vals' from the norm_stats_path.
# ---

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from low_rank_tckae_2d_stable_svd.model import tcKoopmanAutoencoder2D


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained 2D tcKAE (SVD Version) prediction performance.")
    
    # --- Input Paths ---
    io_group = parser.add_argument_group("Input Paths")
    io_group.add_argument("--model-path", type=str, required=True, help="Path to the saved best_model.pth file.")
    io_group.add_argument("--test-data-path", type=str, required=True, help="Path to the contiguous test_set.npz file.")
    io_group.add_argument("--norm-stats-path", type=str, required=True, help="Path to the normalization_stats.npz file (must contain 'mean_vals' and 'std_vals').")

    # --- Output Paths ---
    out_group = parser.add_argument_group("Output Paths")
    out_group.add_argument("--output-dir", type=str, required=True, help="A single directory to save all output files.")
    
    return parser.parse_args()

def main():
    """Main function to generate predictions and evaluate the model."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model_path = Path(args.model_path)
    
    # --- Setup Output Paths ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "prediction_stats.npz"
    pred_path = output_dir / "predicted_timeseries.npz"
    diff_path = output_dir / "difference_timeseries.npz"
    logging.info(f"All outputs will be saved to: {output_dir}")

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
    all_mean_vals = torch.from_numpy(stats['mean_vals']).float().to(device)
    all_std_vals = torch.from_numpy(stats['std_vals']).float().to(device)
    
    logging.info(f"Loading test data from {args.test_data_path}")
    data_handle = np.load(args.test_data_path, allow_pickle=True)
    full_timeseries = data_handle["timeseries"]
    all_channel_names = list(data_handle["labels"])

    if full_timeseries.ndim == 5 and full_timeseries.shape[2] == 1:
        full_timeseries = np.squeeze(full_timeseries, axis=2)
        logging.info(f"Squeezed test data to 4D. New shape: {full_timeseries.shape}")

    if channels_used:
        channel_indices = [all_channel_names.index(name) for name in channels_used]
        mean_vals = all_mean_vals[channel_indices]
        std_vals = all_std_vals[channel_indices]
        channel_names = channels_used
        test_data_np = full_timeseries[..., channel_indices]
    else:
        channel_indices = list(range(len(all_channel_names)))
        mean_vals = all_mean_vals
        std_vals = all_std_vals
        channel_names = all_channel_names
        test_data_np = full_timeseries

    # --- MODIFIED: Reshape and define Mean/Std normalization functions ---
    mean_vals = mean_vals.view(1, -1, 1, 1)
    std_vals = (std_vals + 1e-8).view(1, -1, 1, 1) # Add epsilon for stability, as in training
    
    def normalize(x):
        # x is expected as (B, C, X, Z)
        return (x - mean_vals) / std_vals
        
    def denormalize(x_norm):
        # x_norm is (B, C, X, Z)
        return (x_norm * std_vals) + mean_vals
    # --- END MODIFICATION ---

    test_tensor_cpu = torch.from_numpy(test_data_np).float()
    logging.info(f"Test data loaded and processed. Final shape: {test_tensor_cpu.shape}")

    # --- Autoregressive Rollout ---
    num_timesteps = test_tensor_cpu.shape[0]
    predictions_denorm_cpu = []
    
    # Get initial state in (B, X, Z, C) format, permute to (B, C, X, Z) for model
    initial_state_original = test_tensor_cpu[0].unsqueeze(0).to(device)
    current_state_norm = normalize(initial_state_original.permute(0, 3, 1, 2))

    logging.info(f"Starting autoregressive rollout for {num_timesteps - 1} steps...")
    with torch.no_grad():
        predictions_denorm_cpu.append(initial_state_original.squeeze(0).cpu())
        z_t = model.encode(current_state_norm)

        for _ in tqdm(range(num_timesteps - 1), desc="Evaluating Rollout"):
            z_t = model.koopman_step(z_t)
            predicted_state_norm = model.decode(z_t)
            
            # De-normalize and permute back to (B, X, Z, C) for saving
            predicted_state_denorm = denormalize(predicted_state_norm).permute(0, 2, 3, 1)
            
            predictions_denorm_cpu.append(predicted_state_denorm.squeeze(0).cpu())
            current_state_norm = predicted_state_norm
    
    predicted_timeseries = torch.stack(predictions_denorm_cpu)
    logging.info(f"Rollout complete. Predicted timeseries shape: {predicted_timeseries.shape}")

    # --- Calculate Metrics and Difference ---
    difference_timeseries = predicted_timeseries - test_tensor_cpu
    loss_fn = nn.MSELoss(reduction='none')
    error_tensor = loss_fn(predicted_timeseries, test_tensor_cpu)
    
    avg_rollout_mse_total = error_tensor.mean().item()
    variance_total = test_tensor_cpu.var().item()
    r_squared_total = 1 - (avg_rollout_mse_total / variance_total) if variance_total > 0 else 0.0
    
    avg_mse_per_channel = error_tensor.mean(dim=(0, 1, 2)).numpy()
    r_squared_per_channel = np.zeros(len(channel_names))
    for i in range(len(channel_names)):
        variance_channel = test_tensor_cpu[..., i].var().item()
        r_squared_per_channel[i] = 1 - (avg_mse_per_channel[i] / variance_channel) if variance_channel > 0 else 0.0

    # Per-step error (excluding initial state)
    per_step_channel_error = error_tensor[1:].mean(dim=(1, 2)).numpy()

    # --- Save All Results to Separate Files ---
    
    # 1. Save statistics
    np.savez(
        stats_path,
        channel_names=np.array(channel_names, dtype='U'),
        r_squared_total=r_squared_total,
        r_squared_per_channel=r_squared_per_channel,
        per_step_channel_error=per_step_channel_error,
        avg_rollout_mse_total=avg_rollout_mse_total,
        avg_mse_per_channel=avg_mse_per_channel,
    )
    logging.info(f"Analysis statistics saved to {stats_path}")

    # 2. Save predicted timeseries in original data format
    np.savez(
        pred_path,
        timeseries=predicted_timeseries.numpy(),
        labels=np.array(channel_names, dtype='U'),
    )
    logging.info(f"Predicted timeseries saved to {pred_path}")

    # 3. Save difference timeseries in original data format
    np.savez(
        diff_path,
        timeseries=difference_timeseries.numpy(),
        labels=np.array(channel_names, dtype='U'),
    )
    logging.info(f"Difference timeseries saved to {diff_path}")

    # --- Log Final Metrics ---
    logging.info("--- PREDICTION EVALUATION COMPLETE ---")
    logging.info(f"Overall Average Rollout MSE: {avg_rollout_mse_total:.6f}")
    logging.info(f"Overall R-squared (R²) Score: {r_squared_total:.4f}\n")
    logging.info("--- Per-Channel Metrics ---")
    for i, name in enumerate(channel_names):
        logging.info(f"Channel '{name}':\t Avg MSE = {avg_mse_per_channel[i]:.6f},\t R² = {r_squared_per_channel[i]:.4f}")

if __name__ == "__main__":
    main()
