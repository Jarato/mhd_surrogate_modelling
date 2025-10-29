# -*- coding: utf-8 -*-
#
# This script is based on gen_prediction.py and is modified to
# generate reconstructions instead of predictions.
#
# --- THIS VERSION ---
# 1. Imports the tcKoopmanAutoencoder2D from the *older* mhd_2d_tckae package.
# 2. Uses Mean/Std normalization, loading 'mean_vals' and 'std_vals'.
# 3. The "Autoregressive Rollout" loop is replaced with a batched
#    "Reconstruction" loop that processes each timestep independently
#    (encode -> decode).
# 4. Output files and metrics are renamed to reflect reconstruction.
# ---

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# --- MODIFICATION: Import the older model ---
from mhd_2d_tckae.model import tcKoopmanAutoencoder2D
# --- END MODIFICATION ---


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained 2D tcKAE (OLDER Version) reconstruction performance.")
    
    # --- Input Paths ---
    io_group = parser.add_argument_group("Input Paths")
    io_group.add_argument("--model-path", type=str, required=True, help="Path to the saved best_model.pth file.")
    io_group.add_argument("--test-data-path", type=str, required=True, help="Path to the contiguous test_set.npz file.")
    io_group.add_argument("--norm-stats-path", type=str, required=True, help="Path to the normalization_stats.npz file (must contain 'mean_vals' and 'std_vals').")

    # --- Output Paths ---
    out_group = parser.add_argument_group("Output Paths")
    out_group.add_argument("--output-dir", type=str, required=True, help="A single directory to save all output files.")
    
    # --- Batch Size ---
    perf_group = parser.add_argument_group("Performance")
    perf_group.add_argument("--batch-size", type=int, default=64, help="Batch size for processing the reconstructions.")

    return parser.parse_args()

def main():
    """Main function to generate reconstructions and evaluate the model."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model_path = Path(args.model_path)
    
    # --- Setup Output Paths ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use new names for reconstruction outputs
    stats_path = output_dir / "reconstruction_stats.npz"
    recon_path = output_dir / "reconstructed_timeseries.npz"
    diff_path = output_dir / "difference_timeseries.npz"
    logging.info(f"All reconstruction outputs will be saved to: {output_dir}")

    # --- Load Model from Checkpoint ---
    logging.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config']
    channels_used = checkpoint.get('channels_used')
    
    logging.info(f"Re-creating model with saved config: {model_config}")
    model = tcKoopmanAutoencoder2D(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- Load Data and Stats (Using MIN-MAX normalization) ---
    logging.info(f"Loading normalization stats from {args.norm_stats_path}")
    stats = np.load(args.norm_stats_path)
    # --- MODIFICATION: Load min/max vals ---
    all_min_vals = torch.from_numpy(stats['min_vals']).float().to(device)
    all_max_vals = torch.from_numpy(stats['max_vals']).float().to(device)
    # --- END MODIFICATION ---
    
    logging.info(f"Loading test data from {args.test_data_path}")
    data_handle = np.load(args.test_data_path, allow_pickle=True)
    full_timeseries = data_handle["timeseries"]
    all_channel_names = list(data_handle["labels"])

    if full_timeseries.ndim == 5 and full_timeseries.shape[2] == 1:
        full_timeseries = np.squeeze(full_timeseries, axis=2)
        logging.info(f"Squeezed test data to 4D. New shape: {full_timeseries.shape}")

    if channels_used:
        channel_indices = [all_channel_names.index(name) for name in channels_used]
        # --- MODIFICATION: Use min/max vals ---
        min_vals = all_min_vals[channel_indices]
        max_vals = all_max_vals[channel_indices]
        # --- END MODIFICATION ---
        channel_names = channels_used
        test_data_np = full_timeseries[..., channel_indices]
    else:
        channel_indices = list(range(len(all_channel_names)))
        # --- MODIFICATION: Use min/max vals ---
        min_vals = all_min_vals
        max_vals = all_max_vals
        # --- END MODIFICATION ---
        channel_names = all_channel_names
        test_data_np = full_timeseries

    # --- Reshape and define Min-Max normalization functions ---
    min_vals = min_vals.view(1, -1, 1, 1)
    max_vals = max_vals.view(1, -1, 1, 1)
    data_range = (max_vals - min_vals) + 1e-8 # Add epsilon for stability
    
    def normalize(x):
        # x is expected as (B, C, X, Z)
        # Normalize to [-1, 1]
        return (x - min_vals) / data_range * 2.0 - 1.0
        
    def denormalize(x_norm):
        # x_norm is (B, C, X, Z)
        # Denormalize from [-1, 1]
        return (x_norm + 1.0) / 2.0 * data_range + min_vals
    # --- END NORMALIZATION SECTION ---

    test_tensor_cpu = torch.from_numpy(test_data_np).float()
    logging.info(f"Test data loaded and processed. Final shape: {test_tensor_cpu.shape}")

    # --- Batched Reconstruction ---
    num_timesteps = test_tensor_cpu.shape[0]
    batch_size = args.batch_size
    reconstructions_denorm_cpu = []
    
    logging.info(f"Starting reconstruction for {num_timesteps} steps in batches of {batch_size}...")
    with torch.no_grad():
        for i in tqdm(range(0, num_timesteps, batch_size), desc="Reconstructing"):
            # Get batch in (B, X, Z, C) format
            batch_data_cpu = test_tensor_cpu[i:i + batch_size]
            
            # Permute to (B, C, X, Z) for model, send to device
            batch_data_original = batch_data_cpu.to(device).permute(0, 3, 1, 2)
            
            # Normalize
            batch_data_norm = normalize(batch_data_original)
            
            # --- CORE RECONSTRUCTION ---
            # Encode
            z_t = model.encode(batch_data_norm)
            # Decode
            recon_state_norm = model.decode(z_t)
            # ---------------------------

            # De-normalize and permute back to (B, X, Z, C) for saving
            recon_state_denorm = denormalize(recon_state_norm).permute(0, 2, 3, 1)
            
            reconstructions_denorm_cpu.append(recon_state_denorm.cpu())
    
    reconstructed_timeseries = torch.cat(reconstructions_denorm_cpu, dim=0)
    logging.info(f"Reconstruction complete. Reconstructed timeseries shape: {reconstructed_timeseries.shape}")

    # --- Calculate Metrics and Difference ---
    difference_timeseries = reconstructed_timeseries - test_tensor_cpu
    loss_fn = nn.MSELoss(reduction='none')
    error_tensor = loss_fn(reconstructed_timeseries, test_tensor_cpu)
    
    avg_recon_mse_total = error_tensor.mean().item()
    variance_total = test_tensor_cpu.var().item()
    r_squared_total = 1 - (avg_recon_mse_total / variance_total) if variance_total > 0 else 0.0
    
    avg_mse_per_channel = error_tensor.mean(dim=(0, 1, 2)).numpy()
    r_squared_per_channel = np.zeros(len(channel_names))
    for i in range(len(channel_names)):
        variance_channel = test_tensor_cpu[..., i].var().item()
        r_squared_per_channel[i] = 1 - (avg_mse_per_channel[i] / variance_channel) if variance_channel > 0 else 0.0

    # Per-step error (all states are independent, so we include all)
    # Shape will be (T, C)
    per_step_channel_error = error_tensor.mean(dim=(1, 2)).numpy()

    # --- Save All Results to Separate Files ---
    
    # 1. Save statistics
    np.savez(
        stats_path,
        channel_names=np.array(channel_names, dtype='U'),
        r_squared_total=r_squared_total,
        r_squared_per_channel=r_squared_per_channel,
        per_step_channel_error=per_step_channel_error,
        avg_recon_mse_total=avg_recon_mse_total,
        avg_mse_per_channel=avg_mse_per_channel,
    )
    logging.info(f"Reconstruction statistics saved to {stats_path}")

    # 2. Save reconstructed timeseries in original data format
    np.savez(
        recon_path,
        timeseries=reconstructed_timeseries.numpy(),
        labels=np.array(channel_names, dtype='U'),
    )
    logging.info(f"Reconstructed timeseries saved to {recon_path}")

    # 3. Save difference timeseries in original data format
    np.savez(
        diff_path,
        timeseries=difference_timeseries.numpy(),
        labels=np.array(channel_names, dtype='U'),
    )
    logging.info(f"Difference timeseries saved to {diff_path}")

    # --- Log Final Metrics ---
    logging.info("--- RECONSTRUCTION EVALUATION COMPLETE ---")
    logging.info(f"Overall Average Reconstruction MSE: {avg_recon_mse_total:.6f}")
    logging.info(f"Overall R-squared (R²) Score: {r_squared_total:.4f}\n")
    logging.info("--- Per-Channel Metrics ---")
    for i, name in enumerate(channel_names):
        logging.info(f"Channel '{name}':\t Avg MSE = {avg_mse_per_channel[i]:.6f},\t R² = {r_squared_per_channel[i]:.4f}")

if __name__ == "__main__":
    main()

