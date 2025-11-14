# -*- coding: utf-8 -*-
# experiments/study_6_flow_matching_2d/study_6.1_flow_matching_2d_base/evaluation/prediction/gen_prediction.py
#
# This script is the Flow Matching equivalent of the tcKAE prediction script.
#
# --- Key Differences from tcKAE version ---
# 1. Model: Uses FlowMatchingUNet.
# 2. Rollout: Instead of matrix multiplication in latent space, it solves
#    an ODE (from noise x_0 to data x_1) at every time step, conditioned
#    on the previous frame y_k.
# ------------------------------------------

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Import the Flow Matching model
from flow_matching_2d_base.model import FlowMatchingUNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained 2D Flow Matching model prediction performance.")
    
    # --- Input Paths ---
    io_group = parser.add_argument_group("Input Paths")
    io_group.add_argument("--model-path", type=str, required=True, help="Path to the saved best_model.pth file.")
    io_group.add_argument("--test-data-path", type=str, required=True, help="Path to the contiguous test_set.npz file.")
    io_group.add_argument("--norm-stats-path", type=str, required=True, help="Path to the normalization_stats.npz file.")

    # --- Output Paths ---
    out_group = parser.add_argument_group("Output Paths")
    out_group.add_argument("--output-dir", type=str, required=True, help="Directory to save all output files.")
    
    # --- Flow Matching Specific ---
    fm_group = parser.add_argument_group("Flow Matching Generation Params")
    fm_group.add_argument("--ode-steps", type=int, default=10, help="Number of ODE steps per frame generation (default: 10).")
    fm_group.add_argument("--solver", type=str, default="midpoint", choices=["euler", "midpoint"], help="ODE solver method.")
    fm_group.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility. Samples will use seed, seed+1, ...")
    fm_group.add_argument("--num-samples", type=int, default=1, help="Number of prediction samples to generate.")

    return parser.parse_args()

def solve_flow_ode(model, y_condition, steps, solver, device):
    """
    Solves the Flow Matching ODE to generate y_{k+1} given y_k (y_condition).
    Goes from t=0 (noise) to t=1 (data).
    """
    # 1. Sample initial noise x_0 ~ N(0, I)
    # This relies on the global torch seed set in main() for reproducibility
    x_t = torch.randn_like(y_condition)
    
    dt = 1.0 / steps

    # 2. Integrate from t=0 to t=1
    for i in range(steps):
        t_val = i * dt
        t_batch = torch.full((x_t.shape[0],), t_val, device=device)
        
        if solver == 'euler':
            # v = model(x_t, t, y_k)
            velocity = model(x_t, t_batch, y_condition)
            x_t = x_t + velocity * dt
            
        elif solver == 'midpoint':
            # k1 = v(x_t, t, y_k)
            k1 = model(x_t, t_batch, y_condition)
            
            # k2 = v(x_t + k1*dt/2, t + dt/2, y_k)
            t_mid_val = t_val + dt / 2.0
            x_mid = x_t + k1 * (dt / 2.0)
            t_mid_batch = torch.full((x_t.shape[0],), t_mid_val, device=device)
            k2 = model(x_mid, t_mid_batch, y_condition)
            
            x_t = x_t + k2 * dt

    # At t=1, x_t is our prediction for y_{k+1}
    return x_t

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # --- Setup Base Output Path ---
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Model ---
    logging.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model_config = checkpoint['config']
    channels_used = checkpoint.get('channels_used')
    
    # Filter out any training-only keys if they exist (safety check)
    model_args = {k: v for k, v in model_config.items() if k != 'channels_used'}
    
    logging.info(f"Re-creating FlowMatchingUNet with config: {model_args}")
    model = FlowMatchingUNet(**model_args).to(device)
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

    # Squeeze potential 5D data (T, X, 1, Z, C) -> (T, X, Z, C)
    if full_timeseries.ndim == 5 and full_timeseries.shape[2] == 1:
        full_timeseries = np.squeeze(full_timeseries, axis=2)

    # Handle channel selection if the model was trained on a subset
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

    # Setup Normalization (compatible with B, C, H, W format)
    mean_vals = mean_vals.view(1, -1, 1, 1)
    std_vals = (std_vals + 1e-8).view(1, -1, 1, 1)
    
    def normalize(x): return (x - mean_vals) / std_vals
    def denormalize(x_norm): return (x_norm * std_vals) + mean_vals

    test_tensor_cpu = torch.from_numpy(test_data_np).float()
    num_timesteps = test_tensor_cpu.shape[0]
    logging.info(f"Test data ready. Shape: {test_tensor_cpu.shape}.")

    # --- Main Sample Generation Loop ---
    for i in range(args.num_samples):
        current_seed = args.seed + i
        sample_str = f"sample_{i:02d}"
        logging.info(f"--- Generating Sample {i+1}/{args.num_samples} ({sample_str}) with seed {current_seed} ---")

        # Set seed for reproducibility for this specific sample
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        
        # --- Setup Output Paths for this Sample ---
        sample_output_dir = base_output_dir / sample_str
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        stats_path = sample_output_dir / "prediction_stats.npz"
        pred_path = sample_output_dir / "predicted_timeseries.npz"
        diff_path = sample_output_dir / "difference_timeseries.npz"

        logging.info(f"Starting {args.solver} rollout ({args.ode_steps} steps/frame)...")

        # --- Autoregressive Rollout Loop ---
        predictions_denorm_cpu = []
        
        # 1. Prepare initial state: (X, Z, C) -> (1, X, Z, C) -> to device
        current_state_original = test_tensor_cpu[0].unsqueeze(0).to(device)
        # 2. Permute to model format: (1, C, X, Z) and normalize
        current_state_norm = normalize(current_state_original.permute(0, 3, 1, 2))

        # Add first frame to predictions
        predictions_denorm_cpu.append(current_state_original.squeeze(0).cpu())

        with torch.no_grad():
            for _ in tqdm(range(num_timesteps - 1), desc=f"Generating frames for {sample_str}"):
                # Solve ODE to get next state: y_{k+1} = ODE_Solve(y_k)
                next_state_norm = solve_flow_ode(
                    model=model,
                    y_condition=current_state_norm,
                    steps=args.ode_steps,
                    solver=args.solver,
                    device=device
                )
                
                # Denormalize and permute back to storage format (1, X, Z, C)
                next_state_denorm = denormalize(next_state_norm).permute(0, 2, 3, 1)
                
                # Store and update
                predictions_denorm_cpu.append(next_state_denorm.squeeze(0).cpu())
                current_state_norm = next_state_norm

        predicted_timeseries = torch.stack(predictions_denorm_cpu)
        logging.info(f"Rollout complete for {sample_str}. Shape: {predicted_timeseries.shape}")

        # --- Calculate Metrics (Exact same logic as tcKAE script) ---
        difference_timeseries = predicted_timeseries - test_tensor_cpu
        loss_fn = nn.MSELoss(reduction='none')
        error_tensor = loss_fn(predicted_timeseries, test_tensor_cpu)
        
        avg_rollout_mse_total = error_tensor.mean().item()
        variance_total = test_tensor_cpu.var().item()
        r_squared_total = 1 - (avg_rollout_mse_total / variance_total) if variance_total > 0 else 0.0
        
        avg_mse_per_channel = error_tensor.mean(dim=(0, 1, 2)).numpy()
        r_squared_per_channel = np.zeros(len(channel_names))
        for j in range(len(channel_names)):
            variance_channel = test_tensor_cpu[..., j].var().item()
            r_squared_per_channel[j] = 1 - (avg_mse_per_channel[j] / variance_channel) if variance_channel > 0 else 0.0

        per_step_channel_error = error_tensor[1:].mean(dim=(1, 2)).numpy()

        # --- Save Results ---
        np.savez(stats_path,
            channel_names=np.array(channel_names, dtype='U'),
            r_squared_total=r_squared_total,
            r_squared_per_channel=r_squared_per_channel,
            per_step_channel_error=per_step_channel_error,
            avg_rollout_mse_total=avg_rollout_mse_total,
            avg_mse_per_channel=avg_mse_per_channel,
            ode_settings=np.array([args.solver, str(args.ode_steps)], dtype='U'),
            seed=current_seed
        )
        np.savez(pred_path, timeseries=predicted_timeseries.numpy(), labels=np.array(channel_names, dtype='U'))
        np.savez(diff_path, timeseries=difference_timeseries.numpy(), labels=np.array(channel_names, dtype='U'))

        logging.info(f"--- PREDICTION EVALUATION COMPLETE FOR {sample_str} ---")
        logging.info(f"Solver: {args.solver} ({args.ode_steps} steps)")
        logging.info(f"Overall MSE: {avg_rollout_mse_total:.6f} | R²: {r_squared_total:.4f}\n")

        logging.info("--- Per-Channel Metrics ---")
        for j, name in enumerate(channel_names):
            logging.info(f"Channel '{name}':\t Avg MSE = {avg_mse_per_channel[j]:.6f},\t R² = {r_squared_per_channel[j]:.4f}")
        logging.info("\n" + "="*80 + "\n")

    logging.info("All samples generated.")

if __name__ == "__main__":
    main()