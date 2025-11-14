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
#
# --- MODIFIED FOR PARALLELISM ---
# This version uses multiprocessing to generate samples in parallel.
# - Adds a --num-workers argument.
# - Uses a multiprocessing.Pool to distribute sample generation.
# - Uses a "spawn" context and an initializer to safely load the model
#   and data (once per worker) for use with CUDA.
# ------------------------------------------

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
import multiprocessing
import os

# Import the Flow Matching model
from flow_matching_2d_base.model import FlowMatchingUNet

# --- Globals for Worker Processes ---
# These will be populated by the init_worker function in each child process
g_model = None
g_device = None
g_test_tensor_cpu = None
g_mean_vals = None
g_std_vals = None
g_channel_names = None

def init_worker(model_config, model_state_dict, test_data_np, mean_vals_np, std_vals_np, ch_names):
    """
    Initializer function for each worker process.
    Loads the model and data into the worker's global scope.
    """
    global g_model, g_device, g_test_tensor_cpu, g_mean_vals, g_std_vals, g_channel_names
    
    # Configure logging for this specific worker
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s - [PID {os.getpid()}] - %(levelname)s - %(message)s")
    
    try:
        g_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        g_model = FlowMatchingUNet(**model_config).to(g_device)
        g_model.load_state_dict(model_state_dict)
        g_model.eval()
        
        g_test_tensor_cpu = torch.from_numpy(test_data_np).float()
        g_channel_names = ch_names
        
        g_mean_vals = torch.from_numpy(mean_vals_np).float().to(g_device).view(1, -1, 1, 1)
        g_std_vals = (torch.from_numpy(std_vals_np).float().to(g_device) + 1e-8).view(1, -1, 1, 1)
        
        logging.info(f"Worker PID {os.getpid()} initialized on device '{g_device}'.")
    except Exception as e:
        logging.error(f"Worker PID {os.getpid()} failed to initialize: {e}")
        raise e

def normalize(x):
    """Normalization function using worker-global stats."""
    global g_mean_vals, g_std_vals
    return (x - g_mean_vals) / g_std_vals

def denormalize(x_norm):
    """Denormalization function using worker-global stats."""
    global g_mean_vals, g_std_vals
    return (x_norm * g_std_vals) + g_mean_vals

def solve_flow_ode(y_condition, steps, solver):
    """
    Solves the Flow Matching ODE to generate y_{k+1} given y_k (y_condition).
    Goes from t=0 (noise) to t=1 (data).
    Accesses the model and device from the worker's global scope.
    """
    # 1. Access model and device from global scope
    global g_model, g_device
    
    # This relies on the torch seed set *within the worker task*
    x_t = torch.randn_like(y_condition)
    
    dt = 1.0 / steps

    # 2. Integrate from t=0 to t=1
    for i in range(steps):
        t_val = i * dt
        t_batch = torch.full((x_t.shape[0],), t_val, device=g_device)
        
        if solver == 'euler':
            # v = model(x_t, t, y_k)
            velocity = g_model(x_t, t_batch, y_condition)
            x_t = x_t + velocity * dt
            
        elif solver == 'midpoint':
            # k1 = v(x_t, t, y_k)
            k1 = g_model(x_t, t_batch, y_condition)
            
            # k2 = v(x_t + k1*dt/2, t + dt/2, y_k)
            t_mid_val = t_val + dt / 2.0
            x_mid = x_t + k1 * (dt / 2.0)
            t_mid_batch = torch.full((x_t.shape[0],), t_mid_val, device=g_device)
            k2 = g_model(x_mid, t_mid_batch, y_condition)
            
            x_t = x_t + k2 * dt

    # At t=1, x_t is our prediction for y_{k+1}
    return x_t

def generate_sample_task(sample_index, args):
    """
    The main task function executed by each worker.
    Generates a single sample based on the sample_index.
    """
    # Access global state
    global g_device, g_test_tensor_cpu, g_channel_names
    
    current_seed = args.seed + sample_index
    sample_str = f"sample_{sample_index:02d}"
    is_full_sample = (sample_index < args.num_video_samples)
    
    base_output_dir = Path(args.output_dir)
    sample_output_dir = base_output_dir / sample_str
    stats_path = sample_output_dir / "prediction_stats.npz"
    pred_path = sample_output_dir / "predicted_timeseries.npz"
    diff_path = sample_output_dir / "difference_timeseries.npz"

    # --- Check if sample should be skipped ---
    if not args.force_rerun and stats_path.is_file():
        try:
            stats_data = np.load(stats_path)
            if 'seed' in stats_data and stats_data['seed'] == current_seed:
                logging.info(f"Skipping {sample_str}: Found existing stats file with matching seed {current_seed}.")
                return stats_path # Return path for aggregation
            else:
                logging.warning(f"Found existing stats file for {sample_str}, but seed mismatch or 'seed' key missing. Rerunning...")
        except Exception as e:
            logging.warning(f"Could not load or parse existing stats file {stats_path}. Rerunning... Error: {e}")

    if is_full_sample:
        logging.info(f"--- Generating Full Sample {sample_index+1}/{args.num_stats_samples} ({sample_str}) with seed {current_seed} ---")
    else:
        logging.info(f"--- Generating Stats-Only Sample {sample_index+1}/{args.num_stats_samples} ({sample_str}) with seed {current_seed} ---")

    # Set seed for reproducibility for this specific sample
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Autoregressive Rollout Loop ---
    predictions_denorm_cpu = []
    num_timesteps = g_test_tensor_cpu.shape[0]
    
    # 1. Prepare initial state: (X, Z, C) -> (1, X, Z, C) -> to device
    current_state_original = g_test_tensor_cpu[0].unsqueeze(0).to(g_device)
    # 2. Permute to model format: (1, C, X, Z) and normalize
    current_state_norm = normalize(current_state_original.permute(0, 3, 1, 2))

    # Add first frame to predictions
    predictions_denorm_cpu.append(current_state_original.squeeze(0).cpu())

    with torch.no_grad():
        # Create a TQDM progress bar only if it's sample 0 (or adjust as needed)
        # Note: Multiple TQDM bars in parallel logs can be messy.
        # Disabling per-frame progress bar for parallel execution.
        # for _ in tqdm(range(num_timesteps - 1), desc=f"Generating frames for {sample_str}"):
        
        for _ in range(num_timesteps - 1):
            # Solve ODE to get next state: y_{k+1} = ODE_Solve(y_k)
            next_state_norm = solve_flow_ode(
                y_condition=current_state_norm,
                steps=args.ode_steps,
                solver=args.solver
            )
            
            # Denormalize and permute back to storage format (1, X, Z, C)
            next_state_denorm = denormalize(next_state_norm).permute(0, 2, 3, 1)
            
            # Store and update
            predictions_denorm_cpu.append(next_state_denorm.squeeze(0).cpu())
            current_state_norm = next_state_norm

    predicted_timeseries = torch.stack(predictions_denorm_cpu)
    logging.info(f"Rollout complete for {sample_str}. Shape: {predicted_timeseries.shape}")

    # --- Calculate Metrics ---
    difference_timeseries = predicted_timeseries - g_test_tensor_cpu
    loss_fn = nn.MSELoss(reduction='none')
    error_tensor = loss_fn(predicted_timeseries, g_test_tensor_cpu)
    
    avg_rollout_mse_total = error_tensor.mean().item()
    variance_total = g_test_tensor_cpu.var().item()
    r_squared_total = 1 - (avg_rollout_mse_total / variance_total) if variance_total > 0 else 0.0
    
    avg_mse_per_channel = error_tensor.mean(dim=(0, 1, 2)).numpy()
    r_squared_per_channel = np.zeros(len(g_channel_names))
    for j in range(len(g_channel_names)):
        variance_channel = g_test_tensor_cpu[..., j].var().item()
        r_squared_per_channel[j] = 1 - (avg_mse_per_channel[j] / variance_channel) if variance_channel > 0 else 0.0

    per_step_channel_error = error_tensor[1:].mean(dim=(1, 2)).numpy()

    # --- Save Results ---
    np.savez(stats_path,
        channel_names=np.array(g_channel_names, dtype='U'),
        r_squared_total=r_squared_total,
        r_squared_per_channel=r_squared_per_channel,
        per_step_channel_error=per_step_channel_error,
        avg_rollout_mse_total=avg_rollout_mse_total,
        avg_mse_per_channel=avg_mse_per_channel,
        ode_settings=np.array([args.solver, str(args.ode_steps)], dtype='U'),
        seed=current_seed
    )

    if is_full_sample:
        np.savez(pred_path, timeseries=predicted_timeseries.numpy(), labels=np.array(g_channel_names, dtype='U'))
        np.savez(diff_path, timeseries=difference_timeseries.numpy(), labels=np.array(g_channel_names, dtype='U'))
    
    logging.info(f"--- PREDICTION EVALUATION COMPLETE FOR {sample_str} ---")
    logging.info(f"Overall R²: {r_squared_total:.4f}")
    
    return stats_path


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
    fm_group.add_argument("--num-video-samples", type=int, default=1, help="Number of samples to save full timeseries NPZ files for (for video generation).")
    fm_group.add_argument("--num-stats-samples", type=int, default=1, help="Total number of samples to generate for statistics. Must be >= num-video-samples.")
    fm_group.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers to generate samples. -1 to use all available CPUs.")
    fm_group.add_argument("--force-rerun", action='store_true', help="Force regeneration of all samples, even if they exist with matching seeds.")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Use "spawn" context for safety with CUDA
    # This ensures each child process starts fresh and runs init_worker
    mp_context = multiprocessing.get_context("spawn")
    
    # Configure logging for the main process
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [MAIN] - %(levelname)s - %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Main process using device: {device} (Note: workers will use their own).")

    if args.num_stats_samples < args.num_video_samples:
        logging.warning(f"num-stats-samples ({args.num_stats_samples}) is less than num-video-samples ({args.num_video_samples}).")
        logging.warning(f"Setting num-stats-samples = num-video-samples.")
        args.num_stats_samples = args.num_video_samples
    
    # --- Setup Base Output Path ---
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Model (config and state_dict only) ---
    logging.info(f"Loading model components from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location="cpu") # Load to CPU first
    model_config = checkpoint['config']
    channels_used = checkpoint.get('channels_used')
    model_state_dict = checkpoint['model_state_dict']
    
    # Filter out any training-only keys if they exist (safety check)
    model_config = {k: v for k, v in model_config.items() if k != 'channels_used'}

    # --- Load Data and Stats (as NumPy arrays) ---
    logging.info(f"Loading normalization stats from {args.norm_stats_path}")
    stats = np.load(args.norm_stats_path)
    all_mean_vals_np = stats['mean_vals'].astype(np.float32)
    all_std_vals_np = stats['std_vals'].astype(np.float32)
    
    logging.info(f"Loading test data from {args.test_data_path}")
    data_handle = np.load(args.test_data_path, allow_pickle=True)
    full_timeseries = data_handle["timeseries"]
    all_channel_names = list(data_handle["labels"])

    if full_timeseries.ndim == 5 and full_timeseries.shape[2] == 1:
        full_timeseries = np.squeeze(full_timeseries, axis=2)

    if channels_used:
        channel_indices = [all_channel_names.index(name) for name in channels_used]
        mean_vals_np = all_mean_vals_np[channel_indices]
        std_vals_np = all_std_vals_np[channel_indices]
        channel_names = channels_used
        test_data_np = full_timeseries[..., channel_indices].astype(np.float32)
    else:
        mean_vals_np = all_mean_vals_np
        std_vals_np = all_std_vals_np
        channel_names = all_channel_names
        test_data_np = full_timeseries.astype(np.float32)

    logging.info(f"Test data ready. Shape: {test_data_np.shape}.")
    
    # --- Setup Worker Pool ---
    if args.num_workers == -1:
        try:
            num_workers = multiprocessing.cpu_count()
            logging.info(f"Setting num-workers to all available CPUs: {num_workers}")
        except NotImplementedError:
            num_workers = 4
            logging.warning(f"Could not detect CPU count. Defaulting to {num_workers} workers.")
    else:
        num_workers = max(1, args.num_workers)

    # --- Prepare arguments for worker initializer ---
    init_args = (
        model_config, 
        model_state_dict, 
        test_data_np, 
        mean_vals_np, 
        std_vals_np, 
        channel_names
    )
    
    # Prepare tasks for the pool
    tasks = [(i, args) for i in range(args.num_stats_samples)]
    all_stats_files = []

    # --- Run Generation ---
    if num_workers == 1:
        logging.info("Running in serial mode (num-workers=1). Initializing model in main process...")
        # Need to init globals for the main process
        init_worker(*init_args) 
        # TQDM for serial execution
        all_stats_files = [generate_sample_task(i, args) for i in tqdm(range(args.num_stats_samples), desc="Generating samples")]
    else:
        logging.info(f"Starting worker pool with {num_workers} processes...")
        with mp_context.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
            # Use starmap_async to run tasks and get results
            results_async = pool.starmap_async(generate_sample_task, tasks)
            
            # Use TQDM to show progress as results come in
            results = []
            for res in tqdm(results_async.get(), total=len(tasks), desc="Processing samples"):
                results.append(res)
            all_stats_files = results
    
    # Filter out any 'None' results from failed/skipped tasks
    all_stats_files = [path for path in all_stats_files if path is not None]
    logging.info(f"All {len(all_stats_files)} samples generated (or found).")

    # --- Aggregate Probabilistic Statistics ---
    if not all_stats_files:
        logging.warning("No stats files found or generated. Skipping probabilistic aggregation.")
        return

    logging.info(f"Aggregating statistics from {len(all_stats_files)} samples...")
    
    all_r2_total = []
    all_r2_per_channel = []
    all_mse_total = []
    all_mse_per_channel = []
    all_per_step_channel_error = []
    loaded_channel_names = None

    for f_path in all_stats_files:
        try:
            stats = np.load(f_path)
            all_r2_total.append(stats['r_squared_total'])
            all_r2_per_channel.append(stats['r_squared_per_channel'])
            all_mse_total.append(stats['avg_rollout_mse_total'])
            all_mse_per_channel.append(stats['avg_mse_per_channel'])
            all_per_step_channel_error.append(stats['per_step_channel_error'])
            if loaded_channel_names is None:
                loaded_channel_names = stats['channel_names']
        except Exception as e:
            logging.warning(f"Could not load or parse {f_path} for aggregation. Skipping. Error: {e}")

    if not all_r2_total:
        logging.error("Failed to load any statistics. Aborting aggregation.")
        return
    
    if loaded_channel_names is None:
        loaded_channel_names = np.array(channel_names, dtype='U') # Fallback

    # Stack for easy numpy operations
    all_r2_per_channel_np = np.stack(all_r2_per_channel)
    all_mse_per_channel_np = np.stack(all_mse_per_channel)
    all_per_step_channel_error_np = np.stack(all_per_step_channel_error)

    # Calculate mean and std
    prob_stats = {
        'num_samples': len(all_r2_total),
        'mean_r2_total': np.mean(all_r2_total),
        'std_r2_total': np.std(all_r2_total),
        'mean_mse_total': np.mean(all_mse_total),
        'std_mse_total': np.std(all_mse_total),
        
        'mean_r2_per_channel': np.mean(all_r2_per_channel_np, axis=0),
        'std_r2_per_channel': np.std(all_r2_per_channel_np, axis=0),
        'mean_mse_per_channel': np.mean(all_mse_per_channel_np, axis=0),
        'std_mse_per_channel': np.std(all_mse_per_channel_np, axis=0),
        
        'mean_per_step_channel_error': np.mean(all_per_step_channel_error_np, axis=0),
        'std_per_step_channel_error': np.std(all_per_step_channel_error_np, axis=0),
        
        'channel_names': loaded_channel_names
    }

    # Save aggregated stats
    prob_stats_path = base_output_dir / "probabilistic_stats.npz"
    np.savez(prob_stats_path, **prob_stats)
    
    logging.info("\n" + "="*80)
    logging.info(f"--- PROBABILISTIC EVALUATION COMPLETE (N={prob_stats['num_samples']}) ---")
    logging.info(f"Aggregated stats saved to: {prob_stats_path}")
    logging.info(f"Overall R²:  {prob_stats['mean_r2_total']:.4f} ± {prob_stats['std_r2_total']:.4f}")
    logging.info(f"Overall MSE: {prob_stats['mean_mse_total']:.6f} ± {prob_stats['std_mse_total']:.6f}")
    logging.info("--- Per-Channel Mean R² (Mean ± Std) ---")
    for j, name in enumerate(loaded_channel_names):
        mean_r2 = prob_stats['mean_r2_per_channel'][j]
        std_r2 = prob_stats['std_r2_per_channel'][j]
        logging.info(f"Channel '{name}':\t {mean_r2:.4f} ± {std_r2:.4f}")
    logging.info("="*80 + "\n")


if __name__ == "__main__":
    # This check is crucial for multiprocessing
    main()