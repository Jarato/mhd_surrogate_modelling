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
    parser = argparse.ArgumentParser(description="Evaluate a trained 2D tcKAE's latent space dynamics, generate physical space predictions from high-energy modes, and optionally save individual mode reconstructions.")
    
    # --- Input Paths ---
    io_group = parser.add_argument_group("Input Paths")
    io_group.add_argument("--model-path", type=str, required=True, help="Path to the saved best_model.pth file.")
    io_group.add_argument("--test-data-path", type=str, required=True, help="Path to the contiguous test_set.npz file.")
    io_group.add_argument("--norm-stats-path", type=str, required=True, help="Path to the normalization_stats.npz file.")

    # --- Output Paths ---
    out_group = parser.add_argument_group("Output Paths")
    out_group.add_argument("--output-dir", type=str, default=None, help="Directory to save all analysis and reconstruction files. Defaults to 'eval/' in the model's parent directory.")

    # --- Reconstruction Parameters ---
    recon_group = parser.add_argument_group("Reconstruction Parameters")
    recon_group.add_argument("--reconstruct-modes", action='store_true', help="If set, also creates analysis folders for individual high-energy modes.")
    recon_group.add_argument("--energy-threshold", type=float, default=0.0, help="Minimum Koopman mode magnitude (energy norm) to be included in the physical space prediction. Default: 0.0")
    
    return parser.parse_args()

def decode_timeseries(model, latent_timeseries_tensor, denormalize_fn, batch_size=32, device="cpu"):
    """Decodes and denormalizes a timeseries of latent vectors in batches."""
    model.eval()
    decoded_snapshots = []
    with torch.no_grad():
        for i in tqdm(range(0, latent_timeseries_tensor.shape[0], batch_size), desc="Decoding Timeseries", leave=False, ncols=80):
            batch = latent_timeseries_tensor[i:i+batch_size].to(device)
            decoded_batch_norm = model.decode(batch) # Output: (B, C, X, Z)
            decoded_batch_denorm = denormalize_fn(decoded_batch_norm)
            decoded_batch_permuted = decoded_batch_denorm.permute(0, 2, 3, 1).cpu() # (B, X, Z, C)
            decoded_snapshots.append(decoded_batch_permuted)
    return torch.cat(decoded_snapshots, dim=0)

def analyze_and_save_reconstruction(
    true_recon_latent: torch.Tensor,
    pred_recon_latent: torch.Tensor,
    full_ground_truth_phys: torch.Tensor,
    output_subdir: Path,
    model: nn.Module,
    denormalize_fn,
    channel_names: list[str],
    device: str
):
    """
    Decodes, analyzes, and saves a pair of true/predicted reconstructed timeseries.
    The difference and stats are calculated against the FULL ground truth.
    """
    logging.info(f"--- Analyzing reconstruction for: {output_subdir.name} ---")
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Decode both the "true" reconstruction (for visual comparison) and the predicted reconstruction
    true_recon_phys = decode_timeseries(model, true_recon_latent, denormalize_fn, device=device)
    pred_recon_phys = decode_timeseries(model, pred_recon_latent, denormalize_fn, device=device)
    
    # --- METRICS CALCULATION (against FULL ground truth) ---
    loss_fn = nn.MSELoss(reduction='none')
    diff_phys = pred_recon_phys - full_ground_truth_phys
    error_tensor = loss_fn(pred_recon_phys, full_ground_truth_phys)

    avg_rollout_mse_total = error_tensor.mean().item()
    variance_total = full_ground_truth_phys.var().item()
    r_squared_total = 1 - (avg_rollout_mse_total / variance_total) if variance_total > 0 else 0.0
    
    avg_mse_per_channel = error_tensor.mean(dim=(0, 1, 2)).numpy()
    r_squared_per_channel = np.zeros(len(channel_names))
    for i in range(len(channel_names)):
        variance_channel = full_ground_truth_phys[..., i].var().item()
        r_squared_per_channel[i] = 1 - (avg_mse_per_channel[i] / variance_channel) if variance_channel > 0 else 0.0

    per_step_channel_error = error_tensor[1:].mean(dim=(1, 2)).numpy()

    # --- SAVE FILES ---
    # Save the reconstructed true timeseries (for visual comparison)
    np.savez(output_subdir / "true_timeseries.npz", timeseries=true_recon_phys.numpy(), labels=np.array(channel_names, dtype='U'))
    # Save the predicted timeseries
    np.savez(output_subdir / "predicted_timeseries.npz", timeseries=pred_recon_phys.numpy(), labels=np.array(channel_names, dtype='U'))
    # Save the difference timeseries (calculated against full ground truth)
    np.savez(output_subdir / "difference_timeseries.npz", timeseries=diff_phys.numpy(), labels=np.array(channel_names, dtype='U'))
    
    # Save the statistics (calculated against full ground truth)
    np.savez(
        output_subdir / "prediction_stats.npz",
        channel_names=np.array(channel_names, dtype='U'),
        r_squared_total=r_squared_total,
        r_squared_per_channel=r_squared_per_channel,
        per_step_channel_error=per_step_channel_error,
        avg_rollout_mse_total=avg_rollout_mse_total,
        avg_mse_per_channel=avg_mse_per_channel,
    )
    logging.info(f"Saved all analysis files to {output_subdir}")
    return r_squared_total, avg_rollout_mse_total

def main():
    """Main function to generate latent space predictions and analyze them."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model_path = Path(args.model_path)
    
    # --- Setup Output Paths ---
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_path.parent / "eval"
    
    analysis_dir = output_dir / "latent_space_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_file_path = analysis_dir / "latent_space_analysis.npz"

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
    channel_names = channels_used if channels_used else all_channel_names

    if full_timeseries.ndim == 5 and full_timeseries.shape[2] == 1:
        full_timeseries = np.squeeze(full_timeseries, axis=2)
    if channels_used:
        channel_indices = [all_channel_names.index(name) for name in channels_used]
        test_data_np = full_timeseries[..., channel_indices]
    else:
        test_data_np = full_timeseries
    
    test_tensor_cpu = torch.from_numpy(test_data_np).float()

    min_vals = all_min_vals[channel_indices] if channels_used else all_min_vals
    max_vals = all_max_vals[channel_indices] if channels_used else all_max_vals

    min_vals_norm, max_vals_norm = min_vals.view(1, -1, 1, 1), max_vals.view(1, -1, 1, 1)
    data_range = max_vals_norm - min_vals_norm + 1e-8
    def normalize(x): return (x - min_vals_norm) / data_range * 2.0 - 1.0
    def denormalize(x_norm): return (x_norm + 1.0) / 2.0 * data_range + min_vals_norm

    # --- Encode, Rollout, and Eigendecomposition ---
    true_latent_trajectory = []
    with torch.no_grad():
        for i in tqdm(range(test_data_np.shape[0]), desc="Encoding Ground Truth", ncols=80):
            snapshot_tensor = test_tensor_cpu[i].to(device).permute(2, 0, 1)
            latent_vector = model.encode(normalize(snapshot_tensor.unsqueeze(0)))
            true_latent_trajectory.append(latent_vector.squeeze(0).cpu())
    true_latent_trajectory = torch.stack(true_latent_trajectory)
    logging.info(f"Encoded ground truth trajectory. Latent shape: {true_latent_trajectory.shape}")

    num_timesteps, latent_dim = true_latent_trajectory.shape
    predicted_latent_trajectory = []
    z_t = true_latent_trajectory[0].unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_latent_trajectory.append(z_t.squeeze(0).cpu())
        for _ in tqdm(range(num_timesteps - 1), desc="Latent Rollout", ncols=80):
            z_t = model.koopman_step(z_t)
            predicted_latent_trajectory.append(z_t.squeeze(0).cpu())
    predicted_latent_trajectory = torch.stack(predicted_latent_trajectory)

    logging.info("Performing eigendecomposition and calculating Koopman mode magnitudes...")
    with torch.no_grad():
        K = model.koopman_operator.weight.cpu()
        eigenvalues, eigenvectors = torch.linalg.eig(K)
        koopman_mode_magnitudes = []
        for vec in tqdm(eigenvectors.T, desc="Decoding Eigenvectors", ncols=80):
            vec_complex = vec.cfloat().unsqueeze(0).to(device)
            decoded_real = denormalize(model.decode(vec_complex.real))
            decoded_imag = denormalize(model.decode(vec_complex.imag))
            mode_magnitude = torch.sqrt(torch.linalg.norm(decoded_real).pow(2) + torch.linalg.norm(decoded_imag).pow(2)).item()
            koopman_mode_magnitudes.append(mode_magnitude)
        koopman_mode_magnitudes = np.array(koopman_mode_magnitudes)
        
        try:
            W_inv = torch.linalg.inv(eigenvectors)
            true_projected_traj = (W_inv @ true_latent_trajectory.cfloat().T).T
            pred_projected_traj = (W_inv @ predicted_latent_trajectory.cfloat().T).T
            initial_mode_amplitudes = np.abs(true_projected_traj[0].numpy())
        except torch.linalg.LinAlgError:
            logging.error("Eigenvector matrix is singular; cannot perform projection.")
            true_projected_traj, pred_projected_traj, initial_mode_amplitudes = None, None, None

    # --- Calculate and Save Latent Space Analysis File ---
    loss_fn_latent = nn.MSELoss(reduction='none')
    per_step_latent_error = loss_fn_latent(predicted_latent_trajectory[1:], true_latent_trajectory[1:]).mean(dim=1).numpy()
    
    avg_rollout_mse_latent = np.mean(per_step_latent_error)
    variance_latent = true_latent_trajectory.var().item()
    r_squared_latent = 1 - (avg_rollout_mse_latent / variance_latent) if variance_latent > 0 else 0.0

    np.savez(
        analysis_file_path, 
        per_step_latent_error=per_step_latent_error,
        true_latent_trajectory=true_latent_trajectory.numpy(),
        predicted_latent_trajectory=predicted_latent_trajectory.numpy(),
        eigenvalues=eigenvalues.numpy(),
        koopman_mode_magnitudes=koopman_mode_magnitudes,
        true_projected_trajectory=true_projected_traj.numpy() if true_projected_traj is not None else None,
        pred_projected_trajectory=pred_projected_traj.numpy() if pred_projected_traj is not None else None,
        initial_mode_amplitudes=initial_mode_amplitudes if initial_mode_amplitudes is not None else None,
        avg_rollout_mse=avg_rollout_mse_latent,
        r_squared_latent=r_squared_latent,
    )
    logging.info(f"Saved latent space analysis to {analysis_file_path}")
    
    # --- Reconstruction and Physical Space Analysis ---
    if true_projected_traj is not None:
        high_energy_indices = np.where(koopman_mode_magnitudes > args.energy_threshold)[0]
        
        if len(high_energy_indices) == 0:
            logging.warning("No modes found above the energy threshold. Cannot perform physical space analysis.")
        else:
            # --- New prediction based on spectral formula: a_j(t) = (lambda_j)^t * a_j(0) ---
            logging.info("Calculating predicted trajectory using spectral decomposition formula.")
            initial_amplitudes = true_projected_traj[0]
            num_timesteps = true_projected_traj.shape[0]

            # Create a time evolution matrix of eigenvalues: [lambda_j^t]
            eigenvalues_t = eigenvalues.unsqueeze(0).pow(torch.arange(num_timesteps).unsqueeze(1).to(eigenvalues.device))
            
            # Calculate the time evolution of ALL mode amplitudes using the formula
            pred_projected_traj_from_formula = initial_amplitudes.unsqueeze(0) * eigenvalues_t
            
            # --- 1. Analyze Combined High-Energy Modes ---
            combined_true_recon_latent = torch.zeros_like(true_latent_trajectory)
            combined_pred_recon_latent = torch.zeros_like(predicted_latent_trajectory)
            
            processed_indices_combined = set()
            for idx in high_energy_indices:
                if idx in processed_indices_combined: continue
                is_real = np.isclose(eigenvalues[idx].imag, 0)
                if is_real:
                    combined_true_recon_latent += (true_projected_traj[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                    combined_pred_recon_latent += (pred_projected_traj_from_formula[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                    processed_indices_combined.add(idx)
                else:
                    conj_idx = np.where((np.isclose(eigenvalues.real, eigenvalues[idx].real)) & (np.isclose(eigenvalues.imag, -eigenvalues[idx].imag)))[0][0]
                    combined_true_recon_latent += 2 * (true_projected_traj[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                    combined_pred_recon_latent += 2 * (pred_projected_traj_from_formula[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                    processed_indices_combined.add(idx)
                    processed_indices_combined.add(conj_idx)

            r2_comb, mse_comb = analyze_and_save_reconstruction(
                combined_true_recon_latent, combined_pred_recon_latent, test_tensor_cpu,
                analysis_dir / "combined_high_energy_modes",
                model, denormalize, channel_names, device
            )

            # --- 2. Analyze Individual High-Energy Modes (Optional) ---
            if args.reconstruct_modes:
                energy_sorted_indices = np.argsort(koopman_mode_magnitudes)[::-1]
                high_energy_analysis_indices = [idx for idx in energy_sorted_indices if koopman_mode_magnitudes[idx] >= args.energy_threshold]
                
                processed_indices_individual = set()
                folder_rank_counter = 0
                
                logging.info(f"Found {len(high_energy_analysis_indices)} eigenvectors for individual analysis. Saving reconstructions...")
                
                for idx in tqdm(high_energy_analysis_indices, desc="Analyzing Individual Modes", ncols=80):
                    if idx in processed_indices_individual: continue

                    is_real = np.isclose(eigenvalues[idx].imag, 0)
                    if is_real:
                        true_recon = (true_projected_traj[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                        pred_recon = (pred_projected_traj_from_formula[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                        subdir = analysis_dir / f"mode_rank_{folder_rank_counter}"
                        processed_indices_individual.add(idx)
                    else:
                        conj_idx = np.where((np.isclose(eigenvalues.real, eigenvalues[idx].real)) & (np.isclose(eigenvalues.imag, -eigenvalues[idx].imag)))[0][0]
                        true_recon = 2 * (true_projected_traj[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                        pred_recon = 2 * (pred_projected_traj_from_formula[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                        subdir = analysis_dir / f"mode_pair_rank_{folder_rank_counter}"
                        processed_indices_individual.add(idx)
                        processed_indices_individual.add(conj_idx)

                    analyze_and_save_reconstruction(true_recon, pred_recon, test_tensor_cpu, subdir, model, denormalize, channel_names, device)
                    folder_rank_counter += 1

            # --- Final Summary Logging ---
            logging.info("--- PHYSICAL SPACE EVALUATION (FROM HIGH-ENERGY MODES) ---")
            logging.info(f"R² Score (Combined Modes): {r2_comb:.4f}")
            logging.info(f"Avg MSE (Combined Modes): {mse_comb:.6f}")

    else:
        logging.error("Cannot perform physical space analysis because eigenvector matrix was singular.")

if __name__ == "__main__":
    main()
