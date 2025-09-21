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
    recon_group.add_argument("--reconstruct-modes", action='store_true', help="If set, also saves the timeseries for individual high-energy modes.")
    recon_group.add_argument("--energy-threshold", type=float, default=0.0, help="Minimum Koopman mode magnitude (energy norm) to be included in the physical space prediction. Default: 0.0")
    
    return parser.parse_args()

def decode_timeseries(model, latent_timeseries_tensor, denormalize_fn, batch_size=32, device="cpu"):
    """Decodes and denormalizes a timeseries of latent vectors in batches."""
    model.eval()
    decoded_snapshots = []
    with torch.no_grad():
        for i in tqdm(range(0, latent_timeseries_tensor.shape[0], batch_size), desc="Decoding Timeseries", leave=False):
            batch = latent_timeseries_tensor[i:i+batch_size].to(device)
            decoded_batch_norm = model.decode(batch) # Output: (B, C, X, Z)
            decoded_batch_denorm = denormalize_fn(decoded_batch_norm)
            decoded_batch_permuted = decoded_batch_denorm.permute(0, 2, 3, 1).cpu() # (B, X, Z, C)
            decoded_snapshots.append(decoded_batch_permuted)
    return torch.cat(decoded_snapshots, dim=0)


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

    phys_pred_dir = output_dir / "physical_space_prediction"
    phys_pred_dir.mkdir(parents=True, exist_ok=True)
    stats_path = phys_pred_dir / "prediction_stats.npz"
    pred_path = phys_pred_dir / "predicted_timeseries.npz"
    diff_path = phys_pred_dir / "difference_timeseries.npz"


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

    min_vals_norm, max_vals_norm = min_vals.view(1, -1, 1, 1), max_vals.view(1, -1, 1, 1)
    data_range = max_vals_norm - min_vals_norm + 1e-8
    def normalize(x): return (x - min_vals_norm) / data_range * 2.0 - 1.0
    def denormalize(x_norm): return (x_norm + 1.0) / 2.0 * data_range + min_vals_norm

    # --- Encode the Entire Ground Truth Trajectory ---
    true_latent_trajectory = []
    with torch.no_grad():
        for i in tqdm(range(test_data_np.shape[0]), desc="Encoding Ground Truth"):
            snapshot_np = test_data_np[i]
            snapshot_tensor = torch.from_numpy(snapshot_np).float().to(device)
            snapshot_permuted = snapshot_tensor.permute(2, 0, 1)
            snapshot_norm = normalize(snapshot_permuted.unsqueeze(0))
            latent_vector = model.encode(snapshot_norm)
            true_latent_trajectory.append(latent_vector.squeeze(0).cpu())
    
    true_latent_trajectory = torch.stack(true_latent_trajectory)
    logging.info(f"Encoded ground truth trajectory. Latent shape: {true_latent_trajectory.shape}")

    # --- Autoregressive Rollout in Latent Space ---
    num_timesteps, latent_dim = true_latent_trajectory.shape
    predicted_latent_trajectory = []
    
    z_t = true_latent_trajectory[0].unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_latent_trajectory.append(z_t.squeeze(0).cpu())
        for _ in tqdm(range(num_timesteps - 1), desc="Latent Rollout"):
            z_t = model.koopman_step(z_t)
            predicted_latent_trajectory.append(z_t.squeeze(0).cpu())
    predicted_latent_trajectory = torch.stack(predicted_latent_trajectory)

    # --- Eigendecomposition and Projection (Latent Analysis) ---
    logging.info("Performing eigendecomposition of the Koopman matrix...")
    with torch.no_grad():
        K = model.koopman_operator.weight.cpu()
        eigenvalues, eigenvectors = torch.linalg.eig(K)
        
        logging.info("Decoding latent eigenvectors to calculate Koopman mode magnitudes...")
        koopman_mode_magnitudes = []
        for vec in tqdm(eigenvectors.T, desc="Decoding Eigenvectors"):
            vec_complex = vec.cfloat().unsqueeze(0).to(device)
            decoded_real_norm = model.decode(vec_complex.real)
            decoded_imag_norm = model.decode(vec_complex.imag)
            decoded_real = denormalize(decoded_real_norm)
            decoded_imag = denormalize(decoded_imag_norm)
            norm_real_sq = torch.linalg.norm(decoded_real).pow(2)
            norm_imag_sq = torch.linalg.norm(decoded_imag).pow(2)
            mode_magnitude = torch.sqrt(norm_real_sq + norm_imag_sq).item()
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

    # --- Calculate and Save Latent Space Metrics ---
    loss_fn = nn.MSELoss(reduction='none')
    per_step_latent_error = []
    for t in range(1, num_timesteps):
        error = loss_fn(predicted_latent_trajectory[t], true_latent_trajectory[t]).mean().item()
        per_step_latent_error.append(error)

    avg_rollout_mse_latent = np.mean(per_step_latent_error)
    variance_latent = true_latent_trajectory.var().item()
    r_squared_latent = 1 - (avg_rollout_mse_latent / variance_latent) if variance_latent > 0 else 0.0
    
    np.savez(
        analysis_file_path, 
        per_step_latent_error=np.array(per_step_latent_error),
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
    logging.info(f"Main latent space analysis results saved to {analysis_file_path}")
    
    # --- Reconstruct Latent Trajectory from Filtered Modes for Physical Prediction ---
    final_predicted_latent_trajectory = None
    if true_projected_traj is not None:
        logging.info(f"Filtering for modes with energy > {args.energy_threshold} for physical prediction...")
        high_energy_indices = np.where(koopman_mode_magnitudes > args.energy_threshold)[0]
        
        if len(high_energy_indices) == 0:
            logging.warning("No modes found above the energy threshold. Physical prediction will be a zero field.")
            final_predicted_latent_trajectory = torch.zeros_like(predicted_latent_trajectory, dtype=torch.float32)
        else:
            logging.info(f"Reconstructing predicted trajectory using {len(high_energy_indices)} modes.")
            combined_pred_recon = torch.zeros_like(predicted_latent_trajectory, dtype=torch.float32)
            processed_indices = set()
            for idx in high_energy_indices:
                if idx in processed_indices:
                    continue
                is_real = np.isclose(eigenvalues[idx].imag, 0)
                if is_real:
                    pred_contribution = (pred_projected_traj[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                    processed_indices.add(idx)
                else:
                    conj_idx_candidates = np.where((np.isclose(eigenvalues.real, eigenvalues[idx].real)) & (np.isclose(eigenvalues.imag, -eigenvalues[idx].imag)))[0]
                    conj_idx = conj_idx_candidates[0] if len(conj_idx_candidates) > 0 else idx
                    pred_contribution = 2 * (pred_projected_traj[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                    processed_indices.add(idx)
                    processed_indices.add(conj_idx)
                combined_pred_recon += pred_contribution
            final_predicted_latent_trajectory = combined_pred_recon
    else:
        logging.warning("Eigenvector matrix was singular. Using full latent trajectory for physical prediction.")
        final_predicted_latent_trajectory = predicted_latent_trajectory

    # --- Decode Filtered Prediction and Calculate Physical Space Metrics ---
    logging.info("Decoding filtered predicted trajectory to physical space...")
    predicted_timeseries_phys = decode_timeseries(model, final_predicted_latent_trajectory, denormalize, device=device)
    test_tensor_cpu = torch.from_numpy(test_data_np).float()
    
    difference_timeseries = predicted_timeseries_phys - test_tensor_cpu
    error_tensor = loss_fn(predicted_timeseries_phys, test_tensor_cpu)

    avg_rollout_mse_total = error_tensor.mean().item()
    variance_total = test_tensor_cpu.var().item()
    r_squared_total = 1 - (avg_rollout_mse_total / variance_total) if variance_total > 0 else 0.0
    
    avg_mse_per_channel = error_tensor.mean(dim=(0, 1, 2)).numpy()
    r_squared_per_channel = np.zeros(len(channel_names))
    for i in range(len(channel_names)):
        variance_channel = test_tensor_cpu[..., i].var().item()
        r_squared_per_channel[i] = 1 - (avg_mse_per_channel[i] / variance_channel) if variance_channel > 0 else 0.0

    per_step_channel_error = error_tensor[1:].mean(dim=(1, 2)).numpy()

    # --- Save Physical Space Prediction Files ---
    np.savez(
        stats_path,
        channel_names=np.array(channel_names, dtype='U'),
        r_squared_total=r_squared_total,
        r_squared_per_channel=r_squared_per_channel,
        per_step_channel_error=per_step_channel_error,
        avg_rollout_mse_total=avg_rollout_mse_total,
        avg_mse_per_channel=avg_mse_per_channel,
    )
    logging.info(f"Physical space analysis statistics saved to {stats_path}")

    np.savez(pred_path, timeseries=predicted_timeseries_phys.numpy(), labels=np.array(channel_names, dtype='U'))
    logging.info(f"Predicted physical timeseries saved to {pred_path}")

    np.savez(diff_path, timeseries=difference_timeseries.numpy(), labels=np.array(channel_names, dtype='U'))
    logging.info(f"Difference physical timeseries saved to {diff_path}")

    # --- Optional: Save Individual and Combined Reconstructions ---
    if args.reconstruct_modes:
        if true_projected_traj is not None:
            logging.info(f"Reconstructing and saving individual/combined high-energy mode timeseries files...")
            
            true_recon_dir = analysis_dir / "true_reconstruction"
            pred_recon_dir = analysis_dir / "predicted_reconstruction"
            true_recon_dir.mkdir(exist_ok=True)
            pred_recon_dir.mkdir(exist_ok=True)

            high_energy_indices = np.where(koopman_mode_magnitudes > args.energy_threshold)[0]
            
            # Reconstruct the "true" trajectory from high-energy modes for comparison
            combined_true_recon = torch.zeros_like(true_latent_trajectory, dtype=torch.float32)
            processed_indices = set()

            for idx in tqdm(high_energy_indices, desc="Reconstructing Individual Modes"):
                if idx in processed_indices:
                    continue
                is_real = np.isclose(eigenvalues[idx].imag, 0)
                if is_real:
                    true_recon_mode = (true_projected_traj[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                    pred_recon_mode = (pred_projected_traj[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                    processed_indices.add(idx)
                    filename = f"mode_{idx}.npz"
                else:
                    conj_idx_candidates = np.where((np.isclose(eigenvalues.real, eigenvalues[idx].real)) & (np.isclose(eigenvalues.imag, -eigenvalues[idx].imag)))[0]
                    conj_idx = conj_idx_candidates[0] if len(conj_idx_candidates) > 0 else idx
                    true_recon_mode = 2 * (true_projected_traj[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                    pred_recon_mode = 2 * (pred_projected_traj[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                    processed_indices.add(idx)
                    processed_indices.add(conj_idx)
                    filename = f"mode_pair_{idx}_{conj_idx}.npz"
                
                combined_true_recon += true_recon_mode

                # Decode and save individual mode/pair timeseries
                decoded_true = decode_timeseries(model, true_recon_mode, denormalize, device=device).numpy()
                np.savez(true_recon_dir / filename, timeseries=decoded_true, labels=np.array(channel_names, dtype='U'))
                
                decoded_pred = decode_timeseries(model, pred_recon_mode, denormalize, device=device).numpy()
                np.savez(pred_recon_dir / filename, timeseries=decoded_pred, labels=np.array(channel_names, dtype='U'))
            
            # Decode and save the already-calculated combined predicted timeseries
            np.savez(pred_recon_dir / "combined_modes.npz", timeseries=predicted_timeseries_phys.numpy(), labels=np.array(channel_names, dtype='U'))
            
            # Decode and save the combined true timeseries
            decoded_combined_true = decode_timeseries(model, combined_true_recon, denormalize, device=device).numpy()
            np.savez(true_recon_dir / "combined_modes.npz", timeseries=decoded_combined_true, labels=np.array(channel_names, dtype='U'))

        else:
            logging.warning("Skipping reconstruction file saving because eigenvector matrix was singular.")

    # --- Log Final Metrics ---
    logging.info("--- LATENT SPACE EVALUATION COMPLETE ---")
    logging.info(f"Average Latent Space Rollout MSE: {avg_rollout_mse_latent:.6e}")
    logging.info(f"Latent Space R-squared (R²) Score: {r_squared_latent:.4f}")
    logging.info("--- PHYSICAL SPACE EVALUATION (FROM HIGH-ENERGY MODES) ---")
    logging.info(f"Overall Average Physical Rollout MSE: {avg_rollout_mse_total:.6f}")
    logging.info(f"Overall Physical R-squared (R²) Score: {r_squared_total:.4f}\n")
    logging.info("--- Per-Channel Physical Metrics ---")
    for i, name in enumerate(channel_names):
        logging.info(f"Channel '{name}':\t Avg MSE = {avg_mse_per_channel[i]:.6f},\t R² = {r_squared_per_channel[i]:.4f}")

if __name__ == "__main__":
    main()

