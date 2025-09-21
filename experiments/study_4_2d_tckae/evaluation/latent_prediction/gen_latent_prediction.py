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
    parser = argparse.ArgumentParser(description="Evaluate a trained 2D tcKAE's latent space dynamics and optionally reconstruct high-energy modes.")
    
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
    recon_group.add_argument("--reconstruct-modes", action='store_true', help="If set, reconstructs and saves the timeseries for high-energy modes.")
    recon_group.add_argument("--energy-threshold", type=float, default=0.0, help="Minimum Koopman mode magnitude (energy norm) to be included in reconstruction. Default: 0.0")
    
    return parser.parse_args()

def decode_timeseries(model, latent_timeseries_tensor, batch_size=32, device="cpu"):
    """Decodes a timeseries of latent vectors in batches."""
    model.eval()
    decoded_snapshots = []
    with torch.no_grad():
        for i in tqdm(range(0, latent_timeseries_tensor.shape[0], batch_size), desc="Decoding Timeseries", leave=False):
            batch = latent_timeseries_tensor[i:i+batch_size].to(device)
            decoded_batch = model.decode(batch) # Output: (B, C, X, Z)
            decoded_batch_permuted = decoded_batch.permute(0, 2, 3, 1).cpu().numpy() # (B, X, Z, C)
            decoded_snapshots.append(decoded_batch_permuted)
    return np.vstack(decoded_snapshots)


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

    # --- Eigendecomposition and Projection ---
    logging.info("Performing eigendecomposition of the Koopman matrix...")
    with torch.no_grad():
        K = model.koopman_operator.weight.cpu()
        eigenvalues, eigenvectors = torch.linalg.eig(K)
        
        logging.info("Decoding latent eigenvectors to calculate Koopman mode magnitudes...")
        koopman_mode_magnitudes = []
        for vec in tqdm(eigenvectors.T, desc="Decoding Eigenvectors"):
            vec_complex = vec.cfloat().unsqueeze(0).to(device)
            decoded_real = model.decode(vec_complex.real)
            decoded_imag = model.decode(vec_complex.imag)
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
    
    # --- Save Main Analysis Results ---
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
        avg_rollout_mse=avg_rollout_mse,
        r_squared_latent=r_squared_latent,
    )
    logging.info(f"Main latent space analysis results saved to {analysis_file_path}")

    # --- High-Energy Mode Reconstruction ---
    if args.reconstruct_modes:
        if true_projected_traj is not None:
            logging.info(f"Reconstructing modes with energy > {args.energy_threshold}...")
            
            true_recon_dir = analysis_dir / "true_reconstruction"
            pred_recon_dir = analysis_dir / "predicted_reconstruction"
            true_recon_dir.mkdir(exist_ok=True)
            pred_recon_dir.mkdir(exist_ok=True)

            high_energy_indices = np.where(koopman_mode_magnitudes > args.energy_threshold)[0]
            
            combined_true_recon = torch.zeros_like(true_latent_trajectory, dtype=torch.float32)
            combined_pred_recon = torch.zeros_like(predicted_latent_trajectory, dtype=torch.float32)
            processed_indices = set()

            for idx in tqdm(high_energy_indices, desc="Reconstructing High-Energy Modes"):
                if idx in processed_indices:
                    continue

                is_real = np.isclose(eigenvalues[idx].imag, 0)
                
                if is_real:
                    true_recon_mode = (true_projected_traj[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                    pred_recon_mode = (pred_projected_traj[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                    processed_indices.add(idx)
                    filename = f"mode_{idx}.npz"
                else:
                    conj_idx_candidates = np.where(
                        (np.isclose(eigenvalues.real, eigenvalues[idx].real)) &
                        (np.isclose(eigenvalues.imag, -eigenvalues[idx].imag))
                    )[0]
                    conj_idx = conj_idx_candidates[0] if len(conj_idx_candidates) > 0 else idx

                    true_recon_mode = 2 * (true_projected_traj[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                    pred_recon_mode = 2 * (pred_projected_traj[:, idx].unsqueeze(1) * eigenvectors[:, idx].unsqueeze(0)).real
                    processed_indices.add(idx)
                    processed_indices.add(conj_idx)
                    filename = f"mode_pair_{idx}_{conj_idx}.npz"
                
                combined_true_recon += true_recon_mode
                combined_pred_recon += pred_recon_mode

                decoded_true = decode_timeseries(model, true_recon_mode, device=device)
                np.savez(true_recon_dir / filename, timeseries=decoded_true, labels=np.array(channel_names, dtype='U'))
                
                decoded_pred = decode_timeseries(model, pred_recon_mode, device=device)
                np.savez(pred_recon_dir / filename, timeseries=decoded_pred, labels=np.array(channel_names, dtype='U'))
                
            logging.info("Decoding and saving combined high-energy mode timeseries...")
            decoded_combined_true = decode_timeseries(model, combined_true_recon, device=device)
            np.savez(true_recon_dir / "combined_modes.npz", timeseries=decoded_combined_true, labels=np.array(channel_names, dtype='U'))

            decoded_combined_pred = decode_timeseries(model, combined_pred_recon, device=device)
            np.savez(pred_recon_dir / "combined_modes.npz", timeseries=decoded_combined_pred, labels=np.array(channel_names, dtype='U'))
        else:
            logging.warning("Skipping reconstruction because eigenvector matrix was singular.")

    # --- Log Final Metrics ---
    logging.info("--- LATENT SPACE EVALUATION COMPLETE ---")
    logging.info(f"Average Latent Space Rollout MSE: {avg_rollout_mse:.6e}")
    logging.info(f"Latent Space R-squared (R²) Score: {r_squared_latent:.4f}")

if __name__ == "__main__":
    main()


