# -*- coding: utf-8 -*-
# experiments/study_5_low_rank_tckae_2d/study_5.2_stable_svd/train.py
#
# --- MODIFIED ---
# This script is for the STABLE SVD operator.
# It changes the loss function for singular values from:
# L_sigma = || Sigma - 1 ||^2 (forcing unitarity)
# to:
# L_stability = || ReLU(Sigma - 1) ||^2 (forcing stability, sigma <= 1)
#
# --- MODIFICATION: Optimized Saving Strategy ---
# 1. The "best model" state_dict is now held in CPU RAM to save VRAM and disk I/O.
# 2. The best_model.pth file is only written to disk (from the RAM copy)
#    at the same time a checkpoint is saved, ensuring no data loss on crash.
# --- END MODIFICATION ---

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import the new 2D versions of the data and SVD model classes
# Note: We re-use the *exact same* model and data modules,
# as the architecture is identical. Only the loss function changes.
from low_rank_tckae_2d_stable_svd.data import tcKAEMHDDataset2D, RolloutMHDDataset2D
from low_rank_tckae_2d_stable_svd.model import tcKoopmanAutoencoder2D

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- 1. Argument Parsing ---
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train a Temporally-Consistent 2D Koopman Autoencoder (Stable SVD Version)." # <-- MODIFIED description
    )
    data_group = parser.add_argument_group("Data and I/O")
    data_group.add_argument("--data-path", type=str, required=True, help="Path to the pre-split train_val_set.npz.")
    data_group.add_argument("--norm-stats-path", type=str, default=None, help="Optional path to pre-computed normalization stats.")
    data_group.add_argument("--persistent-dir", type=str, required=True, help="Required path for logs and best model (persistent storage).")
    data_group.add_argument("--scratch-dir", type=str, default=None, help="Optional path for frequent checkpoints (fast, temporary storage).")
    data_group.add_argument("--channels", nargs='+', default=None, help="List of channel names to use for training.")
    data_group.add_argument("--resume", action="store_true", help="Flag to resume training from the latest available checkpoint.")
    
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument("--val-split", type=float, default=0.2, help="Fraction of data for validation.")
    train_group.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split.")
    train_group.add_argument("--checkpoint-save-freq", type=int, default=1, help="Frequency (in epochs) to save the latest checkpoint.")
    train_group.add_argument("--persistent-save-freq", type=int, default=10, help="Frequency (in epochs) to save a checkpoint to persistent storage if scratch-dir is used.")
    train_group.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs.")
    train_group.add_argument("--batch-size", type=int, default=4, help="Number of independent blocks per batch for training.")
    train_group.add_argument("--validation-batch-size", type=int, default=None, help="Batch size for validation. Defaults to training batch size if not set.")
    train_group.add_argument("--num-workers", type=int, default=8, help="Number of worker processes for data loading.")
    train_group.add_argument("--validation-num-workers", type=int, default=None, help="Number of workers for validation. Defaults to num-workers if not set.")
    train_group.add_argument("--validation-rollout-steps", type=int, default=50, help="Number of auto-regressive steps for validation.")
    
    optim_group = parser.add_argument_group("Optimizer and Scheduler")
    optim_group.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    optim_group.add_argument("--patience", type=int, default=20, help="Patience for early stopping.")
    optim_group.add_argument("--lr-patience", type=int, default=8, help="Patience for learning rate scheduler.")
    optim_group.add_argument("--lr-factor", type=float, default=0.1, help="Factor by which to reduce learning rate.")
    optim_group.add_argument("--clip-grad-value", type=float, default=25.0, help="Value to clip gradients to.")
    
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--latent-dim", type=int, default=None, help="Dimension of the latent space (d). Required if --no-flattened-for-koopman is set.")
    model_group.add_argument("--koopman-rank", type=int, default=None, help="Rank 'r' for the SVD Koopman operator. If not set, defaults to --latent-dim (if using bottleneck) or fails otherwise.")
    model_group.add_argument("--bottleneck-dim", type=int, default=None, help="Dimension of the intermediate bottleneck layer. Required if --no-flattened-for-koopman and --use-bottleneck are set.")
    
    model_group.add_argument('--use-flattened-for-koopman', dest='use_flattened_for_koopman', action='store_true', help="Use the high-dim flattened space for Koopman. (Default)")
    model_group.add_argument('--no-flattened-for-koopman', dest='use_flattened_for_koopman', action='store_false', help="Use a lower-dim latent space for Koopman.")
    parser.set_defaults(use_flattened_for_koopman=True)
    
    model_group.add_argument('--use-bottleneck', dest='use_bottleneck', action='store_true', help="Force the use of the bottleneck layer (if not using flattened).")
    model_group.add_argument('--no-bottleneck', dest='use_bottleneck', action='store_false', help="Disable the bottleneck layer (if not using flattened).")
    parser.set_defaults(use_bottleneck=True)
    
    tckae_group = parser.add_argument_group("tcKAE Hyperparameters")
    tckae_group.add_argument("--steps", type=int, default=15, help="Steps for learning forward dynamics (K).")
    tckae_group.add_argument("--sequence-length", type=int, default=8, help="Length of consecutive sequences for tc_loss (M).")
    tckae_group.add_argument("--steps-back", type=int, default=15, help="Steps for learning backward dynamics.")
    tckae_group.add_argument("--steps-tc", type=int, default=8, help="Steps to enforce temporal consistency loss.")
    tckae_group.add_argument("--epoch-trans", type=int, default=20, help="Epoch to start applying the temporal consistency loss.")
    tckae_group.add_argument("--backward", action='store_true', help="Flag to enable training with backward dynamics.")
    
    loss_group = parser.add_argument_group("Loss Weights (Gammas)")
    loss_group.add_argument("--gamma-identity", type=float, default=1.0, help="Weight for the identity/reconstruction loss.")
    loss_group.add_argument("--gamma-fwd", type=float, default=1.0, help="Weight for the forward prediction loss.")
    loss_group.add_argument("--gamma-bwd", type=float, default=1.0, help="Weight for the backward prediction loss.")
    loss_group.add_argument("--gamma-tc", type=float, default=1.0, help="Weight for the temporal consistency loss.")
    # --- MODIFIED ARGUMENT (Renamed for clarity) ---
    loss_group.add_argument("--gamma-svd-ortho", type=float, default=1e-4, help="Weight for SVD orthonormality loss (U^T*U=I, V^T*V=I).")
    loss_group.add_argument("--gamma-svd-stability", type=float, default=1e-4, help="Weight for SVD one-sided stability loss (ReLU(Sigma-1)).")
    
    return parser.parse_args()


# --- 2. Environment and Data Setup ---
def setup_environment(args: argparse.Namespace) -> Tuple[Path, Path | None, SummaryWriter]:
    """Creates directories and initializes the TensorBoard writer."""
    persistent_dir = Path(args.persistent_dir)
    persistent_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=persistent_dir / "logs")

    scratch_dir = Path(args.scratch_dir) if args.scratch_dir else None
    if scratch_dir:
        scratch_dir.mkdir(parents=True, exist_ok=True)

    return persistent_dir, scratch_dir, writer


def setup_data_and_stats(args: argparse.Namespace) -> Dict[str, Any]:
    """Loads raw data, performs train/val split, and computes norm stats."""
    logging.info(f"Loading data from {args.data_path}")
    with np.load(args.data_path, allow_pickle=True) as data:
        timeseries = data["timeseries"]
        all_channel_names = list(data["labels"])

    total_timesteps = timeseries.shape[0]
    val_size = int(total_timesteps * args.val_split)
    train_size = total_timesteps - val_size

    if args.norm_stats_path:
        logging.info(f"Loading pre-computed normalization stats from {args.norm_stats_path}")
        with np.load(args.norm_stats_path) as data:
            norm_stats = {key: data[key] for key in data.files}
    else:
        logging.info(f"Computing normalization stats (min, max, mean, std) on {train_size} training timesteps.")
        train_data_raw = timeseries[:train_size]
        
        stat_axes = tuple(range(train_data_raw.ndim - 1))
        logging.info(
            f"Input data has {train_data_raw.ndim} dimensions. Calculating stats over axes: {stat_axes}"
        )
        
        min_vals = np.min(train_data_raw, axis=stat_axes)
        max_vals = np.max(train_data_raw, axis=stat_axes)
        mean_vals = np.mean(train_data_raw, axis=stat_axes)
        std_vals = np.std(train_data_raw, axis=stat_axes)
        
        norm_stats = {
            "min_vals": min_vals, 
            "max_vals": max_vals, 
            "mean_vals": mean_vals,
            "std_vals": std_vals
        }

    train_block_len = args.sequence_length + args.steps
    val_seq_len = args.validation_rollout_steps + 1
    
    train_indices = np.arange(0, train_size - train_block_len + 1)
    val_indices = np.arange(train_size, total_timesteps - val_seq_len + 1)

    return {
        "timeseries": timeseries,
        "all_channel_names": all_channel_names,
        "norm_stats": norm_stats,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }


def create_dataloaders(args: argparse.Namespace, data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Creates and returns the train and validation dataloaders."""
    process_safe = args.num_workers > 0
    val_process_safe = (args.validation_num_workers if args.validation_num_workers is not None else args.num_workers) > 0

    train_dataset_full = tcKAEMHDDataset2D(
        full_timeseries=data_dict["timeseries"],
        all_channel_names=data_dict["all_channel_names"],
        sequence_length=args.sequence_length,
        steps=args.steps,
        norm_stats=data_dict["norm_stats"],
        channels_to_use=args.channels,
        process_safe_copy=process_safe,
    )
    val_dataset_full = RolloutMHDDataset2D(
        full_timeseries=data_dict["timeseries"],
        all_channel_names=data_dict["all_channel_names"],
        rollout_steps=args.validation_rollout_steps,
        norm_stats=data_dict["norm_stats"],
        channels_to_use=args.channels,
        process_safe_copy=val_process_safe,
    )

    train_dataset = Subset(train_dataset_full, data_dict["train_indices"])
    val_dataset = Subset(val_dataset_full, data_dict["val_indices"])

    val_batch_size = args.validation_batch_size if args.validation_batch_size else args.batch_size
    val_num_workers = args.validation_num_workers if args.validation_num_workers is not None else args.num_workers
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers, pin_memory=True)

    logging.info(f"Data split: {len(train_dataset)} train, {len(val_dataset)} val.")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "channels_used": train_dataset_full.channel_names,
        "sample_input_shape": train_dataset_full[0][0, 0].shape,
    }


# --- 3. Core Training Logic and Evaluation ---
def compute_loss_tckae(
    model: tcKoopmanAutoencoder2D,
    batch_of_blocks: torch.Tensor,
    gammas: Dict[str, float],
    epoch: int,
    epoch_trans: int,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the combined tcKAE loss.
    --- MODIFIED ---
    Replaced the Sigma=1 loss with the one-sided
    ReLU(Sigma - 1) stability loss.
    """
    loss_fn = nn.MSELoss()
    B, M, T, C, X, Z = batch_of_blocks.shape
    model_input = batch_of_blocks[:, :, 0].reshape(B * M, C, X, Z)
    data_list = [batch_of_blocks[:, :, t].reshape(B * M, C, X, Z) for t in range(T)]

    outputs = model(model_input, mode='forward')
    loss_identity = loss_fn(outputs["predicted_states"][-1], data_list[0])
    loss_fwd = sum(loss_fn(outputs["predicted_states"][k], data_list[k + 1]) for k in range(model.steps)) / model.steps
    
    loss_tc = torch.tensor(0.0, device=DEVICE)
    if epoch >= epoch_trans and gammas["tc"] > 0:
        latent_preds = outputs["latent_states"]
        for q in range(1, model.steps_tc):
            loss_q = sum(loss_fn(latent_preds[k].view(B, M, -1)[:, q:, :], latent_preds[k + q].view(B, M, -1)[:, :-q, :]) for k in range(model.steps_tc - q))
            loss_tc += loss_q / (model.steps_tc - q)
        loss_tc /= (model.steps_tc - 1)
        
    loss_bwd = torch.tensor(0.0, device=DEVICE)
    if gammas["bwd"] > 0:
        back_outputs = model(data_list[-1], mode='backward')
        loss_bwd = sum(loss_fn(back_outputs["predicted_states_back"][k], data_list[-(k + 2)]) for k in range(model.steps_back)) / model.steps_back
        
    # --- MODIFIED SVD LOSSES ---
    loss_svd_ortho = torch.tensor(0.0, device=DEVICE)
    loss_svd_stability = torch.tensor(0.0, device=DEVICE) # <-- Renamed
    
    if gammas["svd_ortho"] > 0:
        r = model.koopman_rank
        I_r = torch.eye(r, device=DEVICE)
        U = model.U_learn
        V = model.V_learn
        
        U_T_U = U.t() @ U
        V_T_V = V.t() @ V
        # L_unitarity = || U^T*U - I ||^2 + || V^T*V - I ||^2
        loss_svd_ortho = loss_fn(U_T_U, I_r) + loss_fn(V_T_V, I_r)

    # --- THIS IS THE CORE CHANGE ---
    if gammas["svd_stability"] > 0: # <-- Renamed gamma key
        Sigma_vec = model.Sigma_learn
        
        # L_stability = || ReLU(Sigma_vec - 1) ||^2
        # We want to penalize values > 1.
        # torch.relu(Sigma_vec - 1) gives (sigma - 1) if sigma > 1, and 0 otherwise.
        # We then compute the MSE loss of this penalty term against a vector of zeros.
        
        ones_r = torch.ones_like(Sigma_vec)
        zeros_r = torch.zeros_like(Sigma_vec)

        # Calculate the one-sided penalty term
        penalty_term = torch.relu(Sigma_vec - ones_r) # This is ReLU(Sigma - 1)
        
        # Calculate MSE loss against zero
        loss_svd_stability = loss_fn(penalty_term, zeros_r)
    # --- END SVD STABILITY LOSS MODIFICATION ---

    total_loss = (
        gammas["identity"] * loss_identity + 
        gammas["fwd"] * loss_fwd + 
        gammas["tc"] * loss_tc + 
        gammas["bwd"] * loss_bwd + 
        gammas["svd_ortho"] * loss_svd_ortho +
        gammas["svd_stability"] * loss_svd_stability # <-- Renamed
    )
    
    loss_dict = {
        "total": total_loss.detach(), 
        "identity": loss_identity.detach(), 
        "forward": loss_fwd.detach(), 
        "tc": loss_tc.detach(), 
        "backward": loss_bwd.detach(), 
        "svd_ortho": loss_svd_ortho.detach(),
        "svd_stability": loss_svd_stability.detach() # <-- Renamed
    }
    
    return total_loss, loss_dict


def validate_epoch_rollout(model: tcKoopmanAutoencoder2D, dataloader: DataLoader, rollout_steps: int) -> Dict[str, float]:
    """Performs auto-regressive rollout validation."""
    model.eval()
    loss_fn = nn.MSELoss()
    total_rollout_loss = 0.0
    with torch.no_grad():
        pbar_val = tqdm(dataloader, desc="Validation Rollout", leave=False)
        for batch_sequence in pbar_val:
            batch_sequence = batch_sequence.to(DEVICE)
            initial_conditions, ground_truth = batch_sequence[:, 0], batch_sequence[:, 1:]
            
            z_k = model.encode(initial_conditions)
            batch_rollout_loss = 0.0
            for k in range(rollout_steps):
                z_k = model.koopman_step(z_k)
                x_k_pred = model.decode(z_k)
                batch_rollout_loss += loss_fn(x_k_pred, ground_truth[:, k])
            
            total_rollout_loss += (batch_rollout_loss / rollout_steps).item()
            
    return {"total": total_rollout_loss / len(dataloader)}


# --- 4. Checkpoint Management ---
def find_latest_checkpoint(persistent_dir: Path, scratch_dir: Path | None) -> Path | None:
    """Finds the most recent checkpoint file."""
    persistent_ckpt = persistent_dir / "latest_checkpoint.pth"
    scratch_ckpt = scratch_dir / "latest_checkpoint.pth" if scratch_dir else None
    persistent_exists = persistent_ckpt.exists()
    scratch_exists = scratch_ckpt.exists() if scratch_dir else False

    if not persistent_exists and not scratch_exists: return None
    if scratch_exists and persistent_exists:
        return scratch_ckpt if scratch_ckpt.stat().st_mtime > persistent_ckpt.stat().st_mtime else persistent_ckpt
    return scratch_ckpt if scratch_exists else persistent_ckpt


# --- MODIFIED ---
def save_checkpoint(
    data: Dict[str, Any], 
    epoch: int, 
    save_freq: int, 
    persistent_dir: Path, 
    scratch_dir: Path | None, 
    is_last_epoch: bool = False, 
    is_persistent_save: bool = False
):
    """
    Saves the current training state as a checkpoint.
    
    --- MODIFIED (User Request) ---
    This function ONLY saves the 'latest_checkpoint.pth'.
    The 'best_model.pth' is now saved *only* at the end of the
    training run by the `run_training_loop` function.
    """
    
    should_save_checkpoint_to_scratch = (epoch + 1) % save_freq == 0 or is_last_epoch
    should_save_checkpoint_to_persistent = scratch_dir and (is_persistent_save or is_last_epoch)

    if should_save_checkpoint_to_scratch:
        save_dir = scratch_dir if scratch_dir else persistent_dir
        torch.save(data, save_dir / "latest_checkpoint.pth")
        
    if should_save_checkpoint_to_persistent:
        # This block runs if scratch_dir is used AND it's time for a persistent save.
        torch.save(data, persistent_dir / "latest_checkpoint.pth")
        logging.info(f"Checkpoint synced to persistent storage (epoch {epoch+1}).")
# --- END MODIFICATION ---


# --- 5. Main Training Loop ---
def run_training_loop(
    args: argparse.Namespace,
    training_state: Dict[str, Any],
    data_assets: Dict[str, Any],
    dirs: Dict[str, Path],
    writer: SummaryWriter,
):
    """Executes the main training and validation loop for all epochs."""
    # Unpack assets
    model = training_state["model"]
    optimizer = training_state["optimizer"]
    scheduler = training_state["scheduler"]
    train_loader = data_assets["train_loader"]
    val_loader = data_assets["val_loader"]
    persistent_dir, scratch_dir = dirs["persistent"], dirs["scratch"]
    
    # --- MODIFIED gammas dict ---
    gammas = {
        "identity": args.gamma_identity, 
        "fwd": args.gamma_fwd, 
        "bwd": args.gamma_bwd if args.backward else 0.0, 
        "tc": args.gamma_tc,
        "svd_ortho": args.gamma_svd_ortho,
        "svd_stability": args.gamma_svd_stability # <-- Renamed
    }
    
    # Loop state
    best_val_loss = training_state["best_val_loss"]
    patience_counter = training_state["patience_counter"]
    best_epoch = training_state["best_epoch"]
    losses_at_best_epoch = training_state["losses_at_best_epoch"]
    
    # --- MODIFIED: Load in-memory best model state ---
    best_model_state_dict_cpu = training_state["best_model_state_dict_cpu"]
    best_model_meta_data = training_state["best_model_meta_data"]
    # --- END MODIFICATION ---


    for epoch in range(training_state["start_epoch"], args.epochs):
        model.train()
        
        # --- MODIFIED epoch_losses dict ---
        epoch_losses = {
            "total": 0.0, "identity": 0.0, "forward": 0.0, 
            "tc": 0.0, "backward": 0.0, "svd_ortho": 0.0, "svd_stability": 0.0 # <-- Renamed
        }
        epoch_grad_norm = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)

        # --- Training Step ---
        for batch_of_blocks in pbar:
            batch_of_blocks = batch_of_blocks.to(DEVICE)
            loss, loss_dict = compute_loss_tckae(model, batch_of_blocks, gammas, epoch, args.epoch_trans)
            
            optimizer.zero_grad()
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_value).item() if args.clip_grad_value else sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
            optimizer.step()

            for key in epoch_losses: epoch_losses[key] += loss_dict[key].item()
            epoch_grad_norm += grad_norm
            pbar.set_postfix(loss=loss.item())

        # --- Validation and Logging ---
        avg_train_losses = {key: val / len(train_loader) for key, val in epoch_losses.items()}
        avg_val_losses = validate_epoch_rollout(model, val_loader, args.validation_rollout_steps)
        scheduler.step(avg_val_losses["total"])
        
        logging.info(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_losses['total']:.4f} | Val Rollout Loss: {avg_val_losses['total']:.4f}")
        writer.add_scalars("Loss/Total", {'train': avg_train_losses['total'], 'val': avg_val_losses['total']}, epoch)
        writer.add_scalar("Gradient/Norm", epoch_grad_norm / len(train_loader), epoch)
        for key in avg_train_losses:
            if key != "total": writer.add_scalars(f"Loss/{key}", {'train': avg_train_losses[key]}, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        # --- Model Saving and Early Stopping ---
        if avg_val_losses["total"] < best_val_loss:
            best_val_loss, patience_counter, best_epoch = avg_val_losses["total"], 0, epoch + 1
            losses_at_best_epoch = avg_val_losses
            
            # --- MODIFIED: Save best model to CPU RAM instead of disk ---
            best_model_state_dict_cpu = model.state_dict().to('cpu')
            best_model_meta_data = {
                'config': training_state["model_config"], 
                'channels_used': data_assets["channels_used"], 
                'norm_stats': data_assets["norm_stats"]
            }
            logging.info(f"New best model state captured in RAM (Val Rollout Loss: {best_val_loss:.4f})")
            # --- END MODIFICATION ---
            
        else:
            patience_counter += 1

        # --- MODIFIED: Add in-memory best model state to checkpoint data ---
        checkpoint_data = {
            "epoch": epoch, "config": training_state["model_config"], "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss, "patience_counter": patience_counter, "best_epoch": best_epoch,
            "losses_at_best_epoch": losses_at_best_epoch, "train_indices": data_assets["train_indices"],
            "val_indices": data_assets["val_indices"], "channels_used": data_assets["channels_used"],
            "norm_stats": data_assets["norm_stats"],
            "best_model_state_dict_cpu": best_model_state_dict_cpu, # <-- ADDED
            "best_model_meta_data": best_model_meta_data           # <-- ADDED
        }
        # --- END MODIFICATION ---
        
        is_last = (epoch == args.epochs - 1)
        is_persistent_save = (epoch + 1) % args.persistent_save_freq == 0
        
        # --- MODIFIED: save_checkpoint no longer saves best_model.pth ---
        save_checkpoint(checkpoint_data, epoch, args.checkpoint_save_freq, persistent_dir, scratch_dir, is_last, is_persistent_save)
        # --- END MODIFICATION ---

        if patience_counter >= args.patience:
            logging.info("Early stopping triggered.")
            break
            
    # --- MODIFIED: Save the best model at the end of the run ---
    if best_model_state_dict_cpu and best_model_meta_data:
        logging.info(f"Saving best model (from epoch {best_epoch}) to persistent storage.")
        best_model_save_data = {
            **best_model_meta_data,
            'model_state_dict': best_model_state_dict_cpu
        }
        torch.save(best_model_save_data, persistent_dir / "best_model.pth")
    else:
        logging.warning("Training finished, but no best model state was captured to save.")
    # --- END MODIFICATION ---
            
    return {"best_val_loss": best_val_loss, "best_epoch": best_epoch, "losses_at_best": losses_at_best_epoch}


# --- 6. Orchestration ---
def main():
    """
    Main function to orchestrate the entire training pipeline.
    """
    args = parse_args()
    logging.info(f"Using device: {DEVICE}")

    persistent_dir, scratch_dir, writer = setup_environment(args)
    data_dict = setup_data_and_stats(args)
    
    training_state = {}
    
    resume_checkpoint_path = find_latest_checkpoint(persistent_dir, scratch_dir) if args.resume else None

    if resume_checkpoint_path:
        logging.info(f"Resuming training from {resume_checkpoint_path}")
        # --- MODIFIED: Load checkpoint to CPU first to avoid VRAM spike if resuming on different device ---
        checkpoint = torch.load(resume_checkpoint_path, map_location='cpu')
        
        model_config = checkpoint["config"]
        # --- MODIFIED: Ensure resumed run uses checkpoint's HParams ---
        args.sequence_length, args.steps = model_config["sequence_length"], model_config["steps"]
        args.channels = model_config.get("channels_used")
        
        # Load the new config flags
        args.use_flattened_for_koopman = model_config.get("use_flattened_for_koopman", False) # Default to old behavior if flag missing
        args.use_bottleneck = model_config.get("use_bottleneck", True) # Default to old behavior if flag missing
        args.latent_dim = model_config.get("latent_dim") # Will be None if not present
        args.bottleneck_dim = model_config.get("bottleneck_dim") # Will be None if not present
        args.koopman_rank = model_config.get("koopman_rank") 
        
        if args.koopman_rank is None:
                    # Fallback for checkpoints saved *just* before this change
                    # but after latent_dim was added
                    args.koopman_rank = model_config.get("latent_dim")
                    logging.warning(f"Resumed checkpoint missing 'koopman_rank', falling back to 'latent_dim': {args.koopman_rank}")
        # --- END MODIFIED ---

        data_assets = create_dataloaders(args, {**data_dict, "norm_stats": checkpoint["norm_stats"], "train_indices": checkpoint["train_indices"], "val_indices": checkpoint["val_indices"]})
        
        # --- MODIFIED: Load model to device ---
        model = tcKoopmanAutoencoder2D(**model_config).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        optimizer = Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # --- MODIFIED: Load in-memory best model state ---
        training_state = {
            "model": model, "optimizer": optimizer, "scheduler": scheduler, "model_config": model_config,
            "start_epoch": checkpoint["epoch"] + 1, "best_val_loss": checkpoint["best_val_loss"],
            "patience_counter": checkpoint["patience_counter"], "best_epoch": checkpoint.get("best_epoch", 0),
            "losses_at_best_epoch": checkpoint.get("losses_at_best_epoch", {}),
            "best_model_state_dict_cpu": checkpoint.get("best_model_state_dict_cpu"), # <-- ADDED (will be None if old checkpoint)
            "best_model_meta_data": checkpoint.get("best_model_meta_data")           # <-- ADDED (will be None if old checkpoint)
        }
        # --- END MODIFICATION ---

    else:
        if args.resume:
            logging.warning("Resume flag set, but no checkpoint found. Starting new run.")
        
        data_assets = create_dataloaders(args, data_dict)
        
        koopman_rank_r = args.koopman_rank
        if koopman_rank_r is None:
            if not args.use_flattened_for_koopman and args.latent_dim is not None:
                koopman_rank_r = args.latent_dim
                logging.info(f"Koopman rank not set, defaulting to latent_dim: {koopman_rank_r}")
            else:
                # This is an error. If using flattened, or if latent_dim is also None, rank must be set.
                raise ValueError("Must provide --koopman-rank. (It can only be defaulted to --latent-dim when --no-flattened-for-koopman is set)")
        
        logging.info(f"Initializing model with Koopman Rank (r)={koopman_rank_r}")
        
        model_config = {
            "in_channels": data_assets["sample_input_shape"][0],
            "koopman_rank": koopman_rank_r, # <-- This is 'r'
            "input_spatial_dims": data_assets["sample_input_shape"][1:],
            "steps": args.steps,
            "steps_back": args.steps_back,
            "steps_tc": args.steps_tc,
            "sequence_length": args.sequence_length,
            # Pass the new flags
            "use_flattened_for_koopman": args.use_flattened_for_koopman,
            "use_bottleneck": args.use_bottleneck,
            "latent_dim": args.latent_dim,      # This is 'd' (or None)
            "bottleneck_dim": args.bottleneck_dim,  # This is 'b' (or None)
        }
        
        model = tcKoopmanAutoencoder2D(**model_config).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience)

        # --- MODIFIED: Initialize in-memory best model state as None ---
        training_state = {
            "model": model, "optimizer": optimizer, "scheduler": scheduler, "model_config": model_config,
            "start_epoch": 0, "best_val_loss": float("inf"), "patience_counter": 0, "best_epoch": 0,
            "losses_at_best_epoch": {},
            "best_model_state_dict_cpu": None, # <-- ADDED
            "best_model_meta_data": None       # <-- ADDED
        }
        # --- END MODIFICATION ---

    all_data_assets = {**data_dict, **data_assets}
    
    final_metrics = run_training_loop(
        args, training_state, all_data_assets,
        dirs={"persistent": persistent_dir, "scratch": scratch_dir},
        writer=writer
    )

    # --- Finalize and Log HParams ---
    hparams = vars(args).copy()
    
    # Update hparams with values from model_config to reflect actual running config
    hparams.update({
        'koopman_rank': model_config['koopman_rank'],
        'use_flattened_for_koopman': model_config.get('use_flattened_for_koopman', False),
        'use_bottleneck': model_config.get('use_bottleneck', True),
        'latent_dim': model_config.get('latent_dim'),
        'bottleneck_dim': model_config.get('bottleneck_dim'),
        'steps': model_config['steps'],
        'steps_back': model_config['steps_back'],
        'steps_tc': model_config['steps_tc'],
        'channels': ",".join(data_assets["channels_used"]) if data_assets["channels_used"] is not None else "all"
    })
    
    # Clean up non-scalar values for hparam logging
    for key, value in hparams.items():
        if isinstance(value, (Path, list)): 
            hparams[key] = str(value)
        elif value is None:
            hparams[key] = "None" # Convert None to string for logging

    metric_dict = {
        'hparam/best_val_loss': final_metrics["best_val_loss"],
        'hparam/best_epoch': final_metrics["best_epoch"]
    }
    for key, value in final_metrics["losses_at_best"].items():
        metric_dict[f'hparam/{key}_at_best'] = value

    writer.add_hparams(hparams, metric_dict)
    writer.close()
    logging.info("Training finished.")


if __name__ == "__main__":
    main()

