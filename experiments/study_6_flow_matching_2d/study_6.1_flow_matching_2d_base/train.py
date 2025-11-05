# -*- coding: utf-8 -*-
# experiments/study_6_flow_matching_2d/study_6.1_flow_matching_2d_base/train.py
#
# --- MODIFIED ---
# This script is for training a conditional Flow Matching model (study 6.1).
# It is adapted from the tcKAE training script (study 5.2).
#
# Key changes:
# 1. Imports new FlowMatchingUNet model and datasets.
# 2. Argument parsing updated for UNet features and removed tcKAE args.
# 3. `compute_loss_tckae` replaced with `compute_loss_fm` implementing
#    the flow matching loss (Eq. 3 from paper).
# 4. `validate_epoch_rollout` replaced with `validate_epoch_rollout_fm`
#    which performs auto-regressive validation by solving the FM ODE.
# 5. Data loading adjusted for the new FlowMatchingDataset2D (pairs).
# --- END MODIFICATION ---

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import the new 2D Flow Matching model and data classes
from flow_matching_2d_base.data import FlowMatchingDataset2D, RolloutDataset2D
from flow_matching_2d_base.model import FlowMatchingUNet

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- 1. Argument Parsing ---
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train a 2D Conditional Flow Matching Model." # <-- MODIFIED
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
    train_group.add_argument(
        '--save-best-to-scratch', 
        action='store_true', 
        help="Save the 'best_model.pth' to scratch_dir during training. (Default: persistent_dir)"
    )
    train_group.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs.")
    train_group.add_argument("--batch-size", type=int, default=4, help="Number of (y_k, y_k+1) pairs per batch for training.")
    train_group.add_argument("--validation-batch-size", type=int, default=None, help="Batch size for validation. Defaults to training batch size if not set.")
    train_group.add_argument("--num-workers", type=int, default=8, help="Number of worker processes for data loading.")
    train_group.add_argument("--validation-num-workers", type=int, default=None, help="Number of workers for validation. Defaults to num-workers if not set.")
    train_group.add_argument("--validation-rollout-steps", type=int, default=50, help="Number of auto-regressive steps for validation.")
    # --- NEW ARGUMENT ---
    train_group.add_argument("--validation-ode-steps", type=int, default=10, help="Number of Euler steps to solve the FM ODE during validation.")

    optim_group = parser.add_argument_group("Optimizer and Scheduler")
    optim_group.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    optim_group.add_argument("--patience", type=int, default=20, help="Patience for early stopping.")
    optim_group.add_argument("--lr-patience", type=int, default=8, help="Patience for learning rate scheduler.")
    optim_group.add_argument("--lr-factor", type=float, default=0.1, help="Factor by which to reduce learning rate.")
    optim_group.add_argument("--clip-grad-value", type=float, default=25.0, help="Value to clip gradients to.")
    
    # --- MODIFIED: Model Arguments ---
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--features", type=int, nargs='+', default=[64, 128, 256], help="List of feature channels for the UNet encoder.")
    model_group.add_argument("--time-embed-dim", type=int, default=64, help="Dimension of the sinusoidal time embedding.")
    # --- END MODIFICATION ---
    
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

    # --- MODIFIED: Index calculation for new datasets ---
    # FlowMatchingDataset needs pairs (k, k+1)
    train_indices = np.arange(0, train_size - 1)
    
    # RolloutDataset needs sequences of length (rollout_steps + 1)
    # The last valid index is total_timesteps - (rollout_steps + 1)
    # np.arange(A, B) creates indices up to B-1.
    # So we need B = total_timesteps - (rollout_steps + 1) + 1 = total_timesteps - rollout_steps
    val_indices = np.arange(train_size, total_timesteps - args.validation_rollout_steps)
    # --- END MODIFICATION ---

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

    # --- MODIFIED: Use new datasets ---
    train_dataset_full = FlowMatchingDataset2D(
        full_timeseries=data_dict["timeseries"],
        all_channel_names=data_dict["all_channel_names"],
        norm_stats=data_dict["norm_stats"],
        channels_to_use=args.channels,
        process_safe_copy=process_safe,
    )
    val_dataset_full = RolloutDataset2D(
        full_timeseries=data_dict["timeseries"],
        all_channel_names=data_dict["all_channel_names"],
        rollout_steps=args.validation_rollout_steps,
        norm_stats=data_dict["norm_stats"],
        channels_to_use=args.channels,
        process_safe_copy=val_process_safe,
    )
    # --- END MODIFICATION ---

    train_dataset = Subset(train_dataset_full, data_dict["train_indices"])
    val_dataset = Subset(val_dataset_full, data_dict["val_indices"])

    val_batch_size = args.validation_batch_size if args.validation_batch_size else args.batch_size
    val_num_workers = args.validation_num_workers if args.validation_num_workers is not None else args.num_workers
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers, pin_memory=True)

    logging.info(f"Data split: {len(train_dataset)} train pairs, {len(val_dataset)} val sequences.")

    # Get sample shape from the *first item* of the pair (y_k)
    sample_input_shape = train_dataset_full[0][0].shape

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "channels_used": train_dataset_full.channel_names,
        "sample_input_shape": sample_input_shape, # (C, H, W)
    }


# --- 3. Core Training Logic and Evaluation ---
def compute_loss_fm(
    model: FlowMatchingUNet,
    y_k: torch.Tensor,
    y_k_plus_1: torch.Tensor
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the conditional flow matching loss.
    Loss = || v_theta(x_t, t, y_k) - (x_1 - x_0) ||^2
    
    Args:
        model: The FlowMatchingUNet model.
        y_k: Condition state (B, C, H, W) - Normalized
        y_k_plus_1: Target state (B, C, H, W) - Normalized
    """
    loss_fn = nn.MSELoss()
    B = y_k.shape[0]
    
    # 1. Sample time t ~ U[0, 1]
    # t shape: (B, 1)
    t = torch.rand(B, 1, device=DEVICE)
    
    # 2. Sample noise x_0 ~ N(0, I)
    x_0 = torch.randn_like(y_k)
    
    # 3. Define x_1 (target) and y (condition)
    x_1 = y_k_plus_1
    y_condition = y_k
    
    # 4. Interpolate to find x_t = t*x_1 + (1-t)*x_0
    # Reshape t for broadcasting: (B, 1, 1, 1)
    t_broadcast = t.view(B, 1, 1, 1)
    x_t = t_broadcast * x_1 + (1 - t_broadcast) * x_0
    
    # 5. Define the target velocity (x_1 - x_0)
    target_velocity = x_1 - x_0
    
    # 6. Get the predicted velocity from the model
    # We must squeeze 't' from (B, 1) to (B,) for the time embedding module
    predicted_velocity = model(x_t, t.squeeze(1), y_condition)
    
    # 7. Compute the loss
    loss = loss_fn(predicted_velocity, target_velocity)
    
    loss_dict = {
        "total": loss.detach()
    }
    
    return loss, loss_dict


def validate_epoch_rollout_fm(
    model: FlowMatchingUNet, 
    dataloader: DataLoader, 
    rollout_steps: int,
    ode_steps: int
) -> Dict[str, float]:
    """
    Performs auto-regressive rollout validation by solving the FM ODE.
    """
    model.eval()
    loss_fn = nn.MSELoss()
    total_rollout_loss = 0.0
    dt = 1.0 / ode_steps

    with torch.no_grad():
        pbar_val = tqdm(dataloader, desc="Validation Rollout", leave=False)
        for batch_sequence in pbar_val:
            # batch_sequence: (B, T, C, H, W) where T = rollout_steps + 1
            batch_sequence = batch_sequence.to(DEVICE)
            
            # y_0 is the initial condition for the rollout
            y_0 = batch_sequence[:, 0]
            ground_truth = batch_sequence[:, 1:] # (B, rollout_steps, C, H, W)
            
            current_y = y_0
            batch_rollout_loss = 0.0

            # --- Auto-regressive loop ---
            for k in range(rollout_steps):
                y_condition = current_y # This is y_k
                
                # Sample initial noise x_0 for this step's ODE solve
                x_t = torch.randn_like(y_condition)
                
                # --- Euler ODE Solver Loop ---
                for t_step in range(ode_steps):
                    # t_val is the current time, from 0 to (1 - dt)
                    t_val = t_step * dt
                    # Create a batch of time tensors
                    t = torch.full((x_t.shape[0],), t_val, device=DEVICE)
                    
                    # Get velocity v(x_t, t, y_k)
                    velocity = model(x_t, t, y_condition)
                    
                    # Euler step: x_{t+dt} = x_t + v*dt
                    x_t = x_t + velocity * dt
                # --- End ODE Solver ---
                
                # The final x_t (which is x_1) is our prediction for y_{k+1}
                y_k_plus_1_pred = x_t
                
                # Compare to ground truth
                batch_rollout_loss += loss_fn(y_k_plus_1_pred, ground_truth[:, k])
                
                # Set up for next auto-regressive step
                current_y = y_k_plus_1_pred
            
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

# This function is generic and reused as-is from user's script
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
    Saves the current training state as 'latest_checkpoint.pth'.
    The 'best_model.pth' is saved separately by the main training loop.
    """
    
    should_save_checkpoint_to_scratch = (epoch + 1) % save_freq == 0 or is_last_epoch
    should_save_checkpoint_to_persistent = scratch_dir and (is_persistent_save or is_last_epoch)

    if should_save_checkpoint_to_scratch:
        save_dir = scratch_dir if scratch_dir else persistent_dir
        torch.save(data, save_dir / "latest_checkpoint.pth")
        
    if should_save_checkpoint_to_persistent:
        torch.save(data, persistent_dir / "latest_checkpoint.pth")
        logging.info(f"Checkpoint synced to persistent storage (epoch {epoch+1}).")


# --- 5. Main Training Loop ---
def run_training_loop(
    args: argparse.Namespace,
    training_state: Dict[str, Any],
    data_assets: Dict[str, Any],
    dirs: Dict[str, Path],
    writer: SummaryWriter,
) -> Dict[str, Any]:
    """Executes the main training and validation loop for all epochs."""
    # Unpack assets
    model = training_state["model"]
    optimizer = training_state["optimizer"]
    scheduler = training_state["scheduler"]
    train_loader = data_assets["train_loader"]
    val_loader = data_assets["val_loader"]
    persistent_dir, scratch_dir = dirs["persistent"], dirs["scratch"]
    
    # Determine where to save the best model
    if args.save_best_to_scratch and scratch_dir:
        best_model_save_dir = scratch_dir
        logging.info(f"Will save 'best_model.pth' to scratch directory: {scratch_dir}")
    else:
        best_model_save_dir = persistent_dir
        if args.save_best_to_scratch and not scratch_dir:
            logging.warning("--save-best-to-scratch was set, but --scratch-dir is not. Defaulting to persistent_dir.")
        else:
            logging.info(f"Will save 'best_model.pth' to persistent directory: {persistent_dir}")

    # Loop state
    best_val_loss = training_state["best_val_loss"]
    patience_counter = training_state["patience_counter"]
    best_epoch = training_state["best_epoch"]
    losses_at_best_epoch = training_state["losses_at_best_epoch"]
    
    # In-memory best model state
    best_model_state_dict_cpu = training_state["best_model_state_dict_cpu"]
    best_model_meta_data = training_state["best_model_meta_data"]


    for epoch in range(training_state["start_epoch"], args.epochs):
        model.train()
        
        # --- MODIFIED: Simplified losses ---
        epoch_losses = { "total": 0.0 }
        epoch_grad_norm = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)

        # --- Training Step ---
        for batch in pbar:
            # batch is (y_k, y_k_plus_1)
            y_k, y_k_plus_1 = batch
            y_k, y_k_plus_1 = y_k.to(DEVICE), y_k_plus_1.to(DEVICE)
            
            loss, loss_dict = compute_loss_fm(model, y_k, y_k_plus_1)
            
            optimizer.zero_grad()
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_value).item() if args.clip_grad_value else sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
            optimizer.step()

            for key in epoch_losses: epoch_losses[key] += loss_dict[key].item()
            epoch_grad_norm += grad_norm
            pbar.set_postfix(loss=loss.item())

        # --- Validation and Logging ---
        avg_train_losses = {key: val / len(train_loader) for key, val in epoch_losses.items()}
        
        # --- MODIFIED: Call new validation function ---
        avg_val_losses = validate_epoch_rollout_fm(
            model, val_loader, 
            args.validation_rollout_steps, 
            args.validation_ode_steps
        )
        # --- END MODIFICATION ---
        
        scheduler.step(avg_val_losses["total"])
        
        logging.info(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_losses['total']:.4f} | Val Rollout Loss: {avg_val_losses['total']:.4f}")
        writer.add_scalars("Loss/Total", {'train': avg_train_losses['total'], 'val': avg_val_losses['total']}, epoch)
        writer.add_scalar("Gradient/Norm", epoch_grad_norm / len(train_loader), epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        # --- Model Saving and Early Stopping ---
        if avg_val_losses["total"] < best_val_loss:
            best_val_loss, patience_counter, best_epoch = avg_val_losses["total"], 0, epoch + 1
            # losses_at_best_epoch now just contains 'total'
            losses_at_best_epoch = avg_val_losses
            
            # Save best model to CPU RAM
            best_model_state_dict_cpu = model.state_dict().to('cpu')
            best_model_meta_data = {
                'config': training_state["model_config"], 
                'channels_used': data_assets["channels_used"], 
                'norm_stats': data_assets["norm_stats"]
            }
            logging.info(f"New best model state captured in RAM (Val Rollout Loss: {best_val_loss:.4f})")
            
        else:
            patience_counter += 1

        # Add in-memory best model state to checkpoint data
        checkpoint_data = {
            "epoch": epoch, "config": training_state["model_config"], "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss, "patience_counter": patience_counter, "best_epoch": best_epoch,
            "losses_at_best_epoch": losses_at_best_epoch, "train_indices": data_assets["train_indices"],
            "val_indices": data_assets["val_indices"], "channels_used": data_assets["channels_used"],
            "norm_stats": data_assets["norm_stats"],
            "best_model_state_dict_cpu": best_model_state_dict_cpu,
            "best_model_meta_data": best_model_meta_data
        }
        
        is_last = (epoch == args.epochs - 1)
        is_persistent_save = (epoch + 1) % args.persistent_save_freq == 0
        
        save_checkpoint(checkpoint_data, epoch, args.checkpoint_save_freq, persistent_dir, scratch_dir, is_last, is_persistent_save)

        if patience_counter >= args.patience:
            logging.info("Early stopping triggered.")
            break
            
    # --- Save the best model at the end of the run ---
    final_best_model_path = None
    if best_model_state_dict_cpu and best_model_meta_data:
        logging.info(f"Saving best model (from epoch {best_epoch}) to {best_model_save_dir}.")
        best_model_save_data = {
            **best_model_meta_data,
            'model_state_dict': best_model_state_dict_cpu
        }
        final_best_model_path = best_model_save_dir / "best_model.pth"
        torch.save(best_model_save_data, final_best_model_path)
    else:
        logging.warning("Training finished, but no best model state was captured to save.")
            
    return {
        "best_val_loss": best_val_loss, 
        "best_epoch": best_epoch, 
        "losses_at_best": losses_at_best_epoch,
        "final_best_model_path": final_best_model_path # Return path
    }


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
    model_config = {} # Define in outer scope
    
    resume_checkpoint_path = find_latest_checkpoint(persistent_dir, scratch_dir) if args.resume else None

    if resume_checkpoint_path:
        logging.info(f"Resuming training from {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location='cpu')
        
        model_config = checkpoint["config"]
        # --- MODIFIED: Ensure resumed run uses checkpoint's HParams ---
        args.channels = model_config.get("channels_used")
        args.features = model_config.get("features")
        args.time-embed-dim = model_config.get("time_embed_dim")
        # --- END MODIFIED ---

        data_assets = create_dataloaders(args, {**data_dict, "norm_stats": checkpoint["norm_stats"], "train_indices": checkpoint["train_indices"], "val_indices": checkpoint["val_indices"]})
        
        model = FlowMatchingUNet(**model_config).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        optimizer = Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        training_state = {
            "model": model, "optimizer": optimizer, "scheduler": scheduler, "model_config": model_config,
            "start_epoch": checkpoint["epoch"] + 1, "best_val_loss": checkpoint["best_val_loss"],
            "patience_counter": checkpoint["patience_counter"], "best_epoch": checkpoint.get("best_epoch", 0),
            "losses_at_best_epoch": checkpoint.get("losses_at_best_epoch", {}),
            "best_model_state_dict_cpu": checkpoint.get("best_model_state_dict_cpu"),
            "best_model_meta_data": checkpoint.get("best_model_meta_data")
        }

    else:
        if args.resume:
            logging.warning("Resume flag set, but no checkpoint found. Starting new run.")
        
        data_assets = create_dataloaders(args, data_dict)
        
        # --- MODIFIED: New model config ---
        in_channels = data_assets["sample_input_shape"][0]
        
        model_config = {
            "in_channels": in_channels,
            "condition_channels": in_channels, # Condition y_k has same channels as input x_t
            "time_embed_dim": args.time_embed_dim,
            "features": args.features,
            "channels_used": data_assets["channels_used"] # Save for resuming
        }
        
        logging.info(f"Initializing new model with config: {model_config}")
        
        model = FlowMatchingUNet(**{k: v for k, v in model_config.items() if k != 'channels_used'}).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience)

        training_state = {
            "model": model, "optimizer": optimizer, "scheduler": scheduler, "model_config": model_config,
            "start_epoch": 0, "best_val_loss": float("inf"), "patience_counter": 0, "best_epoch": 0,
            "losses_at_best_epoch": {},
            "best_model_state_dict_cpu": None,
            "best_model_meta_data": None
        }
        # --- END MODIFICATION ---

    all_data_assets = {**data_dict, **data_assets}
    
    final_metrics = run_training_loop(
        args, training_state, all_data_assets,
        dirs={"persistent": persistent_dir, "scratch": scratch_dir},
        writer=writer
    )

    # --- Copy best model from scratch to persistent if needed ---
    final_best_model_path = final_metrics.get("final_best_model_path")
    
    if args.save_best_to_scratch and scratch_dir and final_best_model_path and final_best_model_path.exists():
        if final_best_model_path.parent == scratch_dir:
            try:
                shutil.copy2(final_best_model_path, persistent_dir / "best_model.pth")
                logging.info(f"Copied final 'best_model.pth' from scratch to persistent storage.")
            except Exception as e:
                logging.error(f"Failed to copy 'best_model.pth' from scratch to persistent: {e}")
    elif final_best_model_path and not final_best_model_path.exists():
         logging.warning(f"Training finished, but 'best_model.pth' was not found at {final_best_model_path}. No final copy was made.")

    # --- Finalize and Log HParams ---
    hparams = vars(args).copy()
    
    # Update hparams with values from model_config
    hparams.update({
        'features': str(model_config['features']),
        'time_embed_dim': model_config['time_embed_dim'],
        'channels': ",".join(data_assets["channels_used"]) if data_assets["channels_used"] is not None else "all"
    })
    
    # Clean up non-scalar values
    for key, value in hparams.items():
        if isinstance(value, (Path, list)): 
            hparams[key] = str(value)
        elif value is None:
            hparams[key] = "None" 

    metric_dict = {
        'hparam/best_val_loss': final_metrics["best_val_loss"],
        'hparam/best_epoch': final_metrics["best_epoch"]
    }
    # Add losses at best epoch (e.g., 'total')
    for key, value in final_metrics["losses_at_best"].items():
        metric_dict[f'hparam/{key}_at_best'] = value

    writer.add_hparams(hparams, metric_dict)
    writer.close()
    logging.info("Training finished.")


if __name__ == "__main__":
    main()