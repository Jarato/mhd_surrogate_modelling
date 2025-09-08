# -*- coding: utf-8 -*-
# experiments/study_3_q2d_tckae/train.py

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mhd_q2d_tckae.data import tcKAEMHDDataset
from mhd_q2d_tckae.model import tcKoopmanAutoencoderQ2D

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train a Temporally-Consistent Quasi-2D Koopman Autoencoder."
    )

    # --- Grouped Arguments for Clarity ---
    data_group = parser.add_argument_group("Data and I/O")
    data_group.add_argument("--data-path", type=str, required=True, help="Path to the pre-split train_val_set.npz.")
    data_group.add_argument("--norm-stats-path", type=str, required=True, help="Path to the normalization_stats.npz file.")
    data_group.add_argument("--output-dir", type=str, default="output_tckae", help="Directory to save the best model and logs.")
    data_group.add_argument("--channels", nargs='+', default=None, help="List of channel names to use for training.")
    data_group.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to a checkpoint to resume training.")

    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs.")
    train_group.add_argument("--batch-size", type=int, default=4, help="Number of independent blocks per batch.")
    train_group.add_argument("--num-workers", type=int, default=8, help="Number of worker processes for data loading.")

    optim_group = parser.add_argument_group("Optimizer and Scheduler")
    optim_group.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    optim_group.add_argument("--patience", type=int, default=20, help="Patience for early stopping.")
    optim_group.add_argument("--lr-patience", type=int, default=8, help="Patience for learning rate scheduler.")
    optim_group.add_argument("--lr-factor", type=float, default=0.1, help="Factor by which to reduce learning rate.")
    optim_group.add_argument("--clip-grad-value", type=float, default=25.0, help="Value to clip gradients to.")

    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--latent-dim", type=int, default=128, help="Dimension of the latent space.")
    model_group.add_argument("--bottleneck-dim", type=int, default=4096, help="Dimension of the intermediate bottleneck layer.")
    model_group.add_argument('--use-bottleneck', dest='use_bottleneck', action='store_true', help="Force the use of the bottleneck layer.")
    model_group.add_argument('--no-bottleneck', dest='use_bottleneck', action='store_false', help="Disable the bottleneck layer.")
    parser.set_defaults(use_bottleneck=True)

    tckae_group = parser.add_argument_group("tcKAE Hyperparameters")
    tckae_group.add_argument("--steps", type=int, default=15, help="Steps for learning forward dynamics (K).")
    tckae_group.add_argument("--sequence-length", type=int, default=8, help="Length of consecutive sequences for tc_loss (M).")
    tckae_group.add_argument("--steps-back", type=int, default=15, help="Steps for learning backward dynamics.")
    tckae_group.add_argument("--steps-tc", type=int, default=8, help="Steps to enforce temporal consistency loss.")
    tckae_group.add_argument("--epoch-trans", type=int, default=20, help="Epoch to start applying the temporal consistency loss.")
    tckae_group.add_argument("--backward", action='store_true', help="Flag to enable training with backward dynamics and consistency loss.")

    loss_group = parser.add_argument_group("Loss Weights (Gammas)")
    loss_group.add_argument("--gamma-identity", type=float, default=1.0, help="Weight for the identity/reconstruction loss.")
    loss_group.add_argument("--gamma-fwd", type=float, default=1.0, help="Weight for the forward prediction loss.")
    loss_group.add_argument("--gamma-bwd", type=float, default=1.0, help="Weight for the backward prediction loss.")
    loss_group.add_argument("--gamma-con", type=float, default=1e-4, help="Weight for the backward/forward consistency loss.")
    loss_group.add_argument("--gamma-tc", type=float, default=1.0, help="Weight for the temporal consistency loss.")

    return parser.parse_args()


def compute_loss_tckae(
    model: tcKoopmanAutoencoderQ2D,
    batch_of_blocks: torch.Tensor,
    gammas: Dict[str, float],
    epoch: int,
    epoch_trans: int,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Computes the full tcKAE loss for a batch of sequence blocks."""
    loss_fn = nn.MSELoss()
    batch_size = batch_of_blocks.shape[0]
    total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
    loss_dict = {
        "identity": 0.0, "forward": 0.0, "tc": 0.0,
        "backward": 0.0, "consistency": 0.0
    }

    # Iterate over each independent block in the batch
    for i in range(batch_size):
        block = batch_of_blocks[i]  # Shape: (M, steps+1, C, X, Y, Z)
        data_list = [block[:, k] for k in range(block.shape[1])] # List of M-sized tensors

        # --- Model Forward Pass ---
        outputs = model(data_list[0], mode='forward')

        # --- Identity Loss ---
        loss_identity = loss_fn(outputs["predicted_states"][-1], data_list[0])
        total_loss = total_loss + gammas["identity"] * loss_identity
        loss_dict["identity"] += loss_identity.detach()

        # --- Forward Prediction Loss ---
        loss_fwd = torch.tensor(0.0, device=DEVICE)
        for k in range(model.steps):
            loss_fwd = loss_fwd + loss_fn(outputs["predicted_states"][k], data_list[k + 1])
        loss_fwd = loss_fwd / model.steps
        total_loss = total_loss + gammas["fwd"] * loss_fwd
        loss_dict["forward"] += loss_fwd.detach()

        # --- Temporal Consistency Loss ---
        if epoch >= epoch_trans and gammas["tc"] > 0:
            loss_tc = torch.tensor(0.0, device=DEVICE)
            latent_preds = outputs["latent_states"]
            for q in range(1, model.steps_tc):
                loss_q = torch.tensor(0.0, device=DEVICE)
                for k in range(model.steps_tc - q):
                    loss_q = loss_q + loss_fn(latent_preds[k][q:], latent_preds[k + q][:-q])
                loss_tc = loss_tc + loss_q / (model.steps_tc - q)
            loss_tc = loss_tc / (model.steps_tc - 1)
            total_loss = total_loss + gammas["tc"] * loss_tc
            loss_dict["tc"] += loss_tc.detach()

        # --- Backward and Consistency Loss ---
        if gammas["bwd"] > 0 or gammas["con"] > 0:
            back_outputs = model(data_list[-1], mode='backward')
            loss_bwd = torch.tensor(0.0, device=DEVICE)
            for k in range(model.steps_back):
                loss_bwd = loss_bwd + loss_fn(back_outputs["predicted_states_back"][k], data_list[-(k + 2)])
            loss_bwd = loss_bwd / model.steps_back
            total_loss = total_loss + gammas["bwd"] * loss_bwd
            loss_dict["backward"] += loss_bwd.detach()

            A = model.koopman_operator.weight
            B = model.koopman_operator_backward.weight
            loss_consist = torch.tensor(0.0, device=DEVICE)
            for j in range(1, model.latent_dim + 1):
                I_j = torch.eye(j, device=DEVICE)
                term1 = torch.sum((torch.mm(B[:j, :], A[:, :j]) - I_j) ** 2)
                term2 = torch.sum((torch.mm(A[:j, :], B[:, :j]) - I_j) ** 2)
                loss_consist = loss_consist + (term1 + term2) / (2.0 * j)
            total_loss = total_loss + gammas["con"] * loss_consist
            loss_dict["consistency"] += loss_consist.detach()

    # Average the losses over the batch size
    final_loss = total_loss / batch_size
    final_loss_dict = {key: val / batch_size for key, val in loss_dict.items()}
    final_loss_dict["total"] = final_loss.detach()
    
    return final_loss, final_loss_dict


def validate_epoch(
    model: tcKoopmanAutoencoderQ2D,
    dataloader: DataLoader,
    gammas: Dict[str, float],
    epoch: int,
    epoch_trans: int,
) -> Dict[str, float]:
    """Computes validation loss for one epoch."""
    model.eval()
    total_losses = {
        "total": 0.0, "identity": 0.0, "forward": 0.0, "tc": 0.0,
        "backward": 0.0, "consistency": 0.0
    }
    with torch.no_grad():
        for batch_of_blocks in dataloader:
            batch_of_blocks = batch_of_blocks.to(DEVICE)
            _, loss_dict = compute_loss_tckae(
                model, batch_of_blocks, gammas, epoch, epoch_trans
            )
            for key in total_losses:
                total_losses[key] += loss_dict[key].item()

    # Average over all batches in the validation set
    return {key: val / len(dataloader) for key, val in total_losses.items()}


def main():
    """Main training and validation script."""
    args = parse_args()
    logging.info(f"Using device: {DEVICE}")

    # --- Setup Directories and Logging ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir / "logs")

    # --- Initialize Training State ---
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    channels_used = args.channels
    losses_at_best_epoch = {}
    best_component_losses = {
        "identity": float("inf"), "forward": float("inf"), "tc": float("inf"),
        "backward": float("inf"), "consistency": float("inf")
    }

    # --- Load from Checkpoint if Provided ---
    if args.resume_from_checkpoint:
        logging.info(f"Resuming training from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=DEVICE)
        model_config = checkpoint["config"]
        channels_used = checkpoint.get("channels_used")

        full_dataset = tcKAEMHDDataset(
            file_path=args.data_path,
            steps=model_config["steps"],
            sequence_length=model_config["sequence_length"],
            norm_stats_path=args.norm_stats_path,
            channels_to_use=channels_used,
            process_safe_copy=(args.num_workers > 0),
        )
        train_indices = checkpoint["train_indices"]
        val_indices = checkpoint["val_indices"]
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        model = tcKoopmanAutoencoderQ2D(**model_config).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        optimizer = Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        patience_counter = checkpoint["patience_counter"]
        best_epoch = checkpoint.get("best_epoch", 0)
        losses_at_best_epoch = checkpoint.get("losses_at_best_epoch", {})
        best_component_losses = checkpoint.get("best_component_losses", best_component_losses)

    else:
        # --- Initialize a New Run ---
        full_dataset = tcKAEMHDDataset(
            file_path=args.data_path,
            steps=args.steps,
            sequence_length=args.sequence_length,
            norm_stats_path=args.norm_stats_path,
            channels_to_use=args.channels,
            process_safe_copy=(args.num_workers > 0),
        )
        channels_used = full_dataset.channel_names
        
        stats = np.load(args.norm_stats_path)
        train_indices, val_indices = stats["train_indices"], stats["val_indices"]
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        sample_block = full_dataset[0]
        sample_x = sample_block[0, 0]
        
        model_config = {
            "in_channels": sample_x.shape[0],
            "latent_dim": args.latent_dim,
            "input_spatial_dims": sample_x.shape[1:],
            "steps": args.steps,
            "steps_back": args.steps_back,
            "steps_tc": args.steps_tc,
            "sequence_length": args.sequence_length,
            "bottleneck_dim": args.bottleneck_dim,
            "use_bottleneck": args.use_bottleneck,
        }
        model = tcKoopmanAutoencoderQ2D(**model_config).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience)

    # --- DataLoaders ---
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logging.info(f"Data split: {len(train_dataset)} train, {len(val_dataset)} val.")
    
    gammas = {
        "identity": args.gamma_identity, "fwd": args.gamma_fwd,
        "bwd": args.gamma_bwd if args.backward else 0.0,
        "con": args.gamma_con if args.backward else 0.0,
        "tc": args.gamma_tc,
    }
    logging.info(f"Using loss weights (gammas): {gammas}")

    # =========================================================================
    # --- TRAINING LOOP ---
    # =========================================================================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = { "total": 0.0, "identity": 0.0, "forward": 0.0, "tc": 0.0, "backward": 0.0, "consistency": 0.0 }
        epoch_grad_norm = 0.0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for batch_of_blocks in pbar:
            batch_of_blocks = batch_of_blocks.to(DEVICE)
            
            loss, loss_dict = compute_loss_tckae(model, batch_of_blocks, gammas, epoch, args.epoch_trans)
            
            optimizer.zero_grad()
            loss.backward()

            if args.clip_grad_value:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_value).item()
            else:
                grad_norm = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
            
            optimizer.step()

            # Accumulate losses and gradient norm for logging
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key].item()
            epoch_grad_norm += grad_norm
            pbar.set_postfix(loss=loss.item())

        # --- End of Epoch: Validation and Logging ---
        avg_train_losses = {key: val / len(train_dataloader) for key, val in epoch_losses.items()}
        avg_val_losses = validate_epoch(model, val_dataloader, gammas, epoch, args.epoch_trans)
        scheduler.step(avg_val_losses["total"])

        logging.info(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_losses['total']:.4f} | Val Loss: {avg_val_losses['total']:.4f}")
        
        # --- TensorBoard Logging ---
        writer.add_scalars("Loss/Total", {'train': avg_train_losses['total'], 'val': avg_val_losses['total']}, epoch)
        writer.add_scalar("Gradient/Norm", epoch_grad_norm / len(train_dataloader), epoch)
        for key in avg_train_losses:
            if key != "total":
                writer.add_scalars(f"Loss/{key}", {'train': avg_train_losses[key], 'val': avg_val_losses[key]}, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        # --- Checkpointing and Early Stopping ---
        for key in best_component_losses:
            if key in avg_val_losses:
                 best_component_losses[key] = min(best_component_losses[key], avg_val_losses[key])

        if avg_val_losses["total"] < best_val_loss:
            best_val_loss = avg_val_losses["total"]
            patience_counter = 0
            best_epoch = epoch + 1
            losses_at_best_epoch = avg_val_losses
            torch.save({
                'config': model_config, 
                'model_state_dict': model.state_dict(),
                'channels_used': channels_used
            }, output_dir / "best_model.pth")
            logging.info(f"New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        torch.save({
            "epoch": epoch, "config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss, "patience_counter": patience_counter,
            "best_epoch": best_epoch, "losses_at_best_epoch": losses_at_best_epoch,
            "best_component_losses": best_component_losses,
            "train_indices": train_indices, "val_indices": val_indices,
            "channels_used": channels_used,
        }, output_dir / "latest_checkpoint.pth")

        if patience_counter >= args.patience:
            logging.info("Early stopping triggered.")
            break

    # --- HParam Logging ---
    hparams = {
        **{k: v for k, v in vars(args).items() if k not in [
            'latent_dim', 'bottleneck_dim', 'channels', 'use_bottleneck',
            'steps', 'steps_back', 'steps_tc'
        ]},
        'latent_dim': model_config['latent_dim'],
        'bottleneck_dim': model_config.get('bottleneck_dim', 4096),
        'use_bottleneck': model_config.get('use_bottleneck', True),
        'steps': model_config['steps'],
        'steps_back': model_config['steps_back'],
        'steps_tc': model_config['steps_tc'],
        'channels': ",".join(channels_used) if channels_used is not None else "all",
    }
    
    # Clean up path objects for logging
    for key, value in hparams.items():
        if isinstance(value, Path):
            hparams[key] = str(value)
        elif key.endswith('_path') and value is not None:
             hparams[key] = str(value)

    if hparams.get('resume_from_checkpoint'):
        hparams['resume_from_checkpoint'] = str(hparams['resume_from_checkpoint'])

    final_metrics = {
        'hparam/best_val_loss': best_val_loss,
        'hparam/best_epoch': best_epoch,
    }
    
    for key, value in losses_at_best_epoch.items():
        final_metrics[f'hparam/{key}_at_best'] = value
        
    for key, value in best_component_losses.items():
        final_metrics[f'hparam/best_{key}_loss'] = value

    writer.add_hparams(hparams, final_metrics)
    
    writer.close()
    logging.info("Training finished.")

if __name__ == "__main__":
    main()

