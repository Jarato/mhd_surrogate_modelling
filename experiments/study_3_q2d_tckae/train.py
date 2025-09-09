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

from mhd_q2d_tckae.data import tcKAEMHDDataset, RolloutMHDDataset
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

    data_group = parser.add_argument_group("Data and I/O")
    data_group.add_argument("--data-path", type=str, required=True, help="Path to the pre-split train_val_set.npz.")
    data_group.add_argument("--norm-stats-path", type=str, required=True, help="Path to the normalization_stats.npz file.")
    data_group.add_argument("--persistent-dir", type=str, required=True, help="Required path for logs and best model (persistent storage).")
    data_group.add_argument("--scratch-dir", type=str, default=None, help="Optional path for frequent checkpoints (fast, temporary storage).")
    data_group.add_argument("--channels", nargs='+', default=None, help="List of channel names to use for training.")
    data_group.add_argument("--resume", action="store_true", help="Flag to resume training from the latest available checkpoint.")
    
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument("--checkpoint-save-freq", type=int, default=1, help="Frequency (in epochs) to save the latest checkpoint.")
    train_group.add_argument("--persistent-save-freq", type=int, default=10, help="Frequency (in epochs) to save a checkpoint to persistent storage if scratch-dir is used.")
    train_group.add_argument("--epochs", type=int, default=200, help="Maximum number of training epochs.")
    train_group.add_argument("--batch-size", type=int, default=4, help="Number of independent blocks per batch for training.")
    train_group.add_argument("--validation-batch-size", type=int, default=None, help="Batch size for validation. Defaults to training batch size if not set.")
    train_group.add_argument("--num-workers", type=int, default=8, help="Number of worker processes for data loading.")
    train_group.add_argument("--validation-rollout-steps", type=int, default=50, help="Number of auto-regressive steps for validation.")

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


def find_latest_checkpoint(persistent_dir: Path, scratch_dir: Path | None) -> Path | None:
    """Finds the most recent checkpoint file between scratch and persistent dirs."""
    persistent_ckpt = persistent_dir / "latest_checkpoint.pth"
    scratch_ckpt = scratch_dir / "latest_checkpoint.pth" if scratch_dir else None
    persistent_exists, scratch_exists = persistent_ckpt.exists(), scratch_ckpt.exists() if scratch_dir else False
    if not persistent_exists and not scratch_exists:
        return None
    if scratch_exists and persistent_exists:
        return scratch_ckpt if scratch_ckpt.stat().st_mtime > persistent_ckpt.stat().st_mtime else persistent_ckpt
    return scratch_ckpt if scratch_exists else persistent_ckpt


def compute_loss_tckae(
    model: tcKoopmanAutoencoderQ2D,
    batch_of_blocks: torch.Tensor,
    gammas: Dict[str, float],
    epoch: int,
    epoch_trans: int,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Computes the full tcKAE loss for a batch of sequence blocks in a vectorized manner."""
    loss_fn = nn.MSELoss()
    B, M, T, C, X, Y, Z = batch_of_blocks.shape
    
    model_input = batch_of_blocks[:, :, 0].reshape(B * M, C, X, Y, Z)
    data_list = [batch_of_blocks[:, :, t].reshape(B * M, C, X, Y, Z) for t in range(T)]

    outputs = model(model_input, mode='forward')

    loss_identity = loss_fn(outputs["predicted_states"][-1], data_list[0])

    loss_fwd = torch.tensor(0.0, device=DEVICE)
    for k in range(model.steps):
        loss_fwd = loss_fwd + loss_fn(outputs["predicted_states"][k], data_list[k + 1])
    loss_fwd = loss_fwd / model.steps

    loss_tc = torch.tensor(0.0, device=DEVICE)
    if epoch >= epoch_trans and gammas["tc"] > 0:
        latent_preds = outputs["latent_states"]
        for q in range(1, model.steps_tc):
            loss_q = torch.tensor(0.0, device=DEVICE)
            for k in range(model.steps_tc - q):
                pred_k = latent_preds[k].view(B, M, -1)
                pred_kq = latent_preds[k + q].view(B, M, -1)
                loss_q = loss_q + loss_fn(pred_k[:, q:, :], pred_kq[:, :-q, :])
            loss_tc = loss_tc + loss_q / (model.steps_tc - q)
        loss_tc = loss_tc / (model.steps_tc - 1)

    loss_bwd = torch.tensor(0.0, device=DEVICE)
    loss_consist = torch.tensor(0.0, device=DEVICE)
    if gammas["bwd"] > 0 or gammas["con"] > 0:
        back_outputs = model(data_list[-1], mode='backward')
        for k in range(model.steps_back):
            loss_bwd = loss_bwd + loss_fn(back_outputs["predicted_states_back"][k], data_list[-(k + 2)])
        loss_bwd = loss_bwd / model.steps_back

        A = model.koopman_operator.weight
        B = model.koopman_operator_backward.weight
        for j in range(1, model.latent_dim + 1):
            I_j = torch.eye(j, device=DEVICE)
            term1 = torch.sum((torch.mm(B[:j, :], A[:, :j]) - I_j) ** 2)
            term2 = torch.sum((torch.mm(A[:j, :], B[:, :j]) - I_j) ** 2)
            loss_consist = loss_consist + (term1 + term2) / (2.0 * j)

    total_loss = (
        gammas["identity"] * loss_identity + gammas["fwd"] * loss_fwd +
        gammas["tc"] * loss_tc + gammas["bwd"] * loss_bwd + gammas["con"] * loss_consist
    )
    loss_dict = {
        "total": total_loss.detach(), "identity": loss_identity.detach(),
        "forward": loss_fwd.detach(), "tc": loss_tc.detach(),
        "backward": loss_bwd.detach(), "consistency": loss_consist.detach(),
    }
    return total_loss, loss_dict


def validate_epoch_rollout(
    model: tcKoopmanAutoencoderQ2D,
    dataloader: DataLoader,
    rollout_steps: int,
) -> Dict[str, float]:
    """
    Computes validation loss via auto-regressive rollout. This is a more
    realistic measure of forecasting performance.
    """
    model.eval()
    loss_fn = nn.MSELoss()
    total_rollout_loss = 0.0

    with torch.no_grad():
        for batch_sequence in dataloader:
            batch_sequence = batch_sequence.to(DEVICE)
            
            initial_conditions = batch_sequence[:, 0]
            ground_truth = batch_sequence[:, 1:]

            z_k = model.encode(initial_conditions)
            batch_rollout_loss = 0.0
            for k in range(rollout_steps):
                z_k = model.koopman_step(z_k)
                x_k_pred = model.decode(z_k)
                batch_rollout_loss += loss_fn(x_k_pred, ground_truth[:, k])
            
            total_rollout_loss += (batch_rollout_loss / rollout_steps).item()

    avg_rollout_loss = total_rollout_loss / len(dataloader)
    return {"total": avg_rollout_loss}


def main():
    """Main training and validation script."""
    args = parse_args()
    logging.info(f"Using device: {DEVICE}")

    persistent_dir = Path(args.persistent_dir)
    persistent_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=persistent_dir / "logs")

    scratch_dir = Path(args.scratch_dir) if args.scratch_dir else None
    if scratch_dir:
        scratch_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, best_val_loss, patience_counter, best_epoch = 0, float("inf"), 0, 0
    channels_used = args.channels
    losses_at_best_epoch = {}
    
    resume_checkpoint_path = find_latest_checkpoint(persistent_dir, scratch_dir) if args.resume else None

    if resume_checkpoint_path:
        logging.info(f"Resuming training from {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=DEVICE)
        model_config = checkpoint["config"]
        channels_used = checkpoint.get("channels_used")
        train_dataset_full = tcKAEMHDDataset(
            file_path=args.data_path, steps=model_config["steps"],
            sequence_length=model_config["sequence_length"],
            norm_stats_path=args.norm_stats_path, channels_to_use=channels_used,
            process_safe_copy=(args.num_workers > 0),
        )
        val_dataset_full = RolloutMHDDataset(
            file_path=args.data_path,
            rollout_steps=args.validation_rollout_steps,
            norm_stats_path=args.norm_stats_path,
            channels_to_use=channels_used,
            process_safe_copy=(args.num_workers > 0),
        )
        train_indices, val_indices = checkpoint["train_indices"], checkpoint["val_indices"]
        train_dataset, val_dataset = Subset(train_dataset_full, train_indices), Subset(val_dataset_full, val_indices)
        
        model = tcKoopmanAutoencoderQ2D(**model_config).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch, best_val_loss, patience_counter, best_epoch = checkpoint["epoch"] + 1, checkpoint["best_val_loss"], checkpoint["patience_counter"], checkpoint.get("best_epoch", 0)
        losses_at_best_epoch = checkpoint.get("losses_at_best_epoch", {})
    else:
        if args.resume: logging.error("Resume flag was set, but no valid checkpoint was found. Starting a new run.")
        train_dataset_full = tcKAEMHDDataset(
            file_path=args.data_path, steps=args.steps,
            sequence_length=args.sequence_length,
            norm_stats_path=args.norm_stats_path, channels_to_use=args.channels,
            process_safe_copy=(args.num_workers > 0),
        )
        val_dataset_full = RolloutMHDDataset(
            file_path=args.data_path,
            rollout_steps=args.validation_rollout_steps,
            norm_stats_path=args.norm_stats_path,
            channels_to_use=args.channels,
            process_safe_copy=(args.num_workers > 0),
        )
        channels_used = train_dataset_full.channel_names
        
        stats = np.load(args.norm_stats_path)
        train_indices, val_indices = stats["train_indices"], stats["val_indices"]
        train_dataset = Subset(train_dataset_full, train_indices)
        val_dataset = Subset(val_dataset_full, val_indices)
        
        sample_block = train_dataset_full[0]
        sample_x = sample_block[0, 0]
        model_config = {
            "in_channels": sample_x.shape[0], "latent_dim": args.latent_dim,
            "input_spatial_dims": sample_x.shape[1:], "steps": args.steps,
            "steps_back": args.steps_back, "steps_tc": args.steps_tc,
            "sequence_length": args.sequence_length, "bottleneck_dim": args.bottleneck_dim,
            "use_bottleneck": args.use_bottleneck,
        }
        model = tcKoopmanAutoencoderQ2D(**model_config).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience)

    # --- DataLoaders ---
    val_batch_size = args.validation_batch_size if args.validation_batch_size else args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logging.info(f"Data split: {len(train_dataset)} train, {len(val_dataset)} val.")
    
    gammas = {"identity": args.gamma_identity, "fwd": args.gamma_fwd, "bwd": args.gamma_bwd if args.backward else 0.0, "con": args.gamma_con if args.backward else 0.0, "tc": args.gamma_tc}

    # =========================================================================
    # --- TRAINING LOOP ---
    # =========================================================================
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {"total": 0.0, "identity": 0.0, "forward": 0.0, "tc": 0.0, "backward": 0.0, "consistency": 0.0}
        epoch_grad_norm = 0.0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for batch_of_blocks in pbar:
            batch_of_blocks = batch_of_blocks.to(DEVICE)
            loss, loss_dict = compute_loss_tckae(model, batch_of_blocks, gammas, epoch, args.epoch_trans)
            optimizer.zero_grad(); loss.backward()
            if args.clip_grad_value:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_value).item()
            else:
                grad_norm = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
            optimizer.step()
            for key in epoch_losses: epoch_losses[key] += loss_dict[key].item()
            epoch_grad_norm += grad_norm
            pbar.set_postfix(loss=loss.item())

        avg_train_losses = {key: val / len(train_dataloader) for key, val in epoch_losses.items()}
        avg_val_losses = validate_epoch_rollout(model, val_dataloader, args.validation_rollout_steps)
        scheduler.step(avg_val_losses["total"])

        logging.info(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_losses['total']:.4f} | Val Rollout Loss: {avg_val_losses['total']:.4f}")
        
        writer.add_scalars("Loss/Total", {'train': avg_train_losses['total'], 'val': avg_val_losses['total']}, epoch)
        writer.add_scalar("Gradient/Norm", epoch_grad_norm / len(train_dataloader), epoch)
        for key in avg_train_losses:
            if key != "total": writer.add_scalars(f"Loss/{key}", {'train': avg_train_losses[key]}, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        if avg_val_losses["total"] < best_val_loss:
            best_val_loss, patience_counter, best_epoch = avg_val_losses["total"], 0, epoch + 1
            losses_at_best_epoch = avg_val_losses
            torch.save({'config': model_config, 'model_state_dict': model.state_dict(), 'channels_used': channels_used}, persistent_dir / "best_model.pth")
            logging.info(f"New best model saved to persistent storage (Val Rollout Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        checkpoint_data = {"epoch": epoch, "config": model_config, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "best_val_loss": best_val_loss, "patience_counter": patience_counter, "best_epoch": best_epoch, "losses_at_best_epoch": losses_at_best_epoch, "train_indices": train_indices, "val_indices": val_indices, "channels_used": channels_used}
        
        is_last_epoch = (epoch == args.epochs - 1)
        if (epoch + 1) % args.checkpoint_save_freq == 0 or is_last_epoch:
            save_dir = scratch_dir if scratch_dir else persistent_dir
            torch.save(checkpoint_data, save_dir / "latest_checkpoint.pth")
        
        if scratch_dir and ((epoch + 1) % args.persistent_save_freq == 0 or is_last_epoch):
            torch.save(checkpoint_data, persistent_dir / "latest_checkpoint.pth")

        if patience_counter >= args.patience:
            logging.info("Early stopping triggered."); break

    # --- Final HParam Logging ---
    hparams = {**{k: v for k, v in vars(args).items() if k not in ['latent_dim', 'bottleneck_dim', 'channels', 'use_bottleneck','steps', 'steps_back', 'steps_tc']}, 'latent_dim': model_config['latent_dim'], 'bottleneck_dim': model_config.get('bottleneck_dim', 4096), 'use_bottleneck': model_config.get('use_bottleneck', True), 'steps': model_config['steps'], 'steps_back': model_config['steps_back'], 'steps_tc': model_config['steps_tc'], 'channels': ",".join(channels_used) if channels_used is not None else "all"}
    for key, value in hparams.items():
        if isinstance(value, Path): hparams[key] = str(value)
        elif key.endswith('_path') and value is not None: hparams[key] = str(value)
    if hparams.get('resume_from_checkpoint'): hparams['resume_from_checkpoint'] = str(hparams['resume_from_checkpoint'])
    final_metrics = {'hparam/best_val_loss': best_val_loss, 'hparam/best_epoch': best_epoch}
    for key, value in losses_at_best_epoch.items(): final_metrics[f'hparam/{key}_at_best'] = value
    writer.add_hparams(hparams, final_metrics)
    
    writer.close()
    logging.info("Training finished.")

if __name__ == "__main__":
    main()

