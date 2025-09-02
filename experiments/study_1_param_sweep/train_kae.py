# -*- coding: utf-8 -*-
# experiments/study_1_param_sweep/train_kae.py

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mhd_canonical_kae.model import KoopmanAutoencoder
from mhd_canonical_kae.data import MHDDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Koopman Autoencoder.")
    # --- Data and I/O Arguments ---
    parser.add_argument("--data-path",type=str,required=True,help="Path to the pre-split train_val_set.npz.",)
    parser.add_argument("--norm-stats-path",type=str,required=True,help="Path to the normalization_stats.npz file.",)
    parser.add_argument("--output-dir",type=str,default="output",help="Directory to save the best model and logs.",)
    parser.add_argument("--resume-from-checkpoint",type=str,default=None,help="Path to a 'latest_checkpoint.pth' to resume training.",)
    
    # --- Training Arguments ---
    parser.add_argument("--epochs", type=int, default=128, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    
    # --- Optimizer and Scheduler Arguments ---
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping.")
    parser.add_argument("--lr-patience", type=int, default=8, help="Patience for learning rate scheduler.")
    parser.add_argument("--lr-factor", type=float, default=0.1, help="Factor by which to reduce learning rate.")
    parser.add_argument("--clip-grad-value", type=float, default=25.0, help="Value to clip gradients to.")

    # --- Model and Loss Arguments ---
    parser.add_argument("--latent-dim", type=int, default=128, help="Dimension of the latent space.")
    parser.add_argument("--w-recon", type=float, default=1.0, help="Weight for the reconstruction loss.")
    parser.add_argument("--w-pred", type=float, default=1.0, help="Weight for the prediction loss.")
    parser.add_argument("--w-lin", type=float, default=1.0, help="Weight for the latent linearity loss.")
    parser.add_argument("--w-eig", type=float, default=0.1, help="Weight for the eigenvalue regularization loss.")
    parser.add_argument("--channels", nargs='+', default=None, help="List of channel names to use for training.")
    return parser.parse_args()


def compute_loss(model, batch_x_t, batch_x_t_plus_1, outputs, loss_weights):
    loss_fn = nn.MSELoss()
    loss_recon = loss_fn(outputs["x_t_reconstructed"], batch_x_t)
    loss_pred = loss_fn(outputs["x_t_plus_1_predicted"], batch_x_t_plus_1)
    loss_lin = loss_fn(outputs["z_t_plus_1_predicted"], outputs["z_t_plus_1_encoded"])
    K = model.koopman_operator.weight
    eigenvalues = torch.linalg.eigvals(K)
    eig_loss = torch.mean(torch.relu(torch.abs(eigenvalues) - 1.0))
    total_loss = (loss_weights["recon"]*loss_recon + loss_weights["pred"]*loss_pred + loss_weights["lin"]*loss_lin + loss_weights["eig"]*eig_loss)
    return total_loss, {"total":total_loss.detach(),"recon":loss_recon.detach(),"pred":loss_pred.detach(),"lin":loss_lin.detach(),"eig":eig_loss.detach()}


def validate_epoch(model, dataloader, loss_weights, device):
    model.eval()
    total_losses = {"total": 0.0, "recon": 0.0, "pred": 0.0, "lin": 0.0, "eig": 0.0}
    with torch.no_grad():
        for batch_x_t, batch_x_t_plus_1 in dataloader:
            batch_x_t = batch_x_t.permute(0, 4, 1, 2, 3).to(device)
            batch_x_t_plus_1 = batch_x_t_plus_1.permute(0, 4, 1, 2, 3).to(device)
            outputs = model(batch_x_t, batch_x_t_plus_1)
            _, loss_dict = compute_loss(model, batch_x_t, batch_x_t_plus_1, outputs, loss_weights)
            for key in total_losses: total_losses[key] += loss_dict[key].item()
    return {key: val / len(dataloader) for key, val in total_losses.items()}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir / "logs")

    # --- THE FIX IS HERE (Part 1: Initialize tracking variables) ---
    start_epoch = 0
    patience_counter = 0
    # Track best overall total loss and its epoch
    best_val_loss = float("inf")
    best_epoch = 0
    # Track component losses from the best epoch
    losses_at_best_epoch = {}
    # Track the independent best for each component loss
    best_component_losses = {
        "recon": float("inf"), "pred": float("inf"),
        "lin": float("inf"), "eig": float("inf"),
    }

    channels_used = args.channels

    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model_config = checkpoint["config"]
        channels_used = checkpoint.get("channels_used")
        logging.info(f"Resuming with channels from checkpoint: {channels_used}")
        
        full_dataset = MHDDataset(file_path=args.data_path, norm_stats_path=args.norm_stats_path, channels_to_use=channels_used)
        
        train_indices, val_indices = checkpoint["train_indices"], checkpoint["val_indices"]
        train_dataset, val_dataset = Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)
        
        model = KoopmanAutoencoder(**model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        optimizer = Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        start_epoch, best_val_loss, patience_counter, best_epoch = checkpoint["epoch"] + 1, checkpoint["best_val_loss"], checkpoint["patience_counter"], checkpoint.get("best_epoch", 0)
        losses_at_best_epoch = checkpoint.get("losses_at_best_epoch", {})
        best_component_losses = checkpoint.get("best_component_losses", best_component_losses)
    else:
        full_dataset = MHDDataset(file_path=args.data_path, norm_stats_path=args.norm_stats_path, channels_to_use=args.channels)
        channels_used = full_dataset.channel_names

        stats = np.load(args.norm_stats_path)
        train_indices, val_indices = stats["train_indices"], stats["val_indices"]
        train_dataset, val_dataset = Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)
        
        sample_x, _ = full_dataset[0]
        model_config = {"in_channels": sample_x.shape[-1], "latent_dim": args.latent_dim, "input_spatial_dims": sample_x.shape[:-1]}
        model = KoopmanAutoencoder(**model_config).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    logging.info(f"Data split: {len(train_dataset)} train, {len(val_dataset)} val.")

    loss_weights = {"recon": args.w_recon, "pred": args.w_pred, "lin": args.w_lin, "eig": args.w_eig}
    logging.info(f"Using loss weights: {loss_weights}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_train_losses = {"total": 0.0, "recon": 0.0, "pred": 0.0, "lin": 0.0, "eig": 0.0}
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch_x_t, batch_x_t_plus_1 in pbar:
            batch_x_t, batch_x_t_plus_1 = batch_x_t.permute(0, 4, 1, 2, 3).to(device), batch_x_t_plus_1.permute(0, 4, 1, 2, 3).to(device)
            outputs = model(batch_x_t, batch_x_t_plus_1)
            loss, loss_dict = compute_loss(model, batch_x_t, batch_x_t_plus_1, outputs, loss_weights)
            optimizer.zero_grad(); loss.backward()
            if args.clip_grad_value: total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_value)
            else: total_norm = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None)**0.5
            optimizer.step()
            for key in epoch_train_losses: epoch_train_losses[key] += loss_dict[key].item()
            pbar.set_postfix(loss=loss.item())

        avg_train_losses = {key: val / len(train_dataloader) for key, val in epoch_train_losses.items()}
        avg_val_losses = validate_epoch(model, val_dataloader, loss_weights, device)
        scheduler.step(avg_val_losses["total"])

        logging.info(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_losses['total']:.4f} | Val Loss: {avg_val_losses['total']:.4f}")
        writer.add_scalars("Loss/Total", {'train': avg_train_losses['total'], 'val': avg_val_losses['total']}, epoch)
        writer.add_scalar("Gradient/Norm", total_norm, epoch)
        for key in ["recon", "pred", "lin", "eig"]: writer.add_scalars(f"Loss_Components/{key}", {'train': avg_train_losses[key],'val': avg_val_losses[key]}, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

        # --- THE FIX IS HERE (Part 2: Update tracking logic) ---
        # Track independent best component losses
        for key in best_component_losses:
            best_component_losses[key] = min(best_component_losses[key], avg_val_losses[key])

        if avg_val_losses["total"] < best_val_loss:
            best_val_loss = avg_val_losses["total"]
            patience_counter = 0
            best_epoch = epoch + 1
            losses_at_best_epoch = avg_val_losses # Store all component losses
            best_model_path = output_dir / "best_model.pth"
            torch.save({'config': model_config, 'model_state_dict': model.state_dict(), 'channels_used': channels_used}, best_model_path)
            logging.info(f"New best model saved to {best_model_path} (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            logging.info(f"Validation loss did not improve. Patience: {patience_counter}/{args.patience}")

        latest_checkpoint_path = output_dir / "latest_checkpoint.pth"
        torch.save({"epoch": epoch, "config": model_config, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "best_val_loss": best_val_loss, "patience_counter": patience_counter, "best_epoch": best_epoch, "train_indices": train_dataset.indices, "val_indices": val_dataset.indices, "channels_used": channels_used, "losses_at_best_epoch": losses_at_best_epoch, "best_component_losses": best_component_losses}, latest_checkpoint_path)
        if patience_counter >= args.patience: logging.info("Early stopping triggered."); break

    # --- THE FIX IS HERE (Part 3: Update HParams logging) ---
    hparams = vars(args)
    hparams['channels'] = ",".join(channels_used) if channels_used is not None else "all"
    hparams.update({k: str(v) for k, v in hparams.items() if isinstance(v, Path) or k.endswith('_path')})
    if hparams.get('resume_from_checkpoint'): hparams['resume_from_checkpoint'] = str(hparams['resume_from_checkpoint'])
    
    final_metrics = {
        'hparam/best_val_loss': best_val_loss,
        'hparam/best_epoch': best_epoch,
        'hparam/recon_at_best': losses_at_best_epoch.get('recon', 0),
        'hparam/pred_at_best': losses_at_best_epoch.get('pred', 0),
        'hparam/lin_at_best': losses_at_best_epoch.get('lin', 0),
        'hparam/eig_at_best': losses_at_best_epoch.get('eig', 0),
        'hparam/best_recon_loss': best_component_losses['recon'],
        'hparam/best_pred_loss': best_component_losses['pred'],
        'hparam/best_lin_loss': best_component_losses['lin'],
        'hparam/best_eig_loss': best_component_losses['eig'],
    }
    
    writer.add_hparams(hparams, final_metrics)
    writer.close(); logging.info("Training finished.")

if __name__ == "__main__":
    main()
