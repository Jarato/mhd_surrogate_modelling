# -*- coding: utf-8 -*-
# experiments/study_1_param_sweep/train_kae.py

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# To run this script, you need to have your packages installed in editable mode.
# From the experiment directory, you would run:
# pip install -r requirements.txt
from mhd_canonical_kae.model import KoopmanAutoencoder
from mhd_surrogate_core.data import MHDDataset

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Koopman Autoencoder.")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the pre-split train_val_set.npz data file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save the best model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Dimension of the latent space.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of the train+val data to use for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val split to ensure reproducibility.",
    )
    return parser.parse_args()


def compute_loss(
    batch_x_t: torch.Tensor,
    batch_x_t_plus_1: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Computes the composite loss for the Koopman Autoencoder.
    """
    loss_fn = nn.MSELoss()
    loss_recon = loss_fn(outputs["x_t_reconstructed"], batch_x_t)
    loss_pred = loss_fn(outputs["x_t_plus_1_predicted"], batch_x_t_plus_1)
    loss_lin = loss_fn(outputs["z_t_plus_1_predicted"], outputs["z_t_plus_1_encoded"])

    total_loss = (
        loss_weights["recon"] * loss_recon
        + loss_weights["pred"] * loss_pred
        + loss_weights["lin"] * loss_lin
    )

    loss_dict = {
        "total": total_loss.detach(),
        "recon": loss_recon.detach(),
        "pred": loss_pred.detach(),
        "lin": loss_lin.detach(),
    }
    return total_loss, loss_dict


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_weights: dict[str, float],
    device: str,
) -> float:
    """Runs a validation loop for one epoch."""
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch_x_t, batch_x_t_plus_1 in dataloader:
            batch_x_t = batch_x_t.permute(0, 4, 1, 2, 3).to(device)
            batch_x_t_plus_1 = batch_x_t_plus_1.permute(0, 4, 1, 2, 3).to(device)

            outputs = model(batch_x_t, batch_x_t_plus_1)
            loss, _ = compute_loss(
                batch_x_t, batch_x_t_plus_1, outputs, loss_weights
            )
            total_val_loss += loss.item()

    return total_val_loss / len(dataloader)


def main():
    """Main function to orchestrate the training process."""
    args = parse_args()

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)

    # --- Data Loading and Splitting ---
    train_val_dataset = MHDDataset(file_path=args.data_path)

    val_size = int(len(train_val_dataset) * args.val_split)
    train_size = len(train_val_dataset) - val_size
    
    if train_size < 1 or val_size < 1:
        raise ValueError(
            "The train_val_set is too small to create a non-empty train and val split."
        )

    train_dataset, val_dataset = random_split(
        train_val_dataset, [train_size, val_size], generator=generator
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False
    )
    logging.info(f"Data split: {len(train_dataset)} train, {len(val_dataset)} val.")

    # --- Model Definition ---
    sample_x, _ = train_val_dataset[0]
    in_channels = sample_x.shape[-1]
    input_spatial_dims = sample_x.shape[:-1]
    
    model_config = {
        'in_channels': in_channels,
        'latent_dim': args.latent_dim,
        'input_spatial_dims': input_spatial_dims,
    }
    logging.info(f"Initializing model with config: {model_config}")

    model = KoopmanAutoencoder(**model_config).to(device)

    # --- Optimizer and Loss ---
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_weights = {"recon": 1.0, "pred": 1.0, "lin": 1.0}
    best_val_loss = float("inf")

    # --- Training Loop ---
    logging.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        for batch_idx, (batch_x_t, batch_x_t_plus_1) in enumerate(train_dataloader):
            batch_x_t = batch_x_t.permute(0, 4, 1, 2, 3).to(device)
            batch_x_t_plus_1 = batch_x_t_plus_1.permute(0, 4, 1, 2, 3).to(device)

            outputs = model(batch_x_t, batch_x_t_plus_1)
            loss, _ = compute_loss(
                batch_x_t, batch_x_t_plus_1, outputs, loss_weights
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = validate_epoch(model, val_dataloader, loss_weights, device)

        logging.info(
            f"Epoch [{epoch+1}/{args.epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = output_dir / "best_model.pth"
            
            # Save a dictionary containing the model's config and state
            checkpoint = {
                'config': model_config,
                'model_state_dict': model.state_dict(),
            }
            torch.save(checkpoint, model_path)
            logging.info(f"New best model saved to {model_path} (Val Loss: {best_val_loss:.4f})")

    logging.info("Training finished.")


if __name__ == "__main__":
    main()
