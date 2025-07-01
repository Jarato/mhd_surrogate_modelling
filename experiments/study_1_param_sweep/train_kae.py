# -*- coding: utf-8 -*-
# experiments/study_1_param_sweep/train_kae.py

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

# To run this script, you need to have your packages installed in editable mode.
# From the experiment directory, you would run:
# pip install -e ../../packages/mhd_surrogate_core
# pip install -e ../../packages/mhd_canonical_kae
from mhd_surrogate_core.data import MHDDataset
from mhd_canonical_kae.model import KoopmanAutoencoder

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
        help="Path to the .npz data file.",
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
    # Add other hyperparameters as needed
    return parser.parse_args()


def compute_loss(
    batch_x_t: torch.Tensor,
    batch_x_t_plus_1: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    loss_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Computes the composite loss for the Koopman Autoencoder.

    Args:
        batch_x_t: The input tensor at time t.
        batch_x_t_plus_1: The input tensor at time t+1.
        outputs: The dictionary of outputs from the model's forward pass.
        loss_weights: A dictionary with weights for each loss component.

    Returns:
        A tuple containing:
        - The total weighted loss as a scalar tensor.
        - A dictionary containing the individual, unweighted loss values.
    """
    loss_fn = nn.MSELoss()

    # Reconstruction Loss: How well does the model reconstruct the input?
    loss_recon = loss_fn(outputs["x_t_reconstructed"], batch_x_t)

    # Prediction Loss: How well does the model predict the next state?
    loss_pred = loss_fn(outputs["x_t_plus_1_predicted"], batch_x_t_plus_1)

    # Linearity Loss: How well does the Koopman operator approximate the dynamics
    # in the latent space?
    loss_lin = loss_fn(outputs["z_t_plus_1_predicted"], outputs["z_t_plus_1_encoded"])

    # Total weighted loss
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


def main():
    """Main function to orchestrate the training process."""
    args = parse_args()

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # --- Data Loading ---
    # For now, we assume a single data file. This can be extended to handle
    # training/validation splits later.
    dataset = MHDDataset(file_path=args.data_path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # --- Model Definition ---
    # The number of input channels must match the data.
    # We get this from the first sample in the dataset.
    sample_x, _ = dataset[0]
    in_channels = sample_x.shape[0] 
    # Note: Assumes data is in (C, D, H, W) format after the Dataset.
    # Let's adjust the model and data handling to be consistent.
    # PyTorch CNNs expect (B, C, D, H, W). Our data is (D, H, W, C).
    # We need to permute the dimensions. Let's assume this is done in the Dataset for now.
    # For simplicity, let's hardcode for now.
    # TODO: Make this dynamic based on data.
    in_channels = 20 # As per your data description

    model = KoopmanAutoencoder(
        in_channels=in_channels,
        latent_dim=args.latent_dim,
    ).to(device)

    # --- Optimizer and Loss ---
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_weights = {"recon": 1.0, "pred": 1.0, "lin": 1.0}

    # --- Training Loop ---
    logging.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()  # Set the model to training mode
        total_epoch_loss = 0.0

        for batch_idx, (batch_x_t, batch_x_t_plus_1) in enumerate(dataloader):
            # Move data to the selected device
            # IMPORTANT: Data must be permuted to (B, C, D, H, W) for 3D CNNs
            batch_x_t = batch_x_t.permute(0, 4, 1, 2, 3).to(device)
            batch_x_t_plus_1 = batch_x_t_plus_1.permute(0, 4, 1, 2, 3).to(device)

            # --- Forward Pass ---
            outputs = model(batch_x_t, batch_x_t_plus_1)

            # --- Compute Loss ---
            loss, loss_dict = compute_loss(
                batch_x_t,
                batch_x_t_plus_1,
                outputs,
                loss_weights,
            )

            # --- Backward Pass and Optimization ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

            if batch_idx % 20 == 0:
                logging.info(
                    f"Epoch [{epoch+1}/{args.epochs}] | "
                    f"Batch [{batch_idx}/{len(dataloader)}] | "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_epoch_loss = total_epoch_loss / len(dataloader)
        logging.info(
            f"====> Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f} ===="
        )

    logging.info("Training finished.")
    # TODO: Add model saving logic here


if __name__ == "__main__":
    main()
