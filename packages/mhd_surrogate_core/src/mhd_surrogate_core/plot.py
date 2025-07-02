# -*- coding: utf-8 -*-
# mhd_surrogate_core/src/mhd_surrogate_core/plot.py

import logging
from pathlib import Path

import torch
import matplotlib.pyplot as plt

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def plot_loss_curves(checkpoint_path: Path | str):
    """
    Loads a model checkpoint and plots its training and validation loss curves.

    Args:
        checkpoint_path (Path | str): The path to the saved .pth checkpoint file.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logging.error(f"Checkpoint file not found at: {checkpoint_path}")
        return

    # Load the checkpoint dictionary
    logging.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Extract the loss histories
    train_loss = checkpoint.get('train_loss_history', [])
    val_loss = checkpoint.get('val_loss_history', [])
    
    if not train_loss or not val_loss:
        logging.error("Loss history not found or is empty in the checkpoint.")
        return
        
    logging.info(f"Found {len(train_loss)} epochs of training history.")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(12, 7))
    
    # Plot training loss
    plt.plot(epochs, train_loss, 'o-', label='Training Loss', color='royalblue', alpha=0.8)
    
    # Plot validation loss
    plt.plot(epochs, val_loss, 's-', label='Validation Loss', color='darkorange', alpha=0.8)
    
    # Find the epoch with the best validation loss
    best_epoch = val_loss.index(min(val_loss)) + 1
    plt.axvline(x=best_epoch, color='grey', linestyle='--', label=f'Best Model (Epoch {best_epoch})')

    plt.title('Training and Validation Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
