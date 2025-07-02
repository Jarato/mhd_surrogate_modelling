# -*- coding: utf-8 -*-
# experiments/study_1_param_sweep/evaluate.py

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# To run this script, you need to have your packages installed in editable mode.
from mhd_canonical_kae.model import KoopmanAutoencoder

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Koopman Autoencoder on a contiguous test set."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        required=True,
        help="Path to the contiguous test_set.npz file.",
    )
    return parser.parse_args()


def main():
    """Main function to orchestrate the evaluation process."""
    args = parse_args()

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # --- Load Test Data ---
    logging.info(f"Loading contiguous test set from {args.test_data_path}...")
    with np.load(args.test_data_path) as raw_data:
        test_timeseries = raw_data["timeseries"]
    
    test_tensor = torch.from_numpy(test_timeseries).float()
    logging.info(f"Test data loaded. Shape: {test_tensor.shape}")

    # --- Load Model from Checkpoint ---
    logging.info(f"Loading model checkpoint from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model_config = checkpoint['config']
    
    logging.info(f"Re-creating model with saved config: {model_config}")
    model = KoopmanAutoencoder(**model_config).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- Autoregressive Rollout ---
    num_timesteps = test_tensor.shape[0]
    predictions = []
    
    current_state = test_tensor[0].unsqueeze(0)
    current_state = current_state.permute(0, 4, 1, 2, 3).to(device)

    logging.info(f"Starting autoregressive rollout for {num_timesteps - 1} steps...")
    with torch.no_grad():
        predictions.append(current_state.cpu().permute(0, 2, 3, 4, 1).squeeze(0))
        z_t = model.encode(current_state)

        for _ in range(num_timesteps - 1):
            z_t = model.koopman_step(z_t)
            predicted_state = model.decode(z_t)
            predictions.append(predicted_state.cpu().permute(0, 2, 3, 4, 1).squeeze(0))
            current_state = predicted_state

    predicted_timeseries = torch.stack(predictions)
    logging.info(f"Rollout complete. Predicted timeseries shape: {predicted_timeseries.shape}")

    # --- Calculate Final Error ---
    loss_fn = nn.MSELoss()
    rollout_error = loss_fn(predicted_timeseries, test_tensor)

    logging.info("--- EVALUATION COMPLETE ---")
    logging.info(f"Autoregressive Rollout MSE: {rollout_error.item():.6f}")


if __name__ == "__main__":
    main()
