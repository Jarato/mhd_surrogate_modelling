# -*- coding: utf-8 -*-
# mhd_surrogate_core/src/mhd_surrogate_core/data.py

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class MHDDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MHD simulation data.

    This dataset is designed to work with snapshot data from simulations,
    where each file contains a time series of system states. It returns
    pairs of consecutive snapshots (x_t, x_{t+1}), which are required for
    training Koopman-based models.
    """

    def __init__(
        self,
        file_path: Path | str,
        timeseries_key: str = "timeseries",
        label_key: str = "labels",
    ):
        """
        Initializes the dataset.

        Args:
            file_path (Path | str): The path to the .npz data file.
            timeseries_key (str): The key for the timeseries data.
            label_key (str): The key for the channel names (labels).
        """
        self.file_path = Path(file_path)
        self.timeseries_key = timeseries_key
        self.label_key = label_key

        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found at: {self.file_path}")

        logging.info(f"Loading data from {self.file_path}...")
        try:
            # Load the entire dataset into memory.
            with np.load(self.file_path, allow_pickle=True) as loaded_data:
                self.data = loaded_data[self.timeseries_key]
                self.channel_names = loaded_data[self.label_key]
            logging.info(f"Data loaded successfully. Shape: {self.data.shape}")
            logging.info(f"Channel names loaded: {self.channel_names}")
        except Exception as e:
            logging.error(f"Failed to load data from {self.file_path}: {e}")
            raise

        # The expected shape is (timesteps, nx, ny, nz, channels)
        if self.data.ndim != 5 or self.data.shape[0] < 2:
            raise ValueError(
                "Data must be a 5D array with at least 2 timesteps, "
                f"but got shape {self.data.shape}"
            )

        if self.data.shape[-1] != len(self.channel_names):
            raise ValueError(
                "Mismatch between number of channels in data and number of channel names. "
                f"Got {self.data.shape[-1]} channels and {len(self.channel_names)} names."
            )

        self.num_timesteps = self.data.shape[0]

    def __len__(self) -> int:
        """
        Returns the total number of samples (pairs of consecutive snapshots).
        """
        # If we have N timesteps, we can create N-1 pairs (x_0, x_1), (x_1, x_2), ...
        return self.num_timesteps - 1

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Retrieves a sample from the dataset at the given index.

        A sample consists of two consecutive time steps (x_t, x_{t+1}).

        Args:
            idx (int): The index of the sample.

        Returns:
            A tuple containing two PyTorch tensors: (x_t, x_{t+1}).
        """
        if not 0 <= idx < len(self):
            raise IndexError("Index out of range.")

        # Get the snapshot at time t and t+1
        x_t = self.data[idx]
        x_t_plus_1 = self.data[idx + 1]

        # Convert numpy arrays to PyTorch tensors.
        # The dtype is converted to float32, which is standard for deep learning.
        return (
            torch.from_numpy(x_t).float(),
            torch.from_numpy(x_t_plus_1).float(),
        )


# This block allows you to test the script directly
if __name__ == "__main__":
    # --- How to use the MHDDataset ---

    # Create a dummy .npz file for testing purposes.
    dummy_data_path = Path("dummy_mhd_data.npz")
    logging.info(f"Creating a dummy data file at: {dummy_data_path}")

    # The shape is (timesteps, nx, ny, nz, channels)
    dummy_shape = (100, 10, 8, 8, 3)
    dummy_timeseries = np.random.rand(*dummy_shape).astype(np.float32)
    # Create dummy channel names (labels)
    dummy_channel_names = ["B_x", "B_y", "B_z"]

    # Save the dummy data using np.savez (uncompressed)
    np.savez(
        dummy_data_path,
        timeseries=dummy_timeseries,
        labels=dummy_channel_names,
    )

    logging.info("--- Testing MHDDataset ---")
    try:
        mhd_dataset = MHDDataset(
            file_path=dummy_data_path,
        )

        logging.info(f"Dataset length: {len(mhd_dataset)}")
        logging.info(f"Loaded channel names: {mhd_dataset.channel_names}")

        # Get a single sample
        x_t, x_t_plus_1 = mhd_dataset[0]
        logging.info(f"Shape of x_t: {x_t.shape}")
        logging.info(f"Shape of x_t+1: {x_t_plus_1.shape}")
        logging.info(f"Data type of tensors: {x_t.dtype}")

        # Use it with a PyTorch DataLoader
        logging.info("--- Testing DataLoader ---")
        mhd_dataloader = DataLoader(
            dataset=mhd_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
        )

        # Iterate over one batch from the dataloader
        batch_x_t, batch_x_t_plus_1 = next(iter(mhd_dataloader))

        logging.info(f"Shape of a batch for x_t: {batch_x_t.shape}")
        logging.info(f"Shape of a batch for x_t+1: {batch_x_t_plus_1.shape}")

    except Exception as e:
        logging.error(f"An error occurred during testing: {e}")
    finally:
        # Clean up the dummy file
        if dummy_data_path.exists():
            dummy_data_path.unlink()
            logging.info(f"Cleaned up dummy file: {dummy_data_path}")
