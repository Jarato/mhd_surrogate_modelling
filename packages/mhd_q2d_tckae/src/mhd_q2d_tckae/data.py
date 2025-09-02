# -*- coding: utf-8 -*-
# packages/mhd_q2d_tckae/src/mhd_q2d_tckae/data.py

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class tcKAEMHDDataset(Dataset):
    """
    Custom PyTorch Dataset for the tcKAE model.

    This dataset is specifically designed to provide sequences of timesteps
    required for calculating the multi-step forward loss and temporal
    consistency loss in the tcKAE algorithm.
    """

    def __init__(
        self,
        file_path: Path | str,
        steps: int, # Number of future steps to return for each sample
        norm_stats_path: Path | str | None = None,
        channels_to_use: list[str] | None = None,
        process_safe_copy: bool = False,
        timeseries_key: str = "timeseries",
        label_key: str = "labels",
    ):
        """
        Initializes the dataset.

        Args:
            file_path (Path | str): Path to the .npz data file.
            steps (int): The number of future time steps to include in each sample.
                         Each sample will have a total length of `steps + 1`.
            norm_stats_path (Path | str | None): Path to the normalization stats.
            channels_to_use (list[str] | None): List of channel names to use.
            process_safe_copy (bool): If True, explicitly creates a copy of the
                data in __getitem__. Slower but required for safe multi-process
                data loading (num_workers > 0). Defaults to False.
            timeseries_key (str): Key for the timeseries data in the .npz file.
            label_key (str): Key for the channel names (labels) in the .npz file.
        """
        self.file_path = Path(file_path)
        self.steps = steps
        self.timeseries_key = timeseries_key
        self.label_key = label_key
        self.process_safe_copy = process_safe_copy

        with np.load(self.file_path, allow_pickle=True) as loaded_data:
            full_timeseries_data = loaded_data[self.timeseries_key]
            self.all_channel_names = list(loaded_data[self.label_key])

        if channels_to_use:
            channel_indices = [self.all_channel_names.index(name) for name in channels_to_use]
            self.data = full_timeseries_data[..., channel_indices]
            self.channel_names = channels_to_use
            logging.info(f"Created a subset data array in memory with shape: {self.data.shape}")
        else:
            self.data = full_timeseries_data
            self.channel_names = self.all_channel_names

        self.min_vals = None
        self.max_vals = None
        if norm_stats_path:
            stats = np.load(norm_stats_path)
            all_min_vals = torch.from_numpy(stats['min_vals']).float()
            all_max_vals = torch.from_numpy(stats['max_vals']).float()
            
            if channels_to_use:
                channel_indices = [self.all_channel_names.index(name) for name in channels_to_use]
                self.min_vals = all_min_vals[channel_indices]
                self.max_vals = all_max_vals[channel_indices]
            else:
                self.min_vals = all_min_vals
                self.max_vals = all_max_vals
            self.range = self.max_vals - self.min_vals + 1e-8

    def __len__(self) -> int:
        # The length is the total number of timesteps minus the number of steps
        # needed for a full sequence.
        return self.data.shape[0] - self.steps

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.min_vals is None: return x
        # Reshape for broadcasting: (C) -> (C, 1, 1, 1)
        min_vals = self.min_vals.view(-1, 1, 1, 1)
        range_vals = self.range.view(-1, 1, 1, 1)
        return (x - min_vals) / range_vals * 2.0 - 1.0

    def _denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        if self.min_vals is None: return x_norm
        # Reshape for broadcasting
        min_vals = self.min_vals.view(-1, 1, 1, 1)
        range_vals = self.range.view(-1, 1, 1, 1)
        return (x_norm + 1.0) / 2.0 * range_vals + min_vals

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        """
        Returns a sequence of `steps + 1` consecutive timesteps.
        """
        # Select the sequence of data
        sequence_data = self.data[idx : idx + self.steps + 1]

        # Convert to tensors, handling the process-safe copy flag
        if self.process_safe_copy:
            sequence_tensors = [torch.from_numpy(item.copy()).float() for item in sequence_data]
        else:
            sequence_tensors = [torch.from_numpy(item).float() for item in sequence_data]

        # Permute and normalize each tensor in the sequence
        # Original shape: (X, Y, Z, C). Target shape for model: (C, X, Y, Z)
        processed_sequence = []
        for tensor in sequence_tensors:
            permuted_tensor = tensor.permute(3, 0, 1, 2)
            normalized_tensor = self._normalize(permuted_tensor)
            processed_sequence.append(normalized_tensor)

        return processed_sequence
