# -*- coding: utf-8 -*-
# packages/mhd_q2d_tckae/src/mhd_q2d_tckae/data.py

import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class tcKAEMHDDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MHD data in blocks for tcKAE.
    Each sample is a block of M consecutive sequences of length (steps+1).
    """

    def __init__(
        self,
        file_path: Path | str,
        steps: int,
        sequence_length: int, # This is 'M' from the paper
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
            steps (int): The number of future steps to predict (K in paper).
            sequence_length (int): The number of consecutive sequences per sample (M in paper).
            norm_stats_path (Path | str | None): Path to normalization stats.
            channels_to_use (list[str] | None): List of channel names.
            process_safe_copy (bool): Flag for multi-process data loading.
        """
        self.file_path = Path(file_path)
        self.steps = steps
        self.sequence_length = sequence_length # M
        self.process_safe_copy = process_safe_copy

        with np.load(self.file_path, allow_pickle=True) as loaded_data:
            full_timeseries_data = loaded_data[timeseries_key]
            self.all_channel_names = list(loaded_data[label_key])

        if channels_to_use:
            channel_indices = [self.all_channel_names.index(name) for name in channels_to_use]
            self.data = full_timeseries_data[..., channel_indices]
            self.channel_names = channels_to_use
        else:
            self.data = full_timeseries_data
            self.channel_names = self.all_channel_names

        self.min_vals, self.max_vals, self.range = None, None, None
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

            # Reshape for broadcasting: (C, 1, 1, 1)
            self.min_vals = self.min_vals.view(-1, 1, 1, 1)
            self.max_vals = self.max_vals.view(-1, 1, 1, 1)
            self.range = self.max_vals - self.min_vals + 1e-8

    def __len__(self) -> int:
        # The number of possible starting points for a full block
        return self.data.shape[0] - self.steps - self.sequence_length

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.min_vals is None: return x
        return (x - self.min_vals) / self.range * 2.0 - 1.0

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns a single training sample: a block of M consecutive sequences.
        Shape: (M, steps+1, C, X, Y, Z)
        """
        # A block contains M sequences, each of length (steps+1)
        # Total snapshots needed = M + steps
        block_end = idx + self.sequence_length + self.steps
        snapshots = self.data[idx:block_end]

        if self.process_safe_copy:
            snapshots = snapshots.copy()

        # Create the sequences by striding over the snapshots
        sequences = np.lib.stride_tricks.as_strided(
            snapshots,
            shape=(self.sequence_length, self.steps + 1, *snapshots.shape[1:]),
            strides=(snapshots.strides[0], *snapshots.strides)
        )
        
        sequences_tensor = torch.from_numpy(sequences.copy()).float()
        # Permute from (M, steps+1, X, Y, Z, C) to (M, steps+1, C, X, Y, Z)
        sequences_tensor = sequences_tensor.permute(0, 1, 5, 2, 3, 4)
        
        return self._normalize(sequences_tensor)

