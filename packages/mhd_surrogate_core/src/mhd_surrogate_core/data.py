# -*- coding: utf-8 -*-
# mhd_surrogate_core/src/mhd_surrogate_core/data.py

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class MHDDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and normalizing MHD simulation data.
    """

    def __init__(
        self,
        file_path: Path | str,
        norm_stats_path: Path | str | None = None,
        timeseries_key: str = "timeseries",
        label_key: str = "labels",
    ):
        """
        Initializes the dataset and optionally loads normalization stats.

        Args:
            file_path (Path | str): Path to the .npz data file.
            norm_stats_path (Path | str | None): Path to the .npz file with
                                                 normalization statistics.
            timeseries_key (str): Key for the timeseries data.
            label_key (str): Key for the channel names (labels).
        """
        self.file_path = Path(file_path)
        self.timeseries_key = timeseries_key
        self.label_key = label_key

        # Load data
        with np.load(self.file_path, allow_pickle=True) as loaded_data:
            self.data = loaded_data[self.timeseries_key]
            self.channel_names = loaded_data[self.label_key]

        # Load normalization stats if provided
        self.min_vals = None
        self.max_vals = None
        if norm_stats_path:
            norm_stats_path = Path(norm_stats_path)
            if not norm_stats_path.exists():
                raise FileNotFoundError(f"Normalization stats file not found: {norm_stats_path}")
            
            stats = np.load(norm_stats_path)
            self.min_vals = torch.from_numpy(stats['min_vals']).float()
            self.max_vals = torch.from_numpy(stats['max_vals']).float()
            # Add a small epsilon to avoid division by zero for constant channels
            self.range = self.max_vals - self.min_vals + 1e-8

    def __len__(self) -> int:
        return self.data.shape[0] - 1

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Applies min-max normalization to scale data to [-1, 1]."""
        if self.min_vals is None:
            return x
        # Formula: (value - min) / (max - min) * 2 - 1
        return (x - self.min_vals) / self.range * 2.0 - 1.0

    def _denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Applies inverse normalization to scale data back to original range."""
        if self.min_vals is None:
            return x_norm
        # Inverse Formula: (value_norm + 1) / 2 * (max - min) + min
        return (x_norm + 1.0) / 2.0 * self.range + self.min_vals

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        x_t = torch.from_numpy(self.data[idx]).float()
        x_t_plus_1 = torch.from_numpy(self.data[idx + 1]).float()

        # Apply normalization if stats are available
        x_t = self._normalize(x_t)
        x_t_plus_1 = self._normalize(x_t_plus_1)

        return x_t, x_t_plus_1
