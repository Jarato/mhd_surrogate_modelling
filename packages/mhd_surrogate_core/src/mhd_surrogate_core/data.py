# -*- coding: utf-8 -*-
# mhd_surrogate_core/src/mhd_surrogate_core/data.py

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class MHDDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and normalizing MHD simulation data.
    Can select a subset of channels to use.
    """

    def __init__(
        self,
        file_path: Path | str,
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
            norm_stats_path (Path | str | None): Path to the normalization stats.
            channels_to_use (list[str] | None): List of channel names to use.
            process_safe_copy (bool): If True, explicitly creates a copy of the
                data in __getitem__. This is slower but required for safe
                multi-process data loading (num_workers > 0). Defaults to False.
            timeseries_key (str): Key for the timeseries data.
            label_key (str): Key for the channel names (labels).
        """
        self.file_path = Path(file_path)
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
        return self.data.shape[0] - 1

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.min_vals is None: return x
        return (x - self.min_vals) / self.range * 2.0 - 1.0

    def _denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        if self.min_vals is None: return x_norm
        return (x_norm + 1.0) / 2.0 * self.range + self.min_vals

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        x_t_selected = self.data[idx]
        x_t_plus_1_selected = self.data[idx + 1]

        if self.process_safe_copy:
            x_t = torch.from_numpy(x_t_selected.copy()).float()
            x_t_plus_1 = torch.from_numpy(x_t_plus_1_selected.copy()).float()
        else:
            x_t = torch.from_numpy(x_t_selected).float()
            x_t_plus_1 = torch.from_numpy(x_t_plus_1_selected).float()

        x_t = self._normalize(x_t)
        x_t_plus_1 = self._normalize(x_t_plus_1)

        return x_t, x_t_plus_1
