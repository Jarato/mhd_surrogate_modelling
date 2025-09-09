# -*- coding: utf-8 -*-
# packages/mhd_q2d_tckae/src/mhd_q2d_tckae/data.py

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import as_strided

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class tcKAEMHDDataset(Dataset):
    """
    Custom Dataset for tcKAE training.
    Each sample is a block of M consecutive sequences, each of length K+1.
    """

    def __init__(
        self,
        full_timeseries: np.ndarray,
        all_channel_names: list[str],
        sequence_length: int,  # M
        steps: int,            # K
        norm_stats: Dict[str, np.ndarray] | None = None,
        channels_to_use: list[str] | None = None,
        process_safe_copy: bool = False,
    ):
        self.sequence_length = sequence_length
        self.steps = steps
        self.process_safe_copy = process_safe_copy
        
        self.all_channel_names = all_channel_names

        if channels_to_use:
            indices = [self.all_channel_names.index(name) for name in channels_to_use]
            self.data = full_timeseries[..., indices]
            self.channel_names = channels_to_use
        else:
            self.data = full_timeseries
            self.channel_names = self.all_channel_names

        self.min_vals, self.max_vals, self.range = None, None, None
        if norm_stats:
            all_min = torch.from_numpy(norm_stats['min_vals']).float()
            all_max = torch.from_numpy(norm_stats['max_vals']).float()
            if channels_to_use:
                indices = [self.all_channel_names.index(name) for name in channels_to_use]
                self.min_vals = all_min[indices].view(1, -1, 1, 1, 1)
                self.max_vals = all_max[indices].view(1, -1, 1, 1, 1)
            else:
                self.min_vals = all_min.view(1, -1, 1, 1, 1)
                self.max_vals = all_max.view(1, -1, 1, 1, 1)
            self.range = self.max_vals - self.min_vals + 1e-8

    def __len__(self) -> int:
        # This length is relative to the subset of indices it will be used with
        return self.data.shape[0]

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.min_vals is None: return x
        return (x - self.min_vals) / self.range * 2.0 - 1.0
    
    def _denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        if self.min_vals is None: return x_norm
        return (x_norm + 1.0) / 2.0 * self.range + self.min_vals

    def __getitem__(self, idx: int) -> torch.Tensor:
        block_len = self.sequence_length + self.steps
        snapshots = self.data[idx : idx + block_len]
        if self.process_safe_copy:
            snapshots = snapshots.copy()

        seq_len = self.steps + 1
        num_seq = self.sequence_length
        
        shape = (num_seq, seq_len) + snapshots.shape[1:]
        strides = (snapshots.strides[0],) + snapshots.strides
        sequences = as_strided(snapshots, shape=shape, strides=strides)
        
        tensor_block = torch.from_numpy(sequences.copy()).float().permute(0, 1, 5, 2, 3, 4)
        return self._normalize(tensor_block)


class RolloutMHDDataset(Dataset):
    """
    Simpler Dataset for validation/testing via auto-regressive rollout.
    Each sample is a single sequence of timesteps.
    """
    def __init__(
        self,
        full_timeseries: np.ndarray,
        all_channel_names: list[str],
        rollout_steps: int,
        norm_stats: Dict[str, np.ndarray] | None = None,
        channels_to_use: list[str] | None = None,
        process_safe_copy: bool = False,
    ):
        self.rollout_steps = rollout_steps
        self.process_safe_copy = process_safe_copy
        
        self.all_channel_names = all_channel_names

        if channels_to_use:
            indices = [self.all_channel_names.index(name) for name in channels_to_use]
            self.data = full_timeseries[..., indices]
            self.channel_names = channels_to_use
        else:
            self.data = full_timeseries
            self.channel_names = self.all_channel_names

        self.min_vals, self.max_vals, self.range = None, None, None
        if norm_stats:
            all_min = torch.from_numpy(norm_stats['min_vals']).float()
            all_max = torch.from_numpy(norm_stats['max_vals']).float()
            if channels_to_use:
                indices = [self.all_channel_names.index(name) for name in channels_to_use]
                self.min_vals = all_min[indices].view(1, -1, 1, 1, 1)
                self.max_vals = all_max[indices].view(1, -1, 1, 1, 1)
            else:
                self.min_vals = all_min.view(1, -1, 1, 1, 1)
                self.max_vals = all_max.view(1, -1, 1, 1, 1)
            self.range = self.max_vals - self.min_vals + 1e-8

    def __len__(self) -> int:
        # This length is relative to the subset of indices it will be used with
        return self.data.shape[0]

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.min_vals is None: return x
        return (x - self.min_vals) / self.range * 2.0 - 1.0

    def __getitem__(self, idx: int) -> torch.Tensor:
        sequence = self.data[idx : idx + self.rollout_steps + 1]
        if self.process_safe_copy:
            sequence = sequence.copy()

        tensor_seq = torch.from_numpy(sequence).float().permute(0, 4, 1, 2, 3)
        return self._normalize(tensor_seq)

