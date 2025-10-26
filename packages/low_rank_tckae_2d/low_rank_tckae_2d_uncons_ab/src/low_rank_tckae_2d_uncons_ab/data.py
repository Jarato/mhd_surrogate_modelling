# -*- coding: utf-8 -*-
# packages/mhd_q2d_tckae_uncons_ab/src/mhd_q2d_tckae_uncons_ab/data_2d.py
# Note: This is a modified version for 2D data (X, Z spatial dims).
# UPDATED: Now standardizes the data using mean and std deviation.

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


class tcKAEMHDDataset2D(Dataset):
    """
    Custom Dataset for 2D tcKAE training.
    The Y-dimension of the original data is expected to be 1 and is squeezed out.
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

        # Squeeze the Y dimension, assuming it's of size 1
        if self.data.ndim == 5 and self.data.shape[2] == 1:
            self.data = np.squeeze(self.data, axis=2)
            logging.info(f"Squeezed data shape from 5D to 4D. New shape: {self.data.shape}")


        # --- MODIFIED: Switched to Mean/Std normalization ---
        self.mean_vals, self.std_vals = None, None
        if norm_stats:
            # Reshape stats to be broadcastable over 4D tensor (C, X, Z)
            view_shape = (1, -1, 1, 1)
            all_mean = torch.from_numpy(norm_stats['mean_vals']).float()
            all_std = torch.from_numpy(norm_stats['std_vals']).float() # <-- NEW
            
            if channels_to_use:
                indices = [self.all_channel_names.index(name) for name in channels_to_use]
                self.mean_vals = all_mean[indices].view(*view_shape)
                self.std_vals = all_std[indices].view(*view_shape) # <-- NEW
            else:
                self.mean_vals = all_mean.view(*view_shape)
                self.std_vals = all_std.view(*view_shape) # <-- NEW
            
            # --- NEW SCALING LOGIC (Standardization) ---
            # Add epsilon for numerical stability
            self.std_vals = self.std_vals + 1e-8 
            # --- END MODIFICATION ---
        # --- END MODIFICATION ---

    def __len__(self) -> int:
        return self.data.shape[0]

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_vals is None: return x
        # Center at 0 and scale by standard deviation
        return (x - self.mean_vals) / self.std_vals # <-- MODIFIED
    
    def _denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        if self.mean_vals is None: return x_norm
        # De-scale and add back mean
        return (x_norm * self.std_vals) + self.mean_vals # <-- MODIFIED

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
        
        # Permute to (M, K+1, C, X, Z)
        tensor_block = torch.from_numpy(sequences.copy()).float().permute(0, 1, 4, 2, 3)
        return self._normalize(tensor_block)


class RolloutMHDDataset2D(Dataset):
    """
    Simpler Dataset for validation/testing for 2D data.
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

        # Squeeze the Y dimension, assuming it's of size 1
        if self.data.ndim == 5 and self.data.shape[2] == 1:
            self.data = np.squeeze(self.data, axis=2)

        # --- MODIFIED: Switched to Mean/Std normalization ---
        self.mean_vals, self.std_vals = None, None
        if norm_stats:
            # Reshape stats to be broadcastable over 4D tensor (C, X, Z)
            view_shape = (1, -1, 1, 1)
            all_mean = torch.from_numpy(norm_stats['mean_vals']).float()
            all_std = torch.from_numpy(norm_stats['std_vals']).float() # <-- NEW
            
            if channels_to_use:
                indices = [self.all_channel_names.index(name) for name in channels_to_use]
                self.mean_vals = all_mean[indices].view(*view_shape)
                self.std_vals = all_std[indices].view(*view_shape) # <-- NEW
            else:
                self.mean_vals = all_mean.view(*view_shape)
                self.std_vals = all_std.view(*view_shape) # <-- NEW
            
            # --- NEW SCALING LOGIC (Standardization) ---
            # Add epsilon for numerical stability
            self.std_vals = self.std_vals + 1e-8 
            # --- END MODIFICATION ---
        # --- END MODIFICATION ---

    def __len__(self) -> int:
        return self.data.shape[0]

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_vals is None: return x
        # Center at 0 and scale by standard deviation
        return (x - self.mean_vals) / self.std_vals # <-- MODIFIED

    def __getitem__(self, idx: int) -> torch.Tensor:
        sequence = self.data[idx : idx + self.rollout_steps + 1]
        if self.process_safe_copy:
            sequence = sequence.copy()
        
        # Permute to (T, C, X, Z)
        tensor_seq = torch.from_numpy(sequence).float().permute(0, 3, 1, 2)
        return self._normalize(tensor_seq)
