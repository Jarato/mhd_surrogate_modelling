# -*- coding: utf-8 -*-
# packages/flow_matching_2d/flow_matching_2d_base/src/flow_matching_2d_base/data.py
#
# Datasets for Flow Matching.
# 1. FlowMatchingDataset2D: Returns pairs of (y_k, y_k+1) for training.
# 2. RolloutDataset2D: Returns sequences (y_k, ..., y_k+R) for validation.
#    (Adapted from user's tcKAEMHDDataset2D)

import logging
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class FlowMatchingDataset2D(Dataset):
    """
    Custom Dataset for 2D Flow Matching training.
    Returns pairs of (state_k, state_k+1).
    """

    def __init__(
        self,
        full_timeseries: np.ndarray,
        all_channel_names: list[str],
        norm_stats: Dict[str, np.ndarray] | None = None,
        channels_to_use: list[str] | None = None,
        process_safe_copy: bool = False,
    ):
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
        
        if self.data.ndim != 4:
            logging.warning(f"Expected data to be 4D (T, X, Z, C), but got {self.data.ndim}D.")


        # Use Mean/Std normalization (from user's data.py)
        self.mean_vals, self.std_vals = None, None
        if norm_stats:
            # Reshape stats to be broadcastable (C, 1, 1)
            view_shape = (-1, 1, 1)
            all_mean = torch.from_numpy(norm_stats['mean_vals']).float()
            all_std = torch.from_numpy(norm_stats['std_vals']).float()
            
            if channels_to_use:
                indices = [self.all_channel_names.index(name) for name in channels_to_use]
                self.mean_vals = all_mean[indices].view(*view_shape)
                self.std_vals = all_std[indices].view(*view_shape)
            else:
                self.mean_vals = all_mean.view(*view_shape)
                self.std_vals = all_std.view(*view_shape)
            
            # Add epsilon for numerical stability
            self.std_vals = self.std_vals + 1e-8
            
    def __len__(self) -> int:
        # We need pairs (k, k+1), so length is T-1
        return self.data.shape[0] - 1

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_vals is None: return x
        return (x - self.mean_vals) / self.std_vals
    
    def _denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        if self.mean_vals is None: return x_norm
        return (x_norm * self.std_vals) + self.mean_vals

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get state k (condition) and state k+1 (target)
        state_k = self.data[idx]
        state_k_plus_1 = self.data[idx + 1]
        
        if self.process_safe_copy:
            state_k = state_k.copy()
            state_k_plus_1 = state_k_plus_1.copy()

        # Permute from (X, Z, C) to (C, X, Z)
        y_k_tensor = torch.from_numpy(state_k).float().permute(2, 0, 1)
        y_k_plus_1_tensor = torch.from_numpy(state_k_plus_1).float().permute(2, 0, 1)
        
        return self._normalize(y_k_tensor), self._normalize(y_k_plus_1_tensor)


class RolloutDataset2D(Dataset):
    """
    Simpler Dataset for validation/testing for 2D data.
    Returns a sequence of (y_k, y_k+1, ..., y_k+rollout_steps).
    (Adapted from user's RolloutMHDDataset2D)
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

        # Use Mean/Std normalization (from user's data.py)
        self.mean_vals, self.std_vals = None, None
        if norm_stats:
            # Reshape stats to be broadcastable (1, C, 1, 1) for sequence
            view_shape = (1, -1, 1, 1)
            all_mean = torch.from_numpy(norm_stats['mean_vals']).float()
            all_std = torch.from_numpy(norm_stats['std_vals']).float()
            
            if channels_to_use:
                indices = [self.all_channel_names.index(name) for name in channels_to_use]
                self.mean_vals = all_mean[indices].view(*view_shape)
                self.std_vals = all_std[indices].view(*view_shape)
            else:
                self.mean_vals = all_mean.view(*view_shape)
                self.std_vals = all_std.view(*view_shape)
            
            self.std_vals = self.std_vals + 1e-8

    def __len__(self) -> int:
        # Length is T - (rollout_steps + 1) + 1 = T - rollout_steps
        # This matches the user's original calculation.
        return self.data.shape[0] - self.rollout_steps

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean_vals is None: return x
        return (x - self.mean_vals) / self.std_vals

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Get sequence from idx to idx + rollout_steps + 1 (total R+1 items)
        sequence = self.data[idx : idx + self.rollout_steps + 1]
        if self.process_safe_copy:
            sequence = sequence.copy()
        
        # Permute from (T, X, Z, C) to (T, C, X, Z)
        tensor_seq = torch.from_numpy(sequence).float().permute(0, 3, 1, 2)
        return self._normalize(tensor_seq)
