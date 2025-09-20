# -*- coding: utf-8 -*-
# mhd_surrogate_core/plotting/xz/processed.py

import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# --- Helper to load data for visualization functions ---
def _load_data_for_viz(data_path, timeseries_data, coords):
    """Loads data from file if not provided in memory."""
    if timeseries_data is not None and coords is not None:
        return timeseries_data, coords

    data_path = Path(data_path)
    if not data_path.exists():
        logging.error(f"Data file not found at: {data_path}")
        return None, None
    with np.load(data_path, allow_pickle=True) as data:
        timeseries_data = data['timeseries']
        # Handle 2D coordinates, defaulting to integer ranges if not present
        if 'x_coords' in data and 'z_coords' in data:
            coords = {'labels': list(data['labels']), 'x': data['x_coords'], 'z': data['z_coords']}
        else:
            coords = {
                'labels': list(data['labels']),
                'x': np.arange(timeseries_data.shape[1]),
                'z': np.arange(timeseries_data.shape[2]),
            }
    return timeseries_data, coords


# ==============================================================================
# VISUALIZATION FUNCTIONS FOR 2D PROCESSED NPZ DATA
# ==============================================================================

def plot_z_time_evolution(
    data_path: Path | str,
    channel: str,
    x_index: int,
    timeseries_data: Optional[np.ndarray] = None,
    coords: Optional[Dict] = None,
    channel_alias: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
    figsize: Optional[tuple] = None,
    base_size: float = 8.0,
    min_size: float = 3.0,
    unit_label: str = "",
    cmap: str = "viridis",
):
    """
    Plots the time evolution of a channel along the z-axis (Time-Z plot) for 2D data.
    """
    timeseries_data, coords = _load_data_for_viz(data_path, timeseries_data, coords)
    if timeseries_data is None: return

    try:
        channel_idx = coords['labels'].index(channel)
        display_name = channel_alias if channel_alias else channel
    except ValueError:
        logging.error(f"Channel '{channel}' not found in {coords['labels']}.")
        return

    # Data is shaped (time, x, z, channel)
    data_slice = timeseries_data[:, x_index, :, channel_idx]

    if figsize is None:
        time_range = float(timeseries_data.shape[0])
        z_range = coords['z'].max() - coords['z'].min()
        if time_range > z_range:
            fig_width = base_size
            aspect_ratio = z_range / time_range if time_range > 0 else 1
            fig_height = max(min_size, base_size * aspect_ratio)
        else:
            fig_height = base_size
            aspect_ratio = time_range / z_range if z_range > 0 else 1
            fig_width = max(min_size, base_size * aspect_ratio)
        figsize = (fig_width, fig_height)
        
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    plot_kwargs = {'shading': 'gouraud', 'cmap': cmap}
    if vcenter is not None and vmin is not None and vmax is not None:
        plot_kwargs['norm'] = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(
        range(data_slice.shape[0]),  # Time
        coords['z'],                 # Z-axis
        data_slice.T,                # Transpose for correct orientation
        **plot_kwargs
    )

    cbar_label = f"Value of {display_name}" + (f" [{unit_label}]" if unit_label else "")
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(
        f"Time Evolution of '{display_name}' along Z-axis\n"
        f"at x={coords['x'][x_index]:.2f} (idx={x_index})"
    )
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Z Coordinate")
    plt.tight_layout()
    plt.show()


def plot_x_time_evolution(
    data_path: Path | str,
    channel: str,
    z_index: int,
    timeseries_data: Optional[np.ndarray] = None,
    coords: Optional[Dict] = None,
    channel_alias: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
    figsize: Optional[tuple] = None,
    base_size: float = 8.0,
    min_size: float = 3.0,
    unit_label: str = "",
    cmap: str = "viridis",
):
    """
    Plots the time evolution of a channel along the x-axis (Time-X plot) for 2D data.
    """
    timeseries_data, coords = _load_data_for_viz(data_path, timeseries_data, coords)
    if timeseries_data is None: return

    try:
        channel_idx = coords['labels'].index(channel)
        display_name = channel_alias if channel_alias else channel
    except ValueError:
        logging.error(f"Channel '{channel}' not found in {coords['labels']}.")
        return

    # Data is shaped (time, x, z, channel)
    data_slice = timeseries_data[:, :, z_index, channel_idx]
    
    if figsize is None:
        time_range = float(timeseries_data.shape[0])
        x_range = coords['x'].max() - coords['x'].min()
        if time_range > x_range:
            fig_width = base_size
            aspect_ratio = x_range / time_range if time_range > 0 else 1
            fig_height = max(min_size, base_size * aspect_ratio)
        else:
            fig_height = base_size
            aspect_ratio = time_range / x_range if x_range > 0 else 1
            fig_width = max(min_size, base_size * aspect_ratio)
        figsize = (fig_width, fig_height)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    plot_kwargs = {'shading': 'gouraud', 'cmap': cmap}
    if vcenter is not None and vmin is not None and vmax is not None:
        plot_kwargs['norm'] = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(
        range(data_slice.shape[0]),
        coords['x'],
        data_slice.T,
        **plot_kwargs
    )
    
    cbar_label = f"Value of {display_name}" + (f" [{unit_label}]" if unit_label else "")
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(
        f"Time Evolution of '{display_name}' along X-axis\n"
        f"at z={coords['z'][z_index]:.2f} (idx={z_index})"
    )
    ax.set_xlabel("Time Index")
    ax.set_ylabel("X Coordinate")
    plt.tight_layout()
    plt.show()


def plot_xz_snapshot(
    data_path: Path | str,
    channel: str,
    time_index: int,
    timeseries_data: Optional[np.ndarray] = None,
    coords: Optional[Dict] = None,
    channel_alias: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
    figsize: Optional[tuple] = None,
    base_size: float = 8.0,
    min_size: float = 3.0,
    unit_label: str = "",
    cmap: str = "viridis",
):
    """
    Plots a 2D snapshot in the x-z plane.
    """
    timeseries_data, coords = _load_data_for_viz(data_path, timeseries_data, coords)
    if timeseries_data is None: return

    try:
        channel_idx = coords['labels'].index(channel)
        display_name = channel_alias if channel_alias else channel
    except ValueError:
        logging.error(f"Channel '{channel}' not found in {coords['labels']}.")
        return

    # Data is shaped (time, x, z, channel)
    data_slice = timeseries_data[time_index, :, :, channel_idx]

    if figsize is None:
        x_range = coords['x'].max() - coords['x'].min()
        z_range = coords['z'].max() - coords['z'].min()
        if x_range > z_range:
            fig_width = base_size
            aspect_ratio = z_range / x_range if x_range > 0 else 1
            fig_height = max(min_size, base_size * aspect_ratio)
        else:
            fig_height = base_size
            aspect_ratio = x_range / z_range if z_range > 0 else 1
            fig_width = max(min_size, base_size * aspect_ratio)
        figsize = (fig_width, fig_height)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    plot_kwargs = {'shading': 'gouraud', 'cmap': cmap}
    if vcenter is not None and vmin is not None and vmax is not None:
        plot_kwargs['norm'] = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(
        coords['x'],
        coords['z'],
        data_slice.T,
        **plot_kwargs
    )

    cbar_label = f"Value of {display_name}" + (f" [{unit_label}]" if unit_label else "")
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(
        f"X-Z Snapshot of '{display_name}'\n"
        f"at time index {time_index}"
    )
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Z Coordinate")
    plt.tight_layout()
    plt.show()
