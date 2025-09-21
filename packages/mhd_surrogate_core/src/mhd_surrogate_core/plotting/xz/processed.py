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

def plot_velocity_quiver(
    data_path: Path | str,
    time_index: int,
    u_channel: str = 'vx',
    v_channel: str = 'vz',
    timeseries_data: Optional[np.ndarray] = None,
    coords: Optional[Dict] = None,
    channel_alias_map: Optional[Dict] = None,
    mean_flow_components: Optional[Dict] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[tuple] = None,
    base_size: float = 8.0,
    min_size: float = 3.0,
    unit_label: str = "",
    cmap: str = "viridis",
    quiver_stride: int = 10,
    arrow_width: Optional[float] = None,
):
    """
    Plots a 2D velocity field snapshot where arrows are colored by magnitude.
    """
    timeseries_data, coords = _load_data_for_viz(data_path, timeseries_data, coords)
    if timeseries_data is None: return

    try:
        u_idx = coords['labels'].index(u_channel)
        v_idx = coords['labels'].index(v_channel)
        
        if channel_alias_map:
            u_display = channel_alias_map.get(u_channel, u_channel)
            v_display = channel_alias_map.get(v_channel, v_channel)
        else:
            u_display, v_display = u_channel, v_channel

    except ValueError as e:
        logging.error(f"Required channel not found in {coords['labels']}: {e}")
        return

    # Extract velocity components for the given time index
    u_data = timeseries_data[time_index, :, :, u_idx]
    v_data = timeseries_data[time_index, :, :, v_idx]
    
    # Determine plot dimensions
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

    # --- Prepare data for quiver plot, accounting for mean flow ---
    base_title = f"Velocity Field ({u_display}, {v_display}) Snapshot"
    
    # Prepare fluctuation components for quiver arrows
    u_data_fluctuation = u_data.copy()
    v_data_fluctuation = v_data.copy()
    
    subtitle = f"at time index {time_index}"
    cbar_label = f"Velocity Magnitude" + (f" [{unit_label}]" if unit_label else "")

    if mean_flow_components:
        mean_u = mean_flow_components.get(u_channel, 0.0)
        mean_v = mean_flow_components.get(v_channel, 0.0)
        
        # Only subtract from the interior of the domain, preserving boundaries
        if u_data_fluctuation.shape[0] > 2 and u_data_fluctuation.shape[1] > 2:
             u_data_fluctuation[1:-1, 1:-1] -= mean_u
        if v_data_fluctuation.shape[0] > 2 and v_data_fluctuation.shape[1] > 2:
             v_data_fluctuation[1:-1, 1:-1] -= mean_v

        subtracted_parts = []
        if abs(mean_u) > 1e-9:
            subtracted_parts.append(f"{u_display}={mean_u:.2f}")
        if abs(mean_v) > 1e-9:
            subtracted_parts.append(f"{v_display}={mean_v:.2f}")
            
        if subtracted_parts:
            subtitle += f"\n(Arrows show fluctuations around mean {', '.join(subtracted_parts)})"
            cbar_label = f"Fluctuation Magnitude" + (f" [{unit_label}]" if unit_label else "")

    # Calculate magnitude from the fluctuation components
    magnitude = np.sqrt(u_data_fluctuation**2 + v_data_fluctuation**2)

    # Downsample all data for the quiver plot
    x_coords_q = coords['x'][::quiver_stride]
    z_coords_q = coords['z'][::quiver_stride]
    u_data_q = u_data_fluctuation[::quiver_stride, ::quiver_stride]
    v_data_q = v_data_fluctuation[::quiver_stride, ::quiver_stride]
    magnitude_q = magnitude[::quiver_stride, ::quiver_stride]
    X_q, Z_q = np.meshgrid(x_coords_q, z_coords_q, indexing='ij')

    # --- Plotting: Arrows colored by magnitude ---
    width = arrow_width if arrow_width is not None else 0.0035
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    q = ax.quiver(
        X_q, Z_q, u_data_q, v_data_q, magnitude_q,
        cmap=cmap, norm=norm,
        scale_units='xy', angles='xy', scale=None, width=width
    )
    fig.colorbar(q, ax=ax, label=cbar_label)
    ax.set_facecolor('#F0F0F0') # Use a neutral background

    ax.set_title(f"{base_title}\n{subtitle}")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Z Coordinate")
    plt.tight_layout()
    plt.show()


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
        coords['z'],                # Z-axis
        data_slice.T,               # Transpose for correct orientation
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


