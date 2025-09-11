# mhd_surrogate_core/src/mhd_surrogate_core/plotting/raw.py

"""
Functions for visualizing raw snapshot data from binary files.
These functions include logic to interpolate data from non-uniform
grids onto a uniform grid for plotting.
"""

import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# This module might need the dynamic figsize calculator
from .processed import _calculate_dynamic_figsize

# ==============================================================================
# VISUALIZATION FUNCTIONS FOR RAW SNAPSHOTS (WITH INTERPOLATION)
# ==============================================================================

def plot_interpolated_xz_slice(
    snapshot_data_3d: np.ndarray,
    raw_coords: dict,
    channel: str,
    y_index: int,
    num_interp_points_z: int = 256,
    channel_alias: str = None,
    vmin: float = None,
    vmax: float = None,
    figsize: tuple = None,
    base_size: int = 8,
    min_size: int = 4,
    unit_label: str = None,
):
    """
    Extracts an X-Z slice from raw 3D data, interpolates it onto a uniform
    grid, and plots it.
    """
    display_name = channel_alias if channel_alias else channel
    
    # --- Prepare data and coordinates ---
    try:
        channel_idx = raw_coords['labels'].index(channel)
    except ValueError:
        logging.error(f"Channel '{channel}' not found in {raw_coords['labels']}.")
        return

    # Data is (x, y, z, channel), so slice is (x, z)
    data_slice_raw = snapshot_data_3d[:, y_index, :, channel_idx]
    
    x_coords_raw = raw_coords['x']
    z_coords_raw = raw_coords['z']
    
    # --- Create grids for interpolation ---
    # Source grid (can be non-uniform)
    X_raw, Z_raw = np.meshgrid(x_coords_raw, z_coords_raw, indexing='ij')
    
    # Target grid (uniform)
    x_coords_interp = x_coords_raw # X is already uniform
    z_coords_interp = np.linspace(z_coords_raw.min(), z_coords_raw.max(), num_interp_points_z)
    X_interp, Z_interp = np.meshgrid(x_coords_interp, z_coords_interp, indexing='ij')

    # --- Interpolate data ---
    logging.info(f"Interpolating X-Z slice for '{display_name}' onto a {len(x_coords_interp)}x{len(z_coords_interp)} grid...")
    points_raw = np.array([X_raw.flatten(), Z_raw.flatten()]).T
    values_raw = data_slice_raw.flatten()
    data_interp = griddata(points_raw, values_raw, (X_interp, Z_interp), method='cubic')
    
    # --- Plotting ---
    if figsize is None:
        x_range = x_coords_interp.max() - x_coords_interp.min()
        z_range = z_coords_interp.max() - z_coords_interp.min()
        figsize = _calculate_dynamic_figsize(x_range, z_range, base_size, min_size)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        x_coords_interp,
        z_coords_interp,
        data_interp.T, # Transpose to match pcolormesh convention
        shading='gouraud',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
    )

    cbar_label = f"Value of {display_name}"
    if unit_label:
        cbar_label += f" [{unit_label}]"
    fig.colorbar(im, ax=ax, label=cbar_label)
    
    ax.set_title(
        f"Interpolated X-Z Slice of '{display_name}'\n"
        f"at y={raw_coords['y'][y_index]:.2f} (idx={y_index})"
    )
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Z Coordinate")
    plt.tight_layout()
    plt.show()


def plot_interpolated_xy_slice(
    snapshot_data_3d: np.ndarray,
    raw_coords: dict,
    channel: str,
    z_index: int,
    num_interp_points_y: int = 256,
    channel_alias: str = None,
    vmin: float = None,
    vmax: float = None,
    figsize: tuple = None,
    base_size: int = 8,
    min_size: int = 4,
    unit_label: str = None,
):
    """
    Extracts an X-Y slice, interpolates, and plots.
    """
    display_name = channel_alias if channel_alias else channel
    
    try:
        channel_idx = raw_coords['labels'].index(channel)
    except ValueError:
        logging.error(f"Channel '{channel}' not found in {raw_coords['labels']}.")
        return

    # Data is (x, y, z, channel), so slice is (x, y)
    data_slice_raw = snapshot_data_3d[:, :, z_index, channel_idx]
    
    x_coords_raw = raw_coords['x']
    y_coords_raw = raw_coords['y']
    
    # --- Grids ---
    X_raw, Y_raw = np.meshgrid(x_coords_raw, y_coords_raw, indexing='ij')
    x_coords_interp = x_coords_raw
    y_coords_interp = np.linspace(y_coords_raw.min(), y_coords_raw.max(), num_interp_points_y)
    X_interp, Y_interp = np.meshgrid(x_coords_interp, y_coords_interp, indexing='ij')

    # --- Interpolation ---
    logging.info(f"Interpolating X-Y slice for '{display_name}' onto a {len(x_coords_interp)}x{len(y_coords_interp)} grid...")
    points_raw = np.array([X_raw.flatten(), Y_raw.flatten()]).T
    values_raw = data_slice_raw.flatten()
    data_interp = griddata(points_raw, values_raw, (X_interp, Y_interp), method='cubic')
    
    # --- Plotting ---
    if figsize is None:
        x_range = x_coords_interp.max() - x_coords_interp.min()
        y_range = y_coords_interp.max() - y_coords_interp.min()
        figsize = _calculate_dynamic_figsize(x_range, y_range, base_size, min_size)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        x_coords_interp,
        y_coords_interp,
        data_interp.T,
        shading='gouraud',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
    )

    cbar_label = f"Value of {display_name}"
    if unit_label:
        cbar_label += f" [{unit_label}]"
    fig.colorbar(im, ax=ax, label=cbar_label)
    
    ax.set_title(
        f"Interpolated X-Y Slice of '{display_name}'\n"
        f"at z={raw_coords['z'][z_index]:.2f} (idx={z_index})"
    )
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    plt.tight_layout()
    plt.show()


def plot_interpolated_yz_slice(
    snapshot_data_3d: np.ndarray,
    raw_coords: dict,
    channel: str,
    x_index: int,
    num_interp_points_y: int = 256,
    num_interp_points_z: int = 256,
    channel_alias: str = None,
    vmin: float = None,
    vmax: float = None,
    figsize: tuple = None,
    base_size: int = 8,
    min_size: int = 4,
    unit_label: str = None,
):
    """
    Extracts a Y-Z slice, interpolates, and plots.
    """
    display_name = channel_alias if channel_alias else channel
    
    try:
        channel_idx = raw_coords['labels'].index(channel)
    except ValueError:
        logging.error(f"Channel '{channel}' not found in {raw_coords['labels']}.")
        return

    # Data is (x, y, z, channel), so slice is (y, z)
    data_slice_raw = snapshot_data_3d[x_index, :, :, channel_idx]
    
    y_coords_raw = raw_coords['y']
    z_coords_raw = raw_coords['z']
    
    # --- Grids ---
    Y_raw, Z_raw = np.meshgrid(y_coords_raw, z_coords_raw, indexing='ij')
    y_coords_interp = np.linspace(y_coords_raw.min(), y_coords_raw.max(), num_interp_points_y)
    z_coords_interp = np.linspace(z_coords_raw.min(), z_coords_raw.max(), num_interp_points_z)
    Y_interp, Z_interp = np.meshgrid(y_coords_interp, z_coords_interp, indexing='ij')

    # --- Interpolation ---
    logging.info(f"Interpolating Y-Z slice for '{display_name}' onto a {len(y_coords_interp)}x{len(z_coords_interp)} grid...")
    points_raw = np.array([Y_raw.flatten(), Z_raw.flatten()]).T
    values_raw = data_slice_raw.flatten()
    data_interp = griddata(points_raw, values_raw, (Y_interp, Z_interp), method='cubic')
    
    # --- Plotting ---
    if figsize is None:
        y_range = y_coords_interp.max() - y_coords_interp.min()
        z_range = z_coords_interp.max() - z_coords_interp.min()
        figsize = _calculate_dynamic_figsize(y_range, z_range, base_size, min_size)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(
        y_coords_interp,
        z_coords_interp,
        data_interp.T,
        shading='gouraud',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
    )

    cbar_label = f"Value of {display_name}"
    if unit_label:
        cbar_label += f" [{unit_label}]"
    fig.colorbar(im, ax=ax, label=cbar_label)
    
    ax.set_title(
        f"Interpolated Y-Z Slice of '{display_name}'\n"
        f"at x={raw_coords['x'][x_index]:.2f} (idx={x_index})"
    )
    ax.set_xlabel("Y Coordinate")
    ax.set_ylabel("Z Coordinate")
    plt.tight_layout()
    plt.show()
