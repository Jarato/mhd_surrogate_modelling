import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata

# --- Helper to load a single raw snapshot ---
def _load_single_snapshot(snapshot_file: Path, nx: int, ny: int, nz: int, source_channel_labels: list):
    """Loads and reshapes data from a single raw binary snapshot file."""
    if not snapshot_file.exists():
        logging.error(f"Snapshot file not found at {snapshot_file}")
        return None, None
    
    try:
        input_dtype = np.float64
        with open(snapshot_file, 'rb') as f:
            x_coords_raw = np.fromfile(f, dtype=input_dtype, count=nx)
            y_coords_raw = np.fromfile(f, dtype=input_dtype, count=ny)
            z_coords_raw = np.fromfile(f, dtype=input_dtype, count=nz)
            channel_data_1d = np.fromfile(f, dtype=input_dtype)
        
        num_input_channels = len(source_channel_labels)
        data_4d_physical = channel_data_1d.reshape((nz, num_input_channels, ny, nx))
        snapshot_data_3d = data_4d_physical.transpose(3, 2, 0, 1).astype(np.float32)
        
        raw_coords = {
            'labels': source_channel_labels,
            'x': x_coords_raw, 'y': y_coords_raw, 'z': z_coords_raw
        }
        return snapshot_data_3d, raw_coords
    except Exception as e:
        logging.error(f"Failed to load/process {snapshot_file.name}: {e}")
        return None, None


# --- Spatial Slice Plotting (for 3D data) ---

def plot_interpolated_xz_slice(
    snapshot_data_3d: np.ndarray,
    raw_coords: dict,
    channel: str,
    y_index: int,
    num_interp_points_z: int = 256,
    channel_alias: str = None,
    vmin: float = None,
    vmax: float = None,
    vcenter: float = None,
    figsize: tuple = None,
    base_size: float = 8.0,
    min_size: float = 3.0,
    unit_label: str = "",
    cmap: str = "viridis",
    show_title: bool = True,
    base_font_size: int = 12,
):
    """
    Extracts an x-z slice from raw 3D data, interpolates it onto a uniform
    grid, and plots it.
    """
    if snapshot_data_3d is None or not raw_coords:
        logging.warning("Snapshot data or coordinates not available. Skipping plot.")
        return

    try:
        channel_idx = raw_coords['labels'].index(channel)
        display_name = channel_alias if channel_alias else channel
    except ValueError:
        logging.error(f"Channel '{channel}' not found in labels: {raw_coords['labels']}")
        return

    # --- Data Extraction and Interpolation ---
    data_slice_raw = snapshot_data_3d[:, y_index, :, channel_idx]
    x_coords_raw = raw_coords['x']
    z_coords_raw = raw_coords['z']

    X_raw, Z_raw = np.meshgrid(x_coords_raw, z_coords_raw, indexing='ij')
    
    x_coords_interp = x_coords_raw
    z_coords_interp = np.linspace(z_coords_raw.min(), z_coords_raw.max(), num_interp_points_z)
    X_interp, Z_interp = np.meshgrid(x_coords_interp, z_coords_interp, indexing='ij')

    points_raw = np.array([X_raw.flatten(), Z_raw.flatten()]).T
    values_raw = data_slice_raw.flatten()
    data_interp = griddata(points_raw, values_raw, (X_interp, Z_interp), method='cubic')
    
    if vcenter is not None:
        np.nan_to_num(data_interp, copy=False, nan=vcenter)

    # --- Plotting ---
    if figsize is None:
        x_range = x_coords_raw.max() - x_coords_raw.min()
        z_range = z_coords_raw.max() - z_coords_raw.min()
        if x_range > z_range:
            fig_width = base_size
            aspect_ratio = z_range / x_range if x_range > 0 else 1
            fig_height = max(min_size, base_size * aspect_ratio)
        else:
            fig_height = base_size
            aspect_ratio = x_range / z_range if z_range > 0 else 1
            fig_width = max(min_size, base_size * aspect_ratio)
        figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize)

    plot_kwargs = {'shading': 'gouraud', 'cmap': cmap}
    if vcenter is not None and vmin is not None and vmax is not None:
        plot_kwargs['norm'] = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax
        
    im = ax.pcolormesh(
        x_coords_interp,
        z_coords_interp,
        data_interp.T,
        **plot_kwargs
    )

    cbar_label = f"Velocity {display_name}" + (f" [{unit_label}]" if unit_label else "")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, size=base_font_size)
    cbar.ax.tick_params(labelsize=base_font_size - 2)

    if show_title:
        ax.set_title(f"X-Z Slice of '{display_name}' at y={raw_coords['y'][y_index]:.2f} (idx={y_index})", fontsize=base_font_size + 2)
    ax.set_xlabel("X Coordinate", fontsize=base_font_size)
    ax.set_ylabel("Z Coordinate", fontsize=base_font_size)
    ax.tick_params(axis='both', which='major', labelsize=base_font_size - 2)
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
    vcenter: float = None,
    figsize: tuple = None,
    base_size: float = 8.0,
    min_size: float = 3.0,
    unit_label: str = "",
    cmap: str = "viridis",
    show_title: bool = True,
    base_font_size: int = 12,
):
    """
    Extracts an x-y slice from raw 3D data, interpolates it, and plots it.
    """
    if snapshot_data_3d is None or not raw_coords:
        logging.warning("Snapshot data or coordinates not available. Skipping plot.")
        return

    try:
        channel_idx = raw_coords['labels'].index(channel)
        display_name = channel_alias if channel_alias else channel
    except ValueError:
        logging.error(f"Channel '{channel}' not found in labels: {raw_coords['labels']}")
        return

    data_slice_raw = snapshot_data_3d[:, :, z_index, channel_idx]
    x_coords_raw = raw_coords['x']
    y_coords_raw = raw_coords['y']

    X_raw, Y_raw = np.meshgrid(x_coords_raw, y_coords_raw, indexing='ij')

    x_coords_interp = x_coords_raw
    y_coords_interp = np.linspace(y_coords_raw.min(), y_coords_raw.max(), num_interp_points_y)
    X_interp, Y_interp = np.meshgrid(x_coords_interp, y_coords_interp, indexing='ij')

    points_raw = np.array([X_raw.flatten(), Y_raw.flatten()]).T
    values_raw = data_slice_raw.flatten()
    data_interp = griddata(points_raw, values_raw, (X_interp, Y_interp), method='cubic')
    
    if vcenter is not None:
        np.nan_to_num(data_interp, copy=False, nan=vcenter)

    if figsize is None:
        x_range = x_coords_raw.max() - x_coords_raw.min()
        y_range = y_coords_raw.max() - y_coords_raw.min()
        if x_range > y_range:
            fig_width = base_size
            aspect_ratio = y_range / x_range if x_range > 0 else 1
            fig_height = max(min_size, base_size * aspect_ratio)
        else:
            fig_height = base_size
            aspect_ratio = x_range / y_range if y_range > 0 else 1
            fig_width = max(min_size, base_size * aspect_ratio)
        figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize)
    
    plot_kwargs = {'shading': 'gouraud', 'cmap': cmap}
    if vcenter is not None and vmin is not None and vmax is not None:
        plot_kwargs['norm'] = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax
        
    im = ax.pcolormesh(
        x_coords_interp,
        y_coords_interp,
        data_interp.T,
        **plot_kwargs)

    cbar_label = f"Velocity {display_name}" + (f" [{unit_label}]" if unit_label else "")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, size=base_font_size)
    cbar.ax.tick_params(labelsize=base_font_size - 2)

    if show_title:
        ax.set_title(f"X-Y Slice of '{display_name}' at z={raw_coords['z'][z_index]:.2f} (idx={z_index})", fontsize=base_font_size + 2)
    ax.set_xlabel("X Coordinate", fontsize=base_font_size)
    ax.set_ylabel("Y Coordinate", fontsize=base_font_size)
    ax.tick_params(axis='both', which='major', labelsize=base_font_size - 2)
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
    vcenter: float = None,
    figsize: tuple = None,
    base_size: float = 8.0,
    min_size: float = 3.0,
    unit_label: str = "",
    cmap: str = "viridis",
    show_title: bool = True,
    base_font_size: int = 12,
):
    """
    Extracts a y-z slice from raw 3D data, interpolates it, and plots it.
    """
    if snapshot_data_3d is None or not raw_coords:
        logging.warning("Snapshot data or coordinates not available. Skipping plot.")
        return

    try:
        channel_idx = raw_coords['labels'].index(channel)
        display_name = channel_alias if channel_alias else channel
    except ValueError:
        logging.error(f"Channel '{channel}' not found in labels: {raw_coords['labels']}")
        return

    data_slice_raw = snapshot_data_3d[x_index, :, :, channel_idx]
    y_coords_raw = raw_coords['y']
    z_coords_raw = raw_coords['z']

    Y_raw, Z_raw = np.meshgrid(y_coords_raw, z_coords_raw, indexing='ij')

    y_coords_interp = np.linspace(y_coords_raw.min(), y_coords_raw.max(), num_interp_points_y)
    z_coords_interp = np.linspace(z_coords_raw.min(), z_coords_raw.max(), num_interp_points_z)
    Y_interp, Z_interp = np.meshgrid(y_coords_interp, z_coords_interp, indexing='ij')

    points_raw = np.array([Y_raw.flatten(), Z_raw.flatten()]).T
    values_raw = data_slice_raw.flatten()
    data_interp = griddata(points_raw, values_raw, (Y_interp, Z_interp), method='cubic')
    
    if vcenter is not None:
        np.nan_to_num(data_interp, copy=False, nan=vcenter)

    if figsize is None:
        y_range = y_coords_raw.max() - y_coords_raw.min()
        z_range = z_coords_raw.max() - z_coords_raw.min()
        if y_range > z_range:
            fig_width = base_size
            aspect_ratio = z_range / y_range if y_range > 0 else 1
            fig_height = max(min_size, base_size * aspect_ratio)
        else:
            fig_height = base_size
            aspect_ratio = y_range / z_range if z_range > 0 else 1
            fig_width = max(min_size, base_size * aspect_ratio)
        figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize)
    
    plot_kwargs = {'shading': 'gouraud', 'cmap': cmap}
    if vcenter is not None and vmin is not None and vmax is not None:
        plot_kwargs['norm'] = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(
        y_coords_interp,
        z_coords_interp,
        data_interp.T,
        **plot_kwargs)

    cbar_label = f"Velocity {display_name}" + (f" [{unit_label}]" if unit_label else "")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, size=base_font_size)
    cbar.ax.tick_params(labelsize=base_font_size - 2)

    if show_title:
        ax.set_title(f"Y-Z Slice of '{display_name}' at x={raw_coords['x'][x_index]:.2f} (idx={x_index})", fontsize=base_font_size + 2)
    ax.set_xlabel("Y Coordinate", fontsize=base_font_size)
    ax.set_ylabel("Z Coordinate", fontsize=base_font_size)
    ax.tick_params(axis='both', which='major', labelsize=base_font_size - 2)
    plt.tight_layout()
    plt.show()


# --- Time Evolution Plotting (for 4D data) ---

def plot_interpolated_z_time_evolution(
    time_evolution_data: np.ndarray,
    raw_coords: dict,
    channel: str,
    x_index: int,
    y_index: int,
    output_path: Path,
    num_interp_points_z: int = 256,
    channel_alias: str = None,
    vmin: float = None,
    vmax: float = None,
    vcenter: float = None,
    figsize: tuple = None,
    base_size: float = 8.0,
    min_size: float = 3.0,
    unit_label: str = "",
    cmap: str = "viridis",
    show_title: bool = True,
    base_font_size: int = 12,
):
    """
    Takes a 2D (time, z) data array, interpolates it, and saves a plot.
    """
    display_name = channel_alias if channel_alias else channel
    num_timesteps = time_evolution_data.shape[0]
    time_coords_raw = np.arange(num_timesteps)
    z_coords_raw = raw_coords['z']

    T_raw, Z_raw = np.meshgrid(time_coords_raw, z_coords_raw, indexing='ij')

    time_coords_interp = time_coords_raw
    z_coords_interp = np.linspace(z_coords_raw.min(), z_coords_raw.max(), num_interp_points_z)
    T_interp, Z_interp = np.meshgrid(time_coords_interp, z_coords_interp, indexing='ij')
    
    points_raw = np.array([T_raw.flatten(), Z_raw.flatten()]).T
    values_raw = time_evolution_data.flatten()
    data_interp = griddata(points_raw, values_raw, (T_interp, Z_interp), method='cubic')

    # Handle NaNs from interpolation, which break TwoSlopeNorm
    if vcenter is not None:
        np.nan_to_num(data_interp, copy=False, nan=vcenter)

    if figsize is None:
        time_range = float(num_timesteps)
        z_range = z_coords_raw.max() - z_coords_raw.min()
        if time_range > z_range:
            fig_width = base_size
            aspect_ratio = z_range / time_range if time_range > 0 else 1
            fig_height = max(min_size, base_size * aspect_ratio)
        else:
            fig_height = base_size
            aspect_ratio = time_range / z_range if z_range > 0 else 1
            fig_width = max(min_size, base_size * aspect_ratio)
        figsize = (fig_width, fig_height)
        
    fig, ax = plt.subplots(figsize=figsize)

    plot_kwargs = {'shading': 'gouraud', 'cmap': cmap}
    if vcenter is not None and vmin is not None and vmax is not None:
        plot_kwargs['norm'] = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(
        time_coords_interp,
        z_coords_interp,
        data_interp.T,
        **plot_kwargs
    )
        
    cbar_label = f"Velocity {display_name}" + (f" [{unit_label}]" if unit_label else "")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, size=base_font_size)
    cbar.ax.tick_params(labelsize=base_font_size - 2)

    if show_title:
        ax.set_title(f"Time Evolution of '{display_name}' along Z-axis\nat x={raw_coords['x'][x_index]:.2f}, y={raw_coords['y'][y_index]:.2f}", fontsize=base_font_size + 2)
    ax.set_xlabel("Time Index", fontsize=base_font_size)
    ax.set_ylabel("Z Coordinate", fontsize=base_font_size)
    ax.tick_params(axis='both', which='major', labelsize=base_font_size - 2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_interpolated_y_time_evolution(
    time_evolution_data: np.ndarray,
    raw_coords: dict,
    channel: str,
    x_index: int,
    z_index: int,
    output_path: Path,
    num_interp_points_y: int = 256,
    channel_alias: str = None,
    vmin: float = None,
    vmax: float = None,
    vcenter: float = None,
    figsize: tuple = None,
    base_size: float = 8.0,
    min_size: float = 3.0,
    unit_label: str = "",
    cmap: str = "viridis",
    show_title: bool = True,
    base_font_size: int = 12,
):
    """
    Takes a 2D (time, y) data array, interpolates it, and saves a plot.
    """
    display_name = channel_alias if channel_alias else channel
    num_timesteps = time_evolution_data.shape[0]
    time_coords_raw = np.arange(num_timesteps)
    y_coords_raw = raw_coords['y']

    T_raw, Y_raw = np.meshgrid(time_coords_raw, y_coords_raw, indexing='ij')

    time_coords_interp = time_coords_raw
    y_coords_interp = np.linspace(y_coords_raw.min(), y_coords_raw.max(), num_interp_points_y)
    T_interp, Y_interp = np.meshgrid(time_coords_interp, y_coords_interp, indexing='ij')

    points_raw = np.array([T_raw.flatten(), Y_raw.flatten()]).T
    values_raw = time_evolution_data.flatten()
    data_interp = griddata(points_raw, values_raw, (T_interp, Y_interp), method='cubic')

    # Handle NaNs from interpolation, which break TwoSlopeNorm
    if vcenter is not None:
        np.nan_to_num(data_interp, copy=False, nan=vcenter)

    if figsize is None:
        time_range = float(num_timesteps)
        y_range = y_coords_raw.max() - y_coords_raw.min()
        if time_range > y_range:
            fig_width = base_size
            aspect_ratio = y_range / time_range if time_range > 0 else 1
            fig_height = max(min_size, base_size * aspect_ratio)
        else:
            fig_height = base_size
            aspect_ratio = time_range / y_range if y_range > 0 else 1
            fig_width = max(min_size, base_size * aspect_ratio)
        figsize = (fig_width, fig_height)
        
    fig, ax = plt.subplots(figsize=figsize)

    plot_kwargs = {'shading': 'gouraud', 'cmap': cmap}
    if vcenter is not None and vmin is not None and vmax is not None:
        plot_kwargs['norm'] = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(
        time_coords_interp,
        y_coords_interp,
        data_interp.T,
        **plot_kwargs
    )

    cbar_label = f"Velocity {display_name}" + (f" [{unit_label}]" if unit_label else "")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, size=base_font_size)
    cbar.ax.tick_params(labelsize=base_font_size - 2)

    if show_title:
        ax.set_title(f"Time Evolution of '{display_name}' along Y-axis\nat x={raw_coords['x'][x_index]:.2f}, z={raw_coords['z'][z_index]:.2f}", fontsize=base_font_size + 2)
    ax.set_xlabel("Time Index", fontsize=base_font_size)
    ax.set_ylabel("Y Coordinate", fontsize=base_font_size)
    ax.tick_params(axis='both', which='major', labelsize=base_font_size - 2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_interpolated_x_time_evolution(
    time_evolution_data: np.ndarray,
    raw_coords: dict,
    channel: str,
    y_index: int,
    z_index: int,
    output_path: Path,
    channel_alias: str = None,
    vmin: float = None,
    vmax: float = None,
    vcenter: float = None,
    figsize: tuple = None,
    base_size: float = 8.0,
    min_size: float = 3.0,
    unit_label: str = "",
    cmap: str = "viridis",
    show_title: bool = True,
    base_font_size: int = 12,
):
    """
    Takes a 2D (time, x) data array and saves a plot. No interpolation needed.
    """
    display_name = channel_alias if channel_alias else channel
    num_timesteps = time_evolution_data.shape[0]
    time_coords = np.arange(num_timesteps)
    x_coords = raw_coords['x']

    if figsize is None:
        time_range = float(num_timesteps)
        x_range = x_coords.max() - x_coords.min()
        if time_range > x_range:
            fig_width = base_size
            aspect_ratio = x_range / time_range if time_range > 0 else 1
            fig_height = max(min_size, base_size * aspect_ratio)
        else:
            fig_height = base_size
            aspect_ratio = time_range / x_range if x_range > 0 else 1
            fig_width = max(min_size, base_size * aspect_ratio)
        figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize)
    
    plot_kwargs = {'shading': 'gouraud', 'cmap': cmap}
    if vcenter is not None and vmin is not None and vmax is not None:
        plot_kwargs['norm'] = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(
        time_coords,
        x_coords,
        time_evolution_data.T,
        **plot_kwargs
    )

    cbar_label = f"Velocity {display_name}" + (f" [{unit_label}]" if unit_label else "")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, size=base_font_size)
    cbar.ax.tick_params(labelsize=base_font_size - 2)

    if show_title:
        ax.set_title(f"Time Evolution of '{display_name}' along X-axis\nat y={raw_coords['y'][y_index]:.2f}, z={raw_coords['z'][z_index]:.2f}", fontsize=base_font_size + 2)
    ax.set_xlabel("Time Index", fontsize=base_font_size)
    ax.set_ylabel("X Coordinate", fontsize=base_font_size)
    ax.tick_params(axis='both', which='major', labelsize=base_font_size - 2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

