import logging
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm
import imageio.v2 as imageio

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _get_slice_data(
    snapshot_file: Path,
    nx: int, ny: int, nz: int,
    num_channels: int, channel_idx: int,
    slice_orientation: str, slice_index: int
) -> tuple[np.ndarray, dict]:
    """Loads a single snapshot and extracts the required 2D data slice and coordinates."""
    input_dtype = np.float64
    with open(snapshot_file, 'rb') as f:
        coords = {
            'x': np.fromfile(f, dtype=input_dtype, count=nx),
            'y': np.fromfile(f, dtype=input_dtype, count=ny),
            'z': np.fromfile(f, dtype=input_dtype, count=nz),
        }
        channel_data_1d = np.fromfile(f, dtype=input_dtype)

    data_4d_physical = channel_data_1d.reshape((nz, num_channels, ny, nx))
    snapshot_3d = data_4d_physical.transpose(3, 2, 0, 1) # -> (x, y, z, chan)

    if slice_orientation == 'xz':
        data_slice = snapshot_3d[:, slice_index, :, channel_idx]
        slice_coords = {'x': coords['x'], 'y': coords['z']} # y-axis of plot is z-data
    elif slice_orientation == 'xy':
        data_slice = snapshot_3d[:, :, slice_index, channel_idx]
        slice_coords = {'x': coords['x'], 'y': coords['y']}
    elif slice_orientation == 'yz':
        data_slice = snapshot_3d[slice_index, :, :, channel_idx]
        slice_coords = {'x': coords['y'], 'y': coords['z']} # x-axis of plot is y-data
    else:
        raise ValueError(f"Invalid slice orientation: {slice_orientation}")

    return data_slice.astype(np.float32), slice_coords


def _create_frame(
    frame_path: Path,
    data_slice_raw: np.ndarray,
    slice_coords: dict,
    interp_points: tuple,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    vmin: float,
    vmax: float,
    base_size: float,
    min_size: float,
):
    """Interpolates a 2D slice and plots it to a file."""
    # Interpolate data
    x_coords_raw = slice_coords['x']
    y_coords_raw = slice_coords['y']
    X_raw, Y_raw = np.meshgrid(x_coords_raw, y_coords_raw, indexing='ij')

    x_coords_interp = np.linspace(x_coords_raw.min(), x_coords_raw.max(), data_slice_raw.shape[0] if interp_points[0] is None else interp_points[0])
    y_coords_interp = np.linspace(y_coords_raw.min(), y_coords_raw.max(), data_slice_raw.shape[1] if interp_points[1] is None else interp_points[1])
    X_interp, Y_interp = np.meshgrid(x_coords_interp, y_coords_interp, indexing='ij')
    
    points_raw = np.array([X_raw.flatten(), Y_raw.flatten()]).T
    values_raw = data_slice_raw.flatten()
    data_interp = griddata(points_raw, values_raw, (X_interp, Y_interp), method='cubic')

    # Determine figure size
    dpi = 150 # Standard DPI for saving frames
    macro_block_size = 16 # For video codec compatibility
    
    x_range = x_coords_raw.max() - x_coords_raw.min()
    y_range = y_coords_raw.max() - y_coords_raw.min()
    if x_range > y_range:
        fig_width_in = base_size
        aspect_ratio = y_range / x_range if x_range > 0 else 1
        fig_height_in = max(min_size, base_size * aspect_ratio)
    else:
        fig_height_in = base_size
        aspect_ratio = x_range / y_range if y_range > 0 else 1
        fig_width_in = max(min_size, base_size * aspect_ratio)

    # Adjust size to be divisible by macro_block_size
    width_px = int(fig_width_in * dpi)
    height_px = int(fig_height_in * dpi)
    
    width_px = (width_px + macro_block_size - 1) // macro_block_size * macro_block_size
    height_px = (height_px + macro_block_size - 1) // macro_block_size * macro_block_size
    
    figsize = (width_px / dpi, height_px / dpi)

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(x_coords_interp, y_coords_interp, data_interp.T,
                       shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    
    # Save and close
    plt.savefig(frame_path, dpi=dpi)
    plt.close(fig)


def generate_slice_video(
    snapshot_dir: Path,
    file_prefix: str,
    time_indices: range,
    nx: int, ny: int, nz: int,
    source_channel_labels: list,
    output_path: Path,
    slice_orientation: str,
    slice_index: int,
    channel: str,
    fps: int = 10,
    num_interp_points_y: int = 256,
    num_interp_points_z: int = 256,
    channel_alias: str = None,
    unit_label: str = "",
    base_size: float = 10.0,
    min_size: float = 3.0,
    vmin_override: float = None,
    vmax_override: float = None,
):
    """
    Generates a 2D video of a slice evolving over time from raw snapshot files.
    """
    snapshot_files = [snapshot_dir / f"{file_prefix}{i:06d}" for i in time_indices]
    if not all(f.exists() for f in snapshot_files):
        logging.error("Not all snapshot files were found. Aborting.")
        return

    try:
        channel_idx = source_channel_labels.index(channel)
    except ValueError:
        logging.error(f"Channel '{channel}' not found in source labels. Aborting.")
        return

    # Determine global color scale
    vmin, vmax = vmin_override, vmax_override
    if vmin is None or vmax is None:
        logging.info("Calculating global color scale across all timesteps...")
        all_slice_data = []
        for f_path in tqdm(snapshot_files, desc="Scanning for color scale"):
            data_slice, _ = _get_slice_data(f_path, nx, ny, nz, len(source_channel_labels), 
                                            channel_idx, slice_orientation, slice_index)
            all_slice_data.append(data_slice)
        vmin = min(d.min() for d in all_slice_data)
        vmax = max(d.max() for d in all_slice_data)
        logging.info(f"Global color scale set to: [{vmin:.3f}, {vmax:.3f}]")

    # Setup plot labels
    display_name = channel_alias if channel_alias else channel
    cbar_label = f"Value of {display_name}" + (f" [{unit_label}]" if unit_label else "")
    
    orient_map = {
        'xz': {'xlabel': 'X Coordinate', 'ylabel': 'Z Coordinate', 'interp': (None, num_interp_points_z)},
        'xy': {'xlabel': 'X Coordinate', 'ylabel': 'Y Coordinate', 'interp': (None, num_interp_points_y)},
        'yz': {'xlabel': 'Y Coordinate', 'ylabel': 'Z Coordinate', 'interp': (num_interp_points_y, num_interp_points_z)},
    }
    plot_labels = orient_map[slice_orientation]

    # Generate frames in a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_dir = Path(temp_dir)
        frame_paths = []
        
        for i, f_path in enumerate(tqdm(snapshot_files, desc="Generating frames")):
            data_slice, slice_coords = _get_slice_data(f_path, nx, ny, nz, len(source_channel_labels), 
                                                       channel_idx, slice_orientation, slice_index)
            frame_path = frame_dir / f"frame_{i:04d}.png"
            
            _create_frame(
                frame_path=frame_path,
                data_slice_raw=data_slice,
                slice_coords=slice_coords,
                interp_points=plot_labels['interp'],
                title=f"Slice of '{display_name}' at Time Index {time_indices[i]}",
                xlabel=plot_labels['xlabel'],
                ylabel=plot_labels['ylabel'],
                cbar_label=cbar_label,
                vmin=vmin, vmax=vmax,
                base_size=base_size, min_size=min_size,
            )
            frame_paths.append(frame_path)

        # Assemble video
        logging.info(f"Assembling video at {output_path} with {fps} FPS...")
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame_path in tqdm(frame_paths, desc="Writing video"):
                image = imageio.imread(frame_path)
                writer.append_data(image)

    logging.info("Video generation complete.")

