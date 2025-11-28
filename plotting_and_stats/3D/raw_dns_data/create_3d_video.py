import logging
import tempfile
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import multiprocessing
from functools import partial
from tqdm import tqdm

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Try importing pyvista and handling headless display
try:
    import pyvista as pv
    # Set off-screen to True for server-side generation (headless)
    pv.OFF_SCREEN = True
    
    # CRITICAL FIX FOR HEADLESS CLUSTERS:
    # Try to start a virtual framebuffer (Xvfb). 
    # This mimics a monitor so VTK doesn't complain about missing EGL/Display.
    try:
        pv.start_xvfb()
    except OSError:
        # This happens if 'xvfb' is not installed on the Linux system.
        # We log a warning but proceed, hoping EGL or OSMesa works as fallback.
        logging.warning("Xvfb not found. If rendering fails, please install 'xvfb' (sudo apt install xvfb) or use 'xvfb-run'.")
    except Exception as e:
        logging.warning(f"Attempted to start Xvfb but failed: {e}")

except ImportError:
    pv = None

def _get_volume_data(
    snapshot_file: Path,
    nx: int, ny: int, nz: int,
    num_channels: int, channel_idx: int,
) -> tuple[np.ndarray, dict]:
    """
    Loads a single snapshot and extracts the required 3D volume data.
    """
    input_dtype = np.float64
    with open(snapshot_file, 'rb') as f:
        coords = {
            'x': np.fromfile(f, dtype=input_dtype, count=nx),
            'y': np.fromfile(f, dtype=input_dtype, count=ny),
            'z': np.fromfile(f, dtype=input_dtype, count=nz),
        }
        channel_data_1d = np.fromfile(f, dtype=input_dtype)

    # Reshape logic matches video.py: (nz, num_channels, ny, nx)
    data_4d_physical = channel_data_1d.reshape((nz, num_channels, ny, nx))
    
    # Transpose to (x, y, z) for PyVista standard indexing
    # Original: (z, chan, y, x) -> want (x, y, z)
    volume_3d = data_4d_physical.transpose(3, 2, 0, 1)[:, :, :, channel_idx]

    return volume_3d.astype(np.float32), coords

def _render_3d_frame_worker(args_tuple, common_args):
    """
    Worker function to render a single 3D frame using PyVista.
    """
    if pv is None:
        raise ImportError("PyVista is required for 3D video generation. Install it with 'pip install pyvista'.")

    i, f_path, time_val = args_tuple
    
    # Load data
    volume_data, coords = _get_volume_data(
        snapshot_file=f_path,
        nx=common_args['nx'], ny=common_args['ny'], nz=common_args['nz'],
        num_channels=len(common_args['source_channel_labels']),
        channel_idx=common_args['channel_idx'],
    )

    frame_path = common_args['frame_dir'] / f"frame_{i:04d}.png"

    # Create PyVista Grid
    grid = pv.RectilinearGrid(coords['x'], coords['y'], coords['z'])
    
    # Add data to the grid. 
    # Flatten using order='F' because volume_data is (x, y, z) and VTK iterates x, then y, then z
    grid.point_data["values"] = volume_data.flatten(order='F')

    # Setup Plotter
    # Note: We use the default 'params' to avoid instantiating windows if possible
    plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    plotter.set_background("white")

    # --- Rendering Style ---
    plotter.add_volume(
        grid, 
        scalars="values", 
        cmap=common_args['cmap'], 
        opacity="linear", 
        show_scalar_bar=True,
        scalar_bar_args={'title': common_args['cbar_label'], 'color': 'black'}
    )
    
    # Add outlines/bounds
    plotter.add_mesh(grid.outline(), color="black")

    plotter.view_isometric()
    
    plotter.add_title(
        f"{common_args['display_name']} - Time Index: {time_val}", 
        font_size=12, color="black"
    )
    
    # Save screenshot
    plotter.screenshot(frame_path)
    plotter.close()

    return frame_path

def generate_3d_video(
    snapshot_dir: Path,
    file_prefix: str,
    time_indices: range,
    nx: int, ny: int, nz: int,
    source_channel_labels: list,
    output_path: Path,
    channel: str,
    fps: int = 15,
    channel_alias: str = None,
    unit_label: str = "",
    num_workers: int = 1,
    cmap: str = "viridis",
):
    """
    Generates a 3D Volumetric video of the data evolving over time.
    """
    if pv is None:
        logging.error("PyVista is not installed. Please run `pip install pyvista` to generate 3D videos.")
        return

    snapshot_files = [snapshot_dir / f"{file_prefix}{i:06d}" for i in time_indices]
    if not all(f.exists() for f in snapshot_files):
        logging.error("Not all snapshot files were found. Aborting.")
        return

    try:
        channel_idx = source_channel_labels.index(channel)
    except ValueError:
        logging.error(f"Channel '{channel}' not found in source labels. Aborting.")
        return

    display_name = channel_alias if channel_alias else channel
    cbar_label = f"{display_name}" + (f" [{unit_label}]" if unit_label else "")

    with tempfile.TemporaryDirectory() as temp_dir:
        frame_dir = Path(temp_dir)
        
        logging.info(f"Rendering {len(snapshot_files)} 3D frames. This may take a while...")

        common_args = {
            'nx': nx, 'ny': ny, 'nz': nz, 
            'source_channel_labels': source_channel_labels,
            'channel_idx': channel_idx, 
            'frame_dir': frame_dir, 
            'display_name': display_name,
            'cbar_label': cbar_label, 
            'cmap': cmap,
        }
        
        tasks = [(i, f_path, time_val) for i, (f_path, time_val) in enumerate(zip(snapshot_files, time_indices))]
        
        worker_func = partial(_render_3d_frame_worker, common_args=common_args)
        
        frame_paths = []
        
        # Parallel Processing
        # Using a context manager for the pool to ensure cleanup
        with multiprocessing.Pool(processes=num_workers) as pool:
            with tqdm(total=len(tasks), desc="Rendering 3D frames") as pbar:
                for frame_path in pool.imap_unordered(worker_func, tasks):
                    frame_paths.append(frame_path)
                    pbar.update(1)

        frame_paths.sort()

        logging.info(f"Assembling 3D video at {output_path} with {fps} FPS...")
        with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
            for frame_path in tqdm(frame_paths, desc="Writing video"):
                image = imageio.imread(frame_path)
                writer.append_data(image)

    logging.info("3D Video generation complete.")