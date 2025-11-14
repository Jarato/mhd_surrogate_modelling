# -*- coding: utf-8 -*-
# mhd_surrogate_core/plotting/xz/video.py

import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm
import imageio.v2 as imageio
import multiprocessing
from functools import partial

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variable to hold the shared data for worker processes
worker_data = {}

def _init_worker_from_npz(timeseries_data_shared):
    """Initializer for each worker process. Puts the shared data into a global variable."""
    worker_data['timeseries_data'] = timeseries_data_shared

def _create_frame_from_npz(
    frame_path: Path,
    data_slice: np.ndarray,
    coords: Dict[str, np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    vmin: float,
    vmax: float,
    vcenter: Optional[float],
    base_size: float,
    min_size: float,
    cmap: str = "viridis",
):
    """Plots a 2D data slice from npz data to a file."""
    x_coords = coords['x']
    z_coords = coords['z']
    
    dpi = 150
    macro_block_size = 16
    
    x_range = x_coords.max() - x_coords.min() if len(x_coords) > 1 else 1
    z_range = z_coords.max() - z_coords.min() if len(z_coords) > 1 else 1
    
    if x_range >= z_range:
        fig_width_in = base_size
        aspect_ratio = z_range / x_range if x_range > 0 else 1
        fig_height_in = max(min_size, base_size * aspect_ratio)
    else:
        fig_height_in = base_size
        aspect_ratio = x_range / z_range if z_range > 0 else 1
        fig_width_in = max(min_size, base_size * aspect_ratio)

    width_px = int(fig_width_in * dpi)
    height_px = int(fig_height_in * dpi)
    
    width_px = (width_px + macro_block_size - 1) // macro_block_size * macro_block_size
    height_px = (height_px + macro_block_size - 1) // macro_block_size * macro_block_size
    
    figsize = (width_px / dpi, height_px / dpi)

    fig, ax = plt.subplots(figsize=figsize)
    
    plot_kwargs = {'shading': 'gouraud', 'cmap': cmap}
    if vcenter is not None and vmin is not None and vmax is not None:
        plot_kwargs['norm'] = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    im = ax.pcolormesh(x_coords, z_coords, data_slice.T, **plot_kwargs)
    
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    
    plt.savefig(frame_path, dpi=dpi)
    plt.close(fig)

def _generate_frame_worker_from_npz(relative_time_index, common_args):
    """Wrapper function for multiprocessing to generate a single frame from 2D npz data."""
    timeseries_data = worker_data['timeseries_data']
    
    time_offset = common_args['time_offset']
    absolute_time_index = relative_time_index + time_offset
    
    frame_path = common_args['frame_dir'] / f"frame_{absolute_time_index:05d}.png"
    display_name = common_args['display_name']
    channel_idx = common_args['channel_idx']
    
    # Directly get the 2D data slice (X, Z) for the current timestep and channel
    data_slice = timeseries_data[relative_time_index, :, :, channel_idx]
    
    title = f"'{display_name}' at Time Index {absolute_time_index}"
        
    _create_frame_from_npz(
        frame_path=frame_path,
        data_slice=data_slice,
        coords=common_args['coords'],
        title=title,
        xlabel=common_args['plot_labels']['xlabel'],
        ylabel=common_args['plot_labels']['ylabel'],
        cbar_label=common_args['cbar_label'],
        vmin=common_args['vmin'], vmax=common_args['vmax'], vcenter=common_args['vcenter'],
        base_size=common_args['base_size'], min_size=common_args['min_size'],
        cmap=common_args['cmap'],
    )
    return frame_path

def generate_video_from_npz(
    npz_path: Path,
    output_path: Path,
    channel: str,
    time_start: Optional[int] = None,
    time_end: Optional[int] = None,
    fps: int = 15,
    channel_alias: Optional[str] = None,
    unit_label: str = "",
    base_size: float = 8.0,
    min_size: float = 3.0,
    vmins_override: Optional[Dict[str, float]] = None,
    vmaxs_override: Optional[Dict[str, float]] = None,
    vcenters: Optional[Dict[str, float]] = None,
    num_workers: int = 1,
    cmap: str = "viridis",
):
    """
    Generates a 2D video from a preprocessed .npz file.
    """
    if not npz_path.exists():
        logging.error(f"NPZ file not found at {npz_path}. Aborting.")
        return

    logging.info(f"Loading data from {npz_path}...")
    with np.load(npz_path) as data:
        timeseries_data_full = data['timeseries']
        all_labels = list(data['labels'])
        
        # Handle coordinates: load if they exist, otherwise create integer ranges
        if 'x_coords' in data and 'z_coords' in data:
            coords = {'x': data['x_coords'], 'z': data['z_coords']}
        else:
            logging.warning("Coordinate arrays ('x_coords', 'z_coords') not found in NPZ. Using integer indices.")
            coords = {
                'x': np.arange(timeseries_data_full.shape[1]),
                'z': np.arange(timeseries_data_full.shape[2]),
            }

    # --- Apply time slicing if specified ---
    time_offset = 0
    if time_start is not None or time_end is not None:
        start = time_start if time_start is not None else 0
        end = time_end if time_end is not None else timeseries_data_full.shape[0]
        time_offset = start
        logging.info(f"Slicing timeseries from index {start} to {end}.")
        timeseries_data = timeseries_data_full[start:end]
    else:
        timeseries_data = timeseries_data_full

    try:
        channel_idx = all_labels.index(channel)
    except ValueError:
        logging.error(f"Channel '{channel}' not found in NPZ labels: {all_labels}. Aborting.")
        return

    # --- Determine Color Scale ---
    vmin = vmins_override.get(channel) if vmins_override else None
    vmax = vmaxs_override.get(channel) if vmaxs_override else None
    if vmin is None or vmax is None:
        logging.info("Calculating color scale for the selected time range...")
        channel_data = timeseries_data[..., channel_idx]
        auto_vmin, auto_vmax = np.min(channel_data), np.max(channel_data)
        if vmin is None: vmin = auto_vmin
        if vmax is None: vmax = auto_vmax
        logging.info(f"Final color scale for '{channel}' set to: [{vmin:.3f}, {vmax:.3f}]")
    
    vcenter = vcenters.get(channel) if vcenters else None
    
    display_name = channel_alias if channel_alias else channel
    cbar_label = f"{display_name}" + (f" [{unit_label}]" if unit_label else "")
    plot_labels = {'xlabel': 'X Coordinate', 'ylabel': 'Z Coordinate'}

    # --- Generate and Compile Frames ---
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_dir = Path(temp_dir)
        num_frames = timeseries_data.shape[0]
        
        logging.info(f"Generating {num_frames} frames in parallel using {num_workers} workers...")
        common_args = {
            'coords': coords, 'channel_idx': channel_idx, 'frame_dir': frame_dir,
            'display_name': display_name, 'plot_labels': plot_labels,
            'cbar_label': cbar_label, 'vmin': vmin, 'vmax': vmax, 'vcenter': vcenter,
            'base_size': base_size, 'min_size': min_size, 'time_offset': time_offset,
            'cmap': cmap,
        }
        
        worker_func = partial(_generate_frame_worker_from_npz, common_args=common_args)
        
        frame_paths = []
        with multiprocessing.Pool(processes=num_workers, initializer=_init_worker_from_npz, initargs=(timeseries_data,)) as pool:
            with tqdm(total=num_frames, desc="Generating frames") as pbar:
                for frame_path in pool.imap_unordered(worker_func, range(num_frames)):
                    frame_paths.append(frame_path)
                    pbar.update(1)

        frame_paths.sort()

        logging.info(f"Assembling video at {output_path} with {fps} FPS...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
            for frame_path in tqdm(frame_paths, desc="Writing video"):
                image = imageio.imread(frame_path)
                writer.append_data(image)

    logging.info("Video generation complete.")

# --- NEW FUNCTIONS FOR MULTI-SAMPLE VIDEO ---

def _init_worker_multi_sample(gt_data_shared, pred_data_list_shared):
    """Initializer for multi-sample worker processes."""
    worker_data['gt_data'] = gt_data_shared
    worker_data['pred_data_list'] = pred_data_list_shared

def _create_multi_sample_frame(
    frame_path: Path,
    gt_slice: np.ndarray,
    pred_slices: List[np.ndarray],
    coords: Dict[str, np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    vmin: float,
    vmax: float,
    vcenter: Optional[float],
    base_size: float,
    min_size: float,
    cmap: str = "viridis",
    video_grid_cols: int = 3,
):
    """
    Plots a multi-panel frame (GT + Preds) in a single grid.
    GT is the first panel.
    """
    x_coords = coords['x']
    z_coords = coords['z']
    
    num_samples = len(pred_slices)
    num_panels = 1 + num_samples # GT + N Samples
    
    # --- NEW: Grid Calculation (GT is just another panel) ---
    ncols = video_grid_cols
    nrows = int(np.ceil(num_panels / ncols))
    
    dpi = 150
    macro_block_size = 16
    
    # --- Figure Sizing ---
    x_range = x_coords.max() - x_coords.min() if len(x_coords) > 1 else 1
    z_range = z_coords.max() - z_coords.min() if len(z_coords) > 1 else 1
    
    # Calculate aspect ratio for a single plot
    aspect_ratio = 1.0
    if x_range >= z_range:
        panel_width_in = base_size
        aspect_ratio = z_range / x_range if x_range > 0 else 1
        panel_height_in = max(min_size, base_size * aspect_ratio)
    else:
        panel_height_in = base_size
        aspect_ratio = x_range / z_range if z_range > 0 else 1
        panel_width_in = max(min_size, base_size * aspect_ratio)

    # Total figure size
    fig_width_in = panel_width_in * ncols
    fig_height_in = panel_height_in * nrows
    
    width_px = int(fig_width_in * dpi)
    height_px = int(fig_height_in * dpi)
    
    width_px = (width_px + macro_block_size - 1) // macro_block_size * macro_block_size
    height_px = (height_px + macro_block_size - 1) // macro_block_size * macro_block_size
    
    figsize = (width_px / dpi, height_px / dpi)

    # --- NEW: Use plt.subplots for simple grid ---
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    # --- Plotting ---
    plot_kwargs = {'shading': 'gouraud', 'cmap': cmap}
    if vcenter is not None and vmin is not None and vmax is not None:
        plot_kwargs['norm'] = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax

    all_data_slices = [gt_slice] + pred_slices
    all_titles = ["Ground Truth"] + [f"Sample {i:02d}" for i in range(num_samples)]

    im = None # To hold the mappable for colorbar

    for i in range(num_panels):
        ax = axes_flat[i]
        data_slice = all_data_slices[i]
        
        im = ax.pcolormesh(x_coords, z_coords, data_slice.T, **plot_kwargs)
        ax.set_title(all_titles[i])
        
        # Add labels only to outer plots
        row = i // ncols
        col = i % ncols
        
        if row == nrows - 1: # Last row
            ax.set_xlabel(xlabel)
        
        if col == 0: # First column
            ax.set_ylabel(ylabel)

    # Hide any unused axes
    for i in range(num_panels, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(title, fontsize=16)
    
    # Add a single colorbar
    fig.subplots_adjust(right=0.9, wspace=0.1, hspace=0.2)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    if im: # Only add colorbar if we plotted something
        fig.colorbar(im, cax=cbar_ax, label=cbar_label)
    
    plt.savefig(frame_path, dpi=dpi)
    plt.close(fig)

def _generate_frame_worker_multi_sample(relative_time_index, common_args):
    """Wrapper function for multiprocessing to generate a single multi-sample frame."""
    gt_data = worker_data['gt_data']
    pred_data_list = worker_data['pred_data_list']
    
    time_offset = common_args['time_offset']
    absolute_time_index = relative_time_index + time_offset
    
    frame_path = common_args['frame_dir'] / f"frame_{absolute_time_index:05d}.png"
    display_name = common_args['display_name']
    channel_idx = common_args['channel_idx']
    
    # Get all data slices for this timestep
    gt_slice = gt_data[relative_time_index, :, :, channel_idx]
    pred_slices = [
        pred_data[relative_time_index, :, :, channel_idx]
        for pred_data in pred_data_list
    ]
    
    title = f"'{display_name}' at Time Index {absolute_time_index}"
        
    _create_multi_sample_frame(
        frame_path=frame_path,
        gt_slice=gt_slice,
        pred_slices=pred_slices,
        coords=common_args['coords'],
        title=title,
        xlabel=common_args['plot_labels']['xlabel'],
        ylabel=common_args['plot_labels']['ylabel'],
        cbar_label=common_args['cbar_label'],
        vmin=common_args['vmin'], vmax=common_args['vmax'], vcenter=common_args['vcenter'],
        base_size=common_args['base_size'], min_size=common_args['min_size'],
        cmap=common_args['cmap'],
        video_grid_cols=common_args['video_grid_cols'], # <-- PASS IT HERE
    )
    return frame_path

def generate_multi_sample_video(
    gt_npz_path: Path,
    pred_npz_paths: List[Path],
    output_path: Path,
    channel: str,
    time_start: Optional[int] = None,
    time_end: Optional[int] = None,
    fps: int = 15,
    channel_alias: Optional[str] = None,
    unit_label: str = "",
    base_size: float = 8.0,
    min_size: float = 3.0,
    vmins_override: Optional[Dict[str, float]] = None,
    vmaxs_override: Optional[Dict[str, float]] = None,
    vcenters: Optional[Dict[str, float]] = None,
    num_workers: int = 1,
    cmap: str = "viridis",
    video_grid_cols: int = 3, # <-- NEW ARGUMENT
):
    """
    Generates a 2D video comparing Ground Truth to multiple prediction samples.
    """
    if not gt_npz_path.exists():
        logging.error(f"Ground Truth NPZ file not found at {gt_npz_path}. Aborting.")
        return
    if not pred_npz_paths:
        logging.error("No prediction NPZ files provided. Aborting.")
        return

    logging.info(f"Loading Ground Truth data from {gt_npz_path}...")
    with np.load(gt_npz_path) as data:
        gt_timeseries_full = data['timeseries']
        all_labels = list(data['labels'])
        
        if 'x_coords' in data and 'z_coords' in data:
            coords = {'x': data['x_coords'], 'z': data['z_coords']}
        else:
            logging.warning("Coordinates not found in GT NPZ. Using integer indices.")
            coords = {
                'x': np.arange(gt_timeseries_full.shape[1]),
                'z': np.arange(gt_timeseries_full.shape[2]),
            }

    # --- Load all prediction data ---
    pred_timeseries_list_full = []
    for path in pred_npz_paths:
        if not path.exists():
            logging.warning(f"Prediction file {path} not found. Skipping.")
            continue
        logging.info(f"Loading Prediction data from {path}...")
        with np.load(path) as data:
            pred_timeseries_list_full.append(data['timeseries'])

    if not pred_timeseries_list_full:
        logging.error("No valid prediction data could be loaded. Aborting.")
        return
    
    # --- Apply time slicing ---
    time_offset = 0
    start = time_start if time_start is not None else 0
    end = time_end if time_end is not None else gt_timeseries_full.shape[0]
    time_offset = start
    
    logging.info(f"Slicing all timeseries from index {start} to {end}.")
    gt_timeseries = gt_timeseries_full[start:end]
    pred_timeseries_list = [pred[start:end] for pred in pred_timeseries_list_full]

    try:
        channel_idx = all_labels.index(channel)
    except ValueError:
        logging.error(f"Channel '{channel}' not found in NPZ labels: {all_labels}. Aborting.")
        return

    # --- Determine Color Scale ---
    # IMPORTANT: Color scale is determined ONLY by Ground Truth for consistency
    vmin = vmins_override.get(channel) if vmins_override else None
    vmax = vmaxs_override.get(channel) if vmins_override else None
    if vmin is None or vmax is None:
        logging.info("Calculating color scale from Ground Truth for the selected time range...")
        channel_data = gt_timeseries[..., channel_idx]
        auto_vmin, auto_vmax = np.min(channel_data), np.max(channel_data)
        if vmin is None: vmin = auto_vmin
        if vmax is None: vmax = auto_vmax
        logging.info(f"Final color scale for '{channel}' set to: [{vmin:.3f}, {vmax:.3f}]")
    
    vcenter = vcenters.get(channel) if vcenters else None
    
    display_name = channel_alias if channel_alias else channel
    cbar_label = f"{display_name}" + (f" [{unit_label}]" if unit_label else "")
    plot_labels = {'xlabel': 'X Coordinate', 'ylabel': 'Z Coordinate'}

    # --- Generate and Compile Frames ---
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_dir = Path(temp_dir)
        num_frames = gt_timeseries.shape[0]
        
        logging.info(f"Generating {num_frames} multi-sample frames using {num_workers} workers...")
        common_args = {
            'coords': coords, 'channel_idx': channel_idx, 'frame_dir': frame_dir,
            'display_name': display_name, 'plot_labels': plot_labels,
            'cbar_label': cbar_label, 'vmin': vmin, 'vmax': vmax, 'vcenter': vcenter,
            'base_size': base_size, 'min_size': min_size, 'time_offset': time_offset,
            'cmap': cmap,
            'video_grid_cols': video_grid_cols, # <-- ADD TO COMMON ARGS
        }
        
        worker_func = partial(_generate_frame_worker_multi_sample, common_args=common_args)
        
        frame_paths = []
        init_args = (gt_timeseries, pred_timeseries_list)
        with multiprocessing.Pool(processes=num_workers, initializer=_init_worker_multi_sample, initargs=init_args) as pool:
            with tqdm(total=num_frames, desc="Generating multi-sample frames") as pbar:
                for frame_path in pool.imap_unordered(worker_func, range(num_frames)):
                    frame_paths.append(frame_path)
                    pbar.update(1)

        frame_paths.sort()

        logging.info(f"Assembling multi-sample video at {output_path} with {fps} FPS...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
            for frame_path in tqdm(frame_paths, desc="Writing video"):
                image = imageio.imread(frame_path)
                writer.append_data(image)

    logging.info("Multi-sample video generation complete.")