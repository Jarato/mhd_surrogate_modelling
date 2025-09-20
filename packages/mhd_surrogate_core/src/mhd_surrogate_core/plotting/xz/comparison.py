# -*- coding: utf-8 -*-
# mhd_surrogate_core/plotting/xz/comparison.py

"""
Functions for generating side-by-side comparison videos of 2D data
(e.g., Ground Truth vs. Prediction vs. Difference).
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm
import imageio.v2 as imageio
import multiprocessing
from functools import partial

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global variable to hold shared data for worker processes
worker_data = {}

def _init_worker_comparison(data_tuple):
    """Initializer for each worker process. Puts the shared data into a global variable."""
    worker_data['gt'], worker_data['pred'], worker_data['diff'] = data_tuple

def _create_comparison_frame_from_npz(
    frame_path: Path,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    diff_slice: np.ndarray,
    coords: Dict[str, np.ndarray],
    title: str,
    vmin: float,
    vmax: float,
    vcenter: Optional[float],
    vmin_diff: float,
    vmax_diff: float,
    vcenter_diff: Optional[float],
    cmap: str,
    cmap_diff: str,
):
    """Plots a 3-panel comparison frame (gt, pred, diff) to a file."""
    x_coords = coords['x']
    z_coords = coords['z']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    # Plot 1: Ground Truth
    im1 = axes[0].pcolormesh(x_coords, z_coords, gt_slice.T, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("X Coordinate")
    axes[0].set_ylabel("Z Coordinate")

    # Plot 2: Prediction
    im2 = axes[1].pcolormesh(x_coords, z_coords, pred_slice.T, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("X Coordinate")
    axes[1].set_yticklabels([]) # Hide y-axis labels

    # Plot 3: Difference
    norm_diff = None
    if vcenter_diff is not None and vmin_diff is not None and vmax_diff is not None:
        norm_diff = TwoSlopeNorm(vmin=vmin_diff, vcenter=vcenter_diff, vmax=vmax_diff)
        
    im3 = axes[2].pcolormesh(x_coords, z_coords, diff_slice.T, shading='gouraud', cmap=cmap_diff, norm=norm_diff, vmin=vmin_diff, vmax=vmax_diff)
    axes[2].set_title("Difference (Error)")
    axes[2].set_xlabel("X Coordinate")
    axes[2].set_yticklabels([])

    # Add colorbars
    fig.colorbar(im1, ax=axes[:2], location='bottom', fraction=0.05, pad=0.1, label="Value")
    fig.colorbar(im3, ax=axes[2], location='bottom', fraction=0.05, pad=0.1, label="Error")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(frame_path, dpi=150)
    plt.close(fig)

def _generate_comparison_frame_worker(relative_time_index, common_args):
    """Wrapper function for multiprocessing to generate a single comparison frame."""
    gt_data = worker_data['gt']
    pred_data = worker_data['pred']
    diff_data = worker_data['diff']
    
    time_offset = common_args['time_offset']
    absolute_time_index = relative_time_index + time_offset
    
    frame_path = common_args['frame_dir'] / f"frame_{absolute_time_index:05d}.png"
    channel_idx = common_args['channel_idx']
    
    gt_slice = gt_data[relative_time_index, :, :, channel_idx]
    pred_slice = pred_data[relative_time_index, :, :, channel_idx]
    diff_slice = diff_data[relative_time_index, :, :, channel_idx]
    
    title = f"'{common_args['display_name']}' at Time Index {absolute_time_index}"
        
    _create_comparison_frame_from_npz(
        frame_path=frame_path,
        gt_slice=gt_slice, pred_slice=pred_slice, diff_slice=diff_slice,
        coords=common_args['coords'], title=title,
        vmin=common_args['vmin'], vmax=common_args['vmax'], vcenter=common_args['vcenter'],
        vmin_diff=common_args['vmin_diff'], vmax_diff=common_args['vmax_diff'], vcenter_diff=common_args['vcenter_diff'],
        cmap=common_args['cmap'], cmap_diff=common_args['cmap_diff'],
    )
    return frame_path

def generate_comparison_video_from_npz(
    gt_npz_path: Path,
    pred_npz_path: Path,
    diff_npz_path: Path,
    output_path: Path,
    channel: str,
    time_start: Optional[int] = None,
    time_end: Optional[int] = None,
    fps: int = 15,
    channel_alias: Optional[str] = None,
    vmins_override: Optional[Dict[str, float]] = None,
    vmaxs_override: Optional[Dict[str, float]] = None,
    vcenters: Optional[Dict[str, float]] = None,
    vmins_diff_override: Optional[Dict[str, float]] = None,
    vmaxs_diff_override: Optional[Dict[str, float]] = None,
    vcenters_diff: Optional[Dict[str, float]] = None,
    num_workers: int = 1,
    cmap: str = "viridis",
    cmap_diff: str = "bwr",
):
    """
    Generates a 3-panel comparison video from ground truth, prediction, and difference .npz files.
    """
    logging.info("Loading data for comparison video...")
    with np.load(gt_npz_path) as data:
        gt_full = data['timeseries']
        coords = {'x': data['x_coords'], 'z': data['z_coords'], 'labels': list(data['labels'])}
    with np.load(pred_npz_path) as data:
        pred_full = data['timeseries']
    with np.load(diff_npz_path) as data:
        diff_full = data['timeseries']

    # --- Data Slicing and Validation ---
    time_offset = time_start if time_start is not None else 0
    gt_data = gt_full[time_start:time_end]
    pred_data = pred_full[time_start:time_end]
    diff_data = diff_full[time_start:time_end]

    if not (gt_data.shape == pred_data.shape == diff_data.shape):
        logging.error("Timeseries shapes are inconsistent after slicing. Aborting.")
        return

    try:
        channel_idx = coords['labels'].index(channel)
    except ValueError:
        logging.error(f"Channel '{channel}' not found. Aborting.")
        return

    # --- Determine Color Scales ---
    display_name = channel_alias if channel_alias else channel
    
    # Main plots (GT and Pred)
    vmin = vmins_override.get(channel)
    vmax = vmaxs_override.get(channel)
    if vmin is None or vmax is None:
        gt_chan, pred_chan = gt_data[..., channel_idx], pred_data[..., channel_idx]
        auto_vmin, auto_vmax = min(gt_chan.min(), pred_chan.min()), max(gt_chan.max(), pred_chan.max())
        if vmin is None: vmin = auto_vmin
        if vmax is None: vmax = auto_vmax
        logging.info(f"Auto color scale for '{display_name}': [{vmin:.3f}, {vmax:.3f}]")

    # Difference plot
    vmin_diff, vmax_diff = vmins_diff_override.get(channel), vmaxs_diff_override.get(channel)
    if vmin_diff is None or vmax_diff is None:
        diff_chan = diff_data[..., channel_idx]
        abs_max = np.abs(diff_chan).max()
        auto_vmin_diff, auto_vmax_diff = -abs_max, abs_max
        if vmin_diff is None: vmin_diff = auto_vmin_diff
        if vmax_diff is None: vmax_diff = auto_vmax_diff
        logging.info(f"Auto color scale for difference: [{vmin_diff:.3f}, {vmax_diff:.3f}]")
    
    # --- Generate Frames ---
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_dir = Path(temp_dir)
        num_frames = gt_data.shape[0]
        logging.info(f"Generating {num_frames} frames using {num_workers} workers...")
        
        common_args = {
            'coords': coords, 'channel_idx': channel_idx, 'frame_dir': frame_dir,
            'display_name': display_name, 'time_offset': time_offset,
            'vmin': vmin, 'vmax': vmax, 'vcenter': (vcenters or {}).get(channel),
            'vmin_diff': vmin_diff, 'vmax_diff': vmax_diff, 'vcenter_diff': (vcenters_diff or {}).get(channel, 0.0),
            'cmap': cmap, 'cmap_diff': cmap_diff
        }
        
        worker_func = partial(_generate_comparison_frame_worker, common_args=common_args)
        
        frame_paths = []
        init_args = (gt_data, pred_data, diff_data)
        with multiprocessing.Pool(processes=num_workers, initializer=_init_worker_comparison, initargs=(init_args,)) as pool:
            with tqdm(total=num_frames, desc="Generating frames") as pbar:
                for frame_path in pool.imap_unordered(worker_func, range(num_frames)):
                    frame_paths.append(frame_path)
                    pbar.update(1)

        frame_paths.sort()
        
        logging.info(f"Assembling video at {output_path}...")
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame_path in tqdm(frame_paths, desc="Writing video"):
                writer.append_data(imageio.imread(frame_path))

    logging.info("Comparison video generation complete.")
