# -*- coding: utf-8 -*-
# mhd_surrogate_core/plotting/xz/comparison.py

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

worker_data = {}

def _init_worker_for_comparison(gt_data, pred_data, diff_data):
    """Initializer for worker processes with all three datasets."""
    worker_data['gt'] = gt_data
    worker_data['pred'] = pred_data
    worker_data['diff'] = diff_data

def _create_comparison_frame_from_npz(
    frame_path: Path,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    diff_slice: np.ndarray,
    coords: Dict[str, np.ndarray],
    title: str,
    cmap: str,
    cmap_diff: str,
    vmin: Optional[float],
    vmax: Optional[float],
    vcenter: Optional[float],
    vmin_diff: Optional[float],
    vmax_diff: Optional[float],
    vcenter_diff: Optional[float],
):
    """Plots a 3-panel comparison frame (stacked vertically) and saves it to a file."""
    x_coords = coords.get('x', np.arange(gt_slice.shape[0]))
    z_coords = coords.get('z', np.arange(gt_slice.shape[1]))

    # KEY CHANGE: Create a 3x1 subplot grid for vertical stacking.
    fig, axes = plt.subplots(3, 1, figsize=(8, 18), constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    # --- Setup for Main Plots (Ground Truth & Prediction) ---
    plot_kwargs = {'shading': 'gouraud', 'cmap': cmap}
    if vcenter is not None and vmin is not None and vmax is not None:
        plot_kwargs['norm'] = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        plot_kwargs['vmin'] = vmin
        plot_kwargs['vmax'] = vmax
        
    # --- Setup for Difference Plot ---
    plot_kwargs_diff = {'shading': 'gouraud', 'cmap': cmap_diff}
    if vcenter_diff is not None and vmin_diff is not None and vmax_diff is not None:
        plot_kwargs_diff['norm'] = TwoSlopeNorm(vmin=vmin_diff, vcenter=vcenter_diff, vmax=vmax_diff)
    else:
        plot_kwargs_diff['vmin'] = vmin_diff
        plot_kwargs_diff['vmax'] = vmax_diff

    # --- Plotting ---
    # Ground Truth (Top Plot)
    im1 = axes[0].pcolormesh(x_coords, z_coords, gt_slice.T, **plot_kwargs)
    axes[0].set_title("Ground Truth")
    axes[0].set_ylabel("Z Coordinate")
    axes[0].set_xticklabels([])
    fig.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.1)

    # Prediction (Middle Plot)
    im2 = axes[1].pcolormesh(x_coords, z_coords, pred_slice.T, **plot_kwargs)
    axes[1].set_title("Prediction")
    axes[1].set_ylabel("Z Coordinate")
    axes[1].set_xticklabels([])
    fig.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.1)

    # Difference (Bottom Plot)
    im3 = axes[2].pcolormesh(x_coords, z_coords, diff_slice.T, **plot_kwargs_diff)
    axes[2].set_title("Difference (Error)")
    axes[2].set_xlabel("X Coordinate")
    axes[2].set_ylabel("Z Coordinate")
    fig.colorbar(im3, ax=axes[2], orientation='horizontal', pad=0.1)

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
    display_name = common_args['display_name']
    channel_idx = common_args['channel_idx']
    
    gt_slice = gt_data[relative_time_index, ..., channel_idx]
    pred_slice = pred_data[relative_time_index, ..., channel_idx]
    diff_slice = diff_data[relative_time_index, ..., channel_idx]
    
    title = f"Comparison of '{display_name}' at Time Index {absolute_time_index}"
        
    _create_comparison_frame_from_npz(
        frame_path=frame_path,
        gt_slice=gt_slice,
        pred_slice=pred_slice,
        diff_slice=diff_slice,
        coords=common_args['coords'],
        title=title,
        cmap=common_args['cmap'],
        cmap_diff=common_args['cmap_diff'],
        vmin=common_args['vmin'], vmax=common_args['vmax'], vcenter=common_args['vcenter'],
        vmin_diff=common_args['vmin_diff'], vmax_diff=common_args['vmax_diff'], vcenter_diff=common_args['vcenter_diff'],
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
    """Generates a 3-panel comparison video from ground truth, prediction, and difference npz files."""
    logging.info("Loading data for comparison video...")
    with np.load(gt_npz_path) as data:
        gt_full = data['timeseries']
        if gt_full.ndim == 5: gt_full = np.squeeze(gt_full, axis=2)
        all_labels = list(data['labels'])
        coords = {
            'x': data.get('x_coords', np.arange(gt_full.shape[1])),
            'z': data.get('z_coords', np.arange(gt_full.shape[2])),
        }
    with np.load(pred_npz_path) as data:
        pred_full = data['timeseries']
    with np.load(diff_npz_path) as data:
        diff_full = data['timeseries']

    time_offset = time_start if time_start is not None else 0
    gt_data = gt_full[time_start:time_end]
    pred_data = pred_full[time_start:time_end]
    diff_data = diff_full[time_start:time_end]
    
    try:
        channel_idx = all_labels.index(channel)
    except ValueError:
        logging.error(f"Channel '{channel}' not found in labels: {all_labels}. Aborting.")
        return

    # --- Determine Color Scales ---
    vmin = vmins_override.get(channel) if vmins_override else None
    vmax = vmaxs_override.get(channel) if vmaxs_override else None
    if vmin is None or vmax is None:
        auto_vmin = min(np.min(gt_data[..., channel_idx]), np.min(pred_data[..., channel_idx]))
        auto_vmax = max(np.max(gt_data[..., channel_idx]), np.max(pred_data[..., channel_idx]))
        if vmin is None: vmin = auto_vmin
        if vmax is None: vmax = auto_vmax
    
    vmin_diff = vmins_diff_override.get(channel) if vmins_diff_override else None
    vmax_diff = vmaxs_diff_override.get(channel) if vmaxs_diff_override else None
    if vmin_diff is None or vmax_diff is None:
        abs_max = np.max(np.abs(diff_data[..., channel_idx]))
        if vmin_diff is None: vmin_diff = -abs_max
        if vmax_diff is None: vmax_diff = abs_max

    vcenter = vcenters.get(channel) if vcenters else None
    vcenter_diff = vcenters_diff.get(channel) if vcenters_diff else 0.0
    
    display_name = channel_alias if channel_alias else channel

    with tempfile.TemporaryDirectory() as temp_dir:
        frame_dir = Path(temp_dir)
        num_frames = gt_data.shape[0]
        
        logging.info(f"Generating {num_frames} frames using {num_workers} workers...")
        common_args = {
            'coords': coords, 'channel_idx': channel_idx, 'frame_dir': frame_dir,
            'display_name': display_name, 'time_offset': time_offset,
            'cmap': cmap, 'cmap_diff': cmap_diff,
            'vmin': vmin, 'vmax': vmax, 'vcenter': vcenter,
            'vmin_diff': vmin_diff, 'vmax_diff': vmax_diff, 'vcenter_diff': vcenter_diff,
        }
        
        worker_func = partial(_generate_comparison_frame_worker, common_args=common_args)
        
        initargs = (gt_data, pred_data, diff_data)
        with multiprocessing.Pool(processes=num_workers, initializer=_init_worker_for_comparison, initargs=initargs) as pool:
            frame_paths = list(tqdm(pool.imap_unordered(worker_func, range(num_frames)), total=num_frames, desc="Generating frames"))

        frame_paths.sort()

        logging.info(f"Assembling video at {output_path} with {fps} FPS...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
            for frame_path in tqdm(frame_paths, desc="Writing video"):
                writer.append_data(imageio.imread(frame_path))

    logging.info("Comparison video generation complete.")

