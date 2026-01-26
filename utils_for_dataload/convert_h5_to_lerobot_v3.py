#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script converts raw H5 data files to LeRobot V3 format.

It will:
- Read raw H5 files containing robot actions and camera images
- Align left/right arm data with configurable sync tolerance
- Encode video frames and save as MP4 files (per episode first, then concatenate)
- Generate parquet data files with actions, states, and camera metadata
- Concatenate files based on size thresholds (100MB for data, 500MB for video)
- Generate episode metadata with video timestamps
- Generate tasks.parquet, stats.json, and info.json

Usage:

Convert H5 files to LeRobot V3 format:
```bash
python utils_for_dataload/convert_h5_to_lerobot_v3.py \\
    --input-folder data/raw \\
    --output-folder data/lerobot/my_dataset
```

With custom file size limits:
```bash
python utils_for_dataload/convert_h5_to_lerobot_v3.py \\
    --input-folder data/raw \\
    --output-folder data/lerobot/my_dataset \\
    --data-file-size-in-mb 100 \\
    --video-file-size-in-mb 500
```

Skip video statistics computation (faster):
```bash
python utils_for_dataload/convert_h5_to_lerobot_v3.py \\
    --input-folder data/raw \\
    --output-folder data/lerobot/my_dataset \\
    --skip-video-stats
```
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import tqdm

from lerobot.datasets.utils import (DEFAULT_CHUNK_SIZE,
                                    DEFAULT_DATA_FILE_SIZE_IN_MB,
                                    DEFAULT_DATA_PATH,
                                    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
                                    DEFAULT_VIDEO_PATH,
                                    get_parquet_file_size_in_mb,
                                    update_chunk_file_indices, write_info,
                                    write_stats, write_tasks)
from lerobot.datasets.video_utils import (encode_video_frames,
                                          get_video_duration_in_s)

# Suppress FFmpeg warnings
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"

CODEBASE_VERSION = "v3.0"


# =============================================================================
# Utility Functions
# =============================================================================

def get_file_size_in_mb(file_path: Path) -> float:
    """Get file size on disk in megabytes."""
    return file_path.stat().st_size / (1024 ** 2)


def concatenate_video_files(video_paths: list[Path], output_path: Path):
    """Concatenate multiple video files into one using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a temporary file list for ffmpeg concat
    list_file = output_path.parent / "concat_list.txt"
    with open(list_file, "w") as f:
        for vp in video_paths:
            f.write(f"file '{vp.absolute()}'\n")

    # Use ffmpeg to concatenate
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    list_file.unlink()


# =============================================================================
# H5 Data Processing Functions
# =============================================================================

def decode_image_stream(group, limit: int | None = None) -> tuple[list, np.ndarray]:
    """Decode JPEG images from H5 dataset.

    Args:
        group: H5 group containing 'raw', 'len', 'ts' datasets
        limit: Maximum number of images to decode

    Returns:
        Tuple of (decoded_images, timestamps)
    """
    raw_bytes = group['raw'][:]
    lengths = group['len'][:]
    timestamps = group['ts'][:]

    # Fix timestamp format (nanoseconds to seconds)
    if len(timestamps) > 0 and np.median(timestamps) > 1e14:
        timestamps = timestamps.astype(np.float64) / 1e9
    else:
        timestamps = timestamps.astype(np.float64)

    if limit is not None:
        limit_imgs = min(len(lengths), limit)
        lengths = lengths[:limit_imgs]
        timestamps = timestamps[:limit_imgs]

    decoded_imgs = []
    current_offset = 0

    for length in lengths:
        byte_data = raw_bytes[current_offset:current_offset + length]
        current_offset += length
        img = cv2.imdecode(np.frombuffer(byte_data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            decoded_imgs.append(img)

    return decoded_imgs, timestamps


def fix_timestamp(ts: np.ndarray) -> np.ndarray:
    """Convert timestamps from nanoseconds to seconds if needed."""
    if len(ts) > 0 and np.median(ts) > 1e14:
        return ts.astype(np.float64) / 1e9
    return ts.astype(np.float64)


def align_left_right_data(
    q_l: np.ndarray,
    ts_l: np.ndarray,
    q_r: np.ndarray,
    ts_r: np.ndarray,
    max_sync_diff: float = 0.005
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Align left and right arm data based on timestamps.

    Args:
        q_l: Left arm joint positions
        ts_l: Left arm timestamps
        q_r: Right arm joint positions
        ts_r: Right arm timestamps
        max_sync_diff: Maximum allowed sync difference in seconds

    Returns:
        Tuple of (aligned_q_l, aligned_q_r, aligned_ts, drop_count)
    """
    ts_l = fix_timestamp(ts_l)
    ts_r = fix_timestamp(ts_r)

    # Find nearest right timestamps for each left timestamp
    idxs = np.searchsorted(ts_r, ts_l)
    idxs = np.clip(idxs, 0, len(ts_r) - 1)
    prev_idxs = np.maximum(idxs - 1, 0)

    dist_curr = np.abs(ts_r[idxs] - ts_l)
    dist_prev = np.abs(ts_r[prev_idxs] - ts_l)

    use_prev = dist_prev < dist_curr
    final_indices = np.where(use_prev, prev_idxs, idxs)

    q_r_aligned = q_r[final_indices]

    # Filter by sync error
    sync_errors = np.abs(ts_r[final_indices] - ts_l)
    valid_mask = sync_errors <= max_sync_diff

    drop_count = len(valid_mask) - np.sum(valid_mask)

    return q_l[valid_mask], q_r_aligned[valid_mask], ts_l[valid_mask], drop_count


# =============================================================================
# Statistics Computation Functions
# =============================================================================

def compute_stats(data: np.ndarray) -> dict:
    """Compute statistics for array data."""
    return {
        "mean": np.mean(data, axis=0).astype(np.float32).tolist(),
        "std": np.std(data, axis=0).astype(np.float32).tolist(),
        "min": np.min(data, axis=0).astype(np.float32).tolist(),
        "max": np.max(data, axis=0).astype(np.float32).tolist(),
    }


def compute_scalar_stats(data: list) -> dict:
    """Compute statistics for scalar data."""
    data = np.array(data)
    return {
        "mean": [float(np.mean(data))],
        "std": [float(np.std(data))],
        "min": [float(np.min(data))],
        "max": [float(np.max(data))],
        "count": [int(len(data))],
    }


def compute_video_stats(
    video_dir: Path,
    video_key: str,
    sample_ratio: float = 0.1,
    max_samples: int = 1000
) -> dict | None:
    """Compute pixel statistics from video files.

    Args:
        video_dir: Directory containing video files
        video_key: Video key name (e.g., 'observation.images.cam_left')
        sample_ratio: Ratio of frames to sample
        max_samples: Maximum number of frames to sample

    Returns:
        Statistics dict with shape (3, 1, 1) or None if no videos found
    """
    video_subdir = video_dir / video_key
    video_files = sorted(list(video_subdir.rglob("*.mp4")))
    if not video_files:
        logging.warning(f"No video files found: {video_subdir}")
        return None

    channel_values = [[], [], []]  # R, G, B channels
    total_frames = 0

    # Try imageio first (better AV1 support)
    try:
        import imageio.v3 as iio
        use_imageio = True
    except ImportError:
        use_imageio = False
        logging.info("imageio not installed, using OpenCV for video decoding")

    for video_path in tqdm.tqdm(video_files, desc=f"Computing stats for {video_key}"):
        if use_imageio:
            try:
                frames = iio.imread(str(video_path), plugin="pyav")
                frame_count = len(frames)

                sample_count = min(int(frame_count * sample_ratio), max_samples // len(video_files))
                sample_count = max(1, sample_count)
                sample_indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)

                for idx in sample_indices:
                    frame_rgb = frames[idx]
                    frame_normalized = frame_rgb.astype(np.float32) / 255.0

                    for c in range(3):
                        channel_values[c].extend(frame_normalized[:, :, c].flatten())

                    total_frames += 1

            except Exception as e:
                logging.warning(f"imageio failed, trying OpenCV: {e}")
                use_imageio = False

        if not use_imageio:
            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            sample_count = min(int(frame_count * sample_ratio), max_samples // len(video_files))
            sample_count = max(1, sample_count)
            sample_indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)

            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_normalized = frame_rgb.astype(np.float32) / 255.0

                for c in range(3):
                    channel_values[c].extend(frame_normalized[:, :, c].flatten())

                total_frames += 1

            cap.release()

    if total_frames == 0:
        logging.warning(f"Could not read video frames: {video_subdir}")
        return None

    return {
        "mean": [[[float(np.mean(channel_values[c]))]] for c in range(3)],
        "std": [[[float(np.std(channel_values[c]))]] for c in range(3)],
        "min": [[[float(np.min(channel_values[c]))]] for c in range(3)],
        "max": [[[float(np.max(channel_values[c]))]] for c in range(3)],
        "count": [total_frames],
    }


# =============================================================================
# Phase 1: Convert H5 to per-episode files
# =============================================================================

def convert_h5_to_episodes(
    input_path: Path,
    temp_path: Path,
    max_sync_diff: float,
    debug_limit: int,
    fps: int = 30
) -> tuple[list[dict], int]:
    """Convert H5 files to per-episode parquet and video files.

    Args:
        input_path: Path to folder containing H5 files
        temp_path: Path to temporary output folder
        max_sync_diff: Maximum sync error in seconds
        debug_limit: Limit frames per episode for debugging
        fps: Video frame rate

    Returns:
        Tuple of (episodes_info_list, action_dim)
    """
    h5_files = sorted(list(input_path.glob("raw_*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in: {input_path}")

    logging.info(f"Found {len(h5_files)} H5 files to convert")

    # Create temp directories
    temp_data_dir = temp_path / "data"
    temp_video_dir = temp_path / "videos"
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    episodes_info = []
    global_frame_index = 0
    action_dim = None

    for ep_idx, h5_file in enumerate(tqdm.tqdm(h5_files, desc="Converting H5 files to episodes")):
        with h5py.File(h5_file, 'r') as f:
            q_l = f['act/qpos/left/raw'][:]
            ts_l = f['act/qpos/left/ts'][:]
            q_r = f['act/qpos/right/raw'][:]
            ts_r = f['act/qpos/right/ts'][:]

            q_l, q_r_aligned, ts, drop_count = align_left_right_data(
                q_l, ts_l, q_r, ts_r, max_sync_diff
            )

            if drop_count > 0:
                logging.info(f"Episode {ep_idx}: dropped {drop_count} frames due to sync error")

            img_limit = None if debug_limit == 0 else debug_limit
            if img_limit is not None:
                q_l, q_r_aligned, ts = q_l[:img_limit], q_r_aligned[:img_limit], ts[:img_limit]

            actions = np.concatenate([q_l, q_r_aligned], axis=1)
            ep_len = len(actions)

            if ep_len == 0:
                continue

            if action_dim is None:
                action_dim = actions.shape[1]

            # Save per-episode parquet
            data_dict = {
                "index": np.arange(global_frame_index, global_frame_index + ep_len),
                "episode_index": np.full(ep_len, ep_idx, dtype=np.int64),
                "frame_index": np.arange(global_frame_index, global_frame_index + ep_len),
                "timestamp": ts.astype(np.float32),
                "action": list(actions),
                "observation.state": list(actions.copy()),
                "task_index": np.zeros(ep_len, dtype=np.int64),
            }

            # Process images and compute latency
            for k in ["left", "right"]:
                imgs, img_ts = decode_image_stream(f[f'obs/image/{k}'], limit=img_limit)
                cam_key = f"cam_{k}"
                cam_name = f"observation.images.{cam_key}"

                # Compute frame indices and latencies
                indices = np.searchsorted(img_ts, ts, side='right') - 1
                indices = np.clip(indices, 0, len(img_ts) - 1)
                latencies = ts - img_ts[indices]

                data_dict[f"{cam_name}_frame_index"] = indices.astype(np.int64)
                data_dict[f"{cam_name}_latency"] = latencies.astype(np.float32)

                # Save per-episode video
                tmp_img_dir = temp_path / f"temp_imgs_{ep_idx}_{k}"
                tmp_img_dir.mkdir(parents=True, exist_ok=True)

                for idx, img in enumerate(imgs):
                    cv2.imwrite(str(tmp_img_dir / f"frame-{idx:06d}.png"), img)

                vid_dir = temp_video_dir / cam_name
                vid_dir.mkdir(parents=True, exist_ok=True)
                video_path = vid_dir / f"episode_{ep_idx:06d}.mp4"

                encode_video_frames(
                    imgs_dir=tmp_img_dir,
                    video_path=video_path,
                    fps=fps,
                    overwrite=True
                )
                shutil.rmtree(tmp_img_dir)

            # Save parquet
            pq_path = temp_data_dir / f"episode_{ep_idx:06d}.parquet"
            pd.DataFrame(data_dict).to_parquet(pq_path)

            # Record episode info
            ep_info = {
                "episode_index": ep_idx,
                "length": ep_len,
                "task_index": 0,
                "dataset_from_index": global_frame_index,
                "dataset_to_index": global_frame_index + ep_len,
                "parquet_path": pq_path,
                "video_paths": {
                    "observation.images.cam_left": temp_video_dir / "observation.images.cam_left" / f"episode_{ep_idx:06d}.mp4",
                    "observation.images.cam_right": temp_video_dir / "observation.images.cam_right" / f"episode_{ep_idx:06d}.mp4",
                },
            }
            episodes_info.append(ep_info)
            global_frame_index += ep_len

    logging.info(f"Converted {len(episodes_info)} episodes, total frames: {global_frame_index}")
    return episodes_info, action_dim


# =============================================================================
# Phase 2: Concatenate data files by size
# =============================================================================

def concat_data_files(
    paths_to_cat: list[Path],
    output_path: Path,
    chunk_idx: int,
    file_idx: int
) -> Path:
    """Concatenate multiple parquet files into one."""
    dataframes = [pd.read_parquet(f) for f in paths_to_cat]
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    out_file = output_path / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    concatenated_df.to_parquet(out_file)

    return out_file


def convert_data(
    episodes_info: list[dict],
    output_path: Path,
    data_file_size_in_mb: int
) -> list[dict]:
    """Concatenate per-episode parquet files based on size threshold.

    Returns:
        Updated episodes_info with data chunk/file indices
    """
    logging.info(f"Concatenating data files (threshold: {data_file_size_in_mb} MB)")

    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    paths_to_cat = []

    for ep_info in tqdm.tqdm(episodes_info, desc="Converting data files"):
        ep_path = ep_info["parquet_path"]
        ep_size_in_mb = get_parquet_file_size_in_mb(ep_path)

        # Update episode metadata
        ep_info["data/chunk_index"] = chunk_idx
        ep_info["data/file_index"] = file_idx

        size_in_mb += ep_size_in_mb

        if size_in_mb < data_file_size_in_mb:
            paths_to_cat.append(ep_path)
            continue

        # Size exceeded, save accumulated files first (without current episode)
        if paths_to_cat:
            concat_data_files(paths_to_cat, output_path, chunk_idx, file_idx)

        # Reset for the next file
        size_in_mb = ep_size_in_mb
        paths_to_cat = [ep_path]

        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)

        # Update current episode's chunk/file index
        ep_info["data/chunk_index"] = chunk_idx
        ep_info["data/file_index"] = file_idx

    # Write remaining data if any
    if paths_to_cat:
        concat_data_files(paths_to_cat, output_path, chunk_idx, file_idx)

    return episodes_info


# =============================================================================
# Phase 3: Concatenate video files by size
# =============================================================================

def convert_videos_of_camera(
    episodes_info: list[dict],
    output_path: Path,
    video_key: str,
    video_file_size_in_mb: int
) -> list[dict]:
    """Concatenate per-episode video files for one camera based on size threshold."""
    logging.info(f"Concatenating videos for {video_key} (threshold: {video_file_size_in_mb} MB)")

    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    duration_in_s = 0.0
    paths_to_cat = []

    for ep_idx, ep_info in enumerate(tqdm.tqdm(episodes_info, desc=f"Converting {video_key}")):
        ep_path = ep_info["video_paths"][video_key]
        ep_size_in_mb = get_file_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)

        # Check if adding this episode would exceed the limit
        if size_in_mb + ep_size_in_mb >= video_file_size_in_mb and len(paths_to_cat) > 0:
            # Size limit would be exceeded, save current accumulation WITHOUT this episode
            out_path = output_path / DEFAULT_VIDEO_PATH.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            concatenate_video_files(paths_to_cat, out_path)

            # Update episodes metadata for the file we just saved
            for i, _ in enumerate(paths_to_cat):
                past_ep_idx = ep_idx - len(paths_to_cat) + i
                episodes_info[past_ep_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
                episodes_info[past_ep_idx][f"videos/{video_key}/file_index"] = file_idx

            # Move to next file and start fresh
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_in_mb = 0
            duration_in_s = 0.0
            paths_to_cat = []

        # Add current episode metadata (will be updated when file is saved)
        ep_info[f"videos/{video_key}/chunk_index"] = chunk_idx
        ep_info[f"videos/{video_key}/file_index"] = file_idx
        ep_info[f"videos/{video_key}/from_timestamp"] = duration_in_s
        ep_info[f"videos/{video_key}/to_timestamp"] = duration_in_s + ep_duration_in_s

        # Add current episode to accumulation
        paths_to_cat.append(ep_path)
        size_in_mb += ep_size_in_mb
        duration_in_s += ep_duration_in_s

    # Write remaining videos if any
    if paths_to_cat:
        out_path = output_path / DEFAULT_VIDEO_PATH.format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )
        concatenate_video_files(paths_to_cat, out_path)

        # Update episodes metadata for the final file
        for i, _ in enumerate(paths_to_cat):
            past_ep_idx = len(episodes_info) - len(paths_to_cat) + i
            episodes_info[past_ep_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
            episodes_info[past_ep_idx][f"videos/{video_key}/file_index"] = file_idx

    return episodes_info


def convert_videos(
    episodes_info: list[dict],
    output_path: Path,
    video_file_size_in_mb: int
) -> list[dict]:
    """Concatenate video files for all cameras."""
    video_keys = ["observation.images.cam_left", "observation.images.cam_right"]

    for video_key in video_keys:
        episodes_info = convert_videos_of_camera(
            episodes_info, output_path, video_key, video_file_size_in_mb
        )

    return episodes_info


# =============================================================================
# Phase 4: Generate episode metadata
# =============================================================================

def convert_episodes_metadata(
    episodes_info: list[dict],
    output_path: Path
):
    """Generate episode metadata parquet files."""
    logging.info("Generating episode metadata...")

    # Clean up temporary fields
    clean_episodes = []
    for ep in episodes_info:
        clean_ep = {k: v for k, v in ep.items() if k not in ["parquet_path", "video_paths"]}
        clean_episodes.append(clean_ep)

    # Write to meta/episodes/chunk-000/file-000.parquet
    df = pd.DataFrame(clean_episodes)
    out_path = output_path / "meta/episodes/chunk-000/file-000.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)


# =============================================================================
# Phase 5: Generate stats
# =============================================================================

def generate_stats(
    output_path: Path,
    total_frames: int,
    compute_video_stats_flag: bool = True
) -> dict:
    """Generate dataset statistics."""
    logging.info("Computing dataset statistics...")

    pq_files = sorted(list((output_path / "data").rglob("*.parquet")))

    all_actions = []
    all_states = []
    all_indices = []
    all_episode_indices = []
    all_frame_indices = []
    all_timestamps = []
    all_cam_left_frame_indices = []
    all_cam_left_latencies = []
    all_cam_right_frame_indices = []
    all_cam_right_latencies = []

    for pq in tqdm.tqdm(pq_files, desc="Reading parquet files for stats"):
        df = pd.read_parquet(pq)
        all_actions.append(np.stack(df["action"].values))
        all_states.append(np.stack(df["observation.state"].values))
        all_indices.extend(df["index"].values)
        all_episode_indices.extend(df["episode_index"].values)
        all_frame_indices.extend(df["frame_index"].values)
        all_timestamps.extend(df["timestamp"].values)

        if "observation.images.cam_left_frame_index" in df.columns:
            all_cam_left_frame_indices.extend(df["observation.images.cam_left_frame_index"].values)
        if "observation.images.cam_left_latency" in df.columns:
            latency_data = df["observation.images.cam_left_latency"].values
            if latency_data.dtype == object:
                latency_data = np.array([
                    x if isinstance(x, (int, float)) else x[0] if hasattr(x, '__len__') else 0
                    for x in latency_data
                ])
            all_cam_left_latencies.extend(latency_data)
        if "observation.images.cam_right_frame_index" in df.columns:
            all_cam_right_frame_indices.extend(df["observation.images.cam_right_frame_index"].values)
        if "observation.images.cam_right_latency" in df.columns:
            latency_data = df["observation.images.cam_right_latency"].values
            if latency_data.dtype == object:
                latency_data = np.array([
                    x if isinstance(x, (int, float)) else x[0] if hasattr(x, '__len__') else 0
                    for x in latency_data
                ])
            all_cam_right_latencies.extend(latency_data)

    all_actions = np.concatenate(all_actions, axis=0)
    all_states = np.concatenate(all_states, axis=0)

    stats = {
        "action": compute_stats(all_actions),
        "observation.state": compute_stats(all_states),
        "index": compute_scalar_stats(all_indices),
        "episode_index": compute_scalar_stats(all_episode_indices),
        "frame_index": compute_scalar_stats(all_frame_indices),
        "timestamp": compute_scalar_stats(all_timestamps),
    }

    stats["action"]["count"] = [total_frames]
    stats["observation.state"]["count"] = [total_frames]

    if all_cam_left_frame_indices:
        stats["observation.images.cam_left_frame_index"] = compute_scalar_stats(all_cam_left_frame_indices)
    if all_cam_left_latencies:
        stats["observation.images.cam_left_latency"] = compute_scalar_stats(all_cam_left_latencies)
    if all_cam_right_frame_indices:
        stats["observation.images.cam_right_frame_index"] = compute_scalar_stats(all_cam_right_frame_indices)
    if all_cam_right_latencies:
        stats["observation.images.cam_right_latency"] = compute_scalar_stats(all_cam_right_latencies)

    # Compute video statistics
    if compute_video_stats_flag:
        video_dir = output_path / "videos"
        if video_dir.exists():
            logging.info("Computing video statistics...")

            cam_left_stats = compute_video_stats(video_dir, "observation.images.cam_left")
            if cam_left_stats:
                stats["observation.images.cam_left"] = cam_left_stats
                logging.info("  ✓ cam_left stats complete")

            cam_right_stats = compute_video_stats(video_dir, "observation.images.cam_right")
            if cam_right_stats:
                stats["observation.images.cam_right"] = cam_right_stats
                logging.info("  ✓ cam_right stats complete")

    return stats


# =============================================================================
# Phase 6: Generate info.json
# =============================================================================

def generate_info(
    total_episodes: int,
    total_frames: int,
    action_dim: int,
    fps: int = 30,
    robot_type: str = "aloha",
    video_shape: tuple = (3, 480, 640)
) -> dict:
    """Generate dataset info.json content."""
    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "fps": fps,
        "splits": {
            "train": f"0:{total_episodes}"
        },
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH,
        "features": {
            "index": {"dtype": "int64", "shape": (1,), "names": None, "fps": fps},
            "episode_index": {"dtype": "int64", "shape": (1,), "names": None, "fps": fps},
            "frame_index": {"dtype": "int64", "shape": (1,), "names": None, "fps": fps},
            "timestamp": {"dtype": "float32", "shape": (1,), "names": None, "fps": fps},
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": None,
                "fps": fps
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": None,
                "fps": fps
            },
            "task_index": {"dtype": "int64", "shape": (1,), "names": None, "fps": fps},
            "observation.images.cam_left": {
                "dtype": "video",
                "shape": video_shape,
                "names": ["channel", "height", "width"],
                "video_info": {"video.fps": fps, "video.codec": "av1", "video.pix_fmt": "yuv420p"}
            },
            "observation.images.cam_left_frame_index": {"dtype": "int64", "shape": (1,), "names": None, "fps": fps},
            "observation.images.cam_left_latency": {"dtype": "float32", "shape": (1,), "names": None, "fps": fps},
            "observation.images.cam_right": {
                "dtype": "video",
                "shape": video_shape,
                "names": ["channel", "height", "width"],
                "video_info": {"video.fps": fps, "video.codec": "av1", "video.pix_fmt": "yuv420p"}
            },
            "observation.images.cam_right_frame_index": {"dtype": "int64", "shape": (1,), "names": None, "fps": fps},
            "observation.images.cam_right_latency": {"dtype": "float32", "shape": (1,), "names": None, "fps": fps}
        }
    }


def generate_tasks(output_path: Path):
    """Generate tasks.parquet file."""
    task_df = pd.DataFrame({"task_index": [0]}, index=["Default task."])
    write_tasks(task_df, output_path)


# =============================================================================
# Main Conversion Function
# =============================================================================

def convert_h5_to_lerobot_v3(
    input_folder: str | Path,
    output_folder: str | Path,
    data_file_size_in_mb: int | None = None,
    video_file_size_in_mb: int | None = None,
    max_sync_diff: float = 0.005,
    debug_limit: int = 0,
    skip_video_stats: bool = False,
    fps: int = 30,
    robot_type: str = "aloha"
):
    """Convert raw H5 data to LeRobot V3 format.

    Args:
        input_folder: Path to folder containing raw_*.h5 files
        output_folder: Path to output LeRobot dataset
        data_file_size_in_mb: Max parquet file size in MB (default: 100)
        video_file_size_in_mb: Max video file size in MB (default: 500)
        max_sync_diff: Max sync error in seconds for left/right arm alignment
        debug_limit: Limit frames per episode for debugging (0 = no limit)
        skip_video_stats: Skip computing video pixel statistics
        fps: Video frame rate
        robot_type: Robot type for info.json
    """
    if data_file_size_in_mb is None:
        data_file_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_file_size_in_mb is None:
        video_file_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    temp_path = output_path.parent / f"{output_path.name}_temp"

    # Clean up existing directories
    if output_path.exists():
        logging.warning(f"Output folder exists, removing: {output_path}")
        shutil.rmtree(output_path)
    if temp_path.exists():
        shutil.rmtree(temp_path)

    output_path.mkdir(parents=True, exist_ok=True)
    temp_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Converting H5 files from {input_path} to {output_path}")
    logging.info(f"  Data file size limit: {data_file_size_in_mb} MB")
    logging.info(f"  Video file size limit: {video_file_size_in_mb} MB")
    logging.info(f"  Max sync diff: {max_sync_diff} s")

    # Phase 1: Convert H5 to per-episode files
    logging.info("=" * 60)
    logging.info("Phase 1: Converting H5 files to per-episode files...")
    episodes_info, action_dim = convert_h5_to_episodes(
        input_path=input_path,
        temp_path=temp_path,
        max_sync_diff=max_sync_diff,
        debug_limit=debug_limit,
        fps=fps
    )

    total_episodes = len(episodes_info)
    total_frames = episodes_info[-1]["dataset_to_index"] if episodes_info else 0

    # Phase 2: Concatenate data files by size
    logging.info("=" * 60)
    logging.info("Phase 2: Concatenating data files by size...")
    episodes_info = convert_data(episodes_info, output_path, data_file_size_in_mb)

    # Phase 3: Concatenate video files by size
    logging.info("=" * 60)
    logging.info("Phase 3: Concatenating video files by size...")
    episodes_info = convert_videos(episodes_info, output_path, video_file_size_in_mb)

    # Phase 4: Generate episode metadata
    logging.info("=" * 60)
    logging.info("Phase 4: Generating episode metadata...")
    convert_episodes_metadata(episodes_info, output_path)

    # Phase 5: Generate tasks.parquet
    logging.info("Generating tasks.parquet...")
    generate_tasks(output_path)

    # Phase 6: Generate stats.json
    logging.info("=" * 60)
    logging.info("Phase 5: Generating stats.json...")
    stats = generate_stats(output_path, total_frames, not skip_video_stats)
    write_stats(stats, output_path)

    # Phase 7: Generate info.json
    logging.info("Generating info.json...")
    info = generate_info(
        total_episodes=total_episodes,
        total_frames=total_frames,
        action_dim=action_dim,
        fps=fps,
        robot_type=robot_type
    )
    write_info(info, output_path)

    # Clean up temp directory
    logging.info("Cleaning up temporary files...")
    shutil.rmtree(temp_path)

    logging.info("=" * 60)
    logging.info("Conversion complete!")
    logging.info(f"  Total episodes: {total_episodes}")
    logging.info(f"  Total frames: {total_frames}")
    logging.info(f"  Action dimension: {action_dim}")
    logging.info(f"  Output path: {output_path}")
    logging.info("=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Convert raw H5 data files to LeRobot V3 format."
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default="data/raw",
        help="Path to folder containing raw_*.h5 files."
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="data/lerobot/my_dataset",
        help="Path to output LeRobot dataset."
    )
    parser.add_argument(
        "--data-file-size-in-mb",
        type=int,
        default=None,
        help=f"Max parquet file size in MB (default: {DEFAULT_DATA_FILE_SIZE_IN_MB})."
    )
    parser.add_argument(
        "--video-file-size-in-mb",
        type=int,
        default=None,
        help=f"Max video file size in MB (default: {DEFAULT_VIDEO_FILE_SIZE_IN_MB})."
    )
    parser.add_argument(
        "--max-sync-diff",
        type=float,
        default=0.005,
        help="Max sync error in seconds for left/right arm alignment (default: 0.005)."
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=0,
        help="Limit frames per episode for debugging (0 = no limit)."
    )
    parser.add_argument(
        "--skip-video-stats",
        action="store_true",
        help="Skip computing video pixel statistics (faster conversion)."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate (default: 30)."
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="aloha",
        help="Robot type for info.json (default: aloha)."
    )

    args = parser.parse_args()

    convert_h5_to_lerobot_v3(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        data_file_size_in_mb=args.data_file_size_in_mb,
        video_file_size_in_mb=args.video_file_size_in_mb,
        max_sync_diff=args.max_sync_diff,
        debug_limit=args.debug_limit,
        skip_video_stats=args.skip_video_stats,
        fps=args.fps,
        robot_type=args.robot_type
    )
