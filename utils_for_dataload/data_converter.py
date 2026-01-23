import glob
import shutil
import warnings
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import tqdm

from lerobot.datasets.video_utils import encode_video_frames


def decode_image_stream(group, limit=None):
    raw_bytes = group['raw'][:]
    lengths = group['len'][:]
    timestamps = group['ts'][:]

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
        byte_data = raw_bytes[current_offset : current_offset + length]
        current_offset += length
        img = cv2.imdecode(np.frombuffer(byte_data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            decoded_imgs.append(img)

    return decoded_imgs, timestamps

def convert_to_lerobot_v3_strict(input_folder, output_folder, frames_per_chunk=10000, debug_limit=0, max_sync_diff=0.005):
    input_path = Path(input_folder)
    out_path = Path(output_folder)

    h5_files = sorted(list(input_path.glob("raw_*.h5")))
    if not h5_files:
        print(f"❌ 未找到文件: {input_folder}")
        return

    chunk_idx = 0
    current_chunk_frames = 0

    chunk_actions = []
    chunk_timestamps = []
    chunk_episodes = []
    chunk_frame_indices = []
    chunk_images = {"cam_left": [], "cam_right": []}
    chunk_img_timestamps = {"cam_left": [], "cam_right": []}

    # [修改] 现在 meta_episodes 是 Chunk 级别的累加器
    current_chunk_meta_episodes = []

    global_frame_index = 0

    def save_chunk(c_idx):
        nonlocal current_chunk_frames
        if current_chunk_frames == 0: return

        chunk_str = f"chunk-{c_idx:03d}"
        print(f"\n💾 正在保存 {chunk_str} (包含 {current_chunk_frames} 个有效帧)...")

        # 1. 保存视频 -> videos/observation.images.cam_left/chunk-000/file-000.mp4
        for cam_key, imgs in chunk_images.items():
            tmp_dir = out_path / f"temp_{cam_key}"
            if tmp_dir.exists(): shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)

            for idx, img in enumerate(tqdm.tqdm(imgs, desc=f"Saving {cam_key} images", leave=False)):
                cv2.imwrite(str(tmp_dir / f"frame-{idx:06d}.png"), img)

            # [修改] 符合 V3 规范的视频路径
            vid_dir = out_path / f"videos/observation.images.{cam_key}/{chunk_str}"
            vid_dir.mkdir(parents=True, exist_ok=True)
            encode_video_frames(
                imgs_dir=tmp_dir,
                video_path=vid_dir / "file-000.mp4",
                fps=30,
                overwrite=True
            )
            shutil.rmtree(tmp_dir)

        # 2. 对齐并保存 Parquet 数据 -> data/chunk-000/file-000.parquet
        actions_np = np.concatenate(chunk_actions, axis=0)
        ts_np = np.concatenate(chunk_timestamps, axis=0)
        episodes_np = np.concatenate(chunk_episodes, axis=0)
        frame_idx_np = np.concatenate(chunk_frame_indices, axis=0)

        data_dict = {
            "index": frame_idx_np,
            "episode_index": episodes_np,
            "frame_index": frame_idx_np,
            "timestamp": ts_np,
            "action": list(actions_np),
            "observation.state": list(actions_np.copy()),
            "task_index": np.zeros(len(ts_np), dtype=np.int64),
        }
        warnings.warn("task_index is currently all marked as 0")

        for cam_key in ["cam_left", "cam_right"]:
            img_ts_seq = np.concatenate(chunk_img_timestamps[cam_key], axis=0)
            indices = np.searchsorted(img_ts_seq, ts_np, side='right') - 1
            indices = np.clip(indices, 0, len(img_ts_seq) - 1)
            latencies = ts_np - img_ts_seq[indices]

            cam_name = f"observation.images.{cam_key}"

            data_dict[f"{cam_name}_frame_index"] = indices.astype(np.int64)
            data_dict[f"{cam_name}_latency"] = latencies.astype(np.float32)

        # [修改] 保存为 file-000.parquet
        pq_dir = out_path / f"data/{chunk_str}"
        pq_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(data_dict).to_parquet(pq_dir / "file-000.parquet")

        # 3. [新增] 保存此 Chunk 的 Episodes Meta -> meta/episodes/chunk-000/file-000.parquet
        meta_episodes_df = pd.DataFrame(current_chunk_meta_episodes)
        # 为每一行添加 episode_chunk 字段，这是 V3 读取路径的关键
        meta_episodes_df['episode_chunk'] = c_idx

        ep_meta_dir = out_path / f"meta/episodes/{chunk_str}"
        ep_meta_dir.mkdir(parents=True, exist_ok=True)
        meta_episodes_df.to_parquet(ep_meta_dir / "file-000.parquet")

        # 4. 清空所有累加器
        chunk_actions.clear()
        chunk_timestamps.clear()
        chunk_episodes.clear()
        chunk_frame_indices.clear()
        chunk_images["cam_left"].clear()
        chunk_images["cam_right"].clear()
        chunk_img_timestamps["cam_left"].clear()
        chunk_img_timestamps["cam_right"].clear()
        current_chunk_meta_episodes.clear() # 清空 Meta 累加器
        current_chunk_frames = 0

    # ==========================
    # 主循环：遍历 H5
    # ==========================
    for ep_idx, h5_file in enumerate(tqdm.tqdm(h5_files, desc="Processing files")):
        with h5py.File(h5_file, 'r') as f:
            q_l = f['act/qpos/left/raw'][:]
            ts_l = f['act/qpos/left/ts'][:]
            q_r = f['act/qpos/right/raw'][:]
            ts_r = f['act/qpos/right/ts'][:]

            def fix_ts(ts):
                if len(ts) > 0 and np.median(ts) > 1e14:
                    return ts.astype(np.float64) / 1e9
                return ts.astype(np.float64)

            ts_l = fix_ts(ts_l)
            ts_r = fix_ts(ts_r)

            idxs = np.searchsorted(ts_r, ts_l)
            idxs = np.clip(idxs, 0, len(ts_r) - 1)
            prev_idxs = np.maximum(idxs - 1, 0)

            dist_curr = np.abs(ts_r[idxs] - ts_l)
            dist_prev = np.abs(ts_r[prev_idxs] - ts_l)

            use_prev = dist_prev < dist_curr
            final_indices = np.where(use_prev, prev_idxs, idxs)

            q_r_aligned = q_r[final_indices]

            sync_errors = np.abs(ts_r[final_indices] - ts_l)
            valid_mask = sync_errors <= max_sync_diff

            drop_count = len(valid_mask) - np.sum(valid_mask)
            if drop_count > 0:
                print(f"\n   ✂️ [Episode {ep_idx}] 丢弃 {drop_count} 帧同步异常数据。")

            q_l = q_l[valid_mask]
            q_r_aligned = q_r_aligned[valid_mask]
            ts_l = ts_l[valid_mask]

            img_limit = None if debug_limit == 0 else debug_limit
            if img_limit is not None:
                q_l, q_r_aligned, ts_l = q_l[:img_limit], q_r_aligned[:img_limit], ts_l[:img_limit]

            actions = np.concatenate([q_l, q_r_aligned], axis=1)
            ts = ts_l
            ep_len = len(actions)

            if ep_len == 0: continue

            ep_meta = {
                "episode_index": ep_idx,
                "length": ep_len,
                "task_index": 0,#current_task_idx,
                "timestamp_start": ts[0] if len(ts)>0 else 0.0,
                "dataset_from_index": global_frame_index,
                "dataset_to_index": global_frame_index + ep_len,
            }

            # 为每个相机记录它在当前 MP4 里的起始和结束时间
            # 因为视频是 30FPS，所以 时间 = 帧数 / 30.0
            for k in ["left", "right"]:
                cam_name = f"observation.images.cam_{k}"
                from_ts = current_chunk_frames / 30.0
                to_ts = (current_chunk_frames + ep_len) / 30.0

                ep_meta[f"videos/{cam_name}/chunk_index"] = chunk_idx
                ep_meta[f"videos/{cam_name}/file_index"] = 0  # 固定为 0，因为我们只生成 file-000.mp4
                ep_meta[f"videos/{cam_name}/from_timestamp"] = float(from_ts)
                ep_meta[f"videos/{cam_name}/to_timestamp"] = float(to_ts)

            current_chunk_meta_episodes.append(ep_meta)

            chunk_actions.append(actions)
            chunk_timestamps.append(ts)
            chunk_episodes.append(np.full(ep_len, ep_idx, dtype=int))
            chunk_frame_indices.append(np.arange(global_frame_index, global_frame_index + ep_len))

            for k in ["left", "right"]:
                imgs, img_ts = decode_image_stream(f[f'obs/image/{k}'], limit=img_limit)
                cam_key = f"cam_{k}"
                chunk_images[cam_key].extend(imgs)
                chunk_img_timestamps[cam_key].append(img_ts)

            current_chunk_frames += ep_len
            global_frame_index += ep_len

            if current_chunk_frames >= frames_per_chunk:
                save_chunk(chunk_idx)
                chunk_idx += 1

    save_chunk(chunk_idx)
    print(f"总帧数: {global_frame_index}")

if __name__ == "__main__":
    INPUT_FOLDER = "data/raw"
    OUTPUT_FOLDER = "data/test3"
    convert_to_lerobot_v3_strict(INPUT_FOLDER, OUTPUT_FOLDER, frames_per_chunk=20000, max_sync_diff=0.005)