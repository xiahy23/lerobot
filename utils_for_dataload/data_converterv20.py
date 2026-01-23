import glob
import os
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import tqdm

# 保持导入不变
from lerobot.datasets.video_utils import encode_video_frames


def decode_image_stream(group, limit=None):
    """
    (保持不变) 从 HDF5 的压缩字节流中解码图像
    """
    raw_bytes = group['raw'][:]
    lengths = group['len'][:]
    timestamps = group['ts'][:]

    # 时间戳修复
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
        if img is None:
            continue
        decoded_imgs.append(img)

    return decoded_imgs, timestamps

def process_single_episode(src_h5_path, output_dir, episode_idx, debug_limit=0, max_sync_diff=0.005):
    """
    [Updated] 增加 'max_sync_diff' 参数，自动丢弃不同步的帧
    max_sync_diff: 允许的最大时间差 (秒)，默认 0.005 (5ms)
    """
    src_path = Path(src_h5_path)
    out_path = Path(output_dir)
    ep_id_str = f"episode_{episode_idx:06d}"

    print(f"🔄 [Episode {episode_idx}] 正在转换: {src_path.name} -> {ep_id_str}")

    with h5py.File(src_path, 'r') as f:
        # ==========================================
        # 1. 读取并对齐动作
        # ==========================================
        qpos_left = f['act/qpos/left/raw'][:]
        ts_left = f['act/qpos/left/ts'][:]

        qpos_right = f['act/qpos/right/raw'][:]
        ts_right = f['act/qpos/right/ts'][:]

        # 统一时间单位
        def fix_ts(ts):
            if len(ts) > 0 and np.median(ts) > 1e14:
                return ts.astype(np.float64) / 1e9
            return ts.astype(np.float64)

        ts_left = fix_ts(ts_left)
        ts_right = fix_ts(ts_right)

        # --- 对齐逻辑 ---
        idxs = np.searchsorted(ts_right, ts_left)
        idxs = np.clip(idxs, 0, len(ts_right) - 1)
        prev_idxs = np.maximum(idxs - 1, 0)

        dist_curr = np.abs(ts_right[idxs] - ts_left)
        dist_prev = np.abs(ts_right[prev_idxs] - ts_left)

        use_prev = dist_prev < dist_curr
        final_indices = np.where(use_prev, prev_idxs, idxs)

        # 获取对齐后的右臂数据
        qpos_right_aligned = qpos_right[final_indices]

        # --- [关键新增] 严格同步过滤 (Strict Sync Filter) ---
        # 计算每一帧的同步误差
        sync_errors = np.abs(ts_left - ts_right[final_indices])

        # 生成掩码：保留误差小于阈值 (5ms) 的帧
        valid_mask = sync_errors <= max_sync_diff

        drop_count = len(valid_mask) - np.sum(valid_mask)
        drop_rate = drop_count / len(valid_mask)

        if drop_count > 0:
            print(f"   ✂️ [Strict Sync] 丢弃了 {drop_count} 帧 ({drop_rate:.1%})，因为左右臂误差 > {max_sync_diff*1000}ms")

            # 如果丢弃太多 (比如超过 20%)，发出红色警告
            if drop_rate > 0.2:
                print(f"      ⚠️ 警告: 丢弃率过高！请检查硬件同步或录制脚本！")

            # 应用过滤
            qpos_left = qpos_left[valid_mask]
            qpos_right_aligned = qpos_right_aligned[valid_mask]
            ts_left = ts_left[valid_mask]
            # 注意：act_ts 也要同步过滤，否则后面视频对齐会错位
        else:
            print(f"   ✨ 所有帧同步良好 (误差均 <= {max_sync_diff*1000}ms)")

        # 赋值
        act_ts = ts_left
        actions = np.concatenate([qpos_left, qpos_right_aligned], axis=1)
        states = actions.copy()

        if debug_limit > 0:
            actions = actions[:debug_limit]
            states = states[:debug_limit]
            act_ts = act_ts[:debug_limit]

        # ==========================================
        # 2. 处理视频流 (逻辑不变，自动适配过滤后的 act_ts)
        # ==========================================
        # 注意：这里生成的 MP4 依然是包含所有原始帧的 (30FPS)
        # 但是 Parquet 里的 frame_index 会跳过那些被丢弃的动作帧

        camera_map = {
            "observation.images.cam_left": f['obs/image/left'],
            "observation.images.cam_right": f['obs/image/right']
        }

        video_info_map = {}

        for key, h5_group in camera_map.items():
            img_limit = None
            if debug_limit > 0:
                img_limit = int(debug_limit / 3) + 20

            imgs_bgr, imgs_ts = decode_image_stream(h5_group, limit=img_limit)

            tmp_frame_dir = out_path / f"temp_frames_{key}_{ep_id_str}"
            if tmp_frame_dir.exists(): shutil.rmtree(tmp_frame_dir)
            tmp_frame_dir.mkdir(parents=True)

            for idx, img in enumerate(imgs_bgr):
                save_p = tmp_frame_dir / f"frame-{idx:06d}.png"
                cv2.imwrite(str(save_p), img)

            video_dir = out_path / f"videos/{key}"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"{ep_id_str}.mp4"

            encode_video_frames(
                imgs_dir=tmp_frame_dir,
                video_path=video_path,
                fps=30,
                overwrite=True
            )
            shutil.rmtree(tmp_frame_dir)

            video_info_map[key] = {"timestamps": imgs_ts, "count": len(imgs_bgr)}

        # ==========================================
        # 3. 对齐视频 (自动适配)
        # ==========================================
        # 因为上面的 act_ts 已经是过滤过的了，
        # 所以这里的 searchsorted 会自动跳过那些被丢弃的时间点

        data_dict = {
            "index": np.arange(len(actions)),
            "episode_index": np.full(len(actions), episode_idx, dtype=int),
            "frame_index": np.arange(len(actions)),
            "timestamp": act_ts,
            "action": list(actions),
            "observation.state": list(states),
        }

        for cam_key, info in video_info_map.items():
            img_ts_seq = info['timestamps']
            img_count = info['count']

            indices = np.searchsorted(img_ts_seq, act_ts, side='right') - 1
            indices = np.maximum(indices, 0)
            indices = np.minimum(indices, img_count - 1)

            real_img_ts = img_ts_seq[indices]
            latencies = act_ts - real_img_ts

            data_dict[f"{cam_key}_frame_index"] = indices.astype(np.int64)
            data_dict[f"{cam_key}_latency"] = latencies.astype(np.float32)[:, None].tolist()

        # 保存
        chunk_dir = out_path / "data/chunk-000"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_dir / f"{ep_id_str}.parquet"

        df = pd.DataFrame(data_dict)
        df.to_parquet(parquet_path)

def convert_all_episodes(input_folder, output_folder, debug_limit=0):
    """
    主循环：扫描文件夹并转换所有 h5 文件
    """
    input_path = Path(input_folder)
    out_path = Path(output_folder)

    # 扫描 raw_*.h5 文件并排序
    # 假设文件名是 raw_000000.h5, raw_000001.h5 ...
    h5_files = sorted(list(input_path.glob("raw_*.h5")))

    if not h5_files:
        print(f"❌ 在 {input_folder} 没有找到 raw_*.h5 文件")
        return

    print(f"🔎 发现 {len(h5_files)} 个数据文件，准备转换...")

    # 创建基础目录结构
    (out_path / "videos").mkdir(parents=True, exist_ok=True)
    (out_path / "data/chunk-000").mkdir(parents=True, exist_ok=True)

    # 遍历处理
    # enumerate(h5_files) 会自动给每个文件分配 0, 1, 2... 的索引
    # 这个索引将作为 episode_index
    for idx, h5_file in enumerate(tqdm.tqdm(h5_files, desc="Converting Episodes")):
        try:
            process_single_episode(h5_file, out_path, episode_idx=idx, debug_limit=debug_limit)
        except Exception as e:
            print(f"\n❌ [Error] 转换 {h5_file.name} 失败: {e}")
            # 可以选择 continue 跳过，或者 raise 停止
            # raise e

    print(f"\n🎉 所有转换完成！输出目录: {output_folder}")

if __name__ == "__main__":
    # 输入包含 raw_*.h5 的文件夹路径
    INPUT_DIR = "data/raw"

    # 输出 LeRobot 数据集的路径
    OUTPUT_DIR = "data/test2"

    convert_all_episodes(INPUT_DIR, OUTPUT_DIR, debug_limit=0)