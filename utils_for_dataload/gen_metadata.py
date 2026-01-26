import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# 抑制 FFmpeg 的警告信息
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"


def compute_stats(data):
    """计算数值型数据的统计信息"""
    return {
        "mean": np.mean(data, axis=0).astype(np.float32).tolist(),
        "std": np.std(data, axis=0).astype(np.float32).tolist(),
        "min": np.min(data, axis=0).astype(np.float32).tolist(),
        "max": np.max(data, axis=0).astype(np.float32).tolist(),
    }


def compute_scalar_stats(data):
    """计算标量数据的统计信息（如 index, frame_index 等）"""
    return {
        "mean": [float(np.mean(data))],
        "std": [float(np.std(data))],
        "min": [float(np.min(data))],
        "max": [float(np.max(data))],
        "count": [int(len(data))],
    }


def compute_video_stats(video_dir, video_key, sample_ratio=0.1, max_samples=1000):
    """
    计算视频数据的统计信息

    Args:
        video_dir: 视频目录路径
        video_key: 视频键名（如 'observation.images.cam_left'）
        sample_ratio: 采样比例
        max_samples: 最大采样帧数

    Returns:
        dict: 包含 mean, std, min, max, count 的统计信息，shape 为 (3, 1, 1)
    """
    video_subdir = video_dir / video_key
    video_files = sorted(list(video_subdir.rglob("*.mp4")))
    if not video_files:
        print(f"⚠️ 未找到视频文件: {video_subdir}")
        return None

    # 收集采样的像素值
    channel_values = [[], [], []]  # R, G, B channels
    total_frames = 0

    # 尝试使用 imageio 作为主要后端（对 AV1 支持更好）
    try:
        import imageio.v3 as iio
        use_imageio = True
    except ImportError:
        use_imageio = False
        print("  ℹ️ imageio 未安装，使用 OpenCV 解码视频")

    for video_path in video_files:
        if use_imageio:
            try:
                # 使用 imageio 读取视频
                frames = iio.imread(str(video_path), plugin="pyav")
                frame_count = len(frames)

                # 计算采样帧索引
                sample_count = min(int(frame_count * sample_ratio), max_samples // len(video_files))
                sample_count = max(1, sample_count)
                sample_indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)

                for idx in sample_indices:
                    frame_rgb = frames[idx]
                    # 归一化到 [0, 1]
                    frame_normalized = frame_rgb.astype(np.float32) / 255.0

                    # 收集每个通道的值
                    for c in range(3):
                        channel_values[c].extend(frame_normalized[:, :, c].flatten())

                    total_frames += 1

            except Exception as e:
                print(f"  ⚠️ imageio 读取失败，尝试 OpenCV: {e}")
                use_imageio = False

        if not use_imageio:
            # 使用 OpenCV 作为备选
            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 计算采样帧索引
            sample_count = min(int(frame_count * sample_ratio), max_samples // len(video_files))
            sample_count = max(1, sample_count)
            sample_indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)

            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # OpenCV 读取的是 BGR，转换为 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 归一化到 [0, 1]
                frame_normalized = frame_rgb.astype(np.float32) / 255.0

                # 收集每个通道的值
                for c in range(3):
                    channel_values[c].extend(frame_normalized[:, :, c].flatten())

                total_frames += 1

            cap.release()

    if total_frames == 0:
        print(f"⚠️ 无法读取视频帧: {video_subdir}")
        return None

    # 计算统计信息，shape 为 (3, 1, 1)
    stats = {
        "mean": [[[float(np.mean(channel_values[c]))]] for c in range(3)],
        "std": [[[float(np.std(channel_values[c]))]] for c in range(3)],
        "min": [[[float(np.min(channel_values[c]))]] for c in range(3)],
        "max": [[[float(np.max(channel_values[c]))]] for c in range(3)],
        "count": [total_frames],
    }

    return stats


def generate_lerobot_v3_metadata(dataset_dir, compute_video_stats_flag=True):
    ds_path = Path(dataset_dir)
    meta_dir = ds_path / "meta"

    # 1. 收集全局统计数据
    pq_files = sorted(list((ds_path / "data").rglob("*.parquet")))
    if not pq_files:
        print(f"❌ 未找到任何数据文件: {ds_path / 'data'}")
        return

    all_actions = []
    all_states = []
    total_frames = 0

    for pq in pq_files:
        df = pd.read_parquet(pq)
        all_actions.append(np.stack(df["action"].values))
        all_states.append(np.stack(df["observation.state"].values))
        total_frames += len(df)

    all_actions = np.concatenate(all_actions, axis=0)
    all_states = np.concatenate(all_states, axis=0)

    # ==========================
    # 1. 生成 stats.json
    # ==========================

    # 收集所有数值列的数据用于统计
    all_indices = []
    all_episode_indices = []
    all_frame_indices = []
    all_timestamps = []
    all_cam_left_frame_indices = []
    all_cam_left_latencies = []
    all_cam_right_frame_indices = []
    all_cam_right_latencies = []

    for pq in pq_files:
        df = pd.read_parquet(pq)
        all_indices.extend(df["index"].values)
        all_episode_indices.extend(df["episode_index"].values)
        all_frame_indices.extend(df["frame_index"].values)
        all_timestamps.extend(df["timestamp"].values)

        # 相机相关列可能不存在
        if "observation.images.cam_left_frame_index" in df.columns:
            all_cam_left_frame_indices.extend(df["observation.images.cam_left_frame_index"].values)
        if "observation.images.cam_left_latency" in df.columns:
            latency_data = df["observation.images.cam_left_latency"].values
            # 处理 object 类型的数据
            if latency_data.dtype == object:
                latency_data = np.array([x if isinstance(x, (int, float)) else x[0] if hasattr(x, '__len__') else 0 for x in latency_data])
            all_cam_left_latencies.extend(latency_data)
        if "observation.images.cam_right_frame_index" in df.columns:
            all_cam_right_frame_indices.extend(df["observation.images.cam_right_frame_index"].values)
        if "observation.images.cam_right_latency" in df.columns:
            latency_data = df["observation.images.cam_right_latency"].values
            if latency_data.dtype == object:
                latency_data = np.array([x if isinstance(x, (int, float)) else x[0] if hasattr(x, '__len__') else 0 for x in latency_data])
            all_cam_right_latencies.extend(latency_data)

    stats = {
        # 基础数值统计
        "action": compute_stats(all_actions),
        "observation.state": compute_stats(all_states),

        # 索引和时间统计
        "index": compute_scalar_stats(all_indices),
        "episode_index": compute_scalar_stats(all_episode_indices),
        "frame_index": compute_scalar_stats(all_frame_indices),
        "timestamp": compute_scalar_stats(all_timestamps),
    }

    # 添加 count 到 action 和 observation.state
    stats["action"]["count"] = [total_frames]
    stats["observation.state"]["count"] = [total_frames]

    # 相机帧索引和延迟的统计
    if all_cam_left_frame_indices:
        stats["observation.images.cam_left_frame_index"] = compute_scalar_stats(all_cam_left_frame_indices)
    if all_cam_left_latencies:
        stats["observation.images.cam_left_latency"] = compute_scalar_stats(all_cam_left_latencies)
    if all_cam_right_frame_indices:
        stats["observation.images.cam_right_frame_index"] = compute_scalar_stats(all_cam_right_frame_indices)
    if all_cam_right_latencies:
        stats["observation.images.cam_right_latency"] = compute_scalar_stats(all_cam_right_latencies)

    # 计算视频统计信息
    if compute_video_stats_flag:
        video_dir = ds_path / "videos"
        if video_dir.exists():
            print("📹 正在计算视频统计信息...")

            # 左相机
            cam_left_stats = compute_video_stats(video_dir, "observation.images.cam_left")
            if cam_left_stats:
                stats["observation.images.cam_left"] = cam_left_stats
                print(f"  ✅ cam_left 统计完成")

            # 右相机
            cam_right_stats = compute_video_stats(video_dir, "observation.images.cam_right")
            if cam_right_stats:
                stats["observation.images.cam_right"] = cam_right_stats
                print(f"  ✅ cam_right 统计完成")
        else:
            print(f"⚠️ 视频目录不存在: {video_dir}，跳过视频统计")

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # ==========================
    # 2. 生成 tasks.parquet
    # ==========================
    # 官方格式：task 作为 index，task_index 作为唯一列
    task_df = pd.DataFrame({
        "task_index": [0]
    }, index=["Default task."])
    task_df.to_parquet(meta_dir / "tasks.parquet")

    # ==========================
    # 3. [修改] 读取分布式的 episodes.parquet 碎片
    # ==========================
    # V3 会把 meta/episodes/chunk-000/file-000.parquet 等全部聚合
    ep_files = sorted(list(meta_dir.rglob("episodes/chunk-*/*.parquet")))
    if not ep_files:
        print("❌ 未找到 episode metadata 分片。")
        return

    ep_df = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
    total_episodes = len(ep_df)

    # ==========================
    # 4. 生成 info.json
    # ==========================
    info = {
        "codebase_version": "v3.0",
        "robot_type": "aloha",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": len(pq_files),
        "fps": 30,
        "splits": {
            "train": f"0:{total_episodes}"
        },
        # [修改] V3 终极版路径模板
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            # 1. 基础索引与时间信息
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},

            # 2. 核心状态与动作
            "action": {
                "dtype": "float32",
                "shape": [all_actions.shape[1]],
                "names": ["actions"]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [all_states.shape[1]],
                "names": ["state"]
            },

            # 3. 任务索引
            "task_index": {"dtype": "int64", "shape": [1], "names": None},

            # 4. 左相机流 (主指针、帧索引、延迟)
            "observation.images.cam_left": {
                "dtype": "video",
                "shape": [3, 480, 640],
                "names": ["channel", "height", "width"]
            },
            "observation.images.cam_left_frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.images.cam_left_latency": {"dtype": "float32", "shape": [1], "names": None},

            # 5. 右相机流 (主指针、帧索引、延迟)
            "observation.images.cam_right": {
                "dtype": "video",
                "shape": [3, 480, 640],
                "names": ["channel", "height", "width"]
            },
            "observation.images.cam_right_frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.images.cam_right_latency": {"dtype": "float32", "shape": [1], "names": None}
        }
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    print("✅ V3 Metadata 生成完毕！路径结构已完全对齐。")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成 LeRobot V3 格式的 metadata")
    parser.add_argument("dataset_dir", nargs="?", default="data/lerobot/test3", help="数据集目录路径")
    parser.add_argument("--skip-video-stats", action="store_true", help="跳过视频统计计算")
    args = parser.parse_args()

    generate_lerobot_v3_metadata(args.dataset_dir, compute_video_stats_flag=not args.skip_video_stats)