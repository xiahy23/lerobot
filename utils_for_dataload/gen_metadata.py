import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_stats(data):
    return {
        "mean": np.mean(data, axis=0).astype(np.float32).tolist(),
        "std": np.std(data, axis=0).astype(np.float32).tolist(),
        "min": np.min(data, axis=0).astype(np.float32).tolist(),
        "max": np.max(data, axis=0).astype(np.float32).tolist(),
    }

def generate_lerobot_v3_metadata(dataset_dir):
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
    stats = {
        "action": compute_stats(all_actions),
        "observation.state": compute_stats(all_states)
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # ==========================
    # 2. 生成 tasks.parquet
    # ==========================
    task_df = pd.DataFrame([{
        "task_index": 0,
        "task": "Default task."
    }])
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
    generate_lerobot_v3_metadata("data/test3")