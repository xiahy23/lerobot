import numpy as np
import pandas as pd

# 替换为你实际生成的 parquet 路径
# parquet_path = "data/lerobot/test3/meta/tasks.parquet"
parquet_path = "data/lerobot/HuggingFaceVLA/libero/meta/tasks.parquet"

try:
    df = pd.read_parquet(parquet_path)

    # 1. 查看基本信息
    print(f"📊 数据集总行数 (Actions): {len(df)}")
    print(f"📋 列名列表:\n{df.columns.tolist()}")
    print("-" * 40)
    print(df)

    # # 2. 验证 'Dense Control' 对齐逻辑
    # # 我们取出 时间戳、图像索引、延迟 这三列来看看
    # # 假设你的相机叫 observation.images.cam_left
    # cam_key = "observation.images.cam_left"

    # cols_to_check = [
    #     "episode_index",
    #     "frame_index",
    #     "timestamp",
    #     # f"{cam_key}_frame_index",
    #     # f"{cam_key}_latency"
    # ]

    # # 检查这些列是否存在
    # valid_cols = [c for c in cols_to_check if c in df.columns]

    # if len(valid_cols) == 3:
    #     subset = df[cols_to_check].head(35) #看前35行
    #     print("🔍 前 35 行对齐采样 (注意 Frame Index 的变化):")
    #     print(subset)

    #     # 3. 统计一下 Latency
    #     latencies = np.array(df[f"{cam_key}_latency"].tolist())
    #     print("-" * 40)
    #     print(f"⏱️ 延迟统计 (秒):")
    #     print(f"   Max:  {latencies.max():.4f}")
    #     print(f"   Min:  {latencies.min():.4f}")
    #     print(f"   Mean: {latencies.mean():.4f}")
    # else:
    #     print(f"⚠️ 未找到完整的相机列，现有列: {df.columns}")
    #     print(df.head())

except Exception as e:
    print(f"读取失败: {e}")