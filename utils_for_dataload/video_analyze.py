import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def analyze_intervals(file_path):
    print(f"🕵️ 正在分析文件: {file_path}")

    with h5py.File(file_path, 'r') as f:
        # 检查相机列表
        cameras = []
        if 'obs/image' in f:
            cameras = list(f['obs/image'].keys())

        if not cameras:
            print("❌ 未找到相机数据 (obs/image)")
            return

        plt.figure(figsize=(12, 4 * len(cameras)))

        for idx, cam in enumerate(cameras):
            # 1. 读取时间戳
            ts = f[f'obs/image/{cam}/ts'][:]

            # 2. 单位修正 (纳秒 -> 秒)
            if len(ts) > 0 and np.median(ts) > 1e14:
                ts = ts.astype(np.float64) / 1e9
            else:
                ts = ts.astype(np.float64)

            # 3. 计算相邻帧间隔 (Delta Time)
            # diff[i] = ts[i+1] - ts[i]
            diffs = np.diff(ts)
            diffs_ms = diffs * 1000  # 转毫秒

            # 4. 统计信息
            mean_dt = np.mean(diffs_ms)
            std_dt = np.std(diffs_ms)
            max_dt = np.max(diffs_ms)

            # 理论帧率 (假设 30Hz -> 33.33ms)
            target_ms = 33.33

            print(f"\n📷 相机: {cam}")
            print(f"   总帧数: {len(ts)}")
            print(f"   平均间隔: {mean_dt:.2f} ms (理论: {target_ms:.2f} ms)")
            print(f"   最大间隔: {max_dt:.2f} ms")
            print(f"   标准差:   {std_dt:.2f} ms (抖动程度)")

            # 5. 寻找明显丢帧的位置 (间隔 > 2倍理论值)
            bad_indices = np.where(diffs_ms > (target_ms * 2))[0]
            if len(bad_indices) > 0:
                print(f"   ⚠️ 发现 {len(bad_indices)} 处严重丢帧/卡顿！")
                print(f"      前 5 个卡顿位置 (帧索引): {bad_indices[:5]}")
                print(f"      对应的间隔 (ms): {diffs_ms[bad_indices[:5]]}")
            else:
                print("   ✅ 未发现严重丢帧。")

            # 6. 绘图
            ax = plt.subplot(len(cameras), 1, idx + 1)
            ax.plot(diffs_ms, alpha=0.7, label='Frame Interval')

            # 画一条 33ms 的基准线
            ax.axhline(y=target_ms, color='r', linestyle='--', alpha=0.5, label='Target (33ms)')

            ax.set_title(f"Camera: {cam} (Mean: {mean_dt:.1f}ms, Max: {max_dt:.1f}ms)")
            ax.set_ylabel("Interval (ms)")
            ax.set_xlabel("Frame Index")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 标记出异常点
            if len(bad_indices) > 0:
                ax.scatter(bad_indices, diffs_ms[bad_indices], color='red', s=10, zorder=5)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 替换成你的 h5 文件路径
    analyze_intervals("data/raw/raw_000000.h5")