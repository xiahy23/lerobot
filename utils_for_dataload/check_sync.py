import h5py
import matplotlib.pyplot as plt
import numpy as np


def check_arm_sync(file_path):
    print(f"🕵️ 正在检查文件: {file_path}")

    with h5py.File(file_path, 'r') as f:
        # 读取时间戳
        ts_left = f['act/qpos/left/ts'][:]
        ts_right = f['act/qpos/right/ts'][:]

        # 打印原始长度
        print(f"   左臂帧数: {len(ts_left)}")
        print(f"   右臂帧数: {len(ts_right)}")

        # 1. 自动单位修正 (防止纳秒/秒混用)
        # 取中位数判断，防止首帧为0干扰
        if len(ts_left) > 0 and np.median(ts_left) > 1e14:
            print("   [Info] 左臂时间戳为纳秒，转为秒。")
            ts_left = ts_left.astype(np.float64) / 1e9
        else:
            ts_left = ts_left.astype(np.float64)

        if len(ts_right) > 0 and np.median(ts_right) > 1e14:
            print("   [Info] 右臂时间戳为纳秒，转为秒。")
            ts_right = ts_right.astype(np.float64) / 1e9
        else:
            ts_right = ts_right.astype(np.float64)

        # 2. 对齐长度以便计算差值
        min_len = min(len(ts_left), len(ts_right))
        ts_left_trim = ts_left[:min_len]
        ts_right_trim = ts_right[:min_len]

        # 3. 计算时间差 (Diff)
        # diff = 左 - 右 (如果 diff > 0，说明左臂比右臂 晚 到达该帧)
        diffs = ts_left_trim - ts_right_trim
        diffs_ms = diffs * 1000  # 转换为毫秒，方便看

        # 4. 统计分析
        print("-" * 40)
        print(f"⏱️ 左右臂同步误差统计 (共 {min_len} 帧):")
        print(f"   平均误差 (Mean): {np.mean(diffs_ms):.4f} ms")
        print(f"   最大误差 (Max):  {np.max(np.abs(diffs_ms)):.4f} ms")
        print(f"   标准差 (Std):    {np.std(diffs_ms):.4f} ms")

        # 5. 判定建议
        # 通常 100Hz 的控制频率下，1帧 = 10ms。
        # 如果误差 < 5ms，说明非常同步，直接截断没问题。
        # 如果误差 > 100ms，说明有一只手臂晚启动了，不能直接截断。
        print("-" * 40)
        if np.mean(np.abs(diffs_ms)) < 5.0:
            print("✅ 结论: 同步性良好。可以直接使用 min_len 截断法。")
        else:
            print("⚠️ 结论: 存在较大时间差！")
            print("   建议: 不要直接截断，需要基于时间戳进行 Nearest Neighbor 对齐。")

        # (可选) 画图看看有没有漂移
        # plt.figure(figsize=(10, 3))
        # plt.plot(diffs_ms)
        # plt.title("Left-Right Arm Synchronization Offset (ms)")
        # plt.xlabel("Frame Index")
        # plt.ylabel("Offset (ms)")
        # plt.show()

if __name__ == "__main__":
    check_arm_sync("data/raw/raw_000000.h5")