import cv2  # 需要 opencv-python 库
import h5py
import matplotlib.pyplot as plt
import numpy as np

file_path = "data/raw_000000.h5"

with h5py.File(file_path, "r") as f:
    # --- 1. 可视化机械臂动作 (Action) ---
    # 读取左臂的关节位置数据
    # 形状: (10657, 7)
    joint_data = f['act']['epos']['left']['raw'][:]

    plt.figure(figsize=(12, 4))
    # 绘制所有7个关节的曲线
    for i in range(7):
        plt.plot(joint_data[:, i], label=f'Joint {i+1}')

    plt.title("Left Arm Joint Positions (Action)")
    plt.xlabel("Time steps")
    plt.ylabel("Position (rad or encoder value)")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()

    # --- 2. 可视化摄像头图像 (Observation) ---
    # ALOHA 格式通常是将 JPEG 字节流拼在一起，需要通过长度(len)来切割

    # 获取图像相关的 dataset
    img_grp = f['obs']['image']['left']
    all_bytes = img_grp['raw'][:]  # 那个巨大的 1D 数组
    frame_lens = img_grp['len'][:] # 每张图片的字节长度

    print(f"Total frames: {len(frame_lens)}")

    # 我们只看第 0 帧和第 100 帧作为示例
    current_offset = 0

    plt.figure(figsize=(10, 5))

    for i, length in enumerate(frame_lens):
        if i in [0, 100]:  # 只显示特定帧
            # 切割出该帧的字节
            img_bytes = all_bytes[current_offset : current_offset + length]

            # 解码: 字节 -> 图像
            # 这是一个关键步骤，把 1D uint8 数组转回图片
            image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            # OpenCV 默认是 BGR，Matplotlib 需要 RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            plt.subplot(1, 2, (0 if i==0 else 1) + 1)
            plt.imshow(image)
            plt.title(f"Frame {i}")
            plt.axis('off')

        # 更新偏移量，准备读取下一帧
        current_offset += length

        # 优化：如果我们只需要前100帧，读完就可以退出了，不用循环到底
        if i > 100:
            break

    plt.show()