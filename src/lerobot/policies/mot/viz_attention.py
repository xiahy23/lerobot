import matplotlib.pyplot as plt
import seaborn as sns
import torch

from lerobot.policies.mot.configuration_mot import make_pi0_config
from lerobot.policies.mot.modeling_mot import MoTPolicy, TokenBlock


def visualize_mot_mask():
    # 1. 初始化 Config (以 pi0 为例)
    print("Initializing MoT Config (pi0-like)...")
    config = make_pi0_config()

    # 2. 初始化 Policy
    policy = MoTPolicy(config)
    mask_builder = policy.model.mask_builder

    # 3. 构造 Dummy Token Blocks (模拟真实运行时的 Token 分布)
    # 假设我们有 2 个节点：VLM (Vision+Lang) 和 Action Expert
    batch_size = 1

    # 模拟 VLM 节点输出的 token (例如 256个patch + 48个text = 304)
    vlm_len = 50 # 为了画图清晰，我们缩小一点长度
    vlm_block = TokenBlock(
        node_name="vlm",
        embeddings=torch.randn(batch_size, vlm_len, 1024),
        pad_mask=torch.ones(batch_size, vlm_len, dtype=torch.bool),
        att_mask=torch.zeros(batch_size, vlm_len, dtype=torch.bool) # 0 for full attention usually
    )

    # 模拟 Action Expert 节点输出的 token (例如 50个action = 50)
    expert_len = 30
    expert_block = TokenBlock(
        node_name="action_expert",
        embeddings=torch.randn(batch_size, expert_len, 1024),
        pad_mask=torch.ones(batch_size, expert_len, dtype=torch.bool),
        att_mask=torch.ones(batch_size, expert_len, dtype=torch.bool) # 1 usually indicates distinct groups for prefix
    )

    blocks = [vlm_block, expert_block]

    # 4. 生成 Mask
    # build_attention_mask 返回的是 (Batch, Total_Seq, Total_Seq) 的布尔矩阵
    # True = Attend (可见), False = Masked (不可见)
    mask = mask_builder.build_attention_mask(blocks, device=torch.device("cpu"))

    # 取第一个 batch 的 mask
    mask_2d = mask[0].int().numpy() # 转换成 0/1 整数方便画图

    # 5. 画图
    plt.figure(figsize=(10, 8))

    # 使用 Seaborn 画热力图
    # 1 (白色/浅色) 代表 "Looking at" (可见)
    # 0 (黑色/深色) 代表 "Masked" (不可见)
    sns.heatmap(mask_2d, cmap="Greys_r", cbar=False, square=True, linewidths=0.5, linecolor='gray')

    # 添加辅助线区分 Node 边界
    total_len = vlm_len + expert_len
    plt.axhline(y=vlm_len, color='red', linestyle='--', linewidth=2)
    plt.axvline(x=vlm_len, color='red', linestyle='--', linewidth=2)

    # 标注区域
    # 坐标系：X轴是 Source (Key/Value), Y轴是 Target (Query)
    # Q: Who is looking? (Rows)
    # K: At whom? (Cols)

    plt.text(vlm_len/2, -2, "Source: VLM", ha='center', color='blue', fontsize=12, weight='bold')
    plt.text(vlm_len + expert_len/2, -2, "Source: Expert", ha='center', color='blue', fontsize=12, weight='bold')

    plt.text(-8, vlm_len/2, "Target:\nVLM", va='center', ha='center', color='blue', fontsize=12, weight='bold')
    plt.text(-8, vlm_len + expert_len/2, "Target:\nExpert", va='center', ha='center', color='blue', fontsize=12, weight='bold')

    plt.title(f"MoT Attention Mask Visualization\n(White=Visible, Black=Masked)\nTotal Tokens: {total_len}", fontsize=14)
    plt.xlabel("Key / Value (Source Tokens)")
    plt.ylabel("Query (Target Tokens)")

    # 保存或显示
    save_path = "mot_attention_mask.png"
    plt.savefig(save_path)
    print(f"Attention mask visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_mot_mask()