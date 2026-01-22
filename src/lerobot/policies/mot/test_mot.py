import torch

from lerobot.policies.mot.configuration_mot import (AttentionType,
                                                    BackboneType, DecodingMode,
                                                    MoTBackboneConfig,
                                                    MoTConfig, MoTFlowConfig,
                                                    MoTNodeConfig, NodeType)
from lerobot.policies.mot.modeling_mot import MoTModel, TokenBlock


def test_heterogeneous_core():
    print("=== Test 2: Heterogeneous Core (Backbone Mixing) ===")

    # 1. 定义异构配置: VLM (Hidden=64) + Expert (Hidden=32)
    # 只要 head_dim * num_heads 相同，就能联通
    head_dim = 8
    num_heads = 4
    shared_head_space = head_dim * num_heads # 32

    print(f"[1] Configuring Heterogeneous Backbones (64dim <-> 32dim)...")

    backbone_vlm = MoTBackboneConfig(
        name="backbone_A",
        backbone_type=BackboneType.GEMMA, # 用 Gemma 模拟结构
        variant="gemma_2b",
        hidden_size=64,     # <--- 维度 A
        num_hidden_layers=2,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        intermediate_size=128
    )

    backbone_expert = MoTBackboneConfig(
        name="backbone_B",
        backbone_type=BackboneType.GEMMA,
        variant="gemma_2b",
        hidden_size=32,     # <--- 维度 B (不同!)
        num_hidden_layers=2,
        num_attention_heads=num_heads, # 必须相同
        head_dim=head_dim,             # 必须相同
        intermediate_size=64
    )

    nodes = [
        MoTNodeConfig(name="node_A", backbone_name="backbone_A", input_dim=64, node_type=NodeType.VISION),
        MoTNodeConfig(name="node_B", backbone_name="backbone_B", input_dim=32, node_type=NodeType.ACTION, is_output=True)
    ]

    # 让 B 关注 A (Prefix Attention)
    flows = [
        MoTFlowConfig(source="node_A", target="node_A", attention_type=AttentionType.FULL),
        MoTFlowConfig(source="node_A", target="node_B", attention_type=AttentionType.FULL),
        MoTFlowConfig(source="node_B", target="node_B", attention_type=AttentionType.CAUSAL),
    ]

    config = MoTConfig(
        nodes=nodes, flows=flows, backbones=[backbone_vlm, backbone_expert],
        decoding_mode=DecodingMode.FLOW_MATCHING
    )

    # 2. 初始化模型
    model = MoTModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. 手动构造 Token Blocks (模拟 Embedder 的输出)
    batch_size = 2

    # Block A: 模拟 Vision (64 dim)
    emb_A = torch.randn(batch_size, 10, 64, device=device) # Seq=10
    mask_A = torch.ones(batch_size, 10, dtype=torch.bool, device=device)
    att_A = torch.zeros(batch_size, 10, device=device) # Full

    # Block B: 模拟 Action (32 dim)
    emb_B = torch.randn(batch_size, 5, 32, device=device) # Seq=5
    mask_B = torch.ones(batch_size, 5, dtype=torch.bool, device=device)
    att_B = torch.ones(batch_size, 5, device=device) # Causal

    blocks = [
        TokenBlock("node_A", emb_A, mask_A, att_A),
        TokenBlock("node_B", emb_B, mask_B, att_B)
    ]

    # 4. 测试 Joint Attention
    print("[2] Running Joint Attention...")
    output_blocks, _ = model.joint_attention(blocks)

    out_A = output_blocks[0].embeddings
    out_B = output_blocks[1].embeddings

    print(f"   ✓ Input A: {emb_A.shape} -> Output A: {out_A.shape}")
    print(f"   ✓ Input B: {emb_B.shape} -> Output B: {out_B.shape}")

    # 验证维度保持不变
    assert out_A.shape[-1] == 64, "Output A dimension mismatch!"
    assert out_B.shape[-1] == 32, "Output B dimension mismatch!"

    print("=== Heterogeneous Core Test Passed! ===\n")

if __name__ == "__main__":
    test_heterogeneous_core()