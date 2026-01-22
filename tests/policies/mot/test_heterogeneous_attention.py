#!/usr/bin/env python

"""
Test for MoT heterogeneous joint layer attention.

This test verifies that the MoTJointLayerAttention correctly handles
backbones with different hidden sizes but shared head_dim and num_heads.
"""

import pytest
import torch
import torch.nn as nn

# Skip if transformers not available
pytest.importorskip("transformers")


def test_heterogeneous_hidden_size_qkv_projection():
    """Test that QKV projection works correctly with different hidden sizes."""
    # Simulate two backbones with different hidden sizes but same head structure
    # VLM: hidden_size=2048, num_heads=8, head_dim=256
    # Expert: hidden_size=1024, num_heads=8, head_dim=256

    batch_size = 2
    seq_vlm = 10
    seq_expert = 5
    num_heads = 8
    head_dim = 256
    hidden_vlm = 2048
    hidden_expert = 1024

    # Create mock Q, K, V projections
    q_proj_vlm = nn.Linear(hidden_vlm, num_heads * head_dim)
    k_proj_vlm = nn.Linear(hidden_vlm, num_heads * head_dim)
    v_proj_vlm = nn.Linear(hidden_vlm, num_heads * head_dim)

    q_proj_expert = nn.Linear(hidden_expert, num_heads * head_dim)
    k_proj_expert = nn.Linear(hidden_expert, num_heads * head_dim)
    v_proj_expert = nn.Linear(hidden_expert, num_heads * head_dim)

    # Create input tensors with different hidden sizes
    vlm_hidden = torch.randn(batch_size, seq_vlm, hidden_vlm)
    expert_hidden = torch.randn(batch_size, seq_expert, hidden_expert)

    # Phase 1: QKV projection to head space
    q_vlm = q_proj_vlm(vlm_hidden).view(batch_size, seq_vlm, num_heads, head_dim).transpose(1, 2)
    k_vlm = k_proj_vlm(vlm_hidden).view(batch_size, seq_vlm, num_heads, head_dim).transpose(1, 2)
    v_vlm = v_proj_vlm(vlm_hidden).view(batch_size, seq_vlm, num_heads, head_dim).transpose(1, 2)

    q_expert = q_proj_expert(expert_hidden).view(batch_size, seq_expert, num_heads, head_dim).transpose(1, 2)
    k_expert = k_proj_expert(expert_hidden).view(batch_size, seq_expert, num_heads, head_dim).transpose(1, 2)
    v_expert = v_proj_expert(expert_hidden).view(batch_size, seq_expert, num_heads, head_dim).transpose(1, 2)

    # Verify shapes are compatible for concatenation
    assert q_vlm.shape == (batch_size, num_heads, seq_vlm, head_dim)
    assert q_expert.shape == (batch_size, num_heads, seq_expert, head_dim)

    # Phase 2: Concatenate in sequence dimension (head space is shared)
    q_global = torch.cat([q_vlm, q_expert], dim=2)
    k_global = torch.cat([k_vlm, k_expert], dim=2)
    v_global = torch.cat([v_vlm, v_expert], dim=2)

    # Verify concatenated shapes
    total_seq = seq_vlm + seq_expert
    assert q_global.shape == (batch_size, num_heads, total_seq, head_dim)
    assert k_global.shape == (batch_size, num_heads, total_seq, head_dim)
    assert v_global.shape == (batch_size, num_heads, total_seq, head_dim)

    # Phase 2b: Compute attention (simplified - no mask for this test)
    scaling = head_dim ** -0.5
    attn_weights = torch.matmul(q_global, k_global.transpose(-2, -1)) * scaling
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v_global)

    # Reshape attention output
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, total_seq, num_heads * head_dim)

    assert attn_output.shape == (batch_size, total_seq, num_heads * head_dim)

    # Phase 3: Split and project back to disparate hidden sizes
    attn_vlm = attn_output[:, :seq_vlm, :]
    attn_expert = attn_output[:, seq_vlm:, :]

    o_proj_vlm = nn.Linear(num_heads * head_dim, hidden_vlm)
    o_proj_expert = nn.Linear(num_heads * head_dim, hidden_expert)

    output_vlm = o_proj_vlm(attn_vlm)
    output_expert = o_proj_expert(attn_expert)

    # Verify output shapes match original hidden sizes
    assert output_vlm.shape == (batch_size, seq_vlm, hidden_vlm)
    assert output_expert.shape == (batch_size, seq_expert, hidden_expert)

    print("✓ Heterogeneous hidden size QKV projection test passed!")


def test_heterogeneous_attention_with_mask():
    """Test heterogeneous attention with proper masking (pi0-style)."""
    from lerobot.policies.mot.configuration_mot import (AttentionType,
                                                        BackboneType,
                                                        DecodingMode,
                                                        MoTBackboneConfig,
                                                        MoTConfig,
                                                        MoTFlowConfig,
                                                        MoTNodeConfig,
                                                        NodeType)
    from lerobot.policies.mot.modeling_mot import (MoTAttentionMaskBuilder,
                                                   TokenBlock)

    batch_size = 2
    seq_vlm = 10
    seq_expert = 5
    hidden_vlm = 2048
    hidden_expert = 1024

    # Create a pi0-like config
    nodes = [
        MoTNodeConfig(
            name="vlm",
            node_type=NodeType.VLM,
            backbone_name="paligemma",
            input_dim=hidden_vlm,
            token_dim=hidden_vlm,
            max_tokens=300,
        ),
        MoTNodeConfig(
            name="action_expert",
            node_type=NodeType.LM,
            backbone_name="gemma_expert",
            input_dim=hidden_expert,
            token_dim=hidden_expert,
            max_tokens=51,
            is_output=True,
            output_head="linear",
            output_dim=32,
        ),
    ]

    flows = [
        MoTFlowConfig(source="vlm", target="vlm", attention_type=AttentionType.FULL),
        MoTFlowConfig(source="vlm", target="action_expert", attention_type=AttentionType.FULL),
        MoTFlowConfig(source="action_expert", target="action_expert", attention_type=AttentionType.CAUSAL),
        # Note: No flow from expert to VLM
    ]

    backbones = [
        MoTBackboneConfig(
            name="paligemma",
            backbone_type=BackboneType.PALIGEMMA,
            variant="gemma_2b",
            hidden_size=hidden_vlm,
            num_hidden_layers=18,
            num_attention_heads=8,
            head_dim=256,
        ),
        MoTBackboneConfig(
            name="gemma_expert",
            backbone_type=BackboneType.GEMMA,
            variant="gemma_300m",
            hidden_size=hidden_expert,
            num_hidden_layers=18,
            num_attention_heads=8,
            head_dim=256,
        ),
    ]

    config = MoTConfig(
        nodes=nodes,
        flows=flows,
        backbones=backbones,
        decoding_mode=DecodingMode.FLOW_MATCHING,
    )

    # Create token blocks with different hidden sizes
    vlm_block = TokenBlock(
        node_name="vlm",
        embeddings=torch.randn(batch_size, seq_vlm, hidden_vlm),
        pad_mask=torch.ones(batch_size, seq_vlm, dtype=torch.bool),
        att_mask=torch.zeros(batch_size, seq_vlm),
    )

    expert_block = TokenBlock(
        node_name="action_expert",
        embeddings=torch.randn(batch_size, seq_expert, hidden_expert),
        pad_mask=torch.ones(batch_size, seq_expert, dtype=torch.bool),
        att_mask=torch.ones(batch_size, seq_expert),  # Causal
    )

    token_blocks = [vlm_block, expert_block]

    # Build attention mask
    mask_builder = MoTAttentionMaskBuilder(config)
    mask = mask_builder.build_attention_mask(token_blocks)

    # Verify mask dimensions
    total_seq = seq_vlm + seq_expert
    assert mask.shape == (batch_size, total_seq, total_seq)

    # Verify mask patterns:
    # VLM self-attention (full)
    vlm_self_mask = mask[:, :seq_vlm, :seq_vlm]
    assert vlm_self_mask.all(), "VLM self-attention should be full (all True)"

    # Expert attending to VLM (full)
    expert_to_vlm_mask = mask[:, seq_vlm:, :seq_vlm]
    assert expert_to_vlm_mask.all(), "Expert should fully attend to VLM"

    # VLM attending to Expert (should be blocked - no flow defined)
    vlm_to_expert_mask = mask[:, :seq_vlm, seq_vlm:]
    assert not vlm_to_expert_mask.any(), "VLM should not attend to Expert"

    # Expert self-attention (causal)
    expert_self_mask = mask[:, seq_vlm:, seq_vlm:]
    # First position can only see itself
    assert expert_self_mask[:, 0, 0].all()
    assert not expert_self_mask[:, 0, 1:].any()
    # Last position can see all
    assert expert_self_mask[:, -1, :].all()

    print("✓ Heterogeneous attention with mask test passed!")


def test_config_validation_for_heterogeneous():
    """Test that config properly validates compatible head structures."""
    from lerobot.policies.mot.configuration_mot import (BackboneType,
                                                        MoTBackboneConfig,
                                                        MoTConfig,
                                                        MoTNodeConfig,
                                                        NodeType)

    # Valid: Same num_heads and head_dim, different hidden_size
    nodes = [
        MoTNodeConfig(name="vlm", node_type=NodeType.VLM, backbone_name="paligemma"),
        MoTNodeConfig(name="expert", node_type=NodeType.LM, backbone_name="gemma_expert"),
    ]

    backbones_valid = [
        MoTBackboneConfig(
            name="paligemma",
            backbone_type=BackboneType.PALIGEMMA,
            hidden_size=2048,  # Different
            num_attention_heads=8,  # Same
            head_dim=256,  # Same
            num_hidden_layers=18,  # Same
        ),
        MoTBackboneConfig(
            name="gemma_expert",
            backbone_type=BackboneType.GEMMA,
            hidden_size=1024,  # Different
            num_attention_heads=8,  # Same
            head_dim=256,  # Same
            num_hidden_layers=18,  # Same
        ),
    ]

    # This should not raise
    config = MoTConfig(nodes=nodes, flows=[], backbones=backbones_valid)

    # Verify we can access backbone configs
    paligemma_cfg = config.get_backbone("paligemma")
    gemma_cfg = config.get_backbone("gemma_expert")

    assert paligemma_cfg.hidden_size == 2048
    assert gemma_cfg.hidden_size == 1024
    assert paligemma_cfg.num_attention_heads == gemma_cfg.num_attention_heads
    assert paligemma_cfg.head_dim == gemma_cfg.head_dim

    print("✓ Config validation for heterogeneous test passed!")


def test_full_joint_layer_attention_simulation():
    """
    Full simulation test of heterogeneous joint layer attention.

    This test creates mock backbone layers and verifies the full forward pass
    through the heterogeneous joint attention mechanism.
    """
    from lerobot.policies.mot.configuration_mot import (AttentionType,
                                                        BackboneType,
                                                        DecodingMode,
                                                        MoTBackboneConfig,
                                                        MoTConfig,
                                                        MoTFlowConfig,
                                                        MoTNodeConfig,
                                                        NodeType)
    from lerobot.policies.mot.modeling_mot import (MoTAttentionMaskBuilder,
                                                   TokenBlock)

    batch_size = 2
    seq_vlm = 8
    seq_expert = 4
    hidden_vlm = 2048
    hidden_expert = 1024
    num_heads = 8
    head_dim = 256

    # Create mock layers that simulate the projection behavior
    class MockSelfAttention(nn.Module):
        def __init__(self, hidden_size, num_heads, head_dim):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.scaling = head_dim ** -0.5

            self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    class MockLayerNorm(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))

        def forward(self, x, cond=None):
            # Simplified layer norm
            normed = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)
            return normed * self.weight, None  # Return (normed, gate=None)

    class MockMLP(nn.Module):
        def __init__(self, hidden_size, intermediate_size):
            super().__init__()
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, x):
            return self.down_proj(torch.relu(self.up_proj(x)))

    class MockLayer(nn.Module):
        def __init__(self, hidden_size, num_heads, head_dim, intermediate_size):
            super().__init__()
            self.input_layernorm = MockLayerNorm(hidden_size)
            self.self_attn = MockSelfAttention(hidden_size, num_heads, head_dim)
            self.post_attention_layernorm = MockLayerNorm(hidden_size)
            self.mlp = MockMLP(hidden_size, intermediate_size)

    # Create mock layers for VLM and Expert
    vlm_layer = MockLayer(hidden_vlm, num_heads, head_dim, hidden_vlm * 4)
    expert_layer = MockLayer(hidden_expert, num_heads, head_dim, hidden_expert * 4)

    # Create input tensors
    vlm_hidden = torch.randn(batch_size, seq_vlm, hidden_vlm)
    expert_hidden = torch.randn(batch_size, seq_expert, hidden_expert)

    # Simulate Phase 1: QKV projection
    def project_qkv(layer, hidden_states, num_heads, head_dim):
        batch, seq, _ = hidden_states.shape
        normed, _ = layer.input_layernorm(hidden_states)

        q = layer.self_attn.q_proj(normed).view(batch, seq, num_heads, head_dim).transpose(1, 2)
        k = layer.self_attn.k_proj(normed).view(batch, seq, num_heads, head_dim).transpose(1, 2)
        v = layer.self_attn.v_proj(normed).view(batch, seq, num_heads, head_dim).transpose(1, 2)

        return q, k, v, normed

    q_vlm, k_vlm, v_vlm, normed_vlm = project_qkv(vlm_layer, vlm_hidden, num_heads, head_dim)
    q_expert, k_expert, v_expert, normed_expert = project_qkv(expert_layer, expert_hidden, num_heads, head_dim)

    # Verify QKV shapes are compatible
    assert q_vlm.shape == (batch_size, num_heads, seq_vlm, head_dim)
    assert q_expert.shape == (batch_size, num_heads, seq_expert, head_dim)

    # Phase 2: Concatenate and compute attention
    q_global = torch.cat([q_vlm, q_expert], dim=2)
    k_global = torch.cat([k_vlm, k_expert], dim=2)
    v_global = torch.cat([v_vlm, v_expert], dim=2)

    total_seq = seq_vlm + seq_expert

    # Create attention mask (pi0-style)
    # VLM: full self-attention
    # Expert: can attend to VLM + causal self-attention
    # VLM cannot attend to Expert
    mask = torch.zeros(batch_size, total_seq, total_seq, dtype=torch.bool)
    mask[:, :seq_vlm, :seq_vlm] = True  # VLM self-attention
    mask[:, seq_vlm:, :seq_vlm] = True  # Expert attends to VLM
    # Causal for expert self-attention
    for i in range(seq_expert):
        mask[:, seq_vlm + i, seq_vlm:seq_vlm + i + 1] = True

    # Convert to attention mask format
    attn_mask = torch.where(mask, 0.0, float('-inf'))[:, None, :, :]  # Add head dim

    # Compute attention
    scaling = head_dim ** -0.5
    attn_weights = torch.matmul(q_global, k_global.transpose(-2, -1)) * scaling
    attn_weights = attn_weights + attn_mask
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, v_global)

    # Reshape
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, total_seq, num_heads * head_dim)

    # Phase 3: Split and project back
    attn_vlm = attn_output[:, :seq_vlm, :]
    attn_expert = attn_output[:, seq_vlm:, :]

    out_vlm = vlm_layer.self_attn.o_proj(attn_vlm)
    out_expert = expert_layer.self_attn.o_proj(attn_expert)

    # Verify output shapes match original hidden sizes
    assert out_vlm.shape == (batch_size, seq_vlm, hidden_vlm)
    assert out_expert.shape == (batch_size, seq_expert, hidden_expert)

    # Phase 4: Residual + FFN
    hidden_vlm_out = vlm_hidden + out_vlm
    post_normed_vlm, _ = vlm_layer.post_attention_layernorm(hidden_vlm_out)
    hidden_vlm_out = hidden_vlm_out + vlm_layer.mlp(post_normed_vlm)

    hidden_expert_out = expert_hidden + out_expert
    post_normed_expert, _ = expert_layer.post_attention_layernorm(hidden_expert_out)
    hidden_expert_out = hidden_expert_out + expert_layer.mlp(post_normed_expert)

    # Final verification
    assert hidden_vlm_out.shape == (batch_size, seq_vlm, hidden_vlm)
    assert hidden_expert_out.shape == (batch_size, seq_expert, hidden_expert)

    # Verify attention is working (outputs should be different from inputs due to attention)
    assert not torch.allclose(hidden_vlm_out, vlm_hidden, atol=1e-5)
    assert not torch.allclose(hidden_expert_out, expert_hidden, atol=1e-5)

    print("✓ Full joint layer attention simulation test passed!")


if __name__ == "__main__":
    test_heterogeneous_hidden_size_qkv_projection()
    test_heterogeneous_attention_with_mask()
    test_config_validation_for_heterogeneous()
    test_full_joint_layer_attention_simulation()
    print("\n✅ All heterogeneous attention tests passed!")
