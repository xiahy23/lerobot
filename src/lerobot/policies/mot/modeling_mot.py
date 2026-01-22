#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MoT (Mixture of Transformers) Model Implementation.

This module implements the core MoT architecture components:
- MoTAttentionMaskBuilder: Generates dynamic attention masks from flow configurations
- MoTEmbeddingRouter: Routes and projects inputs to the appropriate token spaces
- MoTBackbone: Manages multiple transformer backbones with joint attention
- MoTPolicy: The main policy class that combines all components

The key innovation is the dynamic mask generation that allows arbitrary attention
patterns between transformer nodes, enabling flexible multi-transformer architectures.
"""

from __future__ import annotations

import builtins
import logging
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoTokenizer
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
from transformers.models.paligemma.modeling_paligemma import \
    PaliGemmaForConditionalGeneration
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.mot.configuration_mot import (AttentionType,
                                                    BackboneType, DecodingMode,
                                                    MoTBackboneConfig,
                                                    MoTConfig, MoTFlowConfig,
                                                    MoTNodeConfig, NodeType)
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import (ACTION, OBS_LANGUAGE_ATTENTION_MASK,
                                     OBS_LANGUAGE_TOKENS, OBS_STATE,
                                     OPENPI_ATTENTION_MASK_VALUE)
from lerobot.utils.import_utils import _transformers_available

logger = logging.getLogger(__name__)


class ActionSelectKwargs(TypedDict, total=False):
    """Keyword arguments for action selection."""
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None
    temperature: float | None


# ==================== Utility Functions ====================

def get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device = "cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type if hasattr(device, 'type') else str(device))
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> Tensor:
    """Sample from Beta distribution."""
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Pad the last dimension of a vector to new_dim with zeros."""
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize image with padding to maintain aspect ratio."""
    if images.shape[-1] <= 4:  # channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    batch_size, channels, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)

    return padded_images


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """
    Create 2D attention masks from padding and attention masks.

    This implements the big_vision attention mask logic that allows flexible
    attention patterns through cumulative masking.
    """
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2D, got {att_masks.ndim}D")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2D, got {pad_masks.ndim}D")

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


# ==================== Core MoT Components ====================

@dataclass
class TokenBlock:
    """Represents a block of tokens from a single node."""
    node_name: str
    embeddings: Tensor        # (batch, seq_len, hidden_dim)
    pad_mask: Tensor          # (batch, seq_len) - True for valid tokens
    att_mask: Tensor          # (batch, seq_len) - Attention mask pattern
    start_idx: int = 0        # Start index in concatenated sequence
    end_idx: int = 0          # End index in concatenated sequence


class MoTAttentionMaskBuilder:
    """
    Builds dynamic attention masks based on flow configurations.

    This is the core innovation of MoT - it generates attention masks that
    implement the specified attention flows between nodes.
    """

    def __init__(self, config: MoTConfig):
        self.config = config
        self.node_map = {n.name: i for i, n in enumerate(config.nodes)}

        # Pre-compute flow lookup for efficiency
        self.flow_lookup: dict[tuple[str, str], MoTFlowConfig] = {}
        for flow in config.flows:
            self.flow_lookup[(flow.source, flow.target)] = flow
            if flow.is_bidirectional:
                # Create reverse flow
                reverse_flow = MoTFlowConfig(
                    source=flow.target,
                    target=flow.source,
                    attention_type=flow.attention_type,
                    layer_range=flow.layer_range,
                    attention_scale=flow.attention_scale,
                )
                self.flow_lookup[(flow.target, flow.source)] = reverse_flow

    def build_attention_mask(
        self,
        token_blocks: list[TokenBlock],
        layer_idx: int | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        """
        Build a full attention mask from token blocks and flow configurations.

        Args:
            token_blocks: List of TokenBlock instances, one per node
            layer_idx: Current layer index (for layer-specific flows)
            device: Device to create mask on

        Returns:
            Attention mask of shape (batch, total_seq, total_seq)
        """
        if not token_blocks:
            raise ValueError("At least one token block required")

        batch_size = token_blocks[0].embeddings.shape[0]
        if device is None:
            device = token_blocks[0].embeddings.device

        # Calculate total sequence length and assign indices
        total_len = 0
        for block in token_blocks:
            block.start_idx = total_len
            block.end_idx = total_len + block.embeddings.shape[1]
            total_len = block.end_idx

        # Initialize mask (False = masked out)
        mask = torch.zeros(batch_size, total_len, total_len, dtype=torch.bool, device=device)

        # Build mask block by block based on flows
        for target_block in token_blocks:
            for source_block in token_blocks:
                flow_key = (source_block.node_name, target_block.node_name)
                flow = self.flow_lookup.get(flow_key)

                if flow is None:
                    # No flow defined = no attention
                    continue

                # Check layer range
                if flow.layer_range is not None and layer_idx is not None:
                    if not (flow.layer_range[0] <= layer_idx < flow.layer_range[1]):
                        continue

                # Get the slice for this block pair
                target_slice = slice(target_block.start_idx, target_block.end_idx)
                source_slice = slice(source_block.start_idx, source_block.end_idx)

                # Build attention pattern based on type
                block_mask = self._build_block_mask(
                    flow.attention_type,
                    target_block,
                    source_block,
                    batch_size,
                    device,
                )

                mask[:, target_slice, source_slice] = block_mask

        # Apply padding masks
        combined_pad_mask = torch.cat([b.pad_mask for b in token_blocks], dim=1)
        pad_2d = combined_pad_mask[:, None, :] & combined_pad_mask[:, :, None]
        mask = mask & pad_2d

        return mask

    def _build_block_mask(
        self,
        attention_type: AttentionType,
        target_block: TokenBlock,
        source_block: TokenBlock,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Build attention mask for a single source-target block pair."""
        target_len = target_block.embeddings.shape[1]
        source_len = source_block.embeddings.shape[1]

        if attention_type == AttentionType.FULL:
            # Full bidirectional attention
            return torch.ones(batch_size, target_len, source_len, dtype=torch.bool, device=device)

        elif attention_type == AttentionType.CAUSAL:
            # Causal attention (only works for self-attention)
            if target_block.node_name == source_block.node_name:
                causal = torch.tril(torch.ones(target_len, source_len, dtype=torch.bool, device=device))
                return causal[None, :, :].expand(batch_size, -1, -1)
            else:
                # Cross-attention with causal would mean attending to all of source
                return torch.ones(batch_size, target_len, source_len, dtype=torch.bool, device=device)

        elif attention_type == AttentionType.PREFIX:
            # Prefix-LM: bidirectional for source, causal for same-block
            if target_block.node_name == source_block.node_name:
                # Use the attention mask from the block
                cumsum = torch.cumsum(target_block.att_mask, dim=1)
                return cumsum[:, None, :] <= cumsum[:, :, None]
            else:
                # Full attention to other blocks (prefix)
                return torch.ones(batch_size, target_len, source_len, dtype=torch.bool, device=device)

        elif attention_type == AttentionType.CROSS:
            # Cross attention only (no self-attention contribution)
            if target_block.node_name == source_block.node_name:
                return torch.zeros(batch_size, target_len, source_len, dtype=torch.bool, device=device)
            else:
                return torch.ones(batch_size, target_len, source_len, dtype=torch.bool, device=device)

        elif attention_type == AttentionType.NONE:
            return torch.zeros(batch_size, target_len, source_len, dtype=torch.bool, device=device)

        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def prepare_4d_mask(self, mask_2d: Tensor) -> Tensor:
        """Convert 2D mask to 4D mask for transformer attention."""
        mask_4d = mask_2d[:, None, :, :]  # Add head dimension
        return torch.where(mask_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)


class MoTInputProjection(nn.Module):
    """Projects input features to token embeddings for a single node."""

    def __init__(self, node_config: MoTNodeConfig, backbone_hidden_size: int):
        super().__init__()
        self.node_config = node_config
        self.hidden_size = backbone_hidden_size

        if node_config.proj_type == "linear":
            self.proj = nn.Linear(node_config.input_dim, backbone_hidden_size)
        elif node_config.proj_type == "mlp":
            hidden_dim = node_config.proj_hidden_dim or backbone_hidden_size * 2
            self.proj = nn.Sequential(
                nn.Linear(node_config.input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, backbone_hidden_size),
            )
        elif node_config.proj_type == "none":
            self.proj = nn.Identity()
        else:
            raise ValueError(f"Unknown projection type: {node_config.proj_type}")

        # Position encoding
        if node_config.position_encoding == "learnable":
            self.pos_embedding = nn.Parameter(
                torch.randn(1, node_config.max_tokens, backbone_hidden_size) * 0.02
            )
        else:
            self.pos_embedding = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim)

        Returns:
            Projected embeddings of shape (batch, seq_len, hidden_size)
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        x = self.proj(x)

        if self.pos_embedding is not None:
            seq_len = x.shape[1]
            x = x + self.pos_embedding[:, :seq_len, :]

        return x


class MoTOutputHead(nn.Module):
    """Output head for action prediction from a node."""

    def __init__(self, node_config: MoTNodeConfig, backbone_hidden_size: int):
        super().__init__()
        self.node_config = node_config

        if node_config.output_head == "linear":
            self.head = nn.Linear(backbone_hidden_size, node_config.output_dim)
        elif node_config.output_head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(backbone_hidden_size, backbone_hidden_size),
                nn.SiLU(),
                nn.Linear(backbone_hidden_size, node_config.output_dim),
            )
        elif node_config.output_head == "none":
            self.head = None
        else:
            raise ValueError(f"Unknown output head type: {node_config.output_head}")

    def forward(self, x: Tensor) -> Tensor | None:
        if self.head is None:
            return None
        return self.head(x)


class MoTBackboneWrapper(nn.Module):
    """
    Wraps a HuggingFace transformer backbone for use in MoT.

    This wrapper provides a unified interface for different backbone types
    (PaliGemma, Gemma, LLaMA, etc.) and handles the specifics of each.
    """

    def __init__(self, backbone_config: MoTBackboneConfig):
        super().__init__()
        self.backbone_config = backbone_config
        self.model = None
        self.vision_encoder = None

        self._init_backbone()

    def _init_backbone(self):
        """Initialize the underlying transformer backbone."""
        if self.backbone_config.backbone_type == BackboneType.PALIGEMMA:
            self._init_paligemma()
        elif self.backbone_config.backbone_type == BackboneType.GEMMA:
            self._init_gemma()
        elif self.backbone_config.backbone_type == BackboneType.LLAMA:
            self._init_llama()
        elif self.backbone_config.backbone_type == BackboneType.BAGEL:
            self._init_bagel()
        elif self.backbone_config.backbone_type == BackboneType.QWEN:
            self._init_qwen()
        else:
            raise ValueError(f"Unknown backbone type: {self.backbone_config.backbone_type}")

    def _init_paligemma(self):
        """Initialize PaliGemma backbone."""
        cfg = self.backbone_config

        hf_config = CONFIG_MAPPING["paligemma"]()
        hf_config._vocab_size = cfg.vocab_size
        hf_config.image_token_index = cfg.vocab_size
        hf_config.text_config.hidden_size = cfg.hidden_size
        hf_config.text_config.intermediate_size = cfg.intermediate_size
        hf_config.text_config.num_attention_heads = cfg.num_attention_heads
        hf_config.text_config.head_dim = cfg.head_dim
        hf_config.text_config.num_hidden_layers = cfg.num_hidden_layers
        hf_config.text_config.num_key_value_heads = cfg.num_key_value_heads
        hf_config.text_config.hidden_activation = cfg.hidden_activation
        hf_config.text_config.torch_dtype = "float32"
        hf_config.text_config.vocab_size = cfg.vocab_size
        hf_config.text_config.use_adarms = cfg.use_adarms
        hf_config.text_config.adarms_cond_dim = cfg.adarms_cond_dim

        if cfg.vision_hidden_size:
            hf_config.vision_config.hidden_size = cfg.vision_hidden_size
            hf_config.vision_config.intermediate_size = 4304
            hf_config.vision_config.projection_dim = cfg.hidden_size
            hf_config.vision_config.projector_hidden_act = "gelu_fast"
        hf_config.vision_config.image_size = cfg.vision_image_size
        hf_config.vision_config.patch_size = cfg.vision_patch_size

        for key, value in cfg.hf_config_kwargs.items():
            setattr(hf_config, key, value)

        self.model = PaliGemmaForConditionalGeneration(config=hf_config)
        self.vision_encoder = self.model.vision_tower
        self.language_model = self.model.language_model

        self._apply_precision(cfg.dtype)

    def _init_gemma(self):
        """Initialize Gemma backbone."""
        cfg = self.backbone_config

        hf_config = CONFIG_MAPPING["gemma"](
            head_dim=cfg.head_dim,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_attention_heads=cfg.num_attention_heads,
            num_hidden_layers=cfg.num_hidden_layers,
            num_key_value_heads=cfg.num_key_value_heads,
            vocab_size=cfg.vocab_size,
            hidden_activation=cfg.hidden_activation,
            torch_dtype="float32",
            use_adarms=cfg.use_adarms,
            adarms_cond_dim=cfg.adarms_cond_dim,
        )

        for key, value in cfg.hf_config_kwargs.items():
            setattr(hf_config, key, value)

        self.model = GemmaForCausalLM(config=hf_config)
        self.model.model.embed_tokens = None  # We use external embeddings
        self.language_model = self.model.model

        self._apply_precision(cfg.dtype)

    def _init_llama(self):
        """Initialize LLaMA backbone."""
        # Similar to Gemma but with LLaMA-specific config
        try:
            from transformers.models.llama.modeling_llama import \
                LlamaForCausalLM

            cfg = self.backbone_config
            hf_config = CONFIG_MAPPING["llama"](
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                num_attention_heads=cfg.num_attention_heads,
                num_hidden_layers=cfg.num_hidden_layers,
                num_key_value_heads=cfg.num_key_value_heads,
                vocab_size=cfg.vocab_size,
            )

            self.model = LlamaForCausalLM(config=hf_config)
            self.language_model = self.model.model
            self._apply_precision(cfg.dtype)
        except ImportError as e:
            raise ImportError("LLaMA backbone requires transformers with LLaMA support") from e

    def _init_bagel(self):
        """Initialize Bagel backbone (placeholder for future implementation)."""
        # Bagel is a newer model - this is a placeholder for when it's available
        # For now, we can use a similar architecture to Gemma
        logger.warning("Bagel backbone not yet implemented, using Gemma as fallback")
        self._init_gemma()

    def _init_qwen(self):
        """Initialize Qwen backbone."""
        try:
            from transformers.models.qwen2.modeling_qwen2 import \
                Qwen2ForCausalLM

            cfg = self.backbone_config
            hf_config = CONFIG_MAPPING["qwen2"](
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                num_attention_heads=cfg.num_attention_heads,
                num_hidden_layers=cfg.num_hidden_layers,
                num_key_value_heads=cfg.num_key_value_heads,
                vocab_size=cfg.vocab_size,
            )

            self.model = Qwen2ForCausalLM(config=hf_config)
            self.language_model = self.model.model
            self._apply_precision(cfg.dtype)
        except ImportError as e:
            raise ImportError("Qwen backbone requires transformers with Qwen2 support") from e

    def _apply_precision(self, dtype: str):
        """Apply precision settings to the model."""
        if dtype == "bfloat16":
            self.model.to(dtype=torch.bfloat16)
            # Keep certain params in float32 for stability
            for name, param in self.model.named_parameters():
                if any(s in name for s in ["layernorm", "norm", "embedding"]):
                    param.data = param.data.to(dtype=torch.float32)
        elif dtype == "float32":
            self.model.to(dtype=torch.float32)

    @property
    def hidden_size(self) -> int:
        return self.backbone_config.hidden_size

    @property
    def num_layers(self) -> int:
        return self.backbone_config.num_hidden_layers

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        """Embed language tokens."""
        if hasattr(self, 'language_model') and hasattr(self.language_model, 'embed_tokens'):
            if self.language_model.embed_tokens is not None:
                return self.language_model.embed_tokens(tokens)
        if hasattr(self.model, 'embed_tokens'):
            return self.model.embed_tokens(tokens)
        raise ValueError("No token embedding layer found")

    def embed_image(self, image: Tensor) -> Tensor:
        """Embed image through vision encoder."""
        if hasattr(self.model, 'get_image_features'):
            return self.model.get_image_features(image)
        if self.vision_encoder is None:
            raise ValueError("This backbone does not have a vision encoder")
        return self.vision_encoder(image)

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """
        Full training forward pass with flow matching loss.
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # Flow matching interpolation
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed inputs
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time
        )

        # Convert to appropriate dtype (Optional check, kept from your code)
        vlm_backbone = list(self.backbones.values())[0]
        if hasattr(vlm_backbone, 'language_model') and vlm_backbone.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # Create token blocks
        token_blocks = [
            TokenBlock(
                node_name="vlm",
                embeddings=prefix_embs,
                pad_mask=prefix_pad_masks,
                att_mask=prefix_att_masks,
            ),
            TokenBlock(
                node_name="action_expert",
                embeddings=suffix_embs,
                pad_mask=suffix_pad_masks,
                att_mask=suffix_att_masks,
            ),
        ]

        # 准备 AdaRMS 条件（如果存在）
        # 需要映射到 backbone name
        adarms_conds = {}
        if adarms_cond is not None and "action_expert" in self.node_to_backbone:
            expert_backbone_name = self.node_to_backbone["action_expert"]
            adarms_conds[expert_backbone_name] = adarms_cond

        # 调用异构联合注意力
        output_blocks, _ = self.joint_attention(
            token_blocks,
            use_cache=False,
            adarms_conds=adarms_conds
        )

        # 提取 Action 输出 (找到 action_expert 对应的块)
        suffix_out = None
        for block in output_blocks:
            if block.node_name == "action_expert":
                suffix_out = block.embeddings
                break

        if suffix_out is None:
            raise ValueError("Action expert output not found!")

        # 截取最后 chunk_size 个 token
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)

        v_t = self.action_out_proj(suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")


class MoTJointLayerAttention(nn.Module):
    """
    Implements joint-layer attention across multiple backbones with heterogeneous hidden sizes.

    This is the key component that enables multiple transformers (e.g., PaliGemma 2B with 2048
    hidden_size and Gemma 300M expert with 1024 hidden_size) to share attention computation
    in a unified head space while maintaining separate parameters.

    The key insight is that while hidden_sizes differ (2048 vs 1024), both models share:
    - num_heads = 8
    - head_dim = 256
    - head_space = num_heads * head_dim = 2048

    Architecture:
    1. Phase 1 (QKV Projection): Each backbone projects from its hidden_size to head space
       - VLM: [B, seq_vlm, 2048] -> Q,K,V: [B, 8, seq_vlm, 256]
       - Expert: [B, seq_exp, 1024] -> Q,K,V: [B, 8, seq_exp, 256]

    2. Phase 2 (Global Attention): Concatenate in sequence dim and compute attention
       - Q: [B, 8, seq_vlm + seq_exp, 256]
       - K, V: same shape
       - Apply RoPE, then attention with mask

    3. Phase 3 (Output Projection): Split and project back to each hidden_size
       - VLM slice: [B, seq_vlm, 2048] -> o_proj -> [B, seq_vlm, 2048]
       - Expert slice: [B, seq_exp, 2048] -> o_proj -> [B, seq_exp, 1024]

    4. Phase 4 (Independent FFN): Each backbone runs its own FFN
    """

    def __init__(
        self,
        backbones: dict[str, MoTBackboneWrapper],
        node_to_backbone: dict[str, str],
        config: MoTConfig,
    ):
        super().__init__()
        self.backbones = nn.ModuleDict(backbones)
        self.node_to_backbone = node_to_backbone
        self.config = config
        self.mask_builder = MoTAttentionMaskBuilder(config)

        # Verify all backbones have same number of layers
        num_layers = list(backbones.values())[0].num_layers
        for name, backbone in backbones.items():
            if backbone.num_layers != num_layers:
                raise ValueError(
                    f"All backbones must have same number of layers. "
                    f"Got {num_layers} for first backbone but {backbone.num_layers} for {name}"
                )
        self.num_layers = num_layers

        # Verify all backbones have compatible head_dim and num_heads
        # This is critical for heterogeneous joint attention
        first_backbone = list(backbones.values())[0]
        first_cfg = first_backbone.backbone_config
        self.num_heads = first_cfg.num_attention_heads
        self.head_dim = first_cfg.head_dim
        self.num_kv_heads = first_cfg.num_key_value_heads

        for name, backbone in backbones.items():
            cfg = backbone.backbone_config
            if cfg.num_attention_heads != self.num_heads:
                raise ValueError(
                    f"All backbones must have same num_attention_heads for joint attention. "
                    f"Expected {self.num_heads} but {name} has {cfg.num_attention_heads}"
                )
            if cfg.head_dim != self.head_dim:
                raise ValueError(
                    f"All backbones must have same head_dim for joint attention. "
                    f"Expected {self.head_dim} but {name} has {cfg.head_dim}"
                )

        # Cache backbone order for consistent processing
        self.backbone_order = list(backbones.keys())

        # Get rotary embeddings from first backbone (they're shared since head_dim is same)
        self._rotary_emb = None

    def _get_rotary_emb(self):
        """Get the rotary embedding module from any backbone."""
        if self._rotary_emb is None:
            for backbone in self.backbones.values():
                if hasattr(backbone, 'language_model') and hasattr(backbone.language_model, 'rotary_emb'):
                    self._rotary_emb = backbone.language_model.rotary_emb
                    break
                elif hasattr(backbone, 'model') and hasattr(backbone.model, 'language_model'):
                    self._rotary_emb = backbone.model.language_model.rotary_emb
                    break
        return self._rotary_emb

    def _get_layer(self, backbone: MoTBackboneWrapper, layer_idx: int) -> nn.Module:
        """Get a specific layer from a backbone."""
        if hasattr(backbone, 'language_model') and hasattr(backbone.language_model, 'layers'):
            return backbone.language_model.layers[layer_idx]
        elif hasattr(backbone, 'model') and hasattr(backbone.model, 'layers'):
            return backbone.model.layers[layer_idx]
        else:
            raise ValueError(f"Cannot find layers in backbone")

    def _get_norm(self, backbone: MoTBackboneWrapper) -> nn.Module:
        """Get the final norm layer from a backbone."""
        if hasattr(backbone, 'language_model') and hasattr(backbone.language_model, 'norm'):
            return backbone.language_model.norm
        elif hasattr(backbone, 'model') and hasattr(backbone.model, 'norm'):
            return backbone.model.norm
        else:
            raise ValueError(f"Cannot find norm in backbone")

    def forward(
        self,
        token_blocks: list[TokenBlock],
        past_key_values: dict[str, Any] | None = None,
        use_cache: bool = False,
        adarms_conds: dict[str, Tensor] | None = None,
    ) -> tuple[list[TokenBlock], dict[str, Any] | None]:
        """
        Forward pass through heterogeneous joint-layer attention.

        This processes all token blocks through their respective backbones
        while allowing attention between blocks according to flow config,
        even when backbones have different hidden sizes.
        """
        if adarms_conds is None:
            adarms_conds = {}

        # Group blocks by backbone, preserving order
        backbone_blocks: dict[str, list[tuple[int, TokenBlock]]] = {
            name: [] for name in self.backbone_order
        }
        for idx, block in enumerate(token_blocks):
            node_config = self.config.get_node(block.node_name)
            backbone_name = node_config.backbone_name
            backbone_blocks[backbone_name].append((idx, block))

        # Build global attention mask (this works regardless of hidden sizes)
        attention_mask = self.mask_builder.build_attention_mask(token_blocks)
        attention_mask_4d = self.mask_builder.prepare_4d_mask(attention_mask)

        # Build position IDs for RoPE
        combined_pad_mask = torch.cat([b.pad_mask for b in token_blocks], dim=1)
        position_ids = torch.cumsum(combined_pad_mask.long(), dim=1) - 1

        # Get rotary embeddings
        rotary_emb = self._get_rotary_emb()

        # Initialize hidden states per backbone
        # Each backbone maintains its own hidden size
        hidden_states_per_backbone: dict[str, list[tuple[int, Tensor]]] = {}
        for backbone_name, indexed_blocks in backbone_blocks.items():
            hidden_states_per_backbone[backbone_name] = [
                (idx, block.embeddings) for idx, block in indexed_blocks
            ]

        # Process through layers
        new_past_key_values = {} if use_cache else None

        for layer_idx in range(self.num_layers):
            hidden_states_per_backbone = self._process_layer_heterogeneous(
                layer_idx=layer_idx,
                hidden_states_per_backbone=hidden_states_per_backbone,
                attention_mask_4d=attention_mask_4d,
                position_ids=position_ids,
                adarms_conds=adarms_conds,
                rotary_emb=rotary_emb,
            )

        # Apply final normalization per backbone
        for backbone_name, indexed_hidden_states in hidden_states_per_backbone.items():
            backbone = self.backbones[backbone_name]
            norm = self._get_norm(backbone)
            adarms_cond = adarms_conds.get(backbone_name)

            new_indexed_hidden_states = []
            for idx, hidden_states in indexed_hidden_states:
                if hasattr(norm, 'cond') and adarms_cond is not None:
                    normed, _ = norm(hidden_states, cond=adarms_cond)
                else:
                    normed = norm(hidden_states)
                    if isinstance(normed, tuple):
                        hidden_state, _ = normed
                    else:
                        hidden_state = normed
                new_indexed_hidden_states.append((idx, hidden_state))

            hidden_states_per_backbone[backbone_name] = new_indexed_hidden_states

        # Reconstruct output blocks in original order
        output_blocks = [None] * len(token_blocks)
        for backbone_name, indexed_hidden_states in hidden_states_per_backbone.items():
            for idx, hidden_states in indexed_hidden_states:
                original_block = token_blocks[idx]
                new_block = TokenBlock(
                    node_name=original_block.node_name,
                    embeddings=hidden_states,
                    pad_mask=original_block.pad_mask,
                    att_mask=original_block.att_mask,
                    start_idx=original_block.start_idx,
                    end_idx=original_block.end_idx,
                )
                output_blocks[idx] = new_block

        return output_blocks, new_past_key_values

    def _process_layer_heterogeneous(
        self,
        layer_idx: int,
        hidden_states_per_backbone: dict[str, list[tuple[int, Tensor]]],
        attention_mask_4d: Tensor,
        position_ids: Tensor,
        adarms_conds: dict[str, Tensor],
        rotary_emb: nn.Module | None,
    ) -> dict[str, list[tuple[int, Tensor]]]:
        """
        Process a single layer with heterogeneous hidden sizes using explicit
        joint attention in head space.

        Phase 1: QKV projection from disparate hidden_size to shared head_space
        Phase 2: Global attention in head_space with RoPE
        Phase 3: Output projection back to disparate hidden_size
        Phase 4: Independent FFN per backbone
        """
        batch_size = attention_mask_4d.shape[0]
        device = attention_mask_4d.device

        # =================================================================
        # Phase 1: QKV Projection
        # Each backbone projects from its hidden_size to head space
        # =================================================================
        all_query_states = []  # Will be concatenated along seq dim
        all_key_states = []
        all_value_states = []

        # Store info needed for Phase 3 & 4
        backbone_info = []  # (backbone_name, layer, seq_len, gate, original_hidden_states, original_indices)

        # Track sequence positions for slicing attention output
        seq_start = 0

        for backbone_name in self.backbone_order:
            indexed_hidden_states = hidden_states_per_backbone.get(backbone_name, [])
            if not indexed_hidden_states:
                continue

            backbone = self.backbones[backbone_name]
            layer = self._get_layer(backbone, layer_idx)
            adarms_cond = adarms_conds.get(backbone_name)

            # Concatenate this backbone's hidden states
            indices = [idx for idx, _ in indexed_hidden_states]
            hidden_list = [h for _, h in indexed_hidden_states]
            hidden_states = torch.cat(hidden_list, dim=1)  # [B, seq_backbone, hidden_size]
            seq_len = hidden_states.shape[1]

            # Apply input layer norm (with optional AdaRMS)
            if hasattr(layer, 'input_layernorm'):
                if hasattr(layer.input_layernorm, 'cond') and adarms_cond is not None:
                    normed, gate = layer.input_layernorm(hidden_states, cond=adarms_cond)
                else:
                    normed = layer.input_layernorm(hidden_states)
                    if isinstance(normed, tuple):
                        normed, gate = normed
                    else:
                        gate = None
            else:
                normed = hidden_states
                gate = None

            # Compute Q, K, V using this layer's projection weights
            # Each backbone has its own q_proj, k_proj, v_proj that map:
            #   hidden_size -> num_heads * head_dim (for q)
            #   hidden_size -> num_kv_heads * head_dim (for k, v)
            input_shape = normed.shape[:-1]  # [B, seq]

            # Q: [B, seq, hidden_size] -> [B, seq, num_heads * head_dim] -> [B, num_heads, seq, head_dim]
            q = layer.self_attn.q_proj(normed)
            q = q.view(*input_shape, self.num_heads, self.head_dim).transpose(1, 2)

            # K, V: similar but with num_kv_heads
            k = layer.self_attn.k_proj(normed)
            k = k.view(*input_shape, self.num_kv_heads, self.head_dim).transpose(1, 2)

            v = layer.self_attn.v_proj(normed)
            v = v.view(*input_shape, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # Store for concatenation
            all_query_states.append(q)
            all_key_states.append(k)
            all_value_states.append(v)

            # Store info for later phases
            # Track per-block sequence lengths for splitting
            block_seq_lens = [h.shape[1] for _, h in indexed_hidden_states]
            backbone_info.append({
                'backbone_name': backbone_name,
                'layer': layer,
                'seq_start': seq_start,
                'seq_len': seq_len,
                'block_seq_lens': block_seq_lens,
                'gate': gate,
                'hidden_states': hidden_states,
                'indices': indices,
                'adarms_cond': adarms_cond,
            })

            seq_start += seq_len

        # =================================================================
        # Phase 2: Global Attention in Head Space
        # Concatenate Q, K, V along sequence dimension and compute attention
        # =================================================================

        # Concatenate: [B, num_heads, total_seq, head_dim]
        query_states = torch.cat(all_query_states, dim=2)
        key_states = torch.cat(all_key_states, dim=2)
        value_states = torch.cat(all_value_states, dim=2)

        # Apply RoPE (Rotary Position Embeddings)
        if rotary_emb is not None:
            # Create dummy tensor for RoPE computation
            dummy_tensor = torch.zeros(
                query_states.shape[0],
                query_states.shape[2],
                query_states.shape[-1],
                device=query_states.device,
                dtype=query_states.dtype,
            )
            cos, sin = rotary_emb(dummy_tensor, position_ids)
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

        # Get scaling factor from first layer
        first_backbone = self.backbones[self.backbone_order[0]]
        first_layer = self._get_layer(first_backbone, layer_idx)
        scaling = first_layer.self_attn.scaling

        # Compute attention using eager_attention_forward
        # This handles the attention computation with the mask
        attn_output, _ = modeling_gemma.eager_attention_forward(
            first_layer.self_attn,  # Module for compatibility
            query_states,
            key_states,
            value_states,
            attention_mask_4d,
            scaling,
        )

        # Reshape attention output: [B, total_seq, num_heads * head_dim]
        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim)

        # =================================================================
        # Phase 3: Output Projection back to Disparate Hidden Sizes
        # Each backbone uses its o_proj to map from head_space to hidden_size
        # =================================================================

        output_hidden_states_per_backbone: dict[str, list[tuple[int, Tensor]]] = {}

        for info in backbone_info:
            backbone_name = info['backbone_name']
            layer = info['layer']
            seq_start = info['seq_start']
            seq_len = info['seq_len']
            gate = info['gate']
            original_hidden_states = info['hidden_states']
            indices = info['indices']
            block_seq_lens = info['block_seq_lens']
            adarms_cond = info['adarms_cond']

            # Slice this backbone's attention output
            backbone_attn_output = attn_output[:, seq_start:seq_start + seq_len, :]

            # Ensure dtype matches o_proj weights
            if backbone_attn_output.dtype != layer.self_attn.o_proj.weight.dtype:
                backbone_attn_output = backbone_attn_output.to(layer.self_attn.o_proj.weight.dtype)

            # Output projection: [B, seq, num_heads * head_dim] -> [B, seq, hidden_size]
            attn_projected = layer.self_attn.o_proj(backbone_attn_output)

            # First residual connection (with optional gating for AdaRMS)
            if gate is not None and hasattr(modeling_gemma, '_gated_residual'):
                hidden_states = modeling_gemma._gated_residual(original_hidden_states, attn_projected, gate)
            else:
                hidden_states = original_hidden_states + attn_projected

            # =================================================================
            # Phase 4: Independent FFN per Backbone
            # =================================================================

            # Save hidden states before FFN for second residual
            after_first_residual = hidden_states.clone()

            # Post-attention layer norm
            if hasattr(layer, 'post_attention_layernorm'):
                if hasattr(layer.post_attention_layernorm, 'cond') and adarms_cond is not None:
                    normed, gate2 = layer.post_attention_layernorm(hidden_states, cond=adarms_cond)
                else:
                    normed = layer.post_attention_layernorm(hidden_states)
                    if isinstance(normed, tuple):
                        normed, gate2 = normed
                    else:
                        gate2 = None
            else:
                normed = hidden_states
                gate2 = None

            # Ensure dtype for MLP
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                normed = normed.to(dtype=torch.bfloat16)

            # FFN (MLP)
            mlp_output = layer.mlp(normed)

            # Second residual connection
            if gate2 is not None and hasattr(modeling_gemma, '_gated_residual'):
                hidden_states = modeling_gemma._gated_residual(after_first_residual, mlp_output, gate2)
            else:
                hidden_states = after_first_residual + mlp_output

            # Split back into individual blocks
            block_start = 0
            output_indexed_hidden_states = []
            for block_idx, block_seq_len in enumerate(block_seq_lens):
                block_hidden = hidden_states[:, block_start:block_start + block_seq_len, :]
                original_idx = indices[block_idx]
                output_indexed_hidden_states.append((original_idx, block_hidden))
                block_start += block_seq_len

            output_hidden_states_per_backbone[backbone_name] = output_indexed_hidden_states

        return output_hidden_states_per_backbone


class MoTModel(nn.Module):
    """
    Core MoT model that combines all components.

    This is the main model class that:
    - Creates and manages multiple transformer backbones
    - Routes inputs through the appropriate projections
    - Performs joint-layer attention
    - Produces outputs through the appropriate heads
    """

    def __init__(self, config: MoTConfig):
        super().__init__()
        self.config = config

        # Initialize backbones
        self.backbones: dict[str, MoTBackboneWrapper] = nn.ModuleDict()
        for backbone_cfg in config.backbones:
            self.backbones[backbone_cfg.name] = MoTBackboneWrapper(backbone_cfg)

        # Build node-to-backbone mapping
        self.node_to_backbone: dict[str, str] = {}
        for node in config.nodes:
            if node.backbone_name:
                self.node_to_backbone[node.name] = node.backbone_name

        # Initialize input projections
        self.input_projections: dict[str, MoTInputProjection] = nn.ModuleDict()
        for node in config.nodes:
            if node.is_input and node.backbone_name:
                backbone = self.backbones[node.backbone_name]
                self.input_projections[node.name] = MoTInputProjection(
                    node, backbone.hidden_size
                )

        # Initialize output heads
        self.output_heads: dict[str, MoTOutputHead] = nn.ModuleDict()
        for node in config.nodes:
            if node.is_output and node.backbone_name:
                backbone = self.backbones[node.backbone_name]
                self.output_heads[node.name] = MoTOutputHead(node, backbone.hidden_size)

        # Initialize joint-layer attention
        self.joint_attention = MoTJointLayerAttention(
            dict(self.backbones),
            self.node_to_backbone,
            config,
        )

        # Flow matching components (for flow_matching decoding mode)
        if config.decoding_mode == DecodingMode.FLOW_MATCHING:
            self._init_flow_matching_components()

        # Mask builder for standalone use
        self.mask_builder = MoTAttentionMaskBuilder(config)

    def _init_flow_matching_components(self):
        """Initialize components needed for flow matching."""
        # Find the action expert backbone
        action_node = None
        for node in self.config.nodes:
            if node.is_output:
                action_node = node
                break

        if action_node is None:
            return

        if action_node.backbone_name not in self.backbones:
            return
        backbone = self.backbones[action_node.backbone_name]

        hidden_size = backbone.hidden_size

        # Action input projection
        self.action_in_proj = nn.Linear(self.config.max_action_dim, hidden_size)

        # Action output projection
        self.action_out_proj = nn.Linear(hidden_size, self.config.max_action_dim)

        # State projection
        self.state_proj = nn.Linear(self.config.max_state_dim, hidden_size)

        # Time MLP for conditioning
        self.action_time_mlp_in = nn.Linear(2 * hidden_size, hidden_size)
        self.action_time_mlp_out = nn.Linear(hidden_size, hidden_size)

    def sample_noise(self, shape: tuple, device: torch.device) -> Tensor:
        """Sample Gaussian noise for flow matching."""
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample time values for flow matching training."""
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            batch_size,
            device,
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Embed prefix inputs (images + language) for VLM backbone.

        Returns:
            Tuple of (embeddings, padding_masks, attention_masks)
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Find VLM backbone
        vlm_backbone = None
        for name, backbone in self.backbones.items():
            if backbone.backbone_config.backbone_type == BackboneType.PALIGEMMA:
                vlm_backbone = backbone
                break

        if vlm_backbone is None:
            raise ValueError("No VLM backbone found for prefix embedding")

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = vlm_backbone.embed_image(img)
            batch_size, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(batch_size, num_img_embs))
            att_masks.extend([0] * num_img_embs)

        # Process language
        lang_emb = vlm_backbone.embed_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks.extend([0] * num_lang_embs)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        batch_size = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(batch_size, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(
        self,
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """
        Embed suffix inputs (state + noisy actions + time) for action expert.

        Returns:
            Tuple of (embeddings, padding_masks, attention_masks, adarms_cond)
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Ensure correct dtype for state projection
        if hasattr(self, 'state_proj') and self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        # State embedding
        state_emb = self.state_proj(state)
        embs.append(state_emb[:, None, :])

        batch_size = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks.append(1)

        # Time embedding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Action embedding with time fusion
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        # MLP fusion
        x = self.action_time_mlp_in(action_time_emb)
        x = F.silu(x)
        action_time_emb = self.action_time_mlp_out(x)

        adarms_cond = None  # Can be computed if needed

        embs.append(action_time_emb)
        batch_size, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(batch_size, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Attention pattern: state can see everything, actions are causal
        att_masks.extend([1] + [0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=device)
        att_masks = att_masks[None, :].expand(batch_size, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """
        Full training forward pass with flow matching loss.
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # Flow matching interpolation
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed inputs
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time
        )

        # Convert to appropriate dtype
        vlm_backbone = list(self.backbones.values())[0]
        if hasattr(vlm_backbone, 'language_model') and vlm_backbone.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        token_blocks = [
            TokenBlock(
                node_name="vlm",
                embeddings=prefix_embs,
                pad_mask=prefix_pad_masks,
                att_mask=prefix_att_masks,
            ),
            TokenBlock(
                node_name="action_expert",
                embeddings=suffix_embs,
                pad_mask=suffix_pad_masks,
                att_mask=suffix_att_masks,
            ),
        ]

        # 准备 AdaRMS 条件
        adarms_conds = {}
        if adarms_cond is not None and "action_expert" in self.node_to_backbone:
            expert_backbone_name = self.node_to_backbone["action_expert"]
            adarms_conds[expert_backbone_name] = adarms_cond

        # 调用异构联合注意力 (不再使用 torch.cat)
        output_blocks, _ = self.joint_attention(
            token_blocks,
            use_cache=False,
            adarms_conds=adarms_conds
        )

        # 提取 Action 输出
        suffix_out = None
        for block in output_blocks:
            if block.node_name == "action_expert":
                suffix_out = block.embeddings
                break

        if suffix_out is None:
            raise ValueError("Action expert output not found!")

        # 截取最后 chunk_size 个 token
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)

        v_t = self.action_out_proj(suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        **kwargs,
    ) -> Tensor:
        """
        Sample actions using flow matching denoising.
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        batch_size = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (batch_size, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Embed prefix once
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(batch_size)

            v_t = self._denoise_step(
                state=state,
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                x_t=x_t,
                timestep=time_tensor,
            )

            x_t = x_t + dt * v_t

        return x_t

    def _denoise_step(
        self,
        state: Tensor,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """Single denoising step."""
        # Embed suffix (state + action + time)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, timestep
        )

        # Convert dtypes if needed
        vlm_backbone = list(self.backbones.values())[0]
        if hasattr(vlm_backbone, 'language_model') and vlm_backbone.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            # prefix_embs 应该已经是正确的 dtype，但确保 suffix 也是
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # 组装 Token Blocks (复用传入的 prefix_embs)
        token_blocks = [
            TokenBlock(
                node_name="vlm",
                embeddings=prefix_embs,
                pad_mask=prefix_pad_masks,
                att_mask=prefix_att_masks,
            ),
            TokenBlock(
                node_name="action_expert",
                embeddings=suffix_embs,
                pad_mask=suffix_pad_masks,
                att_mask=suffix_att_masks,
            ),
        ]

        # 准备 AdaRMS
        adarms_conds = {}
        if adarms_cond is not None and "action_expert" in self.node_to_backbone:
            expert_backbone_name = self.node_to_backbone["action_expert"]
            adarms_conds[expert_backbone_name] = adarms_cond

        # 调用 Joint Attention
        output_blocks, _ = self.joint_attention(
            token_blocks,
            use_cache=False, # 推理时暂不使用 Cache 以确保异构架构的正确性
            adarms_conds=adarms_conds
        )

        # 提取输出
        suffix_out = None
        for block in output_blocks:
            if block.node_name == "action_expert":
                suffix_out = block.embeddings
                break

        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)

        return self.action_out_proj(suffix_out)

# ==================== Policy Class ====================

class MoTPolicy(PreTrainedPolicy):
    """
    MoT (Mixture of Transformers) Policy for LeRobot.

    This is the main policy class that wraps MoTModel and provides
    the standard LeRobot policy interface.
    """

    config_class = MoTConfig
    name = "mot"

    def __init__(self, config: MoTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core model
        self.model = MoTModel(config)
        self.model.to(config.device)

        self.reset()

    @property
    def device(self) -> torch.device:
        """Get the device on which the model is currently loaded."""
        return next(self.parameters()).device

    def reset(self):
        """Reset internal state."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model."""
        images = []
        img_masks = []
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]

            if img.device != device:
                img = img.to(device)
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            is_channels_first = img.shape[1] == 3
            if is_channels_first:
                img = img.permute(0, 2, 3, 1)

            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize to [-1, 1]
            img = img * 2.0 - 1.0

            if is_channels_first:
                img = img.permute(0, 3, 1, 2)

            images.append(img)
            batch_size = img.shape[0]
            mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Add empty images for missing cameras
        for _ in missing_img_keys:
            img = torch.ones_like(images[-1]) * -1
            mask = torch.zeros_like(img_masks[-1])
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Prepare state tensor."""
        return pad_vector(batch[OBS_STATE], self.config.max_state_dim)

    def prepare_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Prepare action tensor."""
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, :self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict a chunk of actions."""
        self.eval()

        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = self.prepare_state(batch)

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, **kwargs
        )

        # Unpad actions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Training forward pass."""
        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions)

        # Truncate to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def get_optim_params(self) -> dict:
        """Get parameters for optimization."""
        return self.parameters()

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """Return default PEFT target modules."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(.*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Load a pretrained MoT policy."""
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model = cls(config, **kwargs)

        # Load weights (implementation similar to PI0Policy)
        # ... weight loading logic ...

        return model
