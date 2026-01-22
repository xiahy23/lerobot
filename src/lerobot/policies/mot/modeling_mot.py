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
                                                    EmbedType,
                                                    MoTBackboneConfig,
                                                    MoTConfig, MoTFlowConfig,
                                                    MoTNodeConfig,
                                                    NodeInputConfig, NodeType)
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


# ==================== Generalized Node Input Embedder ====================

class NodeInputEmbedder(nn.Module):
    """
    Generalized input embedder for any node type.

    This provides a configurable per-node embedding system where each node
    defines its own embedding logic through NodeInputConfig.

    Supported embed types:
    - VISION: Image embedding via vision encoder
    - LANGUAGE: Language token embedding
    - STATE: Linear projection of state vector
    - ACTION: Action embedding with optional time fusion for flow matching
    - FLOW_TIME: Standalone timestep embedding
    - CUSTOM: User-defined embedding function
    """

    def __init__(
        self,
        node_config: MoTNodeConfig,
        backbone: "MoTBackboneWrapper",
        config: MoTConfig,
    ):
        super().__init__()
        self.node_config = node_config
        self.backbone = backbone
        self.config = config
        self.input_config = node_config.input_config or NodeInputConfig()
        self.hidden_size = backbone.hidden_size

        self._init_embedders()

    def _init_embedders(self):
        """Initialize embedding components based on input config."""
        embed_type = self.input_config.embed_type

        if embed_type == EmbedType.VISION:
            # Vision embedding uses backbone's vision encoder
            # No additional components needed
            pass

        elif embed_type == EmbedType.LANGUAGE:
            # Language token embedding uses backbone's token embedder
            # Scale factor stored
            self.vocab_scale = self.input_config.vocab_scale

        elif embed_type == EmbedType.STATE:
            # Linear projection for state
            state_dim = self.input_config.state_dim or self.config.max_state_dim
            self.state_proj = nn.Linear(state_dim, self.hidden_size)

        elif embed_type == EmbedType.ACTION:
            # Action embedding with optional time fusion for flow matching
            action_dim = self.input_config.action_dim or self.config.max_action_dim

            # Action projection
            self.action_proj = nn.Linear(action_dim, self.hidden_size)

            if self.input_config.use_time_embedding:
                # Time-conditioned MLP fusion
                if self.input_config.use_mlp_fusion:
                    self.action_time_mlp_in = nn.Linear(2 * self.hidden_size, self.hidden_size)
                    self.action_time_mlp_out = nn.Linear(self.hidden_size, self.hidden_size)
                # Otherwise use simple addition (no extra layers needed)

        elif embed_type == EmbedType.FLOW_TIME:
            # Flow matching timestep embedding
            # Uses sinusoidal encoding, no learnable parameters needed
            pass

        elif embed_type == EmbedType.CUSTOM:
            # Custom embedding - no default initialization
            pass

    def forward(
        self,
        batch: dict[str, Tensor],
        time: Tensor | None = None,
        noisy_actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Embed inputs for this node.

        Args:
            batch: Input batch dictionary
            time: Time tensor for flow matching (optional)
            noisy_actions: Noisy action tensor for flow matching (optional)

        Returns:
            Tuple of (embeddings, pad_mask, att_mask)
        """
        embed_type = self.input_config.embed_type

        if embed_type == EmbedType.VISION:
            return self._embed_vision(batch)
        elif embed_type == EmbedType.LANGUAGE:
            return self._embed_tokens(batch)
        elif embed_type == EmbedType.STATE:
            return self._embed_state(batch)
        elif embed_type == EmbedType.ACTION:
            return self._embed_action_flow(batch, time, noisy_actions)
        elif embed_type == EmbedType.FLOW_TIME:
            return self._embed_flow_time(batch, time)
        else:
            raise ValueError(f"Unsupported embed type: {embed_type}")

    # ========== Public API methods for embed_all_nodes ==========

    def embed_vision(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Embed vision inputs (images).

        This is the public API used by embed_all_nodes for VISION embed type.

        Args:
            images: List of image tensors
            img_masks: List of image mask tensors (indicating valid images)

        Returns:
            Tuple of (embeddings, pad_mask, att_mask)
        """
        embs = []
        pad_masks = []
        att_masks = []

        device = next(self.backbone.parameters()).device

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.backbone.embed_image(img)
            batch_size, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(batch_size, num_img_embs))
            att_masks.extend([0] * num_img_embs)

        if not embs:
            raise ValueError(f"No images provided for vision node {self.node_config.name}")

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        batch_size = pad_masks.shape[0]
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)
        att_masks = att_masks[None, :].expand(batch_size, -1)

        return embs, pad_masks, att_masks

    def embed_language(
        self,
        tokens: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Embed language tokens.

        This is the public API used by embed_all_nodes for LANGUAGE embed type.

        Args:
            tokens: Token IDs tensor
            masks: Attention mask tensor

        Returns:
            Tuple of (embeddings, pad_mask, att_mask)
        """
        device = tokens.device

        lang_emb = self.backbone.embed_tokens(tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        batch_size, seq_len = lang_emb.shape[:2]

        # Attention pattern: bidirectional for language
        att_masks = torch.zeros(batch_size, seq_len, device=device)

        return lang_emb, masks, att_masks

    def embed_state(
        self,
        state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Embed state vector.

        This is the public API used by embed_all_nodes for STATE embed type.

        Args:
            state: State tensor

        Returns:
            Tuple of (embeddings, pad_mask, att_mask)
        """
        device = state.device
        batch_size = state.shape[0]

        # Pad state if needed
        state = pad_vector(state, self.config.max_state_dim)

        if hasattr(self, 'state_proj') and self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :]  # Add sequence dimension

        pad_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        att_mask = torch.ones(batch_size, 1, device=device)  # State is start of causal chain

        return state_emb, pad_mask, att_mask

    def embed_action(
        self,
        actions: Tensor,
        timestep: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Embed action tokens with optional time conditioning.

        This is the public API used by embed_all_nodes for ACTION embed type.
        Creates action embeddings with time fusion for flow matching.

        Args:
            actions: Action tensor (batch, chunk_size, action_dim)
            timestep: Optional timestep tensor for time conditioning

        Returns:
            Tuple of (embeddings, pad_mask, att_mask)
        """
        device = actions.device
        batch_size = actions.shape[0]

        # Action projection
        action_emb = self.action_proj(actions)

        if timestep is not None and self.input_config.use_time_embedding:
            # Time embedding
            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.hidden_size,
                min_period=self.config.min_period,
                max_period=self.config.max_period,
                device=device,
            )
            time_emb = time_emb.to(dtype=timestep.dtype)
            time_emb_expanded = time_emb[:, None, :].expand_as(action_emb)

            if self.input_config.use_mlp_fusion:
                action_time_emb = torch.cat([action_emb, time_emb_expanded], dim=2)
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)
                action_emb = self.action_time_mlp_out(x)
            else:
                action_emb = action_emb + time_emb_expanded

        seq_len = action_emb.shape[1]
        pad_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # Causal attention for action tokens
        att_mask = torch.zeros(batch_size, seq_len, device=device)

        return action_emb, pad_mask, att_mask

    def embed_flow_time(
        self,
        timestep: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Embed flow matching timestep as a standalone token.

        This is the public API used by embed_all_nodes for FLOW_TIME embed type.

        Args:
            timestep: Timestep tensor

        Returns:
            Tuple of (embeddings, pad_mask, att_mask)
        """
        device = timestep.device
        batch_size = timestep.shape[0]

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=device,
        )
        time_emb = time_emb.to(dtype=timestep.dtype)
        time_emb = time_emb[:, None, :]  # Add sequence dimension

        pad_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        att_mask = torch.zeros(batch_size, 1, device=device)

        return time_emb, pad_mask, att_mask

    def _embed_vision(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Embed vision inputs (images + optional language)."""
        embs = []
        pad_masks = []
        att_masks = []

        device = next(self.backbone.parameters()).device

        # Process images
        image_keys = self.input_config.image_keys or list(self.config.image_features.keys())

        for key in image_keys:
            if key not in batch:
                continue
            img = batch[key]
            img_emb = self.backbone.embed_image(img)
            batch_size, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            # Create image mask (all valid)
            mask = torch.ones(batch_size, num_img_embs, dtype=torch.bool, device=device)
            pad_masks.append(mask)
            # Bidirectional attention for images
            att_masks.extend([0] * num_img_embs)

        # Process language tokens if present
        if OBS_LANGUAGE_TOKENS in batch:
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch.get(OBS_LANGUAGE_ATTENTION_MASK)

            lang_emb = self.backbone.embed_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            lang_emb = lang_emb * math.sqrt(lang_emb_dim) * self.input_config.vocab_scale

            embs.append(lang_emb)
            if lang_masks is not None:
                pad_masks.append(lang_masks)
            else:
                pad_masks.append(torch.ones(lang_emb.shape[:2], dtype=torch.bool, device=device))

            num_lang_embs = lang_emb.shape[1]
            att_masks.extend([0] * num_lang_embs)

        if not embs:
            raise ValueError(f"No valid inputs found for vision node {self.node_config.name}")

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        batch_size = pad_masks.shape[0]
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=device)
        att_masks = att_masks[None, :].expand(batch_size, -1)

        return embs, pad_masks, att_masks

    def _embed_tokens(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Embed language tokens."""
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch.get(OBS_LANGUAGE_ATTENTION_MASK)
        device = lang_tokens.device

        lang_emb = self.backbone.embed_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim) * self.input_config.vocab_scale

        batch_size, seq_len = lang_emb.shape[:2]

        if lang_masks is None:
            lang_masks = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # Attention pattern
        if self.input_config.attention_pattern == "causal":
            att_masks = torch.ones(batch_size, seq_len, device=device)
        else:
            att_masks = torch.zeros(batch_size, seq_len, device=device)

        return lang_emb, lang_masks, att_masks

    def _embed_state(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Embed state vector."""
        state = batch[OBS_STATE]
        device = state.device
        batch_size = state.shape[0]

        # Pad state if needed
        state = pad_vector(state, self.config.max_state_dim)

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :]  # Add sequence dimension

        pad_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        att_mask = torch.zeros(batch_size, 1, device=device)

        return state_emb, pad_mask, att_mask

    def _embed_action_flow(
        self,
        batch: dict[str, Tensor],
        time: Tensor | None,
        noisy_actions: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Embed action + time for flow matching (used by action expert nodes).

        NOTE: This is the internal method for batch-based API.
        For the public API, use embed_action().

        This creates: [action_token_1, action_token_2, ..., action_token_n]
        (State should be handled by a separate STATE node if needed)
        """
        if time is None or noisy_actions is None:
            raise ValueError("ACTION embed type requires time and noisy_actions")

        device = time.device
        batch_size = time.shape[0]

        # Time embedding
        time_emb = create_sinusoidal_pos_embedding(
            time,
            self.hidden_size,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=device,
        )
        time_emb = time_emb.to(dtype=time.dtype)

        # Action embedding with time fusion
        action_emb = self.action_proj(noisy_actions)
        time_emb_expanded = time_emb[:, None, :].expand_as(action_emb)

        if self.input_config.use_mlp_fusion:
            action_time_emb = torch.cat([action_emb, time_emb_expanded], dim=2)
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            action_time_emb = self.action_time_mlp_out(x)
        else:
            action_time_emb = action_emb + time_emb_expanded

        action_seq_len = action_time_emb.shape[1]
        pad_mask = torch.ones(batch_size, action_seq_len, dtype=torch.bool, device=device)

        # Causal attention for action tokens
        att_mask = torch.zeros(batch_size, action_seq_len, device=device)

        return action_time_emb, pad_mask, att_mask

    def _embed_flow_time(
        self,
        batch: dict[str, Tensor],
        time: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed flow matching timestep as standalone token."""
        if time is None:
            raise ValueError("FLOW_TIME embed type requires time tensor")

        return self.embed_flow_time(time)

        return embs, pad_masks, att_masks

    def get_adarms_cond(
        self,
        time: Tensor | None = None,
        batch: dict[str, Tensor] | None = None,
    ) -> Tensor | None:
        """Get AdaRMS conditioning tensor if applicable."""
        # This can be extended based on backbone requirements
        return None


# ==================== Backbone Interface ====================

class BackboneInterface:
    """
    Interface that all backbone wrappers must implement.

    This ensures consistent behavior across different transformer types
    (PaliGemma, Gemma, LLaMA, Qwen, Bagel, etc.).
    """

    @property
    def hidden_size(self) -> int:
        """Return the hidden dimension of this backbone."""
        raise NotImplementedError

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        raise NotImplementedError

    @property
    def num_heads(self) -> int:
        """Return the number of attention heads."""
        raise NotImplementedError

    @property
    def head_dim(self) -> int:
        """Return the dimension of each attention head."""
        raise NotImplementedError

    @property
    def num_kv_heads(self) -> int:
        """Return the number of key/value heads (for GQA)."""
        raise NotImplementedError

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        """Embed language tokens."""
        raise NotImplementedError

    def embed_image(self, image: Tensor) -> Tensor:
        """Embed images (only for VLM backbones)."""
        raise NotImplementedError

    def get_layer(self, layer_idx: int) -> nn.Module:
        """Get a specific transformer layer."""
        raise NotImplementedError

    def get_norm(self) -> nn.Module:
        """Get the final normalization layer."""
        raise NotImplementedError

    def get_rotary_emb(self) -> nn.Module | None:
        """Get the rotary embedding module if available."""
        raise NotImplementedError

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        config: MoTBackboneConfig,
        **kwargs,
    ) -> "BackboneInterface":
        """Load pretrained weights."""
        raise NotImplementedError


class MoTBackboneWrapper(nn.Module, BackboneInterface):
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

    @property
    def num_heads(self) -> int:
        return self.backbone_config.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self.backbone_config.head_dim

    @property
    def num_kv_heads(self) -> int:
        return self.backbone_config.num_key_value_heads

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

    def get_layer(self, layer_idx: int) -> nn.Module:
        """Get a specific transformer layer."""
        if hasattr(self, 'language_model') and hasattr(self.language_model, 'layers'):
            return self.language_model.layers[layer_idx]
        elif hasattr(self, 'model') and hasattr(self.model, 'layers'):
            return self.model.layers[layer_idx]
        else:
            raise ValueError("Cannot find layers in backbone")

    def get_norm(self) -> nn.Module:
        """Get the final normalization layer."""
        if hasattr(self, 'language_model') and hasattr(self.language_model, 'norm'):
            return self.language_model.norm
        elif hasattr(self, 'model') and hasattr(self.model, 'norm'):
            return self.model.norm
        else:
            raise ValueError("Cannot find norm in backbone")

    def get_rotary_emb(self) -> nn.Module | None:
        """Get the rotary embedding module if available."""
        if hasattr(self, 'language_model') and hasattr(self.language_model, 'rotary_emb'):
            return self.language_model.rotary_emb
        elif hasattr(self, 'model') and hasattr(self.model, 'language_model'):
            lm = self.model.language_model
            if hasattr(lm, 'rotary_emb'):
                return lm.rotary_emb
        return None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        backbone_config: MoTBackboneConfig,
        **kwargs,
    ) -> "MoTBackboneWrapper":
        """
        Load a backbone from pretrained weights.

        Args:
            pretrained_path: Path to pretrained weights (local or HuggingFace hub)
            backbone_config: Configuration for the backbone
            **kwargs: Additional arguments for loading

        Returns:
            Initialized backbone wrapper with loaded weights
        """
        # Create wrapper with config
        wrapper = cls(backbone_config)

        # Load weights based on backbone type
        if backbone_config.backbone_type == BackboneType.PALIGEMMA:
            wrapper._load_paligemma_weights(pretrained_path, **kwargs)
        elif backbone_config.backbone_type == BackboneType.GEMMA:
            wrapper._load_gemma_weights(pretrained_path, **kwargs)
        elif backbone_config.backbone_type == BackboneType.LLAMA:
            wrapper._load_llama_weights(pretrained_path, **kwargs)
        elif backbone_config.backbone_type == BackboneType.QWEN:
            wrapper._load_qwen_weights(pretrained_path, **kwargs)
        else:
            logger.warning(f"No pretrained loading implemented for {backbone_config.backbone_type}")

        return wrapper

    def _load_paligemma_weights(self, pretrained_path: str, **kwargs):
        """Load PaliGemma pretrained weights."""
        from transformers import \
            PaliGemmaForConditionalGeneration as HFPaliGemma

        try:
            pretrained_model = HFPaliGemma.from_pretrained(
                pretrained_path,
                torch_dtype=torch.float32 if self.backbone_config.dtype == "float32" else torch.bfloat16,
                **kwargs,
            )

            # Copy weights
            missing, unexpected = self.model.load_state_dict(
                pretrained_model.state_dict(), strict=False
            )

            if missing:
                logger.warning(f"Missing keys when loading PaliGemma weights: {missing[:10]}...")
            if unexpected:
                logger.warning(f"Unexpected keys when loading PaliGemma weights: {unexpected[:10]}...")

            del pretrained_model

        except Exception as e:
            logger.error(f"Failed to load PaliGemma weights from {pretrained_path}: {e}")
            raise

    def _load_gemma_weights(self, pretrained_path: str, **kwargs):
        """Load Gemma pretrained weights."""
        from transformers import GemmaForCausalLM as HFGemma

        try:
            pretrained_model = HFGemma.from_pretrained(
                pretrained_path,
                torch_dtype=torch.float32 if self.backbone_config.dtype == "float32" else torch.bfloat16,
                **kwargs,
            )

            missing, unexpected = self.model.load_state_dict(
                pretrained_model.state_dict(), strict=False
            )

            if missing:
                logger.warning(f"Missing keys when loading Gemma weights: {missing[:10]}...")
            if unexpected:
                logger.warning(f"Unexpected keys when loading Gemma weights: {unexpected[:10]}...")

            del pretrained_model

        except Exception as e:
            logger.error(f"Failed to load Gemma weights from {pretrained_path}: {e}")
            raise

    def _load_llama_weights(self, pretrained_path: str, **kwargs):
        """Load LLaMA pretrained weights."""
        try:
            from transformers import LlamaForCausalLM as HFLlama

            pretrained_model = HFLlama.from_pretrained(
                pretrained_path,
                torch_dtype=torch.float32 if self.backbone_config.dtype == "float32" else torch.bfloat16,
                **kwargs,
            )

            missing, unexpected = self.model.load_state_dict(
                pretrained_model.state_dict(), strict=False
            )

            if missing:
                logger.warning(f"Missing keys when loading LLaMA weights: {missing[:10]}...")

            del pretrained_model

        except Exception as e:
            logger.error(f"Failed to load LLaMA weights from {pretrained_path}: {e}")
            raise

    def _load_qwen_weights(self, pretrained_path: str, **kwargs):
        """Load Qwen pretrained weights."""
        try:
            from transformers import Qwen2ForCausalLM as HFQwen

            pretrained_model = HFQwen.from_pretrained(
                pretrained_path,
                torch_dtype=torch.float32 if self.backbone_config.dtype == "float32" else torch.bfloat16,
                **kwargs,
            )

            missing, unexpected = self.model.load_state_dict(
                pretrained_model.state_dict(), strict=False
            )

            if missing:
                logger.warning(f"Missing keys when loading Qwen weights: {missing[:10]}...")

            del pretrained_model

        except Exception as e:
            logger.error(f"Failed to load Qwen weights from {pretrained_path}: {e}")
            raise


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

        # Initialize KV cache structure
        self._kv_cache: dict[str, list[tuple[Tensor, Tensor]]] | None = None

    def _get_rotary_emb(self):
        """Get the rotary embedding module from any backbone."""
        if self._rotary_emb is None:
            for backbone in self.backbones.values():
                rotary = backbone.get_rotary_emb()
                if rotary is not None:
                    self._rotary_emb = rotary
                    break
        return self._rotary_emb

    def reset_kv_cache(self):
        """Reset the KV cache."""
        self._kv_cache = None

    def forward(
        self,
        token_blocks: list[TokenBlock],
        past_key_values: dict[str, list[tuple[Tensor, Tensor]]] | None = None,
        use_cache: bool = False,
        adarms_conds: dict[str, Tensor] | None = None,
    ) -> tuple[list[TokenBlock], dict[str, list[tuple[Tensor, Tensor]]] | None]:
        """
        Forward pass through heterogeneous joint-layer attention.

        This processes all token blocks through their respective backbones
        while allowing attention between blocks according to flow config,
        even when backbones have different hidden sizes.

        Args:
            token_blocks: List of TokenBlock, one per input node
            past_key_values: Dict of cached K/V per backbone, each is a list
                             of (key, value) tuples per layer
            use_cache: Whether to return new_past_key_values for caching
            adarms_conds: Dict of AdaRMS conditioning tensors per backbone

        Returns:
            Tuple of (output_blocks, new_past_key_values)
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

        # Adjust position IDs for incremental decoding (when using cache)
        if past_key_values is not None:
            # Get cache sequence length from first backbone's first layer
            cache_seq_len = 0
            for backbone_name in self.backbone_order:
                if backbone_name in past_key_values and len(past_key_values[backbone_name]) > 0:
                    cache_seq_len = past_key_values[backbone_name][0][0].shape[2]
                    break
            # Start position IDs from cache_seq_len
            position_ids = torch.cumsum(combined_pad_mask.long(), dim=1) - 1 + cache_seq_len
        else:
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
        new_past_key_values: dict[str, list[tuple[Tensor, Tensor]]] | None = None
        if use_cache:
            new_past_key_values = {name: [] for name in self.backbone_order}

        for layer_idx in range(self.num_layers):
            # Get past KV for this layer (if any)
            layer_past_kv = None
            if past_key_values is not None:
                layer_past_kv = {}
                for backbone_name in self.backbone_order:
                    if backbone_name in past_key_values and layer_idx < len(past_key_values[backbone_name]):
                        layer_past_kv[backbone_name] = past_key_values[backbone_name][layer_idx]
                    else:
                        layer_past_kv[backbone_name] = None

            hidden_states_per_backbone, layer_new_kv = self._process_layer_heterogeneous(
                layer_idx=layer_idx,
                hidden_states_per_backbone=hidden_states_per_backbone,
                attention_mask_4d=attention_mask_4d,
                position_ids=position_ids,
                adarms_conds=adarms_conds,
                rotary_emb=rotary_emb,
                past_key_values=layer_past_kv,
                use_cache=use_cache,
            )

            # Store new KV cache for this layer
            if use_cache and layer_new_kv is not None:
                for backbone_name, kv in layer_new_kv.items():
                    new_past_key_values[backbone_name].append(kv)

        # Apply final normalization per backbone
        for backbone_name, indexed_hidden_states in hidden_states_per_backbone.items():
            backbone = self.backbones[backbone_name]
            norm = backbone.get_norm()
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
        past_key_values: dict[str, tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[dict[str, list[tuple[int, Tensor]]], dict[str, tuple[Tensor, Tensor]] | None]:
        """
        Process a single layer with heterogeneous hidden sizes using explicit
        joint attention in head space.

        Phase 1: QKV projection from disparate hidden_size to shared head_space
        Phase 2: Global attention in head_space with RoPE
        Phase 3: Output projection back to disparate hidden_size
        Phase 4: Independent FFN per backbone

        Args:
            layer_idx: Current layer index
            hidden_states_per_backbone: Dict mapping backbone name to list of (idx, tensor)
            attention_mask_4d: Attention mask [B, 1, seq, seq]
            position_ids: Position IDs for RoPE
            adarms_conds: AdaRMS conditioning per backbone
            rotary_emb: Rotary position embedding module
            past_key_values: Dict mapping backbone name to (past_key, past_value) for this layer
            use_cache: Whether to return updated KV cache

        Returns:
            Tuple of (updated hidden_states_per_backbone, new_kv_cache_for_this_layer)
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

        # KV cache storage for this layer
        new_kv_cache: dict[str, tuple[Tensor, Tensor]] | None = {} if use_cache else None

        for backbone_name in self.backbone_order:
            indexed_hidden_states = hidden_states_per_backbone.get(backbone_name, [])
            if not indexed_hidden_states:
                continue

            backbone = self.backbones[backbone_name]
            layer = backbone.get_layer(layer_idx)
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

            # Handle KV cache for this backbone
            if past_key_values is not None and backbone_name in past_key_values:
                past_kv = past_key_values[backbone_name]
                if past_kv is not None:
                    past_k, past_v = past_kv
                    # Concatenate past K/V with current K/V
                    k = torch.cat([past_k, k], dim=2)
                    v = torch.cat([past_v, v], dim=2)

            # Store new KV for cache (include full K/V including past)
            if use_cache:
                new_kv_cache[backbone_name] = (k.clone(), v.clone())

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
        # Note: When using cache, K/V are already extended with past values
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
        first_layer = first_backbone.get_layer(layer_idx)
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

        return output_hidden_states_per_backbone, new_kv_cache


class MoTModel(nn.Module):
    """
    Core MoT model that combines all components.

    This is the main model class that:
    - Creates and manages multiple transformer backbones
    - Routes inputs through the appropriate projections via NodeInputEmbedder
    - Performs joint-layer attention
    - Produces outputs through the appropriate heads

    The key abstraction is that each node has its own NodeInputEmbedder that
    defines how to embed its inputs into TokenBlocks for joint attention.
    """

    def __init__(self, config: MoTConfig):
        super().__init__()
        self.config = config

        # Initialize backbones (with optional pretrained weights)
        self.backbones: dict[str, MoTBackboneWrapper] = nn.ModuleDict()
        for backbone_cfg in config.backbones:
            if backbone_cfg.pretrained_path:
                # Load from pretrained
                self.backbones[backbone_cfg.name] = MoTBackboneWrapper.from_pretrained(
                    backbone_cfg.pretrained_path,
                    backbone_cfg,
                )
            else:
                # Initialize from scratch
                self.backbones[backbone_cfg.name] = MoTBackboneWrapper(backbone_cfg)

        # Build node-to-backbone mapping
        self.node_to_backbone: dict[str, str] = {}
        for node in config.nodes:
            if node.backbone_name:
                self.node_to_backbone[node.name] = node.backbone_name

        # Initialize input embedders per node
        self.input_embedders: dict[str, NodeInputEmbedder] = nn.ModuleDict()
        for node in config.nodes:
            if node.is_input and node.backbone_name:
                backbone = self.backbones[node.backbone_name]
                self.input_embedders[node.name] = NodeInputEmbedder(
                    node, backbone, config
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

    def embed_all_nodes(
        self,
        inputs: dict[str, Any],
    ) -> list[TokenBlock]:
        """
        Embed inputs for all nodes using their configured input embedders.

        Each node's embedding is handled by its NodeInputEmbedder based on
        the node's input_config.embed_type. Returns a list of TokenBlocks.        Args:
            inputs: Dictionary containing all input data:
                - "images": list[Tensor] for vision nodes
                - "img_masks": list[Tensor] for vision nodes
                - "lang_tokens": Tensor for language nodes
                - "lang_masks": Tensor for language nodes
                - "state": Tensor for state nodes
                - "noisy_actions": Tensor for action nodes (flow matching)
                - "timestep": Tensor for action nodes with time fusion

        Returns:
            List of TokenBlock, one per configured input node.
        """
        token_blocks = []

        for node in self.config.nodes:
            if not node.is_input or node.name not in self.input_embedders:
                continue

            embedder = self.input_embedders[node.name]
            embed_type = node.input_config.embed_type if node.input_config else EmbedType.CUSTOM

            # Dispatch to appropriate embedding method based on embed_type
            if embed_type == EmbedType.VISION:
                embs, pad_mask, att_mask = embedder.embed_vision(
                    images=inputs.get("images", []),
                    img_masks=inputs.get("img_masks", []),
                )
            elif embed_type == EmbedType.LANGUAGE:
                embs, pad_mask, att_mask = embedder.embed_language(
                    tokens=inputs["lang_tokens"],
                    masks=inputs["lang_masks"],
                )
            elif embed_type == EmbedType.STATE:
                embs, pad_mask, att_mask = embedder.embed_state(
                    state=inputs["state"],
                )
            elif embed_type == EmbedType.ACTION:
                embs, pad_mask, att_mask = embedder.embed_action(
                    actions=inputs.get("noisy_actions", inputs.get("actions")),
                    timestep=inputs.get("timestep"),
                )
            elif embed_type == EmbedType.FLOW_TIME:
                embs, pad_mask, att_mask = embedder.embed_flow_time(
                    timestep=inputs["timestep"],
                )
            else:
                # Custom - use the embedder's forward method if defined
                node_inputs = {
                    k: inputs[k] for k in node.input_config.input_keys
                    if k in inputs
                } if node.input_config else {}
                embs, pad_mask, att_mask = embedder(node_inputs)

            token_blocks.append(TokenBlock(
                node_name=node.name,
                embeddings=embs,
                pad_mask=pad_mask,
                att_mask=att_mask,
            ))

        return token_blocks

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

        Uses embed_all_nodes to create TokenBlocks for all configured nodes,
        then processes through joint attention.
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # Flow matching interpolation
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Prepare inputs dict for embed_all_nodes
        inputs = {
            "images": images,
            "img_masks": img_masks,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "state": state,
            "noisy_actions": x_t,
            "timestep": time,
        }

        # Embed all nodes into TokenBlocks
        token_blocks = self.embed_all_nodes(inputs)

        # Convert to appropriate dtype if needed
        vlm_backbone = list(self.backbones.values())[0]
        target_dtype = None
        if hasattr(vlm_backbone, 'language_model'):
            weight = vlm_backbone.language_model.layers[0].self_attn.q_proj.weight
            if weight.dtype == torch.bfloat16:
                target_dtype = torch.bfloat16

        if target_dtype is not None:
            for block in token_blocks:
                block.embeddings = block.embeddings.to(dtype=target_dtype)

        # Prepare AdaRMS conditions (if any node provides them)
        adarms_conds = {}
        for node in self.config.nodes:
            if node.name in self.input_embedders:
                embedder = self.input_embedders[node.name]
                cond = embedder.get_adarms_cond(time=time, batch=inputs)
                if cond is not None and node.backbone_name:
                    adarms_conds[node.backbone_name] = cond

        # Run joint attention
        output_blocks, _ = self.joint_attention(
            token_blocks,
            use_cache=False,
            adarms_conds=adarms_conds
        )

        # Find the action output node
        action_out = None
        action_node_name = None
        for node in self.config.nodes:
            if node.is_output and node.node_type in [NodeType.ACTION, NodeType.LM]:
                action_node_name = node.name
                break

        for block in output_blocks:
            if block.node_name == action_node_name:
                action_out = block.embeddings
                break

        if action_out is None:
            raise ValueError(f"Action output node '{action_node_name}' not found in output blocks!")

        # Extract last chunk_size tokens for action prediction
        action_out = action_out[:, -self.config.chunk_size:]
        action_out = action_out.to(dtype=torch.float32)

        v_t = self.action_out_proj(action_out)

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

        Embeds static inputs (images, language) once, then iteratively
        denoises the action sequence.
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        batch_size = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (batch_size, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Prepare static inputs (images + language) - embedded once
        static_inputs = {
            "images": images,
            "img_masks": img_masks,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
        }

        # Get static token blocks (VLM nodes)
        static_token_blocks = []
        for node in self.config.nodes:
            if node.name not in self.input_embedders:
                continue
            embedder = self.input_embedders[node.name]
            embed_type = node.input_config.embed_type if node.input_config else EmbedType.CUSTOM

            # Only process vision/language nodes as static
            if embed_type == EmbedType.VISION:
                embs, pad_mask, att_mask = embedder.embed_vision(
                    images=images, img_masks=img_masks
                )
                # Also add language if this is a VLM node
                if node.node_type == NodeType.VLM:
                    lang_embs, lang_pad, lang_att = embedder.embed_language(
                        tokens=lang_tokens, masks=lang_masks
                    )
                    embs = torch.cat([embs, lang_embs], dim=1)
                    pad_mask = torch.cat([pad_mask, lang_pad], dim=1)
                    att_mask = torch.cat([att_mask, lang_att], dim=1)

                static_token_blocks.append(TokenBlock(
                    node_name=node.name,
                    embeddings=embs,
                    pad_mask=pad_mask,
                    att_mask=att_mask,
                ))
            elif embed_type == EmbedType.LANGUAGE:
                embs, pad_mask, att_mask = embedder.embed_language(
                    tokens=lang_tokens, masks=lang_masks
                )
                static_token_blocks.append(TokenBlock(
                    node_name=node.name,
                    embeddings=embs,
                    pad_mask=pad_mask,
                    att_mask=att_mask,
                ))

        # Convert static blocks to target dtype
        vlm_backbone = list(self.backbones.values())[0]
        target_dtype = None
        if hasattr(vlm_backbone, 'language_model'):
            weight = vlm_backbone.language_model.layers[0].self_attn.q_proj.weight
            if weight.dtype == torch.bfloat16:
                target_dtype = torch.bfloat16

        if target_dtype is not None:
            for block in static_token_blocks:
                block.embeddings = block.embeddings.to(dtype=target_dtype)

        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(time_val, dtype=torch.float32, device=device).expand(batch_size)

            v_t = self._denoise_step(
                state=state,
                static_token_blocks=static_token_blocks,
                x_t=x_t,
                timestep=time_tensor,
                target_dtype=target_dtype,
            )

            x_t = x_t + dt * v_t

        return x_t

    def _denoise_step(
        self,
        state: Tensor,
        static_token_blocks: list[TokenBlock],
        x_t: Tensor,
        timestep: Tensor,
        target_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """
        Single denoising step.

        Combines pre-computed static token blocks with dynamic action embeddings.
        """
        # Find action/state node and embed dynamic inputs
        dynamic_token_blocks = []
        adarms_conds = {}

        for node in self.config.nodes:
            if node.name not in self.input_embedders:
                continue
            embedder = self.input_embedders[node.name]
            embed_type = node.input_config.embed_type if node.input_config else EmbedType.CUSTOM

            # Process state and action nodes dynamically
            if embed_type == EmbedType.STATE:
                embs, pad_mask, att_mask = embedder.embed_state(state=state)
                dynamic_token_blocks.append(TokenBlock(
                    node_name=node.name,
                    embeddings=embs,
                    pad_mask=pad_mask,
                    att_mask=att_mask,
                ))
            elif embed_type == EmbedType.ACTION:
                embs, pad_mask, att_mask = embedder.embed_action(
                    actions=x_t, timestep=timestep
                )
                dynamic_token_blocks.append(TokenBlock(
                    node_name=node.name,
                    embeddings=embs,
                    pad_mask=pad_mask,
                    att_mask=att_mask,
                ))
                # Get AdaRMS condition if applicable
                cond = embedder.get_adarms_cond(time=timestep)
                if cond is not None and node.backbone_name:
                    adarms_conds[node.backbone_name] = cond

        # Convert dynamic blocks to target dtype
        if target_dtype is not None:
            for block in dynamic_token_blocks:
                block.embeddings = block.embeddings.to(dtype=target_dtype)

        # Combine static and dynamic token blocks in node order
        all_blocks = []
        static_dict = {b.node_name: b for b in static_token_blocks}
        dynamic_dict = {b.node_name: b for b in dynamic_token_blocks}

        for node in self.config.nodes:
            if node.name in static_dict:
                all_blocks.append(static_dict[node.name])
            elif node.name in dynamic_dict:
                all_blocks.append(dynamic_dict[node.name])

        # Run joint attention
        output_blocks, _ = self.joint_attention(
            all_blocks,
            use_cache=False,
            adarms_conds=adarms_conds
        )

        # Find action output
        action_out = None
        action_node_name = None
        for node in self.config.nodes:
            if node.is_output and node.node_type in [NodeType.ACTION, NodeType.LM]:
                action_node_name = node.name
                break

        for block in output_blocks:
            if block.node_name == action_node_name:
                action_out = block.embeddings
                break

        action_out = action_out[:, -self.config.chunk_size:]
        action_out = action_out.to(dtype=torch.float32)

        return self.action_out_proj(action_out)

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
