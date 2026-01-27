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
MoT (Multistream-VLA) Configuration

This module defines the configuration classes for the MoT architecture.
The configuration follows the "mechanism vs policy separation" philosophy:

- MoTStreamConfig: Describes a single stream (e.g., vision-language, action)
- MoTConfig: Contains the list of streams and attention implementation settings

The core framework only handles the "form" of tensor flow, not the "semantics" of content.
Concrete policies (like Pi0) inherit from this config and define their specific streams.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig


class AttentionType(str, Enum):
    """Types of attention mechanisms supported by MoT."""

    # Standard: Each stream attends only to itself
    INDEPENDENT = "independent"

    # Joint: All streams share attention computation (concatenated)
    JOINT = "joint"


class StreamRole(str, Enum):
    """Role of a stream in the MoT architecture."""

    # Vision-Language stream (typically the main stream for VLM)
    VISION_LANGUAGE = "vision_language"

    # Action stream (for action prediction)
    ACTION = "action"

    # State stream (for proprioceptive state encoding)
    STATE = "state"

    # Auxiliary stream (for additional modalities)
    AUXILIARY = "auxiliary"


@dataclass
class MoTStreamConfig:
    """
    Configuration for a single stream in the MoT architecture.

    A stream represents a distinct processing pathway in the multi-stream
    architecture. Each stream has its own backbone (transformer) and can
    optionally interact with other streams through joint/cross attention.

    Attributes:
        name: Unique identifier for this stream.
        role: The role of this stream (vision_language, action, state, auxiliary).
        backbone_type: Type of backbone ("hf" for HuggingFace, "custom" for PyTorch).
        model_path: Path or identifier for loading pretrained models (HF only).
        preset: Preset configuration name (e.g., "gemma", "paligemma"). If
        you are loading a HuggingFace model, this can help auto-configure attribute paths.

        # Architecture parameters (can be auto-detected for HF models)
        hidden_size: Hidden dimension size.
        head_dim: Attention head dimension.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of key-value heads (for GQA/MQA).
        num_layers: Number of transformer layers.

        # Attribute paths for HF models
        layers_attr: Path to transformer layers (e.g., "model.layers").
        norm_attr: Path to final normalization layer.
        embed_tokens_attr: Path to token embedding layer.
        rotary_emb_attr: Path to rotary embeddings.

        # Stream-specific settings
        use_adarms: Whether to use adaptive RMS normalization.
        dropout: Dropout probability for this stream.
        use_gradient_checkpointing: Whether to use gradient checkpointing.
    """

    # Basic identification
    name: str = "stream"
    role: StreamRole | str = StreamRole.AUXILIARY

    # Backbone configuration
    backbone_type: Literal["hf", "custom"] = "hf"
    model_path: str | None = None
    preset: str | None = None

    # Architecture parameters (can be auto-detected from HF models)
    hidden_size: int | None = None
    head_dim: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    num_layers: int | None = None

    # HF model attribute paths
    layers_attr: str = "model.layers"
    norm_attr: str = "model.norm"
    embed_tokens_attr: str | None = "model.embed_tokens"
    rotary_emb_attr: str | None = "model.rotary_emb"

    # Stream-specific settings
    use_adarms: bool = False
    dropout: float = 0.0
    use_gradient_checkpointing: bool = False

    # Precision settings
    dtype: str = "bfloat16"  # "float32", "bfloat16", "float16"

    def __post_init__(self):
        # Convert string role to enum if needed
        if isinstance(self.role, str):
            self.role = StreamRole(self.role)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "role": self.role.value if isinstance(self.role, StreamRole) else self.role,
            "backbone_type": self.backbone_type,
            "model_path": self.model_path,
            "preset": self.preset,
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_layers": self.num_layers,
            "layers_attr": self.layers_attr,
            "norm_attr": self.norm_attr,
            "embed_tokens_attr": self.embed_tokens_attr,
            "rotary_emb_attr": self.rotary_emb_attr,
            "use_adarms": self.use_adarms,
            "dropout": self.dropout,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MoTStreamConfig":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class MoTJointAttentionConfig:
    """
    Configuration for joint attention between streams.

    Joint attention allows multiple streams to share attention computation
    by concatenating their sequences and computing attention jointly.

    Attributes:
        enabled: Whether joint attention is enabled.
        streams: List of stream names to include in joint attention.
        layers: Layer indices where joint attention is applied (None = all layers).
        bidirectional: Whether attention is bidirectional between streams.
    """

    enabled: bool = True
    streams: list[str] | None = None  # None means all streams
    layers: list[int] | None = None  # None means all layers
    bidirectional: bool = False  # True = streams can attend to each other


@dataclass
class MoTConfig(PreTrainedConfig):
    """
    Base configuration for MoT (Multistream-VLA) models.

    This configuration defines the multi-stream architecture where multiple
    transformer backbones process different modalities (vision-language, action)
    and can optionally interact through joint attention mechanisms.

    Core Philosophy: "Mechanism vs Policy Separation"
    - MoT Core handles mechanism (synchronized layer loops, joint attention, KV cache)
    - Concrete policies (Pi0, etc.) handle business logic (embeddings, masks, loss)

    Attributes:
        streams: List of stream configurations.
        attn_implementation: Attention implementation type.
        joint_attention: Joint attention configuration.

        # Training settings
        gradient_checkpointing: Enable gradient checkpointing.
        compile_model: Whether to use torch.compile.
        compile_mode: Torch compile mode.

    Example:
        >>> config = MoTConfig(
        ...     streams=[
        ...         MoTStreamConfig(name="vlm", role="vision_language", preset="paligemma"),
        ...         MoTStreamConfig(name="action", role="action", preset="gemma"),
        ...     ],
        ...     attn_implementation="joint",
        ... )
    """

    # Stream configurations
    streams: list[MoTStreamConfig] = field(default_factory=list)

    # Attention implementation
    attn_implementation: AttentionType | str = AttentionType.JOINT

    # Joint attention settings
    joint_attention: MoTJointAttentionConfig = field(
        default_factory=MoTJointAttentionConfig
    )

    # Inference settings
    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict
    n_action_steps: int = 50  # Number of action steps to execute

    # Padding dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Training settings
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Optimizer settings
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        # Convert string attention type to enum if needed
        if isinstance(self.attn_implementation, str):
            self.attn_implementation = AttentionType(self.attn_implementation)

        # Validate streams
        self._validate_streams()

    def _validate_streams(self):
        """Validate stream configurations."""
        if not self.streams:
            return

        # Check for duplicate stream names
        names = [s.name for s in self.streams]
        if len(names) != len(set(names)):
            raise ValueError("Stream names must be unique")

        # Check layer alignment for joint attention
        if self.attn_implementation == AttentionType.JOINT:
            layer_counts = [s.num_layers for s in self.streams if s.num_layers is not None]
            if layer_counts and len(set(layer_counts)) > 1:
                raise ValueError(
                    f"All streams must have the same number of layers for joint attention. "
                    f"Found: {dict(zip(names, layer_counts))}"
                )

    def get_stream(self, name: str) -> MoTStreamConfig | None:
        """Get a stream configuration by name."""
        for stream in self.streams:
            if stream.name == name:
                return stream
        return None

    def get_streams_by_role(self, role: StreamRole | str) -> list[MoTStreamConfig]:
        """Get all streams with a specific role."""
        if isinstance(role, str):
            role = StreamRole(role)
        return [s for s in self.streams if s.role == role]

    @property
    def num_streams(self) -> int:
        """Return the number of streams."""
        return len(self.streams)

    @property
    def stream_names(self) -> list[str]:
        """Return the names of all streams."""
        return [s.name for s in self.streams]

    @property
    def num_layers(self) -> int | None:
        """Return the number of layers (assuming all streams are aligned)."""
        if not self.streams:
            return None
        for stream in self.streams:
            if stream.num_layers is not None:
                return stream.num_layers
        return None

    def validate_features(self) -> None:
        """Validate input/output features. To be overridden by subclasses."""
        pass

    def get_optimizer_preset(self) -> OptimizerConfig:
        """Return optimizer configuration."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """Return scheduler configuration."""
        from lerobot.optim.schedulers import \
            CosineDecayWithWarmupSchedulerConfig

        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        """Return observation delta indices (None for MoT)."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Return action delta indices."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """Return reward delta indices (None for MoT)."""
        return None


# Convenience functions for creating common stream configurations

def create_paligemma_stream(
    name: str = "vision_language",
    variant: str = "gemma_2b",
    use_adarms: bool = False,
    **kwargs,
) -> MoTStreamConfig:
    """
    Create a PaliGemma stream configuration.

    Args:
        name: Stream name.
        variant: Model variant ("gemma_300m" or "gemma_2b").
        use_adarms: Whether to use AdaRMS normalization.
        **kwargs: Additional configuration overrides.

    Returns:
        MoTStreamConfig for PaliGemma.
    """
    # Determine dimensions based on variant
    if variant == "gemma_300m":
        hidden_size = 1024
        num_attention_heads = 8
        head_dim = 256
        num_layers = 18
    elif variant == "gemma_2b":
        hidden_size = 2048
        num_attention_heads = 8
        head_dim = 256
        num_layers = 18
    else:
        raise ValueError(f"Unknown PaliGemma variant: {variant}")

    return MoTStreamConfig(
        name=name,
        role=StreamRole.VISION_LANGUAGE,
        backbone_type="hf",
        preset="paligemma_language",
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=1,
        num_layers=num_layers,
        use_adarms=use_adarms,
        **kwargs,
    )


def create_gemma_stream(
    name: str = "action",
    variant: str = "gemma_300m",
    use_adarms: bool = False,
    **kwargs,
) -> MoTStreamConfig:
    """
    Create a Gemma stream configuration (typically for action expert).

    Args:
        name: Stream name.
        variant: Model variant ("gemma_300m" or "gemma_2b").
        use_adarms: Whether to use AdaRMS normalization.
        **kwargs: Additional configuration overrides.

    Returns:
        MoTStreamConfig for Gemma.
    """
    # Determine dimensions based on variant
    if variant == "gemma_300m":
        hidden_size = 1024
        num_attention_heads = 8
        head_dim = 256
        num_layers = 18
    elif variant == "gemma_2b":
        hidden_size = 2048
        num_attention_heads = 8
        head_dim = 256
        num_layers = 18
    else:
        raise ValueError(f"Unknown Gemma variant: {variant}")

    return MoTStreamConfig(
        name=name,
        role=StreamRole.ACTION,
        backbone_type="hf",
        preset="gemma_lm",
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=1,
        num_layers=num_layers,
        use_adarms=use_adarms,
        **kwargs,
    )
