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
MoT Backbone Adapters

This module provides the factory function and exports for the backbone adapter layer.
It allows the MoT Core to work with different transformer implementations through
a unified interface.

The backbone adapter layer consists of:
- MoTBackboneWrapper: Abstract base class defining the protocol
- HuggingFaceBackboneWrapper: Adapter for HuggingFace Transformers models
- GenericPyTorchWrapper: Adapter for custom PyTorch modules

Usage:
    >>> from lerobot.policies.mot.backbones import build_backbone, MoTBackboneWrapper
    >>> backbone = build_backbone(config)

Or use specific wrappers directly:
    >>> from lerobot.policies.mot.backbones import HuggingFaceBackboneWrapper
    >>> wrapper = HuggingFaceBackboneWrapper(model, layers_attr="model.layers")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from torch import nn

from lerobot.policies.mot.backbones import rope_utils
from lerobot.policies.mot.backbones.custom import (GenericPyTorchWrapper,
                                                   SimpleTransformerBlock,
                                                   create_simple_backbone)
from lerobot.policies.mot.backbones.hf_mixin import (
    HF_MODEL_PRESETS, HuggingFaceBackboneWrapper, get_hf_preset)
from lerobot.policies.mot.backbones.wrapper import (BackboneWrapperConfig,
                                                    MoTBackboneWrapper)

if TYPE_CHECKING:
    from lerobot.policies.mot.configuration_mot import MoTStreamConfig

logger = logging.getLogger(__name__)

__all__ = [
    # Base classes
    "MoTBackboneWrapper",
    "BackboneWrapperConfig",
    # HuggingFace adapter
    "HuggingFaceBackboneWrapper",
    "HF_MODEL_PRESETS",
    "get_hf_preset",
    # Custom PyTorch adapter
    "GenericPyTorchWrapper",
    "SimpleTransformerBlock",
    "create_simple_backbone",
    # Utilities
    "rope_utils",
    # Factory
    "build_backbone",
    "build_backbone_from_config",
]


def build_backbone(
    model: nn.Module | None = None,
    backbone_type: str = "hf",
    layers: nn.ModuleList | None = None,
    norm: nn.Module | None = None,
    hidden_size: int | None = None,
    head_dim: int | None = None,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
    layers_attr: str = "model.layers",
    norm_attr: str = "model.norm",
    embed_tokens_attr: str | None = "model.embed_tokens",
    rotary_emb_attr: str | None = "model.rotary_emb",
    use_adarms: bool = False,
    preset: str | None = None,
    **kwargs: Any,
) -> MoTBackboneWrapper:
    """
    Factory function to create a backbone wrapper.

    This function creates the appropriate wrapper based on the backbone_type parameter.
    It provides a unified interface for instantiating different backbone adapters.

    Args:
        model: The model to wrap (required for "hf" type, optional for "custom").
        backbone_type: Type of backbone wrapper to create.
            - "hf": HuggingFace Transformers model
            - "custom": Generic PyTorch module
        layers: ModuleList of transformer layers (required for "custom" type).
        norm: Final normalization layer (required for "custom" type).
        hidden_size: Hidden dimension size (required for "custom", auto-detected for "hf").
        head_dim: Attention head dimension.
        num_attention_heads: Number of attention heads (required for "custom").
        num_key_value_heads: Number of key-value heads.
        layers_attr: Attribute path to layers (for "hf" type).
        norm_attr: Attribute path to final norm (for "hf" type).
        embed_tokens_attr: Attribute path to token embeddings (for "hf" type).
        rotary_emb_attr: Attribute path to rotary embeddings (for "hf" type).
        use_adarms: Whether the model uses AdaRMS normalization.
        preset: Optional preset name for HuggingFace models (e.g., "gemma", "llama").
        **kwargs: Additional arguments passed to the wrapper.

    Returns:
        MoTBackboneWrapper: The instantiated backbone wrapper.

    Raises:
        ValueError: If required parameters are missing or backbone_type is invalid.

    Examples:
        Create a HuggingFace wrapper:
        >>> from transformers import GemmaModel
        >>> model = GemmaModel.from_pretrained("google/gemma-2b")
        >>> wrapper = build_backbone(model, backbone_type="hf", preset="gemma")

        Create a custom wrapper:
        >>> layers = nn.ModuleList([...])
        >>> norm = nn.LayerNorm(256)
        >>> wrapper = build_backbone(
        ...     backbone_type="custom",
        ...     layers=layers,
        ...     norm=norm,
        ...     hidden_size=256,
        ...     num_attention_heads=4,
        ... )
    """
    if backbone_type == "hf":
        if model is None:
            raise ValueError("model is required for HuggingFace backbone wrapper")

        # Apply preset if specified
        if preset is not None:
            preset_config = get_hf_preset(preset)
            layers_attr = preset_config.get("layers_attr", layers_attr)
            norm_attr = preset_config.get("norm_attr", norm_attr)
            embed_tokens_attr = preset_config.get("embed_tokens_attr", embed_tokens_attr)
            rotary_emb_attr = preset_config.get("rotary_emb_attr", rotary_emb_attr)

        return HuggingFaceBackboneWrapper(
            model=model,
            layers_attr=layers_attr,
            norm_attr=norm_attr,
            embed_tokens_attr=embed_tokens_attr,
            rotary_emb_attr=rotary_emb_attr,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            use_adarms=use_adarms,
        )

    elif backbone_type == "custom":
        if layers is None:
            raise ValueError("layers is required for custom backbone wrapper")
        if norm is None:
            raise ValueError("norm is required for custom backbone wrapper")
        if hidden_size is None:
            raise ValueError("hidden_size is required for custom backbone wrapper")
        if num_attention_heads is None:
            raise ValueError("num_attention_heads is required for custom backbone wrapper")

        # Try to infer head_dim if not provided
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads

        # Get optional embed_tokens from model if provided
        embed_tokens = None
        if model is not None and hasattr(model, "embed_tokens"):
            embed_tokens = model.embed_tokens

        # Get optional rotary_emb from model if provided
        rotary_emb = None
        if model is not None and hasattr(model, "rotary_emb"):
            rotary_emb = model.rotary_emb

        return GenericPyTorchWrapper(
            layers=layers,
            norm=norm,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            embed_tokens=embed_tokens,
            rotary_emb=rotary_emb,
            use_adarms=use_adarms,
        )

    else:
        raise ValueError(
            f"Unknown backbone_type: '{backbone_type}'. "
            f"Supported types: 'hf', 'custom'"
        )


def build_backbone_from_config(
    stream_config: "MoTStreamConfig",
    model: nn.Module | None = None,
    layers: nn.ModuleList | None = None,
    norm: nn.Module | None = None,
) -> MoTBackboneWrapper:
    """
    Build a backbone wrapper from a MoTStreamConfig.

    This function is a convenience wrapper around build_backbone that
    extracts parameters from a MoTStreamConfig object.

    Args:
        stream_config: The stream configuration containing backbone parameters.
        model: The model to wrap (for HF backbones).
        layers: The transformer layers (for custom backbones).
        norm: The final normalization layer (for custom backbones).

    Returns:
        MoTBackboneWrapper: The instantiated backbone wrapper.
    """
    return build_backbone(
        model=model,
        backbone_type=stream_config.backbone_type,
        layers=layers,
        norm=norm,
        hidden_size=stream_config.hidden_size,
        head_dim=stream_config.head_dim,
        num_attention_heads=stream_config.num_attention_heads,
        num_key_value_heads=stream_config.num_key_value_heads,
        layers_attr=stream_config.layers_attr,
        norm_attr=stream_config.norm_attr,
        embed_tokens_attr=stream_config.embed_tokens_attr,
        rotary_emb_attr=stream_config.rotary_emb_attr,
        use_adarms=stream_config.use_adarms,
        preset=stream_config.preset,
    )
