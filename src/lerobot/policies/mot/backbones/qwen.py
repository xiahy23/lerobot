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
Qwen backbone adapter for MoT.

This module implements the adapter for Alibaba's Qwen family of language models.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

from .base import BackboneAdapter

if TYPE_CHECKING:
    from lerobot.policies.mot.configuration_mot import MoTBackboneConfig

logger = logging.getLogger(__name__)

# ==================== Qwen Model Presets ====================

QWEN2_0_5B_DEFAULTS = {
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "intermediate_size": 4864,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "head_dim": 64,
}

QWEN2_1_5B_DEFAULTS = {
    "hidden_size": 1536,
    "num_hidden_layers": 28,
    "intermediate_size": 8960,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "head_dim": 128,
}

QWEN2_7B_DEFAULTS = {
    "hidden_size": 3584,
    "num_hidden_layers": 28,
    "intermediate_size": 18944,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "head_dim": 128,
}

QWEN_PRESETS = {
    "qwen2_0.5b": QWEN2_0_5B_DEFAULTS,
    "qwen2_1.5b": QWEN2_1_5B_DEFAULTS,
    "qwen2_7b": QWEN2_7B_DEFAULTS,
}


def get_qwen_preset(variant: str) -> dict[str, Any]:
    """
    Get preset configuration for a known Qwen variant.

    Args:
        variant: Variant name (e.g., "qwen2_0.5b", "qwen2_7b")

    Returns:
        Dictionary of configuration values

    Raises:
        ValueError: If variant is not recognized
    """
    if variant not in QWEN_PRESETS:
        available = list(QWEN_PRESETS.keys())
        raise ValueError(f"Unknown Qwen variant: {variant}. Available: {available}")
    return QWEN_PRESETS[variant].copy()


class QwenAdapter(BackboneAdapter):
    """
    Adapter for Alibaba's Qwen family of language models.

    This adapter handles:
    - Qwen2-specific HuggingFace config creation
    - Proper weight loading from HuggingFace hub or local paths
    """

    def setup_model(self) -> None:
        """Initialize the Qwen model structure."""
        try:
            from transformers.models.auto import CONFIG_MAPPING
            from transformers.models.qwen2.modeling_qwen2 import \
                Qwen2ForCausalLM
        except ImportError as e:
            raise ImportError(
                "Qwen backbone requires transformers with Qwen2 support. "
                "Please upgrade transformers: pip install -U transformers"
            ) from e

        cfg = self.config

        # Create HuggingFace config for Qwen2
        hf_config = CONFIG_MAPPING["qwen2"](
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_attention_heads=cfg.num_attention_heads,
            num_hidden_layers=cfg.num_hidden_layers,
            num_key_value_heads=cfg.num_key_value_heads,
            vocab_size=cfg.vocab_size,
        )

        # Apply any extra config kwargs
        for key, value in cfg.hf_config_kwargs.items():
            setattr(hf_config, key, value)

        # Initialize model
        self.model = Qwen2ForCausalLM(config=hf_config)
        self.language_model = self.model.model

        # Apply precision settings
        self.apply_precision(cfg.dtype)

        logger.info(
            f"Initialized Qwen backbone: "
            f"hidden_size={cfg.hidden_size}, "
            f"num_layers={cfg.num_hidden_layers}, "
            f"num_heads={cfg.num_attention_heads}"
        )

    def load_pretrained(self, path: str, **kwargs) -> None:
        """
        Load pretrained Qwen weights.

        Args:
            path: Path to pretrained weights (local or HuggingFace hub)
            **kwargs: Additional arguments for from_pretrained()
        """
        try:
            from transformers import Qwen2ForCausalLM as HFQwen
        except ImportError as e:
            raise ImportError(
                "Qwen backbone requires transformers with Qwen2 support."
            ) from e

        try:
            # Determine dtype for loading
            torch_dtype = (
                torch.float32 if self.config.dtype == "float32" else torch.bfloat16
            )

            # Load pretrained model
            pretrained_model = HFQwen.from_pretrained(
                path,
                torch_dtype=torch_dtype,
                **kwargs,
            )

            # Copy weights to our model
            missing, unexpected = self.model.load_state_dict(
                pretrained_model.state_dict(), strict=False
            )

            if missing:
                logger.warning(
                    f"Missing keys when loading Qwen weights: {missing[:10]}..."
                )
            if unexpected:
                logger.warning(
                    f"Unexpected keys when loading Qwen weights: {unexpected[:10]}..."
                )

            # Clean up
            del pretrained_model

            logger.info(f"Successfully loaded Qwen weights from {path}")

        except Exception as e:
            logger.error(f"Failed to load Qwen weights from {path}: {e}")
            raise

    @property
    def layers(self) -> nn.ModuleList:
        """Return Qwen's transformer layers."""
        return self.language_model.layers

    @property
    def norm(self) -> nn.Module:
        """Return Qwen's final normalization layer."""
        return self.language_model.norm

    @property
    def rotary_emb(self) -> nn.Module | None:
        """Return Qwen's rotary embedding module if available."""
        if hasattr(self.language_model, "rotary_emb"):
            return self.language_model.rotary_emb
        return None

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        """
        Embed language tokens using Qwen's embedding layer.

        Args:
            tokens: Token IDs of shape (batch, seq_len)

        Returns:
            Token embeddings of shape (batch, seq_len, hidden_size)
        """
        if self.language_model.embed_tokens is not None:
            return self.language_model.embed_tokens(tokens)
        raise ValueError("Qwen embed_tokens layer not found")
