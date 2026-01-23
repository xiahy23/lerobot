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
Gemma backbone adapter for MoT.

This module implements the adapter for Google's Gemma language models,
including model-specific configuration defaults and weight loading logic.
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

# ==================== Gemma Model Presets ====================

GEMMA_300M_DEFAULTS = {
    "hidden_size": 1024,
    "num_hidden_layers": 18,
    "intermediate_size": 4096,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
}

GEMMA_2B_DEFAULTS = {
    "hidden_size": 2048,
    "num_hidden_layers": 18,
    "intermediate_size": 16384,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
}

GEMMA_PRESETS = {
    "gemma_300m": GEMMA_300M_DEFAULTS,
    "gemma_2b": GEMMA_2B_DEFAULTS,
}


def get_gemma_preset(variant: str) -> dict[str, Any]:
    """
    Get preset configuration for a known Gemma variant.

    Args:
        variant: Variant name (e.g., "gemma_300m", "gemma_2b")

    Returns:
        Dictionary of configuration values

    Raises:
        ValueError: If variant is not recognized
    """
    if variant not in GEMMA_PRESETS:
        available = list(GEMMA_PRESETS.keys())
        raise ValueError(f"Unknown Gemma variant: {variant}. Available: {available}")
    return GEMMA_PRESETS[variant].copy()


class GemmaAdapter(BackboneAdapter):
    """
    Adapter for Google's Gemma language models.

    This adapter handles:
    - Gemma-specific HuggingFace config creation
    - AdaRMS conditioning support (for pi0.5-style models)
    - Proper weight loading from HuggingFace hub or local paths
    """

    def setup_model(self) -> None:
        """Initialize the Gemma model structure."""
        from transformers.models.auto import CONFIG_MAPPING
        from transformers.models.gemma.modeling_gemma import GemmaForCausalLM

        cfg = self.config

        # Create HuggingFace config for Gemma
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
            # Gemma-specific: AdaRMS for pi0.5 style conditioning
            use_adarms=cfg.use_adarms,
            adarms_cond_dim=cfg.adarms_cond_dim,
        )

        # Apply any extra config kwargs
        for key, value in cfg.hf_config_kwargs.items():
            setattr(hf_config, key, value)

        # Initialize model
        self.model = GemmaForCausalLM(config=hf_config)

        # For Gemma in MoT, we typically use external embeddings
        # so we set embed_tokens to None to save memory
        self.model.model.embed_tokens = None
        self.language_model = self.model.model

        # Apply precision settings
        self.apply_precision(cfg.dtype)

        logger.info(
            f"Initialized Gemma backbone: "
            f"hidden_size={cfg.hidden_size}, "
            f"num_layers={cfg.num_hidden_layers}, "
            f"num_heads={cfg.num_attention_heads}"
        )

    def load_pretrained(self, path: str, **kwargs) -> None:
        """
        Load pretrained Gemma weights.

        Args:
            path: Path to pretrained weights (local or HuggingFace hub)
            **kwargs: Additional arguments for from_pretrained()
        """
        from transformers import GemmaForCausalLM as HFGemma

        try:
            # Determine dtype for loading
            torch_dtype = (
                torch.float32 if self.config.dtype == "float32" else torch.bfloat16
            )

            # Load pretrained model
            pretrained_model = HFGemma.from_pretrained(
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
                    f"Missing keys when loading Gemma weights: {missing[:10]}..."
                )
            if unexpected:
                logger.warning(
                    f"Unexpected keys when loading Gemma weights: {unexpected[:10]}..."
                )

            # Clean up
            del pretrained_model

            logger.info(f"Successfully loaded Gemma weights from {path}")

        except Exception as e:
            logger.error(f"Failed to load Gemma weights from {path}: {e}")
            raise

    @property
    def layers(self) -> nn.ModuleList:
        """Return Gemma's transformer layers."""
        return self.language_model.layers

    @property
    def norm(self) -> nn.Module:
        """Return Gemma's final normalization layer."""
        return self.language_model.norm

    @property
    def rotary_emb(self) -> nn.Module | None:
        """Return Gemma's rotary embedding module if available."""
        if hasattr(self.language_model, "rotary_emb"):
            return self.language_model.rotary_emb
        return None

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        """
        Embed language tokens using Gemma's embedding layer.

        Note: In MoT, we typically use external embeddings, so this may not be used.
        """
        if self.language_model.embed_tokens is not None:
            return self.language_model.embed_tokens(tokens)
        raise ValueError(
            "Gemma embed_tokens is None. "
            "In MoT, external embeddings are typically used."
        )
