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
LLaMA backbone adapter for MoT.

This module implements the adapter for Meta's LLaMA family of language models.
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

# ==================== LLaMA Model Presets ====================

LLAMA_7B_DEFAULTS = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "intermediate_size": 11008,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "head_dim": 128,
}

LLAMA_13B_DEFAULTS = {
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    "intermediate_size": 13824,
    "num_attention_heads": 40,
    "num_key_value_heads": 40,
    "head_dim": 128,
}

LLAMA2_7B_DEFAULTS = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "intermediate_size": 11008,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,  # LLaMA 2 7B uses MHA, not GQA
    "head_dim": 128,
}

LLAMA3_8B_DEFAULTS = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,  # LLaMA 3 uses GQA
    "head_dim": 128,
}

LLAMA_PRESETS = {
    "llama_7b": LLAMA_7B_DEFAULTS,
    "llama_13b": LLAMA_13B_DEFAULTS,
    "llama2_7b": LLAMA2_7B_DEFAULTS,
    "llama3_8b": LLAMA3_8B_DEFAULTS,
}


def get_llama_preset(variant: str) -> dict[str, Any]:
    """
    Get preset configuration for a known LLaMA variant.

    Args:
        variant: Variant name (e.g., "llama_7b", "llama3_8b")

    Returns:
        Dictionary of configuration values

    Raises:
        ValueError: If variant is not recognized
    """
    if variant not in LLAMA_PRESETS:
        available = list(LLAMA_PRESETS.keys())
        raise ValueError(f"Unknown LLaMA variant: {variant}. Available: {available}")
    return LLAMA_PRESETS[variant].copy()


class LlamaAdapter(BackboneAdapter):
    """
    Adapter for Meta's LLaMA family of language models.

    This adapter handles:
    - LLaMA-specific HuggingFace config creation
    - Support for LLaMA 1, 2, and 3 variants
    - Proper weight loading from HuggingFace hub or local paths
    """

    def setup_model(self) -> None:
        """Initialize the LLaMA model structure."""
        try:
            from transformers.models.auto import CONFIG_MAPPING
            from transformers.models.llama.modeling_llama import \
                LlamaForCausalLM
        except ImportError as e:
            raise ImportError(
                "LLaMA backbone requires transformers with LLaMA support. "
                "Please upgrade transformers: pip install -U transformers"
            ) from e

        cfg = self.config

        # Create HuggingFace config for LLaMA
        hf_config = CONFIG_MAPPING["llama"](
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
        self.model = LlamaForCausalLM(config=hf_config)
        self.language_model = self.model.model

        # Apply precision settings
        self.apply_precision(cfg.dtype)

        logger.info(
            f"Initialized LLaMA backbone: "
            f"hidden_size={cfg.hidden_size}, "
            f"num_layers={cfg.num_hidden_layers}, "
            f"num_heads={cfg.num_attention_heads}"
        )

    def load_pretrained(self, path: str, **kwargs) -> None:
        """
        Load pretrained LLaMA weights.

        Args:
            path: Path to pretrained weights (local or HuggingFace hub)
            **kwargs: Additional arguments for from_pretrained()
        """
        try:
            from transformers import LlamaForCausalLM as HFLlama
        except ImportError as e:
            raise ImportError(
                "LLaMA backbone requires transformers with LLaMA support."
            ) from e

        try:
            # Determine dtype for loading
            torch_dtype = (
                torch.float32 if self.config.dtype == "float32" else torch.bfloat16
            )

            # Load pretrained model
            pretrained_model = HFLlama.from_pretrained(
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
                    f"Missing keys when loading LLaMA weights: {missing[:10]}..."
                )
            if unexpected:
                logger.warning(
                    f"Unexpected keys when loading LLaMA weights: {unexpected[:10]}..."
                )

            # Clean up
            del pretrained_model

            logger.info(f"Successfully loaded LLaMA weights from {path}")

        except Exception as e:
            logger.error(f"Failed to load LLaMA weights from {path}: {e}")
            raise

    @property
    def layers(self) -> nn.ModuleList:
        """Return LLaMA's transformer layers."""
        return self.language_model.layers

    @property
    def norm(self) -> nn.Module:
        """Return LLaMA's final normalization layer."""
        return self.language_model.norm

    @property
    def rotary_emb(self) -> nn.Module | None:
        """Return LLaMA's rotary embedding module if available."""
        if hasattr(self.language_model, "rotary_emb"):
            return self.language_model.rotary_emb
        return None

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        """
        Embed language tokens using LLaMA's embedding layer.

        Args:
            tokens: Token IDs of shape (batch, seq_len)

        Returns:
            Token embeddings of shape (batch, seq_len, hidden_size)
        """
        if self.language_model.embed_tokens is not None:
            return self.language_model.embed_tokens(tokens)
        raise ValueError("LLaMA embed_tokens layer not found")
