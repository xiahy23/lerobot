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
PaliGemma backbone adapter for MoT.

This module implements the adapter for Google's PaliGemma vision-language model,
including both vision encoder and language model components.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

from .base import BackboneAdapter
from .gemma import GEMMA_2B_DEFAULTS

if TYPE_CHECKING:
    from lerobot.policies.mot.configuration_mot import MoTBackboneConfig

logger = logging.getLogger(__name__)

# ==================== PaliGemma Model Presets ====================

PALIGEMMA_VISION_DEFAULTS = {
    "vision_hidden_size": 1152,
    "vision_num_layers": 27,
    "vision_patch_size": 14,
    "vision_image_size": 224,
}

# PaliGemma 2B uses Gemma 2B text backbone with SigLIP vision encoder
PALIGEMMA_2B_DEFAULTS = {
    **GEMMA_2B_DEFAULTS,
    **PALIGEMMA_VISION_DEFAULTS,
}

PALIGEMMA_PRESETS = {
    "paligemma_2b": PALIGEMMA_2B_DEFAULTS,
}


def get_paligemma_preset(variant: str) -> dict[str, Any]:
    """
    Get preset configuration for a known PaliGemma variant.

    Args:
        variant: Variant name (e.g., "paligemma_2b")

    Returns:
        Dictionary of configuration values

    Raises:
        ValueError: If variant is not recognized
    """
    if variant not in PALIGEMMA_PRESETS:
        available = list(PALIGEMMA_PRESETS.keys())
        raise ValueError(f"Unknown PaliGemma variant: {variant}. Available: {available}")
    return PALIGEMMA_PRESETS[variant].copy()


class PaliGemmaAdapter(BackboneAdapter):
    """
    Adapter for Google's PaliGemma vision-language model.

    PaliGemma combines:
    - SigLIP vision encoder for image understanding
    - Gemma language model for text processing and generation

    This adapter handles:
    - PaliGemma-specific HuggingFace config creation
    - Vision encoder initialization and image embedding
    - Language model access for text processing
    - AdaRMS conditioning support
    """

    def setup_model(self) -> None:
        """Initialize the PaliGemma model structure."""
        from transformers.models.auto import CONFIG_MAPPING
        from transformers.models.paligemma.modeling_paligemma import \
            PaliGemmaForConditionalGeneration

        cfg = self.config

        # Create HuggingFace config for PaliGemma
        hf_config = CONFIG_MAPPING["paligemma"]()

        # Set vocabulary config
        hf_config._vocab_size = cfg.vocab_size
        hf_config.image_token_index = cfg.vocab_size

        # Configure text model (Gemma)
        hf_config.text_config.hidden_size = cfg.hidden_size
        hf_config.text_config.intermediate_size = cfg.intermediate_size
        hf_config.text_config.num_attention_heads = cfg.num_attention_heads
        hf_config.text_config.head_dim = cfg.head_dim
        hf_config.text_config.num_hidden_layers = cfg.num_hidden_layers
        hf_config.text_config.num_key_value_heads = cfg.num_key_value_heads
        hf_config.text_config.hidden_activation = cfg.hidden_activation
        hf_config.text_config.torch_dtype = "float32"
        hf_config.text_config.vocab_size = cfg.vocab_size

        # PaliGemma-specific: AdaRMS for pi0.5 style conditioning
        hf_config.text_config.use_adarms = cfg.use_adarms
        hf_config.text_config.adarms_cond_dim = cfg.adarms_cond_dim

        # Configure vision encoder (SigLIP)
        if cfg.vision_hidden_size:
            hf_config.vision_config.hidden_size = cfg.vision_hidden_size
            hf_config.vision_config.intermediate_size = 4304
            hf_config.vision_config.projection_dim = cfg.hidden_size
            hf_config.vision_config.projector_hidden_act = "gelu_fast"

        hf_config.vision_config.image_size = cfg.vision_image_size
        hf_config.vision_config.patch_size = cfg.vision_patch_size

        # Apply any extra config kwargs
        for key, value in cfg.hf_config_kwargs.items():
            setattr(hf_config, key, value)

        # Initialize model
        self.model = PaliGemmaForConditionalGeneration(config=hf_config)
        self.vision_encoder = self.model.vision_tower
        self.language_model = self.model.language_model

        # Apply precision settings
        self.apply_precision(cfg.dtype)

        logger.info(
            f"Initialized PaliGemma backbone: "
            f"hidden_size={cfg.hidden_size}, "
            f"num_layers={cfg.num_hidden_layers}, "
            f"vision_size={cfg.vision_image_size}"
        )

    def load_pretrained(self, path: str, **kwargs) -> None:
        """
        Load pretrained PaliGemma weights.

        Args:
            path: Path to pretrained weights (local or HuggingFace hub)
            **kwargs: Additional arguments for from_pretrained()
        """
        from transformers import \
            PaliGemmaForConditionalGeneration as HFPaliGemma

        try:
            # Determine dtype for loading
            torch_dtype = (
                torch.float32 if self.config.dtype == "float32" else torch.bfloat16
            )

            # Load pretrained model
            pretrained_model = HFPaliGemma.from_pretrained(
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
                    f"Missing keys when loading PaliGemma weights: {missing[:10]}..."
                )
            if unexpected:
                logger.warning(
                    f"Unexpected keys when loading PaliGemma weights: {unexpected[:10]}..."
                )

            # Clean up
            del pretrained_model

            logger.info(f"Successfully loaded PaliGemma weights from {path}")

        except Exception as e:
            logger.error(f"Failed to load PaliGemma weights from {path}: {e}")
            raise

    @property
    def layers(self) -> nn.ModuleList:
        """Return PaliGemma's language model transformer layers."""
        return self.language_model.model.layers

    @property
    def norm(self) -> nn.Module:
        """Return PaliGemma's language model final normalization layer."""
        return self.language_model.model.norm

    @property
    def rotary_emb(self) -> nn.Module | None:
        """Return PaliGemma's rotary embedding module if available."""
        lm_model = self.language_model.model
        if hasattr(lm_model, "rotary_emb"):
            return lm_model.rotary_emb
        return None

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        """
        Embed language tokens using PaliGemma's embedding layer.

        Args:
            tokens: Token IDs of shape (batch, seq_len)

        Returns:
            Token embeddings of shape (batch, seq_len, hidden_size)
        """
        embed_layer = self.language_model.model.embed_tokens
        if embed_layer is not None:
            return embed_layer(tokens)
        raise ValueError("PaliGemma embed_tokens layer not found")

    def embed_image(self, image: Tensor) -> Tensor:
        """
        Embed images through PaliGemma's vision encoder.

        Args:
            image: Image tensor of shape (batch, channels, height, width)

        Returns:
            Image embeddings of shape (batch, num_patches, hidden_size)
        """
        if hasattr(self.model, "get_image_features"):
            return self.model.get_image_features(image)

        if self.vision_encoder is None:
            raise ValueError("PaliGemma vision encoder not initialized")

        return self.vision_encoder(image)
