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
Base adapter interface for MoT backbone models.

This module defines the abstract base class that all backbone adapters must implement,
ensuring a consistent interface across different transformer architectures.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from lerobot.policies.mot.configuration_mot import MoTBackboneConfig

logger = logging.getLogger(__name__)


class BackboneAdapter(nn.Module, ABC):
    """
    Abstract base class for all backbone adapters.

    This interface defines what the MoT core engine needs from any backbone,
    regardless of the underlying implementation (PaliGemma, Gemma, LLaMA, etc.).

    Subclasses must implement:
        - setup_model(): Initialize the HuggingFace model structure
        - load_pretrained(): Load pretrained weights
        - layers: Property returning transformer layers
        - norm: Property returning final LayerNorm

    Optional methods to override:
        - embed_tokens(): For language models
        - embed_image(): For vision-language models
        - rotary_emb: Property for rotary position embeddings
    """

    def __init__(self, config: MoTBackboneConfig):
        """
        Initialize the backbone adapter.

        Args:
            config: Configuration for this backbone
        """
        super().__init__()
        self.config = config
        self.model = None
        self.language_model = None
        self.vision_encoder = None

    @abstractmethod
    def setup_model(self) -> None:
        """
        Initialize the underlying HuggingFace model structure.

        This method should:
        1. Create the HF config from self.config
        2. Instantiate the model
        3. Set self.model and self.language_model appropriately
        4. Apply precision settings if needed
        """
        pass

    @abstractmethod
    def load_pretrained(self, path: str, **kwargs) -> None:
        """
        Load pretrained weights into the model.

        Args:
            path: Path to pretrained weights (local path or HuggingFace hub ID)
            **kwargs: Additional arguments for loading (e.g., torch_dtype, device_map)
        """
        pass

    # ==================== Core Interface Properties ====================

    @property
    @abstractmethod
    def layers(self) -> nn.ModuleList:
        """
        Return the list of transformer layers.

        Returns:
            nn.ModuleList of transformer decoder layers
        """
        pass

    @property
    @abstractmethod
    def norm(self) -> nn.Module:
        """
        Return the final normalization layer.

        Returns:
            The final LayerNorm/RMSNorm module
        """
        pass

    @property
    def rotary_emb(self) -> nn.Module | None:
        """
        Return the rotary position embedding module if available.

        Returns:
            Rotary embedding module or None if not used
        """
        return None

    # ==================== Derived Properties ====================

    @property
    def hidden_size(self) -> int:
        """Return the hidden size of the transformer."""
        return self.config.hidden_size

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        return self.config.num_hidden_layers

    @property
    def num_heads(self) -> int:
        """Return the number of attention heads."""
        return self.config.num_attention_heads

    @property
    def head_dim(self) -> int:
        """Return the dimension of each attention head."""
        return self.config.head_dim

    @property
    def num_kv_heads(self) -> int:
        """Return the number of key-value heads (for GQA)."""
        return self.config.num_key_value_heads

    # ==================== Optional Methods ====================

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        """
        Embed language tokens.

        Args:
            tokens: Token IDs of shape (batch, seq_len)

        Returns:
            Token embeddings of shape (batch, seq_len, hidden_size)

        Raises:
            NotImplementedError: If this backbone does not support text embedding
        """
        raise NotImplementedError(
            f"Backbone {self.__class__.__name__} does not support token embedding"
        )

    def embed_image(self, image: Tensor) -> Tensor:
        """
        Embed images through the vision encoder.

        Args:
            image: Image tensor of shape (batch, channels, height, width)

        Returns:
            Image embeddings of shape (batch, num_patches, hidden_size)

        Raises:
            NotImplementedError: If this backbone does not have a vision encoder
        """
        raise NotImplementedError(
            f"Backbone {self.__class__.__name__} does not support vision embedding"
        )

    def get_layer(self, layer_idx: int) -> nn.Module:
        """
        Get a specific transformer layer by index.

        Args:
            layer_idx: Index of the layer (0-indexed)

        Returns:
            The transformer layer module
        """
        return self.layers[layer_idx]

    def get_norm(self) -> nn.Module:
        """
        Get the final normalization layer.

        Returns:
            The final LayerNorm/RMSNorm module
        """
        return self.norm

    def get_rotary_emb(self) -> nn.Module | None:
        """
        Get the rotary embedding module.

        Returns:
            Rotary embedding module or None
        """
        return self.rotary_emb

    # ==================== Utility Methods ====================

    def apply_precision(self, dtype: str) -> None:
        """
        Apply precision settings to the model.

        Args:
            dtype: Target dtype string ("float32", "bfloat16", "float16")
        """
        if self.model is None:
            logger.warning("Cannot apply precision: model not initialized")
            return

        if dtype == "bfloat16":
            self.model.to(dtype=torch.bfloat16)
            # Keep certain parameters in float32 for stability
            for name, param in self.model.named_parameters():
                if any(s in name for s in ["layernorm", "norm", "embedding"]):
                    param.data = param.data.to(dtype=torch.float32)
        elif dtype == "float16":
            self.model.to(dtype=torch.float16)
            for name, param in self.model.named_parameters():
                if any(s in name for s in ["layernorm", "norm", "embedding"]):
                    param.data = param.data.to(dtype=torch.float32)
        elif dtype == "float32":
            self.model.to(dtype=torch.float32)

    def freeze(self) -> None:
        """Freeze all parameters in this backbone."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters in this backbone."""
        for param in self.parameters():
            param.requires_grad = True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        config: MoTBackboneConfig,
        **kwargs,
    ) -> BackboneAdapter:
        """
        Create a backbone adapter and load pretrained weights.

        Args:
            pretrained_path: Path to pretrained weights
            config: Configuration for the backbone
            **kwargs: Additional arguments for loading

        Returns:
            Initialized backbone adapter with loaded weights
        """
        adapter = cls(config)
        adapter.setup_model()
        adapter.load_pretrained(pretrained_path, **kwargs)
        return adapter
