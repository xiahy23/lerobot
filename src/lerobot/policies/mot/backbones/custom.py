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
Generic PyTorch Backbone Adapter

This module provides an adapter for wrapping custom PyTorch nn.Module implementations
to conform to the MoT backbone wrapper protocol. It's designed for research experiments
where you want to quickly test custom transformer architectures without packaging them
into HuggingFace format.

Usage:
    - Pass your custom layers (nn.ModuleList) and norm (nn.Module) directly.
    - No need to wrap your model in HuggingFace's PreTrainedModel interface.
"""

import logging
from typing import Any, Callable

import torch
from torch import Tensor, nn

from lerobot.policies.mot.backbones.wrapper import (BackboneWrapperConfig,
                                                    MoTBackboneWrapper)

logger = logging.getLogger(__name__)


class GenericPyTorchWrapper(MoTBackboneWrapper):
    """
    Backbone wrapper for generic PyTorch nn.Module implementations.

    This wrapper allows you to use custom transformer implementations that don't
    follow the HuggingFace format. You simply provide the layers ModuleList and
    the final normalization module directly.

    This is useful for:
    - Rapid prototyping of custom architectures
    - Research experiments with novel transformer designs
    - Testing without HuggingFace dependencies
    - Using models from other frameworks (e.g., converted from JAX)

    Example usage:
        >>> layers = nn.ModuleList([
        ...     TransformerBlock(hidden_size=256) for _ in range(6)
        ... ])
        >>> norm = nn.LayerNorm(256)
        >>> wrapper = GenericPyTorchWrapper(
        ...     layers=layers,
        ...     norm=norm,
        ...     hidden_size=256,
        ...     head_dim=64,
        ...     num_attention_heads=4,
        ... )

    Attributes:
        layers: The ModuleList of transformer layers.
        norm: The final normalization layer.
        embed_tokens_module: Optional token embedding module.
        rotary_emb_module: Optional rotary embedding module.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        norm: nn.Module,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int | None = None,
        embed_tokens: nn.Module | None = None,
        rotary_emb: nn.Module | None = None,
        use_adarms: bool = False,
    ):
        """
        Initialize the generic PyTorch backbone wrapper.

        Args:
            layers: ModuleList containing transformer layers/blocks.
            norm: The final layer normalization module.
            hidden_size: The hidden dimension size.
            head_dim: The attention head dimension.
            num_attention_heads: Number of attention heads.
            num_key_value_heads: Number of key-value heads (defaults to num_attention_heads).
            embed_tokens: Optional token embedding module.
            rotary_emb: Optional rotary position embedding module.
            use_adarms: Whether the norm layer uses AdaRMS (adaptive RMS norm).
        """
        super().__init__()

        if not isinstance(layers, nn.ModuleList):
            raise TypeError(f"layers must be nn.ModuleList, got {type(layers)}")

        if len(layers) == 0:
            raise ValueError("layers ModuleList cannot be empty")

        self.layers = layers
        self.norm = norm
        self._hidden_size = hidden_size
        self._head_dim = head_dim
        self._num_attention_heads = num_attention_heads
        self._num_key_value_heads = num_key_value_heads or num_attention_heads
        self._use_adarms = use_adarms

        # Optional modules
        self.embed_tokens_module = embed_tokens
        self.rotary_emb_module = rotary_emb

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        return len(self.layers)

    @property
    def hidden_size(self) -> int:
        """Return the hidden dimension size."""
        return self._hidden_size

    @property
    def head_dim(self) -> int:
        """Return the attention head dimension."""
        return self._head_dim

    @property
    def num_attention_heads(self) -> int:
        """Return the number of attention heads."""
        return self._num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        """Return the number of key-value heads."""
        return self._num_key_value_heads

    def get_layer(self, idx: int) -> nn.Module:
        """
        Get the transformer layer at the specified index.

        Args:
            idx: Layer index (0-indexed).

        Returns:
            The transformer layer module.

        Raises:
            IndexError: If idx is out of bounds.
        """
        if idx < 0 or idx >= len(self.layers):
            raise IndexError(f"Layer index {idx} out of range [0, {len(self.layers)})")
        return self.layers[idx]

    def forward_norm(
        self,
        hidden_states: Tensor,
        cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Apply the final layer normalization.

        Args:
            hidden_states: Hidden states of shape (batch, seq_len, hidden_size).
            cond: Optional conditioning tensor for AdaRMS.

        Returns:
            tuple: (normalized hidden states, gate tensor or None)
        """
        if self._use_adarms and cond is not None:
            # AdaRMS normalization returns (output, gate)
            return self.norm(hidden_states, cond=cond)
        else:
            # Standard normalization
            normalized = self.norm(hidden_states)
            return normalized, None

    def embed_tokens(self, input_ids: Tensor) -> Tensor | None:
        """
        Get token embeddings from input IDs.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Token embeddings or None if not available.
        """
        if self.embed_tokens_module is None:
            return None
        return self.embed_tokens_module(input_ids)

    def get_rotary_embedding(self) -> nn.Module | None:
        """Return the rotary embedding module if available."""
        return self.rotary_emb_module

    def supports_adarms(self) -> bool:
        """Check if this backbone uses AdaRMS normalization."""
        return self._use_adarms


class SimpleTransformerBlock(nn.Module):
    """
    A simple transformer block for testing and prototyping.

    This is a minimal implementation that can be used to quickly test
    the MoT framework without loading full pretrained models.

    Architecture:
        - Input LayerNorm
        - Multi-Head Self-Attention
        - Residual connection
        - Post-Attention LayerNorm
        - Feed-Forward Network (MLP)
        - Residual connection

    Note: This is primarily for testing. For production use, consider
    using proper implementations from HuggingFace or other libraries.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int | None = None,
        head_dim: int | None = None,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ):
        """
        Initialize the simple transformer block.

        Args:
            hidden_size: Hidden dimension size.
            num_attention_heads: Number of attention heads.
            intermediate_size: Size of the FFN intermediate layer (default: 4*hidden_size).
            head_dim: Attention head dimension (default: hidden_size // num_attention_heads).
            dropout: Dropout probability.
            layer_norm_eps: Epsilon for layer normalization.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.intermediate_size = intermediate_size or (4 * hidden_size)

        # Normalization layers
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # Self-attention
        self.self_attn = SimpleAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            head_dim=self.head_dim,
            dropout=dropout,
        )

        # MLP
        self.mlp = SimpleMLP(
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """
        Forward pass through the transformer block.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs (currently unused).
            past_key_value: Optional cached key-value pair.
            use_cache: Whether to return key-value cache.
            **kwargs: Additional arguments (ignored).

        Returns:
            tuple: (output hidden states, optional key-value cache)
        """
        residual = hidden_states

        # Pre-attention normalization
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        attn_output, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # First residual connection
        hidden_states = residual + self.dropout(attn_output)
        residual = hidden_states

        # Post-attention normalization
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        mlp_output = self.mlp(hidden_states)

        # Second residual connection
        hidden_states = residual + self.dropout(mlp_output)

        return hidden_states, present_key_value if use_cache else None


class SimpleAttention(nn.Module):
    """Simple multi-head attention module for testing."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Forward pass through attention."""
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle cached key-values for incremental decoding
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        present_key_value = (key, value) if use_cache else None

        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scaling

        # Apply attention mask
        if attention_mask is not None:
            # Expand mask for heads dimension
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        attn_output = torch.matmul(attn_weights, value)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


class SimpleMLP(nn.Module):
    """Simple MLP/FFN module for testing."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MLP with SiLU gating."""
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


def create_simple_backbone(
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    intermediate_size: int | None = None,
    head_dim: int | None = None,
    dropout: float = 0.0,
    layer_norm_eps: float = 1e-6,
    vocab_size: int | None = None,
) -> GenericPyTorchWrapper:
    """
    Create a simple backbone for testing purposes.

    This is a convenience function to quickly create a minimal transformer
    backbone for testing the MoT framework.

    Args:
        num_layers: Number of transformer layers.
        hidden_size: Hidden dimension size.
        num_attention_heads: Number of attention heads.
        intermediate_size: FFN intermediate size (default: 4*hidden_size).
        head_dim: Attention head dimension (default: hidden_size // num_attention_heads).
        dropout: Dropout probability.
        layer_norm_eps: Layer norm epsilon.
        vocab_size: Vocabulary size for token embeddings (optional).

    Returns:
        GenericPyTorchWrapper wrapping the simple transformer.

    Example:
        >>> wrapper = create_simple_backbone(
        ...     num_layers=6,
        ...     hidden_size=256,
        ...     num_attention_heads=4,
        ... )
        >>> print(wrapper.num_layers)  # 6
    """
    head_dim = head_dim or (hidden_size // num_attention_heads)
    intermediate_size = intermediate_size or (4 * hidden_size)

    # Create layers
    layers = nn.ModuleList([
        SimpleTransformerBlock(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        for _ in range(num_layers)
    ])

    # Create final norm
    norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    # Create optional embedding layer
    embed_tokens = None
    if vocab_size is not None:
        embed_tokens = nn.Embedding(vocab_size, hidden_size)

    return GenericPyTorchWrapper(
        layers=layers,
        norm=norm,
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_attention_heads=num_attention_heads,
        embed_tokens=embed_tokens,
    )
