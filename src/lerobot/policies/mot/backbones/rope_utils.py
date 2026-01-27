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
Rotary Position Embedding Utilities

This module provides default implementations for Rotary Position Embeddings (RoPE)
that can be used as fallbacks when model-specific implementations are not available.

These implementations are model-agnostic and follow the standard RoPE formulation.
Specific model adapters (e.g., HuggingFaceBackboneWrapper) can override these with
model-specific implementations when needed.
"""

import torch
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    """
    Rotate half the hidden dims of the input.

    This is a standard operation in RoPE where we split the input into two halves
    and rotate them: [x1, x2] -> [-x2, x1]

    Args:
        x: Input tensor of shape (..., head_dim).

    Returns:
        Rotated tensor of the same shape.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[Tensor, Tensor]:
    """
    Apply Rotary Position Embeddings (RoPE) to query and key states.

    This is a standard implementation of RoPE that can be used as a fallback.
    Model-specific implementations (e.g., for Gemma) may handle precision
    differently.

    Args:
        query: Query states of shape (batch, num_heads, seq_len, head_dim).
        key: Key states of shape (batch, num_kv_heads, seq_len, head_dim).
        cos: Cosine part of rotary embeddings of shape (seq_len, head_dim) or
             (batch, seq_len, head_dim).
        sin: Sine part of rotary embeddings of shape (seq_len, head_dim) or
             (batch, seq_len, head_dim).
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting.
                       Default is 1 to match (batch, num_heads, seq_len, head_dim).

    Returns:
        tuple[Tensor, Tensor]: Rotary-embedded query and key tensors.

    Note:
        The formula for RoPE is:
        q_embed = q * cos + rotate_half(q) * sin
        k_embed = k * cos + rotate_half(k) * sin
    """
    # Expand cos and sin for broadcasting with query/key
    # cos/sin shape: (seq_len, head_dim) or (batch, seq_len, head_dim)
    # target shape: (batch, num_heads, seq_len, head_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Apply rotary embeddings
    query_embed = (query * cos) + (rotate_half(query) * sin)
    key_embed = (key * cos) + (rotate_half(key) * sin)

    return query_embed, key_embed


def apply_rotary_pos_emb_with_precision(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueeze_dim: int = 1,
    compute_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Apply Rotary Position Embeddings with explicit precision handling.

    This version allows specifying a compute dtype for intermediate calculations,
    which is important for models like Gemma that require float32 precision
    for accurate RoPE computation even when the rest of the model uses bfloat16.

    Args:
        query: Query states of shape (batch, num_heads, seq_len, head_dim).
        key: Key states of shape (batch, num_kv_heads, seq_len, head_dim).
        cos: Cosine part of rotary embeddings.
        sin: Sine part of rotary embeddings.
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting.
        compute_dtype: Optional dtype for intermediate computations.
                       If None, uses the input dtype.

    Returns:
        tuple[Tensor, Tensor]: Rotary-embedded query and key tensors.
    """
    original_query_dtype = query.dtype
    original_key_dtype = key.dtype

    if compute_dtype is not None:
        query = query.to(compute_dtype)
        key = key.to(compute_dtype)
        cos = cos.to(compute_dtype)
        sin = sin.to(compute_dtype)

    # Expand cos and sin for broadcasting
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Apply rotary embeddings
    query_embed = (query * cos) + (rotate_half(query) * sin)
    key_embed = (key * cos) + (rotate_half(key) * sin)

    # Convert back to original dtype if we changed it
    if compute_dtype is not None:
        query_embed = query_embed.to(original_query_dtype)
        key_embed = key_embed.to(original_key_dtype)

    return query_embed, key_embed
