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
MoT Backbone Wrapper Protocol

This module defines the abstract base class for backbone wrappers in the MoT (Multistream-VLA) architecture.
The wrapper provides a unified interface for accessing transformer layers from different model implementations
(HuggingFace Transformers, custom PyTorch modules, etc.).

The core philosophy is "mechanism vs policy separation":
- MoT Core handles the mechanism (synchronized layer loops, Joint Attention, KV Cache)
- Backbone Wrappers shield the core from underlying model differences
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.mot.backbones import rope_utils


class MoTBackboneWrapper(nn.Module, ABC):
    """
    Abstract base class defining the protocol for MoT backbone wrappers.

    This wrapper provides a unified interface for the MoT Core to interact with different
    transformer backbone implementations. It abstracts away the differences between
    HuggingFace Transformers, custom PyTorch modules, and other implementations.

    All concrete implementations must implement the following interface:
    - num_layers: Total number of transformer layers
    - hidden_size: Hidden dimension size
    - head_dim: Attention head dimension
    - num_attention_heads: Number of attention heads
    - num_key_value_heads: Number of KV heads (for GQA/MQA)
    - get_layer(idx): Get the transformer block at index idx
    - forward_norm(hidden_states): Apply final layer normalization
    - embed_tokens(input_ids): Get token embeddings (optional, may return None)

    Example usage:
        >>> wrapper = SomeBackboneWrapper(model)
        >>> for layer_idx in range(wrapper.num_layers):
        ...     layer = wrapper.get_layer(layer_idx)
        ...     hidden_states = layer(hidden_states, attention_mask=mask)
        >>> output = wrapper.forward_norm(hidden_states)
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """
        Return the total number of transformer layers in the backbone.

        Returns:
            int: Number of transformer layers.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """
        Return the hidden dimension size of the backbone.

        Returns:
            int: Hidden dimension size.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def head_dim(self) -> int:
        """
        Return the dimension of each attention head.

        Returns:
            int: Attention head dimension.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_attention_heads(self) -> int:
        """
        Return the number of attention heads.

        Returns:
            int: Number of attention heads.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_key_value_heads(self) -> int:
        """
        Return the number of key-value heads (for GQA/MQA support).

        For standard multi-head attention, this equals num_attention_heads.
        For grouped-query attention (GQA), this is less than num_attention_heads.
        For multi-query attention (MQA), this is 1.

        Returns:
            int: Number of key-value heads.
        """
        raise NotImplementedError

    @property
    def model(self) -> nn.Module | None:
        """
        Return the underlying model if available.

        This property provides access to the wrapped model for config inspection
        and other purposes. Subclasses should override to return their model.

        Note: This is NOT registered as a submodule to avoid duplicate parameters.

        Returns:
            nn.Module | None: The underlying model, or None if not available.
        """
        return None

    @abstractmethod
    def get_layer(self, idx: int) -> nn.Module:
        """
        Return the transformer layer (block) at the specified index.

        Args:
            idx: Index of the layer to retrieve (0-indexed).

        Returns:
            nn.Module: The transformer layer at the specified index.

        Raises:
            IndexError: If idx is out of bounds.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_norm(
        self,
        hidden_states: Tensor,
        cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Apply the final layer normalization to the hidden states.

        This typically corresponds to the final RMSNorm or LayerNorm layer
        after all transformer blocks have been processed.

        Args:
            hidden_states: The hidden states tensor of shape (batch, seq_len, hidden_size).
            cond: Optional conditioning tensor for adaptive normalization (e.g., AdaRMS).

        Returns:
            tuple[Tensor, Tensor | None]:
                - The normalized hidden states tensor.
                - Optional gate tensor (for adaptive normalization, None otherwise).
        """
        raise NotImplementedError

    @abstractmethod
    def embed_tokens(self, input_ids: Tensor) -> Tensor | None:
        """
        Get token embeddings from input IDs.

        This method is optional and may return None if the backbone doesn't
        have a token embedding layer (e.g., for action experts).

        Args:
            input_ids: Token IDs tensor of shape (batch, seq_len).

        Returns:
            Tensor | None: Token embeddings of shape (batch, seq_len, hidden_size),
                          or None if not available.
        """
        raise NotImplementedError

    @abstractmethod
    def get_rotary_embedding(self) -> nn.Module | None:
        """
        Return the rotary position embedding module if available.

        Returns:
            nn.Module | None: The rotary embedding module, or None if not available.
        """
        raise NotImplementedError

    def get_rope_config(self) -> dict | None:
        """
        Return the RoPE (Rotary Position Embedding) configuration for this backbone.

        This is used to validate RoPE compatibility across streams in Joint Attention.
        For Joint Attention to be mathematically valid, all streams must share the same
        RoPE space geometry (theta, scaling, head_dim).

        Returns:
            dict | None: Dictionary containing RoPE config with keys:
                - rope_theta: Base theta for rotary embeddings
                - rope_scaling: Scaling configuration (if any)
                - head_dim: Head dimension for position encoding
                Returns None if RoPE is not used or config not available.
        """
        # Default implementation tries to extract from model config
        model = self.model
        if model is None:
            return None

        config = getattr(model, "config", None)
        if config is None:
            return None

        # Try to get RoPE parameters from config
        rope_config = {}

        # Common parameter names across different model architectures
        theta_names = ["rope_theta", "rotary_emb_base", "rope_base"]
        for name in theta_names:
            if hasattr(config, name):
                rope_config["rope_theta"] = getattr(config, name)
                break

        # Scaling config
        scaling_names = ["rope_scaling", "rotary_scaling"]
        for name in scaling_names:
            if hasattr(config, name):
                rope_config["rope_scaling"] = getattr(config, name)
                break

        # Head dimension
        rope_config["head_dim"] = self.head_dim

        return rope_config if rope_config else None

    def get_layer_input_layernorm(self, idx: int) -> nn.Module | None:
        """
        Return the input layer normalization module for a specific layer.

        This is useful for separate access to normalization during layer computation.

        Args:
            idx: Index of the layer.

        Returns:
            nn.Module | None: The input layer norm, or None if not separately accessible.
        """
        layer = self.get_layer(idx)
        if hasattr(layer, "input_layernorm"):
            return layer.input_layernorm
        return None

    def get_layer_post_attention_layernorm(self, idx: int) -> nn.Module | None:
        """
        Return the post-attention layer normalization module for a specific layer.

        Args:
            idx: Index of the layer.

        Returns:
            nn.Module | None: The post-attention layer norm, or None if not separately accessible.
        """
        layer = self.get_layer(idx)
        if hasattr(layer, "post_attention_layernorm"):
            return layer.post_attention_layernorm
        return None

    def get_layer_self_attn(self, idx: int) -> nn.Module | None:
        """
        Return the self-attention module for a specific layer.

        Args:
            idx: Index of the layer.

        Returns:
            nn.Module | None: The self-attention module, or None if not separately accessible.
        """
        layer = self.get_layer(idx)
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        return None

    def get_layer_mlp(self, idx: int) -> nn.Module | None:
        """
        Return the MLP/FFN module for a specific layer.

        Args:
            idx: Index of the layer.

        Returns:
            nn.Module | None: The MLP module, or None if not separately accessible.
        """
        layer = self.get_layer(idx)
        if hasattr(layer, "mlp"):
            return layer.mlp
        return None

    def project_qkv(
        self,
        hidden_states: Tensor,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Project hidden states to query, key, value tensors.

        This method abstracts the QKV projection to support different model architectures:
        - Gemma/Llama: Uses separate q_proj, k_proj, v_proj Linear layers
        - GPT-2/Bloom: Uses combined c_attn Conv1D
        - Custom: May have different naming conventions

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            layer_idx: Index of the current layer.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Query, Key, Value tensors.
                Each of shape (batch, num_heads, seq_len, head_dim) after reshaping.
        """
        self_attn = self.get_layer_self_attn(layer_idx)
        if self_attn is None:
            raise ValueError(f"Could not get self_attn module for layer {layer_idx}")

        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        # Try standard Llama/Gemma style (separate projections)
        if hasattr(self_attn, "q_proj") and hasattr(self_attn, "k_proj") and hasattr(self_attn, "v_proj"):
            query = self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key = self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            return query, key, value

        # Try GPT-2/Bloom style (combined c_attn)
        if hasattr(self_attn, "c_attn"):
            # c_attn projects to 3 * hidden_size, then split
            qkv = self_attn.c_attn(hidden_states)
            # Split into Q, K, V
            split_size = qkv.shape[-1] // 3
            query, key, value = qkv.split(split_size, dim=-1)
            query = query.view(hidden_shape).transpose(1, 2)
            key = key.view(hidden_shape).transpose(1, 2)
            value = value.view(hidden_shape).transpose(1, 2)
            return query, key, value

        # Try combined qkv_proj style
        if hasattr(self_attn, "qkv_proj"):
            qkv = self_attn.qkv_proj(hidden_states)
            split_size = qkv.shape[-1] // 3
            query, key, value = qkv.split(split_size, dim=-1)
            query = query.view(hidden_shape).transpose(1, 2)
            key = key.view(hidden_shape).transpose(1, 2)
            value = value.view(hidden_shape).transpose(1, 2)
            return query, key, value

        raise NotImplementedError(
            f"Unknown attention projection style for {type(self_attn)}. "
            "Please override project_qkv() in your wrapper subclass."
        )

    def project_output(
        self,
        attn_output: Tensor,
        layer_idx: int,
    ) -> Tensor:
        """
        Apply output projection after attention computation.

        Args:
            attn_output: Attention output of shape (batch, seq_len, num_heads * head_dim).
            layer_idx: Index of the current layer.

        Returns:
            Tensor: Projected output of shape (batch, seq_len, hidden_size).
        """
        self_attn = self.get_layer_self_attn(layer_idx)
        if self_attn is None:
            raise ValueError(f"Could not get self_attn module for layer {layer_idx}")

        # Try standard o_proj (Llama/Gemma style)
        if hasattr(self_attn, "o_proj"):
            return self_attn.o_proj(attn_output)

        # Try GPT-2 style c_proj
        if hasattr(self_attn, "c_proj"):
            return self_attn.c_proj(attn_output)

        # Try out_proj style
        if hasattr(self_attn, "out_proj"):
            return self_attn.out_proj(attn_output)

        raise NotImplementedError(
            f"Unknown output projection style for {type(self_attn)}. "
            "Please override project_output() in your wrapper subclass."
        )

    def apply_input_layernorm(
        self,
        hidden_states: Tensor,
        layer_idx: int,
        cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Apply input layer normalization with optional AdaRMS conditioning.

        This method safely handles both standard LayerNorm and AdaRMS variants.

        Args:
            hidden_states: Input tensor.
            layer_idx: Index of the current layer.
            cond: Optional conditioning tensor for AdaRMS.

        Returns:
            tuple[Tensor, Tensor | None]: (normalized output, gate tensor or None)
        """
        input_ln = self.get_layer_input_layernorm(layer_idx)
        if input_ln is None:
            return hidden_states, None

        if self.supports_adarms() and cond is not None:
            # Try to call with cond parameter (AdaRMS style)
            try:
                result = input_ln(hidden_states, cond=cond)
                # AdaRMS returns (output, gate) tuple
                if isinstance(result, tuple):
                    return result
                else:
                    return result, None
            except TypeError:
                # Fallback if cond is not supported
                pass

        # Standard LayerNorm call
        result = input_ln(hidden_states)
        # Handle case where LayerNorm might still return a tuple (e.g., modified models)
        if isinstance(result, tuple):
            return result
        return result, None

    def apply_post_attention_layernorm(
        self,
        hidden_states: Tensor,
        layer_idx: int,
        cond: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Apply post-attention layer normalization with optional AdaRMS conditioning.

        Args:
            hidden_states: Input tensor.
            layer_idx: Index of the current layer.
            cond: Optional conditioning tensor for AdaRMS.

        Returns:
            tuple[Tensor, Tensor | None]: (normalized output, gate tensor or None)
        """
        post_ln = self.get_layer_post_attention_layernorm(layer_idx)
        if post_ln is None:
            return hidden_states, None

        if self.supports_adarms() and cond is not None:
            # Try to call with cond parameter (AdaRMS style)
            try:
                result = post_ln(hidden_states, cond=cond)
                # AdaRMS returns (output, gate) tuple
                if isinstance(result, tuple):
                    return result
                else:
                    return result, None
            except TypeError:
                # Fallback if cond is not supported
                pass

        # Standard LayerNorm call
        result = post_ln(hidden_states)
        # Handle case where LayerNorm might still return a tuple (e.g., modified models)
        if isinstance(result, tuple):
            return result
        return result, None

    def get_scaling(self, idx: int) -> float:
        """
        Return the attention scaling factor for a specific layer.

        Args:
            idx: Index of the layer.

        Returns:
            float: The attention scaling factor (typically 1/sqrt(head_dim)).
        """
        attn = self.get_layer_self_attn(idx)
        if attn is not None and hasattr(attn, "scaling"):
            return attn.scaling
        # Default scaling
        return 1.0 / (self.head_dim ** 0.5)

    def supports_adarms(self) -> bool:
        """
        Check if the backbone supports adaptive RMS normalization (AdaRMS).

        Returns:
            bool: True if AdaRMS is supported, False otherwise.
        """
        return False

    # ========================================================================
    # Math Operations Interface
    # ========================================================================
    # These methods provide a unified interface for mathematical operations
    # that may differ across model architectures. The MoT Core uses these
    # interfaces to remain model-agnostic.
    # ========================================================================

    def apply_rotary_pos_emb(
        self,
        query: Tensor,
        key: Tensor,
        cos: Tensor,
        sin: Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """
        Apply Rotary Position Embeddings (RoPE) to query and key states.

        This method provides a unified interface for RoPE application.
        Subclasses can override this to use model-specific implementations
        (e.g., Gemma uses float32 precision for RoPE computation).

        Args:
            query: Query states of shape (batch, num_heads, seq_len, head_dim).
            key: Key states of shape (batch, num_kv_heads, seq_len, head_dim).
            cos: Cosine part of rotary embeddings.
            sin: Sine part of rotary embeddings.
            unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting.

        Returns:
            tuple[Tensor, Tensor]: Rotary-embedded (query, key) tensors.
        """
        # Default implementation uses the standard RoPE utility
        return rope_utils.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim)

    def compute_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        scaling: float,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Compute Scaled Dot-Product Attention.

        This method provides a unified interface for attention computation.
        Subclasses can override this to use model-specific implementations
        (e.g., Gemma's eager_attention_forward).

        Args:
            query: Query states of shape (batch, num_heads, seq_len, head_dim).
            key: Key states of shape (batch, num_kv_heads, kv_len, head_dim).
            value: Value states of shape (batch, num_kv_heads, kv_len, head_dim).
            attention_mask: Optional attention mask of shape (batch, 1, seq_len, kv_len)
                           or broadcastable. Values should be 0 or -inf.
            scaling: Attention scaling factor (typically 1/sqrt(head_dim)).

        Returns:
            tuple[Tensor, Tensor | None]:
                - Attention output of shape (batch, num_heads, seq_len, head_dim).
                - Optional attention weights (may be None for efficiency).
        """
        # Default implementation: standard scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def apply_gated_residual(
        self,
        original: Tensor,
        update: Tensor,
        gate: Tensor | None,
    ) -> Tensor:
        """
        Apply residual connection, optionally with gating.

        This method provides a unified interface for residual connections.
        Some models (like Gemma2 with AdaRMS) use gated residuals where the
        update is scaled by a gate before adding to the original.

        Args:
            original: Original tensor before the transformation.
            update: The update/output from the transformation (e.g., attention or MLP).
            gate: Optional gate tensor for scaling the update.
                  If None, performs standard residual: original + update.

        Returns:
            Tensor: Result of the residual connection.
        """
        if gate is not None:
            # Gated residual: original + gate * update
            # This is the pattern used in Gemma2/Pi0 with AdaRMS
            return original + gate * update
        else:
            # Standard residual connection
            return original + update

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
        cond: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]] | None]:
        """
        Default forward pass through all transformer layers.

        This provides a standard implementation that iterates through all layers.
        Subclasses may override this for custom behavior.

        Args:
            hidden_states: Input hidden states of shape (batch, seq_len, hidden_size).
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs for positional encoding.
            past_key_values: Optional cached key-value pairs for incremental decoding.
            use_cache: Whether to return key-value cache.
            cond: Optional conditioning tensor for adaptive normalization.
            **kwargs: Additional arguments passed to each layer.

        Returns:
            tuple[Tensor, list[tuple[Tensor, Tensor]] | None]:
                - Output hidden states after all layers and final norm.
                - Optional list of key-value caches if use_cache is True.
        """
        all_key_values = [] if use_cache else None

        for idx in range(self.num_layers):
            layer = self.get_layer(idx)

            layer_past = past_key_values[idx] if past_key_values is not None else None

            # Standard layer forward call
            # Note: actual layer calling convention may vary by model architecture
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
                **kwargs,
            )

            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                if use_cache and len(layer_outputs) > 1:
                    all_key_values.append(layer_outputs[1])
            else:
                hidden_states = layer_outputs

        # Apply final normalization
        hidden_states, _ = self.forward_norm(hidden_states, cond=cond)

        return hidden_states, all_key_values


class BackboneWrapperConfig:
    """
    Configuration for backbone wrapper instantiation.

    This class holds the necessary information to create a backbone wrapper,
    including the type of wrapper to use and model-specific parameters.

    Attributes:
        backbone_type: Type of backbone ("hf" for HuggingFace, "custom" for PyTorch).
        model_path: Path or identifier for loading the model (for HF models).
        layers_attr: Attribute path to access layers (e.g., "model.layers").
        norm_attr: Attribute path to access final norm (e.g., "model.norm").
        embed_tokens_attr: Attribute path to access embedding layer (optional).
        rotary_emb_attr: Attribute path to access rotary embeddings (optional).
        config_overrides: Dictionary of config overrides for model initialization.
    """

    def __init__(
        self,
        backbone_type: str = "hf",
        model_path: str | None = None,
        layers_attr: str = "model.layers",
        norm_attr: str = "model.norm",
        embed_tokens_attr: str | None = "model.embed_tokens",
        rotary_emb_attr: str | None = "model.rotary_emb",
        config_overrides: dict[str, Any] | None = None,
    ):
        self.backbone_type = backbone_type
        self.model_path = model_path
        self.layers_attr = layers_attr
        self.norm_attr = norm_attr
        self.embed_tokens_attr = embed_tokens_attr
        self.rotary_emb_attr = rotary_emb_attr
        self.config_overrides = config_overrides or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "backbone_type": self.backbone_type,
            "model_path": self.model_path,
            "layers_attr": self.layers_attr,
            "norm_attr": self.norm_attr,
            "embed_tokens_attr": self.embed_tokens_attr,
            "rotary_emb_attr": self.rotary_emb_attr,
            "config_overrides": self.config_overrides,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BackboneWrapperConfig":
        """Create config from dictionary."""
        return cls(**d)
