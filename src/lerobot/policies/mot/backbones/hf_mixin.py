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
HuggingFace Transformers Backbone Adapter

This module provides an adapter for wrapping HuggingFace Transformers models
to conform to the MoT backbone wrapper protocol. It allows seamless integration
of pretrained models like PaliGemma, Gemma, Llama, etc.

The adapter uses string-based attribute paths to locate layers and normalization
modules within the HuggingFace model hierarchy.

Strategy Pattern:
    This module uses the Strategy Pattern with registries for model-specific
    operations (RoPE, gated residual). Instead of if-else chains that check
    model type at runtime, we bind the appropriate strategy function during
    initialization. This follows the Open-Closed Principle - to support a new
    model, just add an entry to the registry without modifying existing code.
"""

import logging
from functools import reduce
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.mot.backbones import rope_utils
from lerobot.policies.mot.backbones.wrapper import (BackboneWrapperConfig,
                                                    MoTBackboneWrapper)

logger = logging.getLogger(__name__)


# Preset configurations for common HuggingFace models
HF_MODEL_PRESETS: dict[str, dict[str, str]] = {
    "gemma": {
        "layers_attr": "model.layers",
        "norm_attr": "model.norm",
        "embed_tokens_attr": "model.embed_tokens",
        "rotary_emb_attr": "model.rotary_emb",
    },
    "gemma_lm": {
        "layers_attr": "model.layers",
        "norm_attr": "model.norm",
        "embed_tokens_attr": "embed_tokens",
        "rotary_emb_attr": "model.rotary_emb",
    },
    "paligemma": {
        "layers_attr": "language_model.model.layers",
        "norm_attr": "language_model.model.norm",
        "embed_tokens_attr": "language_model.embed_tokens",
        "rotary_emb_attr": "language_model.model.rotary_emb",
    },
    "paligemma_language": {
        "layers_attr": "model.layers",
        "norm_attr": "model.norm",
        "embed_tokens_attr": "embed_tokens",
        "rotary_emb_attr": "model.rotary_emb",
    },
    "llama": {
        "layers_attr": "model.layers",
        "norm_attr": "model.norm",
        "embed_tokens_attr": "model.embed_tokens",
        "rotary_emb_attr": "model.rotary_emb",
    },
    "mistral": {
        "layers_attr": "model.layers",
        "norm_attr": "model.norm",
        "embed_tokens_attr": "model.embed_tokens",
        "rotary_emb_attr": "model.rotary_emb",
    },
}


def get_hf_preset(preset_name: str) -> dict[str, str]:
    """
    Get a preset configuration for a common HuggingFace model.

    Args:
        preset_name: Name of the preset (e.g., "gemma", "paligemma", "llama").

    Returns:
        Dictionary of attribute paths.

    Raises:
        ValueError: If preset name is not recognized.
    """
    if preset_name not in HF_MODEL_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {list(HF_MODEL_PRESETS.keys())}"
        )
    return HF_MODEL_PRESETS[preset_name].copy()


# ============================================================================
# Strategy Functions for RoPE (Rotary Position Embeddings)
# ============================================================================
# Each strategy function has the same signature:
#   (query, key, cos, sin, unsqueeze_dim) -> (query_embed, key_embed)
# ============================================================================

def _default_rope_strategy(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[Tensor, Tensor]:
    """Default RoPE strategy using our own implementation."""
    return rope_utils.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim)


def _gemma_rope_strategy(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[Tensor, Tensor]:
    """Gemma-family RoPE strategy with float32 precision handling."""
    try:
        from transformers.models.gemma.modeling_gemma import \
            apply_rotary_pos_emb as gemma_apply_rotary_pos_emb
        return gemma_apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=unsqueeze_dim)
    except ImportError:
        logger.warning(
            "Could not import Gemma-specific apply_rotary_pos_emb, "
            "falling back to default implementation"
        )
        return _default_rope_strategy(query, key, cos, sin, unsqueeze_dim)


def _llama_rope_strategy(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[Tensor, Tensor]:
    """Llama-family RoPE strategy."""
    try:
        from transformers.models.llama.modeling_llama import \
            apply_rotary_pos_emb as llama_apply_rotary_pos_emb
        return llama_apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=unsqueeze_dim)
    except ImportError:
        logger.warning(
            "Could not import Llama-specific apply_rotary_pos_emb, "
            "falling back to default implementation"
        )
        return _default_rope_strategy(query, key, cos, sin, unsqueeze_dim)


# ============================================================================
# Strategy Functions for Gated Residual Connections
# ============================================================================
# Each strategy function has the same signature:
#   (original, update, gate) -> result
# ============================================================================

def _default_gated_residual_strategy(
    original: Tensor,
    update: Tensor,
    gate: Tensor | None,
) -> Tensor:
    """Default gated residual: original + gate * update (or just original + update)."""
    if gate is not None:
        return original + gate * update
    return original + update


def _gemma_gated_residual_strategy(
    original: Tensor,
    update: Tensor,
    gate: Tensor | None,
) -> Tensor:
    """Gemma-family gated residual strategy."""
    if gate is not None:
        try:
            from transformers.models.gemma.modeling_gemma import \
                _gated_residual as gemma_gated_residual
            return gemma_gated_residual(original, update, gate)
        except (ImportError, AttributeError):
            # If not available, use the default implementation
            pass
    return _default_gated_residual_strategy(original, update, gate)


# ============================================================================
# Strategy Registries
# ============================================================================
# Maps model_type (from HF config) to the corresponding strategy function.
# To support a new model, simply add an entry here - no need to modify the
# HuggingFaceBackboneWrapper class itself (Open-Closed Principle).
# ============================================================================

# RoPE Strategy Registry: model_type -> rope_strategy_function
ROPE_STRATEGY_REGISTRY: dict[str, Callable] = {
    # Gemma family
    "gemma": _gemma_rope_strategy,
    "gemma2": _gemma_rope_strategy,
    "paligemma": _gemma_rope_strategy,
    # Llama family
    "llama": _llama_rope_strategy,
    "mistral": _llama_rope_strategy,
    # Add more models here as needed:
    # "qwen2": _qwen_rope_strategy,
    # "phi": _phi_rope_strategy,
}

# Gated Residual Strategy Registry: model_type -> gated_residual_strategy_function
GATED_RESIDUAL_STRATEGY_REGISTRY: dict[str, Callable] = {
    # Gemma family (uses special gated residual)
    "gemma": _gemma_gated_residual_strategy,
    "gemma2": _gemma_gated_residual_strategy,
    "paligemma": _gemma_gated_residual_strategy,
    # Other models use default (no special gating)
    # "llama": _default_gated_residual_strategy,  # Not needed, default is used
}


# ============================================================================
# Utility Functions
# ============================================================================


def get_nested_attr(obj: Any, attr_path: str) -> Any:
    """
    Get a nested attribute using dot-separated path.

    Args:
        obj: The object to get the attribute from.
        attr_path: Dot-separated path like "model.layers" or "language_model.model.norm".

    Returns:
        The attribute at the specified path.

    Raises:
        AttributeError: If any part of the path doesn't exist.
    """
    return reduce(getattr, attr_path.split("."), obj)


def has_nested_attr(obj: Any, attr_path: str) -> bool:
    """
    Check if a nested attribute exists.

    Args:
        obj: The object to check.
        attr_path: Dot-separated path.

    Returns:
        bool: True if the attribute exists, False otherwise.
    """
    try:
        get_nested_attr(obj, attr_path)
        return True
    except AttributeError:
        return False


class HuggingFaceBackboneWrapper(MoTBackboneWrapper):
    """
    Backbone wrapper for HuggingFace Transformers models.

    This wrapper adapts HuggingFace PreTrainedModel instances to the MoT backbone
    protocol. It uses configurable attribute paths to locate the transformer layers,
    normalization modules, and embedding layers within different HF model architectures.

    Supported model types:
    - Gemma / Gemma2
    - PaliGemma
    - Llama / Llama2 / Llama3
    - Mistral
    - And other models with similar architecture

    Example usage:
        >>> from transformers import GemmaForCausalLM
        >>> hf_model = GemmaForCausalLM.from_pretrained("google/gemma-2b")
        >>> wrapper = HuggingFaceBackboneWrapper(
        ...     model=hf_model,
        ...     layers_attr="model.layers",
        ...     norm_attr="model.norm",
        ... )
        >>> print(wrapper.num_layers)  # e.g., 18

    Attributes:
        model: The wrapped HuggingFace model.
        layers: The ModuleList of transformer layers.
        norm: The final normalization layer.
        embed_tokens_module: The token embedding layer (if available).
        rotary_emb_module: The rotary embedding module (if available).

    Note:
        This wrapper does NOT own the model - it only holds references to it.
        The model should be owned by the parent policy class.
        We use _model, _layers, _norm, etc. (with underscores) to prevent
        PyTorch from registering them as submodules.
    """

    def __init__(
        self,
        model: nn.Module,
        layers_attr: str = "model.layers",
        norm_attr: str = "model.norm",
        embed_tokens_attr: str | None = "model.embed_tokens",
        rotary_emb_attr: str | None = "model.rotary_emb",
        hidden_size: int | None = None,
        head_dim: int | None = None,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        use_adarms: bool = False,
    ):
        """
        Initialize the HuggingFace backbone wrapper.

        Args:
            model: HuggingFace PreTrainedModel instance.
            layers_attr: Dot-separated path to the transformer layers ModuleList.
            norm_attr: Dot-separated path to the final normalization layer.
            embed_tokens_attr: Dot-separated path to the token embedding layer (optional).
            rotary_emb_attr: Dot-separated path to the rotary embedding module (optional).
            hidden_size: Override for hidden size (auto-detected if None).
            head_dim: Override for head dimension (auto-detected if None).
            num_attention_heads: Override for number of attention heads (auto-detected if None).
            num_key_value_heads: Override for number of KV heads (auto-detected if None).
            use_adarms: Whether this model uses adaptive RMS normalization.
        """
        super().__init__()

        # IMPORTANT: Use object.__setattr__ to bypass PyTorch's module registration.
        # The wrapper does NOT own these modules - it only holds references to them.
        # The actual ownership is in the parent policy (e.g., MoTPI0Policy.paligemma_with_expert).
        # This prevents duplicate parameter registration which causes optimizer errors.
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_use_adarms", use_adarms)

        # Get layers (store as non-module attribute)
        try:
            layers = get_nested_attr(model, layers_attr)
        except AttributeError as e:
            raise ValueError(
                f"Could not find layers at '{layers_attr}'. "
                f"Please check the model architecture and provide the correct path."
            ) from e

        if not isinstance(layers, nn.ModuleList):
            raise TypeError(
                f"Expected layers to be nn.ModuleList, got {type(layers)}. "
                f"Check the layers_attr path: '{layers_attr}'"
            )
        object.__setattr__(self, "_layers", layers)

        # Get norm (store as non-module attribute)
        try:
            norm = get_nested_attr(model, norm_attr)
        except AttributeError as e:
            raise ValueError(
                f"Could not find norm at '{norm_attr}'. "
                f"Please check the model architecture and provide the correct path."
            ) from e
        object.__setattr__(self, "_norm", norm)

        # Get optional embed_tokens (store as non-module attribute)
        embed_tokens = None
        if embed_tokens_attr and has_nested_attr(model, embed_tokens_attr):
            embed_tokens = get_nested_attr(model, embed_tokens_attr)
        object.__setattr__(self, "_embed_tokens_module", embed_tokens)

        # Get optional rotary embeddings (store as non-module attribute)
        rotary_emb = None
        if rotary_emb_attr and has_nested_attr(model, rotary_emb_attr):
            rotary_emb = get_nested_attr(model, rotary_emb_attr)
        object.__setattr__(self, "_rotary_emb_module", rotary_emb)

        # Auto-detect or use provided dimensions (these are plain values, not modules)
        self._hidden_size = hidden_size
        self._head_dim = head_dim
        self._num_attention_heads = num_attention_heads
        self._num_key_value_heads = num_key_value_heads

        if self._hidden_size is None:
            self._hidden_size = self._detect_hidden_size()
        if self._head_dim is None:
            self._head_dim = self._detect_head_dim()
        if self._num_attention_heads is None:
            self._num_attention_heads = self._detect_num_attention_heads()
        if self._num_key_value_heads is None:
            self._num_key_value_heads = self._detect_num_key_value_heads()

        # ====================================================================
        # Strategy Binding (Decision made once at initialization)
        # ====================================================================
        # Detect model type and bind appropriate strategy functions.
        # This avoids runtime if-else checks on every forward pass.
        model_type = self._detect_model_type()
        self._model_type = model_type

        # Bind RoPE strategy
        self._rope_strategy = ROPE_STRATEGY_REGISTRY.get(
            model_type, _default_rope_strategy
        )

        # Bind gated residual strategy
        self._gated_residual_strategy = GATED_RESIDUAL_STRATEGY_REGISTRY.get(
            model_type, _default_gated_residual_strategy
        )

        logger.debug(
            f"HuggingFaceBackboneWrapper initialized for model_type='{model_type}' "
            f"with rope_strategy={self._rope_strategy.__name__}, "
            f"gated_residual_strategy={self._gated_residual_strategy.__name__}"
        )

    def _detect_model_type(self) -> str:
        """
        Detect the model type from HuggingFace config.

        Returns:
            str: The model type (e.g., "gemma", "llama", "mistral").
                 Returns "unknown" if detection fails.
        """
        if hasattr(self._model, "config"):
            model_type = getattr(self._model.config, "model_type", "")
            if model_type:
                return model_type.lower()
        logger.warning(
            "Could not detect model_type from HuggingFace config. "
            "Please specify manually if needed in the registries. "
            "Defaulting to 'unknown'."
        )
        return "unknown"

    def _detect_hidden_size(self) -> int:
        """Auto-detect hidden size from model config or layer weights."""
        # Try to get from config
        if hasattr(self._model, "config"):
            config = self._model.config
            if hasattr(config, "hidden_size"):
                return config.hidden_size
            if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
                return config.text_config.hidden_size

        # Try to infer from first layer's weights
        if len(self._layers) > 0:
            layer = self._layers[0]
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
                return layer.self_attn.q_proj.weight.shape[1]
            if hasattr(layer, "input_layernorm") and hasattr(layer.input_layernorm, "weight"):
                return layer.input_layernorm.weight.shape[0]

        raise ValueError("Could not auto-detect hidden_size. Please provide it explicitly.")

    def _detect_head_dim(self) -> int:
        """Auto-detect head dimension from model config or layer weights."""
        if hasattr(self._model, "config"):
            config = self._model.config
            if hasattr(config, "head_dim"):
                return config.head_dim
            if hasattr(config, "text_config") and hasattr(config.text_config, "head_dim"):
                return config.text_config.head_dim

        # Try to infer from first layer
        if len(self._layers) > 0:
            layer = self._layers[0]
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "head_dim"):
                return layer.self_attn.head_dim

        # Default fallback
        return self._hidden_size // self._num_attention_heads if self._num_attention_heads else 64

    def _detect_num_attention_heads(self) -> int:
        """Auto-detect number of attention heads."""
        if hasattr(self._model, "config"):
            config = self._model.config
            if hasattr(config, "num_attention_heads"):
                return config.num_attention_heads
            if hasattr(config, "text_config") and hasattr(config.text_config, "num_attention_heads"):
                return config.text_config.num_attention_heads

        if len(self._layers) > 0:
            layer = self._layers[0]
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "num_heads"):
                return layer.self_attn.num_heads

        return 8  # Default fallback

    def _detect_num_key_value_heads(self) -> int:
        """Auto-detect number of key-value heads for GQA/MQA."""
        if hasattr(self._model, "config"):
            config = self._model.config
            if hasattr(config, "num_key_value_heads"):
                return config.num_key_value_heads
            if hasattr(config, "text_config") and hasattr(config.text_config, "num_key_value_heads"):
                return config.text_config.num_key_value_heads

        if len(self._layers) > 0:
            layer = self._layers[0]
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "num_key_value_heads"):
                return layer.self_attn.num_key_value_heads

        return self._num_attention_heads  # Default to MHA

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        return len(self._layers)

    @property
    def model(self) -> nn.Module:
        """Return the underlying HuggingFace model (for config inspection)."""
        return self._model

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
        if idx < 0 or idx >= len(self._layers):
            raise IndexError(f"Layer index {idx} out of range [0, {len(self._layers)})")
        return self._layers[idx]

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
            return self._norm(hidden_states, cond=cond)
        else:
            # Standard normalization
            result = self._norm(hidden_states)
            # Handle case where norm returns a tuple (e.g., Gemma2 style with gate)
            if isinstance(result, tuple):
                return result
            return result, None

    def embed_tokens(self, input_ids: Tensor) -> Tensor | None:
        """
        Get token embeddings from input IDs.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Token embeddings or None if not available.
        """
        if self._embed_tokens_module is None:
            return None
        return self._embed_tokens_module(input_ids)

    def get_rotary_embedding(self) -> nn.Module | None:
        """Return the rotary embedding module if available."""
        return self._rotary_emb_module

    def supports_adarms(self) -> bool:
        """Check if this backbone uses AdaRMS normalization."""
        return self._use_adarms

    # ========================================================================
    # Model-Specific Math Operations (Strategy Pattern)
    # ========================================================================
    # These methods use strategies bound at initialization time.
    # No runtime if-else checks - just direct function calls.
    # ========================================================================

    @property
    def model_type(self) -> str:
        """Return the detected model type (e.g., 'gemma', 'llama')."""
        return self._model_type

    def apply_rotary_pos_emb(
        self,
        query: Tensor,
        key: Tensor,
        cos: Tensor,
        sin: Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """
        Apply Rotary Position Embeddings using the bound strategy.

        The appropriate strategy function was selected at initialization time
        based on the model type. This avoids runtime if-else checks.

        Args:
            query: Query states.
            key: Key states.
            cos: Cosine part of rotary embeddings.
            sin: Sine part of rotary embeddings.
            unsqueeze_dim: Dimension to unsqueeze for broadcasting.

        Returns:
            tuple[Tensor, Tensor]: Rotary-embedded (query, key) tensors.
        """
        return self._rope_strategy(query, key, cos, sin, unsqueeze_dim)

    def compute_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        scaling: float,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Compute attention using the default implementation.

        For joint attention in MoT, we always use the default scaled dot-product
        attention implementation. This is because:
        1. Joint attention concatenates Q/K/V from multiple streams
        2. Model-specific implementations may require module attributes
           (e.g., num_key_value_groups) that don't apply in joint context
        3. The default implementation handles the math correctly for all cases

        Args:
            query: Query states.
            key: Key states.
            value: Value states.
            attention_mask: Optional attention mask.
            scaling: Attention scaling factor.

        Returns:
            tuple[Tensor, Tensor | None]: (attention output, optional weights)
        """
        # Use default implementation from parent class
        return super().compute_attention(query, key, value, attention_mask, scaling)

    def apply_gated_residual(
        self,
        original: Tensor,
        update: Tensor,
        gate: Tensor | None,
    ) -> Tensor:
        """
        Apply residual connection using the bound strategy.

        The appropriate strategy function was selected at initialization time
        based on the model type. This avoids runtime if-else checks.

        Args:
            original: Original tensor before transformation.
            update: The update from transformation (attention or MLP output).
            gate: Optional gate tensor for gated residual.

        Returns:
            Tensor: Result of the residual connection.
        """
        return self._gated_residual_strategy(original, update, gate)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        model_class: type | None = None,
        layers_attr: str = "model.layers",
        norm_attr: str = "model.norm",
        embed_tokens_attr: str | None = "model.embed_tokens",
        rotary_emb_attr: str | None = "model.rotary_emb",
        use_adarms: bool = False,
        **model_kwargs: Any,
    ) -> "HuggingFaceBackboneWrapper":
        """
        Create a wrapper from a pretrained model path.

        Args:
            model_name_or_path: HuggingFace model identifier or local path.
            model_class: The HF model class to use (e.g., GemmaForCausalLM).
                        If None, will use AutoModel.
            layers_attr: Path to layers ModuleList.
            norm_attr: Path to final norm.
            embed_tokens_attr: Path to token embeddings.
            rotary_emb_attr: Path to rotary embeddings.
            use_adarms: Whether to use AdaRMS normalization.
            **model_kwargs: Additional kwargs passed to from_pretrained.

        Returns:
            HuggingFaceBackboneWrapper instance.
        """
        if model_class is None:
            from transformers import AutoModel
            model_class = AutoModel

        model = model_class.from_pretrained(model_name_or_path, **model_kwargs)

        return cls(
            model=model,
            layers_attr=layers_attr,
            norm_attr=norm_attr,
            embed_tokens_attr=embed_tokens_attr,
            rotary_emb_attr=rotary_emb_attr,
            use_adarms=use_adarms,
        )

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: BackboneWrapperConfig,
    ) -> "HuggingFaceBackboneWrapper":
        """
        Create a wrapper from a BackboneWrapperConfig.

        Args:
            model: The HuggingFace model instance.
            config: BackboneWrapperConfig with attribute paths.

        Returns:
            HuggingFaceBackboneWrapper instance.
        """
        return cls(
            model=model,
            layers_attr=config.layers_attr,
            norm_attr=config.norm_attr,
            embed_tokens_attr=config.embed_tokens_attr,
            rotary_emb_attr=config.rotary_emb_attr,
            use_adarms=config.config_overrides.get("use_adarms", False),
        )
