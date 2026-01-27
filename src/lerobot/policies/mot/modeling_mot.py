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
MoT (Multistream-VLA) Model Implementation

This module implements the core MoT model architecture following the
"mechanism vs policy separation" philosophy:

- MoTModel: The synchronized layer executor that handles tensor flow
- MoTPolicy: Base class for policies that use the MoT architecture

The MoT Core is responsible for:
- Synchronized layer-by-layer execution across multiple streams
- Joint attention computation when enabled
- KV cache management for efficient inference

What MoT Core does NOT do:
- Tokenization or image processing
- Embedding computation
- Mask construction (handled by concrete policies)
- Loss computation (handled by concrete policies)

IMPORTANT: This module is model-agnostic. It does NOT import any model-specific
code (e.g., transformers.models.gemma). All model-specific operations are
delegated to the backbone adapters through the MoTBackboneWrapper protocol.
"""

from __future__ import annotations

import abc
import logging
from collections import deque
from typing import Any, TypedDict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.mot.backbones import (MoTBackboneWrapper,
                                            build_backbone_from_config)
from lerobot.policies.mot.configuration_mot import (AttentionType, MoTConfig,
                                                    MoTStreamConfig)
from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


class MoTModelOutput(TypedDict, total=False):
    """Output type for MoTModel forward pass."""

    hidden_states: dict[str, Tensor]  # Stream name -> final hidden states
    past_key_values: dict[str, list[tuple[Tensor, Tensor]]] | None
    attentions: dict[str, Tensor] | None


class ActionSelectKwargs(TypedDict, total=False):
    """Keyword arguments for action selection."""

    noise: Tensor | None
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def compute_joint_layer(
    layer_idx: int,
    inputs_embeds: list[Tensor],
    attention_mask: Tensor,
    position_ids: Tensor,
    adapters: list[MoTBackboneWrapper],
    adarms_cond: list[Tensor | None] | None = None,
    past_key_values: list[tuple[Tensor, Tensor]] | None = None,
    use_cache: bool = False,
    use_rotary: bool = True,
) -> tuple[list[Tensor], list[tuple[Tensor, Tensor]] | None]:
    """
    Compute a single layer with joint attention across all streams.

    This function concatenates the hidden states from all streams,
    computes attention jointly, and then splits the output back
    to individual streams.

    IMPORTANT: This function is model-agnostic. All model-specific operations
    (RoPE, attention computation, gated residuals) are delegated to the
    backbone adapters through the MoTBackboneWrapper protocol.

    In Joint Attention, we use the first adapter as the "primary physics engine"
    for shared operations like RoPE and attention. Each stream's adapter handles
    its own residual connections.

    Args:
        layer_idx: Index of the current layer.
        inputs_embeds: List of input embeddings for each stream.
        attention_mask: Joint attention mask for all streams.
        position_ids: Position IDs for positional encoding.
        adapters: List of backbone adapters for each stream.
        adarms_cond: Optional list of AdaRMS conditioning tensors.
        past_key_values: Optional cached (key, value) tuple from previous forward.
                        Shape: (batch, num_heads, past_seq_len, head_dim) each.
        use_cache: Whether to return current layer's KV cache.
        use_rotary: Whether to apply rotary position embeddings.

    Returns:
        tuple[list[Tensor], list[tuple[Tensor, Tensor]] | None]:
            - List of output embeddings for each stream.
            - Optional (key, value) cache tuple if use_cache is True.
    """
    if adarms_cond is None:
        adarms_cond = [None] * len(adapters)

    batch_size = inputs_embeds[0].shape[0]

    # Use the first adapter as the "primary physics engine" for shared operations
    # In Joint Attention, all streams must share the same RoPE space and attention math
    primary_adapter = adapters[0]

    # Collect Q, K, V and gates from all streams using abstracted projection
    query_states = []
    key_states = []
    value_states = []
    gates = []
    normalized_hidden_list = []

    for i, (hidden_states, adapter) in enumerate(zip(inputs_embeds, adapters)):
        # Input LayerNorm (with optional AdaRMS) - using abstracted method
        normalized_hidden, gate = adapter.apply_input_layernorm(
            hidden_states, layer_idx, cond=adarms_cond[i]
        )
        gates.append(gate)
        normalized_hidden_list.append(normalized_hidden)

        # QKV projection using abstracted method (supports different architectures)
        query_state, key_state, value_state = adapter.project_qkv(normalized_hidden, layer_idx)

        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    # Concatenate across sequence dimension for joint attention
    query_concat = torch.cat(query_states, dim=2)
    key_concat = torch.cat(key_states, dim=2)
    value_concat = torch.cat(value_states, dim=2)

    # ========== KV Cache Handling ==========
    # If past_key_values is provided, concatenate with current K/V
    # This is CRITICAL for efficient inference (avoiding O(N^2) recomputation)
    if past_key_values is not None:
        past_key, past_value = past_key_values
        # Concatenate along sequence dimension (dim=2)
        # past: (batch, num_heads, past_seq_len, head_dim)
        # current: (batch, num_heads, cur_seq_len, head_dim)
        # result: (batch, num_heads, past_seq_len + cur_seq_len, head_dim)
        key_concat = torch.cat([past_key, key_concat], dim=2)
        value_concat = torch.cat([past_value, value_concat], dim=2)

    # Store current KV for caching if requested
    present_key_value = (key_concat, value_concat) if use_cache else None

    # ========== Rotary Position Embeddings ==========
    # Apply rotary embeddings using the primary adapter's implementation
    if use_rotary:
        rotary_emb = primary_adapter.get_rotary_embedding()
        if rotary_emb is not None:
            # Create dummy tensor for cos/sin computation
            # Note: For cached inference, position_ids should account for past length
            dummy = torch.zeros(
                query_concat.shape[0],
                query_concat.shape[2],
                query_concat.shape[-1],
                device=query_concat.device,
                dtype=query_concat.dtype,
            )
            cos, sin = rotary_emb(dummy, position_ids)

            # For cached inference, we only apply rotary to the query and current keys
            # The past keys already have rotary applied from their original computation
            if past_key_values is not None:
                # Extract only the current portion of keys for rotary
                past_len = past_key_values[0].shape[2]
                current_keys = key_concat[:, :, past_len:, :]

                # Apply rotary to query and current keys using adapter interface
                query_concat, current_keys = primary_adapter.apply_rotary_pos_emb(
                    query_concat, current_keys, cos, sin, unsqueeze_dim=1
                )

                # Concatenate rotary-applied keys back with past keys
                key_concat = torch.cat([key_concat[:, :, :past_len, :], current_keys], dim=2)
            else:
                # Apply rotary to all query and keys using adapter interface
                query_concat, key_concat = primary_adapter.apply_rotary_pos_emb(
                    query_concat, key_concat, cos, sin, unsqueeze_dim=1
                )

    # ========== Attention Computation ==========
    # Get scaling factor and compute attention using primary adapter
    scaling = primary_adapter.get_scaling(layer_idx)

    # Use the primary adapter's compute_attention method
    # This delegates to model-specific implementations (e.g., Gemma's eager_attention)
    attn_output, _ = primary_adapter.compute_attention(
        query_concat,
        key_concat,
        value_concat,
        attention_mask,
        scaling,
    )

    # Reshape attention output from (batch, num_heads, seq_len, head_dim)
    # to (batch, seq_len, num_heads * head_dim)
    head_dim = primary_adapter.head_dim
    num_heads = primary_adapter.num_attention_heads
    attn_output = attn_output.reshape(batch_size, -1, num_heads * head_dim)

    # ========== Split & Process Per-Stream ==========
    # Split attention output back to individual streams and apply output projection
    outputs_embeds = []
    start_pos = 0

    for i, (original_hidden, adapter) in enumerate(zip(inputs_embeds, adapters)):
        seq_len = original_hidden.shape[1]
        end_pos = start_pos + seq_len

        # Slice attention output for this stream
        stream_attn_output = attn_output[:, start_pos:end_pos]

        # Get output projection weight dtype for type conversion
        self_attn = adapter.get_layer_self_attn(layer_idx)
        o_proj_dtype = self_attn.o_proj.weight.dtype if hasattr(self_attn, 'o_proj') else stream_attn_output.dtype

        # Type conversion if needed
        if stream_attn_output.dtype != o_proj_dtype:
            stream_attn_output = stream_attn_output.to(o_proj_dtype)

        # Output projection using abstracted method
        out_emb = adapter.project_output(stream_attn_output, layer_idx)

        # First residual connection using adapter interface
        # Each stream uses its own adapter for residual (may have different gating)
        out_emb = adapter.apply_gated_residual(original_hidden, out_emb, gates[i])

        after_first_residual = out_emb.clone()

        # Post-attention LayerNorm using abstracted method
        out_emb, mlp_gate = adapter.apply_post_attention_layernorm(
            out_emb, layer_idx, cond=adarms_cond[i]
        )

        # MLP
        mlp = adapter.get_layer_mlp(layer_idx)
        if mlp is not None:
            # Type conversion for MLP if needed
            if hasattr(mlp, 'up_proj') and mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)
            out_emb = mlp(out_emb)

        # Second residual connection using adapter interface
        out_emb = adapter.apply_gated_residual(after_first_residual, out_emb, mlp_gate)

        outputs_embeds.append(out_emb)
        start_pos = end_pos

    return outputs_embeds, present_key_value


class MoTModel(nn.Module):
    """
    MoT (Multistream-VLA) Model - Synchronized Layer Executor.

    This model handles the synchronized execution of transformer layers
    across multiple streams. It does NOT contain:
    - Tokenizers
    - Image Processors
    - Embedding Layers

    The model receives pre-computed embeddings for each stream and outputs
    the final hidden states after processing through all layers.

    Key Features:
    - Synchronized layer-by-layer execution
    - Optional joint attention between streams
    - Gradient checkpointing support
    - KV cache management for inference

    Architecture:
        For each layer_idx in 0..N:
            For each stream:
                Apply layer_module to hidden_states
            (Optional) Apply Joint-Attention across streams
        Apply Final LayerNorm to each stream

    Attributes:
        config: MoTConfig with stream configurations.
        stream_adapters: Dictionary mapping stream names to backbone wrappers.
        num_layers: Number of transformer layers.
        gradient_checkpointing_enabled: Whether gradient checkpointing is active.
    """

    def __init__(
        self,
        config: MoTConfig,
        stream_adapters: dict[str, MoTBackboneWrapper] | None = None,
    ):
        """
        Initialize the MoT model.

        Args:
            config: MoTConfig with stream configurations.
            stream_adapters: Optional pre-built stream adapters. If not provided,
                            they will be built from the config.
        """
        super().__init__()

        self.config = config
        self.stream_adapters: dict[str, MoTBackboneWrapper] = {}
        self.gradient_checkpointing_enabled = False

        # Use provided adapters or leave empty for external initialization
        if stream_adapters is not None:
            self.stream_adapters = nn.ModuleDict(stream_adapters)  # type: ignore
        else:
            self.stream_adapters = nn.ModuleDict()

        # Validate layer alignment for joint attention
        if self.config.attn_implementation == AttentionType.JOINT:
            self._validate_layer_alignment()

    def _validate_layer_alignment(self):
        """Validate that all streams have the same number of layers for joint attention."""
        if not self.stream_adapters:
            return

        num_layers = None
        for name, adapter in self.stream_adapters.items():
            if num_layers is None:
                num_layers = adapter.num_layers
            elif adapter.num_layers != num_layers:
                raise ValueError(
                    f"Layer count mismatch for joint attention: "
                    f"'{name}' has {adapter.num_layers} layers, expected {num_layers}"
                )

    def _validate_rope_compatibility(self):
        """
        Validate that all streams use compatible RoPE (Rotary Position Embedding) configurations.

        For Joint Attention to be mathematically valid, all streams must share the same
        RoPE space geometry. This includes:
        - RoPE base (theta)
        - RoPE scaling factor
        - Head dimension

        If streams have incompatible RoPE configs, Joint Attention will produce incorrect
        position encodings for some streams.

        Raises:
            Warning if RoPE configs are mismatched (not an error to allow experimentation).
        """
        if not self.stream_adapters or self.config.attn_implementation != AttentionType.JOINT:
            return

        rope_configs = {}
        for name, adapter in self.stream_adapters.items():
            rope_config = adapter.get_rope_config()
            if rope_config is not None:
                rope_configs[name] = rope_config

        if len(rope_configs) <= 1:
            return  # No comparison needed

        # Compare all RoPE configs
        reference_name = next(iter(rope_configs.keys()))
        reference_config = rope_configs[reference_name]

        mismatches = []
        for name, config in rope_configs.items():
            if name == reference_name:
                continue

            # Check critical RoPE parameters
            for key in ["rope_theta", "rope_scaling", "head_dim"]:
                ref_val = reference_config.get(key)
                cur_val = config.get(key)
                if ref_val != cur_val:
                    mismatches.append(
                        f"  - {name}.{key}={cur_val} vs {reference_name}.{key}={ref_val}"
                    )

        if mismatches:
            logger.warning(
                f"RoPE configuration mismatch detected in Joint Attention streams!\n"
                f"Joint Attention requires all streams to share the same RoPE space geometry.\n"
                f"Mismatches:\n" + "\n".join(mismatches) + "\n"
                f"This may cause incorrect position encodings for some streams.\n"
                f"If streams are from the same model family (e.g., both Gemma), this is usually safe."
            )

    def register_adapter(self, name: str, adapter: MoTBackboneWrapper):
        """
        Register a stream adapter.

        Args:
            name: Name of the stream.
            adapter: The backbone wrapper for this stream.
        """
        self.stream_adapters[name] = adapter
        if self.config.attn_implementation == AttentionType.JOINT:
            self._validate_layer_alignment()
            self._validate_rope_compatibility()

    @property
    def num_layers(self) -> int:
        """Return the number of layers (from first adapter)."""
        if not self.stream_adapters:
            return 0
        return next(iter(self.stream_adapters.values())).num_layers

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        logger.info("Enabled gradient checkpointing for MoTModel")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        logger.info("Disabled gradient checkpointing for MoTModel")

    def forward(
        self,
        inputs_embeds_dict: dict[str, Tensor],
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: dict[str, list[tuple[Tensor, Tensor]]] | None = None,
        use_cache: bool = False,
        adarms_cond_dict: dict[str, Tensor] | None = None,
        output_attentions: bool = False,
    ) -> MoTModelOutput:
        """
        Forward pass through the MoT model.

        Args:
            inputs_embeds_dict: Dictionary mapping stream names to input embeddings.
                               Shape: {stream_name: (batch, seq_len, hidden_size)}
                               During cached inference (decode phase), streams with cached KV
                               can pass None to skip recomputing their embeddings.
            attention_mask: Joint attention mask for all streams.
                           Shape: (batch, 1, total_seq_len, total_seq_len) or similar
            position_ids: Position IDs for positional encoding.
                         Shape: (batch, total_seq_len)
            past_key_values: Optional cached key-value pairs for each stream.
            use_cache: Whether to return key-value caches.
            adarms_cond_dict: Optional AdaRMS conditioning tensors per stream.
            output_attentions: Whether to output attention weights.

        Returns:
            MoTModelOutput containing:
                - hidden_states: Dict of final hidden states per stream
                - past_key_values: Optional KV caches
                - attentions: Optional attention weights
        """
        # Filter out None inputs (for cached inference decode phase)
        active_stream_names = [
            name for name, emb in inputs_embeds_dict.items()
            if emb is not None
        ]

        # Validate active input streams match registered adapters
        for name in active_stream_names:
            if name not in self.stream_adapters:
                raise ValueError(
                    f"Unknown stream '{name}'. Registered streams: {list(self.stream_adapters.keys())}"
                )

        # Get ordered lists for processing (only active streams)
        stream_names = active_stream_names
        inputs_embeds = [inputs_embeds_dict[name] for name in stream_names]
        adapters = [self.stream_adapters[name] for name in stream_names]

        # Prepare AdaRMS conditioning
        adarms_cond = None
        if adarms_cond_dict is not None:
            adarms_cond = [adarms_cond_dict.get(name) for name in stream_names]

        # Initialize cache storage
        all_key_values = {name: [] for name in stream_names} if use_cache else None

        # Execute layers based on attention implementation
        if self.config.attn_implementation == AttentionType.JOINT:
            # For joint attention, past_key_values is stored under "joint" key
            joint_past = None
            if past_key_values is not None and "joint" in past_key_values:
                joint_past = past_key_values["joint"]

            inputs_embeds, joint_kv_cache = self._forward_joint_attention(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                adapters=adapters,
                adarms_cond=adarms_cond,
                past_key_values=joint_past,
                use_cache=use_cache,
            )
            # For joint attention, we store the unified KV cache
            if use_cache and joint_kv_cache is not None:
                all_key_values = {"joint": joint_kv_cache}
        elif self.config.attn_implementation == AttentionType.INDEPENDENT:
            inputs_embeds = self._forward_independent_attention(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                adapters=adapters,
                adarms_cond=adarms_cond,
                past_key_values=past_key_values,
                all_key_values=all_key_values,
                use_cache=use_cache,
            )
        else:
            raise NotImplementedError(
                f"Attention implementation '{self.config.attn_implementation}' not yet supported"
            )

        # Apply final normalization
        outputs_embeds = []
        for i, (hidden_states, adapter) in enumerate(zip(inputs_embeds, adapters)):
            cond = adarms_cond[i] if adarms_cond is not None else None
            normalized, _ = adapter.forward_norm(hidden_states, cond=cond)
            outputs_embeds.append(normalized)

        # Build output dictionary
        hidden_states_dict = {name: emb for name, emb in zip(stream_names, outputs_embeds)}

        return MoTModelOutput(
            hidden_states=hidden_states_dict,
            past_key_values=all_key_values,
            attentions=None,  # TODO: implement attention output if needed
        )

    def _forward_joint_attention(
        self,
        inputs_embeds: list[Tensor],
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
        adapters: list[MoTBackboneWrapper],
        adarms_cond: list[Tensor | None] | None,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[list[Tensor], list[tuple[Tensor, Tensor]] | None]:
        """Forward pass with joint attention across all streams.

        Args:
            inputs_embeds: List of input embeddings per stream
            attention_mask: Joint attention mask
            position_ids: Position IDs for rotary embeddings
            adapters: List of backbone wrappers per stream
            adarms_cond: Optional conditioning tensors per stream
            past_key_values: List of (key, value) tuples per layer from previous forward pass
            use_cache: Whether to return KV cache for incremental decoding

        Returns:
            Tuple of:
                - List of output hidden states per stream
                - List of (key, value) tuples per layer if use_cache, else None
        """
        num_layers = adapters[0].num_layers
        new_key_values = [] if use_cache else None

        for layer_idx in range(num_layers):
            # Get past KV for this layer
            layer_past = None
            if past_key_values is not None and layer_idx < len(past_key_values):
                layer_past = past_key_values[layer_idx]

            if self.gradient_checkpointing_enabled and self.training:
                # During training with gradient checkpointing, we don't use cache
                inputs_embeds, _ = torch.utils.checkpoint.checkpoint(
                    compute_joint_layer,
                    layer_idx,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    adapters,
                    adarms_cond,
                    None,  # past_key_value - not used during training with grad checkpoint
                    False,  # use_cache - disabled during gradient checkpointing
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                inputs_embeds, layer_kv = compute_joint_layer(
                    layer_idx=layer_idx,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    adapters=adapters,
                    adarms_cond=adarms_cond,
                    past_key_values=layer_past,
                    use_cache=use_cache,
                )
                if use_cache and layer_kv is not None:
                    new_key_values.append(layer_kv)

        return inputs_embeds, new_key_values

    def _forward_independent_attention(
        self,
        inputs_embeds: list[Tensor],
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
        adapters: list[MoTBackboneWrapper],
        adarms_cond: list[Tensor | None] | None,
        past_key_values: dict[str, list[tuple[Tensor, Tensor]]] | None,
        all_key_values: dict[str, list] | None,
        use_cache: bool,
    ) -> list[Tensor]:
        """Forward pass with independent attention for each stream."""
        stream_names = list(self.stream_adapters.keys())
        num_layers = adapters[0].num_layers

        for layer_idx in range(num_layers):
            new_embeds = []
            for i, (hidden_states, adapter, name) in enumerate(
                zip(inputs_embeds, adapters, stream_names)
            ):
                layer = adapter.get_layer(layer_idx)

                # Get past key-value for this stream and layer
                layer_past = None
                if past_key_values is not None and name in past_key_values:
                    stream_kv = past_key_values[name]
                    if layer_idx < len(stream_kv):
                        layer_past = stream_kv[layer_idx]

                # Prepare kwargs for layer forward
                layer_kwargs = {
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "past_key_value": layer_past,
                    "use_cache": use_cache,
                }

                # Add AdaRMS conditioning if available
                cond = adarms_cond[i] if adarms_cond is not None else None
                if adapter.supports_adarms() and cond is not None:
                    layer_kwargs["cond"] = cond

                # Forward through layer
                layer_output = layer(hidden_states, **layer_kwargs)

                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                    if use_cache and len(layer_output) > 1 and all_key_values is not None:
                        all_key_values[name].append(layer_output[1])
                else:
                    hidden_states = layer_output

                new_embeds.append(hidden_states)

            inputs_embeds = new_embeds

        return inputs_embeds


class MoTPolicy(PreTrainedPolicy, abc.ABC):
    """
    Base class for policies using the MoT (Multistream-VLA) architecture.

    This class provides the common structure for MoT-based policies while
    leaving the specific implementation details (embeddings, masks, loss)
    to concrete subclasses.

    Core Philosophy: "Mechanism vs Policy Separation"
    - MoT Core (MoTModel) handles mechanism: synchronized layers, joint attention, KV cache
    - Concrete policies handle business logic: embeddings, masks, loss computation

    Subclasses must implement:
    - get_input_embeddings(batch): Compute embeddings for each stream
    - _make_attention_mask(batch): Construct the attention mask
    - forward(batch): Full forward pass with loss computation

    Attributes:
        config: MoTConfig or subclass with policy configuration.
        model: The MoTModel for synchronized layer execution.
    """

    config_class = MoTConfig
    name = "mot"

    def __init__(self, config: MoTConfig, *args, **kwargs):
        """
        Initialize the MoT policy.

        Args:
            config: MoTConfig with policy and stream configurations.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(config, *args, **kwargs)

        self.config = config
        self.model: MoTModel | None = None

        # Action chunk cache for inference
        self._action_queue: deque = deque()

    def _init_model(self, stream_adapters: dict[str, MoTBackboneWrapper] | None = None):
        """
        Initialize the MoTModel with stream adapters.

        This should be called by subclasses after building the stream adapters.

        Args:
            stream_adapters: Dictionary of stream name to backbone wrapper.
        """
        self.model = MoTModel(config=self.config, stream_adapters=stream_adapters)

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    @abc.abstractmethod
    def get_input_embeddings(
        self, batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """
        Compute input embeddings for each stream.

        This method is responsible for:
        - Processing images through vision encoder
        - Tokenizing and embedding text
        - Projecting actions and states
        - Assembling embeddings for each stream

        Args:
            batch: Input batch containing images, text, states, actions.

        Returns:
            Dictionary mapping stream names to embeddings.
            Shape: {stream_name: (batch, seq_len, hidden_size)}
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _make_attention_mask(
        self, batch: dict[str, Tensor]
    ) -> Tensor:
        """
        Construct the attention mask for the model.

        This method is responsible for building the potentially complex
        2D attention mask (e.g., Prefix LM + Causal Action for Pi0).

        Args:
            batch: Input batch for mask construction.

        Returns:
            Attention mask tensor.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict | None]:
        """
        Forward pass with loss computation.

        This method should:
        1. Call get_input_embeddings() to get stream embeddings
        2. Call _make_attention_mask() to get the mask
        3. Call self.model() to process through layers
        4. Compute and return the loss

        Args:
            batch: Input batch with all required data.

        Returns:
            tuple: (loss, optional info dict)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """
        Predict an action chunk for inference.

        Args:
            batch: Observation batch.
            **kwargs: Additional inference arguments.

        Returns:
            Action chunk tensor of shape (batch, chunk_size, action_dim).
        """
        raise NotImplementedError

    def reset(self):
        """Reset the policy state (clear action queue)."""
        self._action_queue.clear()

    def select_action(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """
        Select a single action from the action chunk.

        This method handles action chunking by caching predicted chunks
        and returning actions sequentially.

        Args:
            batch: Observation batch.
            **kwargs: Additional inference arguments.

        Returns:
            Single action tensor of shape (batch, action_dim).
        """
        # If queue is empty, predict new chunk
        if len(self._action_queue) == 0:
            action_chunk = self.predict_action_chunk(batch, **kwargs)
            # Add each action in the chunk to the queue
            for i in range(action_chunk.shape[1]):
                self._action_queue.append(action_chunk[:, i])

        # Return and remove the first action from queue
        return self._action_queue.popleft()

    def get_optim_params(self) -> dict:
        """
        Return optimization parameters for the optimizer.

        Returns:
            Dictionary with parameter groups and settings.
        """
        return {"params": self.parameters()}

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing on the model."""
        if self.model is not None:
            self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the model."""
        if self.model is not None:
            self.model.gradient_checkpointing_disable()
