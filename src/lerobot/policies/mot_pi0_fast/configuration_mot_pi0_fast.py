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
MoT-PI0-Fast Configuration.

This module defines the configuration for the MoT-PI0-Fast architecture, which is
a specialized MoT configuration for autoregressive action token generation.

PI0-FAST uses:
- PaliGemma 2B as VLM (vision + language + action tokens)
- Autoregressive token generation instead of flow matching
- Action tokenizer for discretized action representation
- Prefix-LM attention pattern
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.mot.configuration_mot import (GEMMA_2B_CONFIG,
                                                    PALIGEMMA_2B_VISION_CONFIG,
                                                    AttentionType,
                                                    BackboneType, DecodingMode,
                                                    MoTBackboneConfig,
                                                    MoTConfig, MoTFlowConfig,
                                                    MoTNodeConfig, NodeType)


def _make_pi0fast_nodes() -> list[MoTNodeConfig]:
    """Create default nodes for PI0-FAST architecture."""
    return [
        MoTNodeConfig(
            name="vlm",
            node_type=NodeType.VLM,
            backbone_name="paligemma",
            input_dim=2048,
            token_dim=2048,
            max_tokens=256 + 200,  # image patches + longer language/action tokens
            is_input=True,
            is_output=True,
            output_head="tokenizer",  # Uses tokenizer for autoregressive output
        ),
    ]


def _make_pi0fast_flows() -> list[MoTFlowConfig]:
    """Create default attention flows for PI0-FAST architecture."""
    return [
        # VLM self-attention with prefix-LM pattern (bidirectional for prefix, causal for generation)
        MoTFlowConfig(source="vlm", target="vlm", attention_type=AttentionType.PREFIX),
    ]


def _make_pi0fast_backbones() -> list[MoTBackboneConfig]:
    """Create default backbones for PI0-FAST architecture."""
    return [
        MoTBackboneConfig(
            name="paligemma",
            backbone_type=BackboneType.PALIGEMMA,
            variant="gemma_2b",
            **GEMMA_2B_CONFIG,
            **PALIGEMMA_2B_VISION_CONFIG,
        ),
    ]


@PreTrainedConfig.register_subclass("mot_pi0_fast")
@dataclass
class MoTPI0FastConfig(MoTConfig):
    """
    Configuration for the MoT-PI0-Fast architecture.

    This is a specialized MoT configuration that implements the PI0-FAST architecture
    using autoregressive token generation for faster inference.

    Key characteristics:
    - Single PaliGemma VLM handles both input processing and action generation
    - Autoregressive decoding with action tokenization
    - Prefix-LM attention pattern (bidirectional prefix, causal generation)
    - No separate expert model - simpler architecture

    Attributes:
        paligemma_variant: Variant of PaliGemma to use ("gemma_2b" or "gemma_300m")
        action_tokenizer_name: Name of the action tokenizer to use
        max_decoding_steps: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        max_action_tokens: Maximum number of action tokens
        fast_skip_tokens: Number of tokens to skip in FAST tokenizer

    Example:
        >>> config = MoTPI0FastConfig(
        ...     max_action_dim=7,
        ...     chunk_size=50,
        ...     temperature=0.0,
        ... )
    """

    # PI0-FAST specific settings
    paligemma_variant: str = "gemma_2b"

    # Override defaults for autoregressive mode
    action_tokenizer_name: str | None = "physical-intelligence/fast"
    max_decoding_steps: int = 256
    max_action_tokens: int = 256
    temperature: float = 0.0
    tokenizer_max_length: int = 200
    fast_skip_tokens: int = 128

    # Whether to validate that decoded action tokens start with "Action: " prefix
    validate_action_token_prefix: bool = True

    def __post_init__(self):
        # Set PI0-FAST specific defaults before parent validation
        if not self.nodes:
            self.nodes = _make_pi0fast_nodes()
        if not self.flows:
            self.flows = _make_pi0fast_flows()
        if not self.backbones:
            self.backbones = _make_pi0fast_backbones()

        # Ensure autoregressive mode
        self.decoding_mode = DecodingMode.AUTOREGRESSIVE

        # Set default normalization for PI0-FAST
        if self.normalization_mapping == MoTConfig.__dataclass_fields__['normalization_mapping'].default_factory():
            self.normalization_mapping = {
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.MEAN_STD,
                "ACTION": NormalizationMode.MEAN_STD,
            }

        # Update backbone variant if specified differently
        self._update_backbone_variants()

        # Call parent __post_init__ for validation
        super().__post_init__()

    def _update_backbone_variants(self):
        """Update backbone configurations based on variant selections."""
        variant_configs = {
            "gemma_300m": {
                "hidden_size": 1024,
                "num_hidden_layers": 18,
                "intermediate_size": 4096,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256,
            },
            "gemma_2b": GEMMA_2B_CONFIG,
        }

        for backbone in self.backbones:
            if backbone.name == "paligemma" and self.paligemma_variant in variant_configs:
                for key, value in variant_configs[self.paligemma_variant].items():
                    setattr(backbone, key, value)
                for key, value in PALIGEMMA_2B_VISION_CONFIG.items():
                    setattr(backbone, key, value)

        # Update node dimensions to match backbone
        for node in self.nodes:
            if node.name == "vlm":
                node.input_dim = variant_configs.get(self.paligemma_variant, GEMMA_2B_CONFIG)["hidden_size"]
                node.token_dim = node.input_dim
