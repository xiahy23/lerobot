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
MoT-PI0 Configuration.

This module defines the configuration for the MoT-PI0 architecture, which is
a specialized MoT configuration for PI0-style flow matching models.

PI0 uses:
- PaliGemma 2B as VLM (vision + language)
- Gemma 300M as action expert
- Joint layer attention between VLM and expert
- Flow matching for action decoding
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.mot.configuration_mot import (GEMMA_2B_CONFIG,
                                                    GEMMA_300M_CONFIG,
                                                    PALIGEMMA_2B_VISION_CONFIG,
                                                    AttentionType,
                                                    BackboneType, DecodingMode,
                                                    MoTBackboneConfig,
                                                    MoTConfig, MoTFlowConfig,
                                                    MoTNodeConfig, NodeType)


def _make_pi0_nodes(max_action_dim: int = 32) -> list[MoTNodeConfig]:
    """Create default nodes for PI0 architecture."""
    return [
        MoTNodeConfig(
            name="vlm",
            node_type=NodeType.VLM,
            backbone_name="paligemma",
            input_dim=2048,
            token_dim=2048,
            max_tokens=256 + 48,  # image patches + language tokens
            is_input=True,
            is_output=False,
        ),
        MoTNodeConfig(
            name="action_expert",
            node_type=NodeType.LM,
            backbone_name="gemma_expert",
            input_dim=1024,
            token_dim=1024,
            max_tokens=51,  # state + action tokens
            is_input=True,
            is_output=True,
            output_head="linear",
            output_dim=max_action_dim,
        ),
    ]


def _make_pi0_flows() -> list[MoTFlowConfig]:
    """Create default attention flows for PI0 architecture."""
    return [
        # VLM self-attention (bidirectional for images/language)
        MoTFlowConfig(source="vlm", target="vlm", attention_type=AttentionType.FULL),
        # Expert can attend to VLM (prefix-style)
        MoTFlowConfig(source="vlm", target="action_expert", attention_type=AttentionType.FULL),
        # Expert self-attention (causal for actions)
        MoTFlowConfig(source="action_expert", target="action_expert", attention_type=AttentionType.CAUSAL),
    ]


def _make_pi0_backbones() -> list[MoTBackboneConfig]:
    """Create default backbones for PI0 architecture."""
    return [
        MoTBackboneConfig(
            name="paligemma",
            backbone_type=BackboneType.PALIGEMMA,
            variant="gemma_2b",
            **GEMMA_2B_CONFIG,
            **PALIGEMMA_2B_VISION_CONFIG,
        ),
        MoTBackboneConfig(
            name="gemma_expert",
            backbone_type=BackboneType.GEMMA,
            variant="gemma_300m",
            **GEMMA_300M_CONFIG,
        ),
    ]


@PreTrainedConfig.register_subclass("mot_pi0")
@dataclass
class MoTPI0Config(MoTConfig):
    """
    Configuration for the MoT-PI0 architecture.

    This is a specialized MoT configuration that implements the PI0 architecture
    using flow matching for action decoding. It inherits all parameters from
    MoTConfig and sets appropriate defaults for PI0.

    Key characteristics:
    - PaliGemma 2B VLM for vision and language processing
    - Gemma 300M expert for action generation
    - Flow matching decoding with configurable inference steps
    - Joint layer attention between VLM and expert

    Attributes:
        paligemma_variant: Variant of PaliGemma to use ("gemma_2b" or "gemma_300m")
        action_expert_variant: Variant of Gemma expert ("gemma_2b" or "gemma_300m")

    Example:
        >>> config = MoTPI0Config(
        ...     max_action_dim=7,
        ...     chunk_size=50,
        ...     num_inference_steps=10,
        ... )
    """

    # PI0-specific variant selection
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    def __post_init__(self):
        # Set PI0-specific defaults before parent validation
        if not self.nodes:
            self.nodes = _make_pi0_nodes(self.max_action_dim)
        if not self.flows:
            self.flows = _make_pi0_flows()
        if not self.backbones:
            self.backbones = _make_pi0_backbones()

        # Ensure flow matching mode
        self.decoding_mode = DecodingMode.FLOW_MATCHING

        # Set default normalization for PI0
        if self.normalization_mapping == MoTConfig.__dataclass_fields__['normalization_mapping'].default_factory():
            self.normalization_mapping = {
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.MEAN_STD,
                "ACTION": NormalizationMode.MEAN_STD,
            }

        # Update backbone variants if specified differently
        self._update_backbone_variants()

        # Call parent __post_init__ for validation
        super().__post_init__()

    def _update_backbone_variants(self):
        """Update backbone configurations based on variant selections."""
        variant_configs = {
            "gemma_300m": GEMMA_300M_CONFIG,
            "gemma_2b": GEMMA_2B_CONFIG,
        }

        for backbone in self.backbones:
            if backbone.name == "paligemma" and self.paligemma_variant in variant_configs:
                for key, value in variant_configs[self.paligemma_variant].items():
                    setattr(backbone, key, value)
                for key, value in PALIGEMMA_2B_VISION_CONFIG.items():
                    setattr(backbone, key, value)

            elif backbone.name == "gemma_expert" and self.action_expert_variant in variant_configs:
                for key, value in variant_configs[self.action_expert_variant].items():
                    setattr(backbone, key, value)

        # Update node dimensions to match backbone
        for node in self.nodes:
            if node.name == "vlm":
                node.input_dim = variant_configs.get(self.paligemma_variant, GEMMA_2B_CONFIG)["hidden_size"]
                node.token_dim = node.input_dim
            elif node.name == "action_expert":
                node.input_dim = variant_configs.get(self.action_expert_variant, GEMMA_300M_CONFIG)["hidden_size"]
                node.token_dim = node.input_dim
