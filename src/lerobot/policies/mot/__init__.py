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
MoT (Mixture of Transformers) Policy for LeRobot.

This module provides a highly configurable multi-transformer architecture that allows:
- Defining arbitrary numbers of transformer nodes (vision, state, action, etc.)
- Configuring attention flows between nodes via configuration
- Swapping underlying transformer implementations (PaliGemma, Gemma, LLaMA, Qwen, etc.)
- Supporting pi0, pi0.5, pi0FAST architectures through configuration alone

The backbone adapter system (in `backbones/`) uses a registry pattern for extensibility:
- To add new backbone types, create an adapter in `backbones/` and register it
- See `backbones/base.py` for the interface all adapters must implement

Example usage:
    from lerobot.policies.mot import MoTConfig, MoTPolicy

    # Create a pi0-like configuration
    config = MoTConfig(
        nodes=[
            MoTNodeConfig(name="vision", node_type="vlm", ...),
            MoTNodeConfig(name="action_expert", node_type="lm", ...),
        ],
        flows=[
            MoTFlowConfig(source="vision", target="action_expert", attention_type="full"),
        ],
    )
    policy = MoTPolicy(config)

    # To add a custom backbone:
    from lerobot.policies.mot.backbones import BackboneAdapter, BackboneRegistry

    @BackboneRegistry.register("my_backbone")
    class MyBackboneAdapter(BackboneAdapter):
        def setup_model(self):
            ...
        def load_pretrained(self, path, **kwargs):
            ...
        @property
        def layers(self):
            ...
        @property
        def norm(self):
            ...
"""

# Expose backbone system for extensibility
from .backbones import (BackboneAdapter, BackboneRegistry, build_backbone,
                        get_backbone_preset)
from .configuration_mot import (MoTBackboneConfig, MoTConfig, MoTFlowConfig,
                                MoTNodeConfig)
from .modeling_mot import MoTPolicy
from .processor_mot import make_mot_pre_post_processors

__all__ = [
    # Configuration
    "MoTConfig",
    "MoTNodeConfig",
    "MoTFlowConfig",
    "MoTBackboneConfig",
    # Policy
    "MoTPolicy",
    # Processors
    "make_mot_pre_post_processors",
    # Backbone system
    "BackboneAdapter",
    "BackboneRegistry",
    "build_backbone",
    "get_backbone_preset",
]
