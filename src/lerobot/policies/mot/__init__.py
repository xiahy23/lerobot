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
MoT (Multistream-VLA) Core Framework

This module provides the core framework for building multi-stream vision-language-action
models. It follows the "mechanism vs policy separation" philosophy:

- MoT Core: Handles mechanism (synchronized layer loops, joint attention, KV cache)
- Backbones: Shield the core from underlying model differences
- Concrete Policies: Handle business logic (embeddings, masks, loss computation)

Components:
- MoTConfig: Configuration for multi-stream architecture
- MoTStreamConfig: Configuration for individual streams
- MoTModel: Synchronized layer executor
- MoTPolicy: Base class for MoT-based policies
- Backbone Wrappers: Adapters for different model implementations

Usage:
    >>> from lerobot.policies.mot import MoTConfig, MoTStreamConfig, MoTModel, MoTPolicy
    >>> from lerobot.policies.mot.backbones import build_backbone, HuggingFaceBackboneWrapper
"""

from lerobot.policies.mot.configuration_mot import (AttentionType, MoTConfig,
                                                    MoTJointAttentionConfig,
                                                    MoTStreamConfig,
                                                    StreamRole,
                                                    create_gemma_stream,
                                                    create_paligemma_stream)
from lerobot.policies.mot.modeling_mot import (ActionSelectKwargs, MoTModel,
                                               MoTModelOutput, MoTPolicy,
                                               compute_joint_layer)

__all__ = [
    # Configuration
    "MoTConfig",
    "MoTStreamConfig",
    "MoTJointAttentionConfig",
    "AttentionType",
    "StreamRole",
    # Convenience functions for stream creation
    "create_paligemma_stream",
    "create_gemma_stream",
    # Model
    "MoTModel",
    "MoTModelOutput",
    "MoTPolicy",
    "compute_joint_layer",
    # Types
    "ActionSelectKwargs",
]
