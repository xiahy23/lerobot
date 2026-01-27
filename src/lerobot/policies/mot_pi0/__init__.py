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
MoT-Pi0 Policy

This module provides the Pi0 policy implemented on top of the MoT (Multistream-VLA) framework.

Pi0 is a vision-language-action policy that uses:
- PaliGemma for vision and language understanding
- A Gemma-based action expert for action prediction
- Flow matching for action generation

The MoT framework provides:
- Modular backbone adapters for different model implementations
- Synchronized layer-by-layer execution across streams
- Joint attention mechanism for cross-stream interaction

Usage:
    >>> from lerobot.policies.mot_pi0 import MoTPI0Config, MoTPI0Policy
    >>> config = MoTPI0Config()
    >>> policy = MoTPI0Policy(config)
"""

from lerobot.policies.mot_pi0.configuration_mot_pi0 import (
    DEFAULT_IMAGE_SIZE, GemmaModelConfig, MoTPI0Config, get_gemma_model_config)
from lerobot.policies.mot_pi0.modeling_mot_pi0 import (
    MoTPI0Policy, PaliGemmaWithExpertWrapper, create_sinusoidal_pos_embedding,
    make_att_2d_masks, pad_vector, resize_with_pad_torch, sample_beta)
from lerobot.policies.mot_pi0.processor_mot_pi0 import (
    MoTPI0NewLineProcessor, make_mot_pi0_pre_post_processors)

__all__ = [
    # Configuration
    "MoTPI0Config",
    "GemmaModelConfig",
    "get_gemma_model_config",
    "DEFAULT_IMAGE_SIZE",
    # Policy
    "MoTPI0Policy",
    "PaliGemmaWithExpertWrapper",
    # Processor
    "MoTPI0NewLineProcessor",
    "make_mot_pi0_pre_post_processors",
    # Utilities
    "create_sinusoidal_pos_embedding",
    "make_att_2d_masks",
    "pad_vector",
    "resize_with_pad_torch",
    "sample_beta",
]
