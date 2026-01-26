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
MoT-PI0-Fast Model Implementation.

This module provides the MoTPI0FastPolicy class, which is a specialized wrapper
around the base MoTPolicy for PI0-FAST autoregressive models.

The actual model implementation is shared with the base MoT architecture.
This module mainly provides the policy class registration and any PI0-FAST-specific
customizations.
"""

from lerobot.policies.mot.modeling_mot import MoTPolicy
from lerobot.policies.mot_pi0_fast.configuration_mot_pi0_fast import \
    MoTPI0FastConfig


class MoTPI0FastPolicy(MoTPolicy):
    """
    Policy implementation for MoT-PI0-Fast architecture.

    This is a thin wrapper around MoTPolicy that uses MoTPI0FastConfig.
    The core model implementation is shared with the base MoT architecture.

    MoT-PI0-Fast implements the PI0-FAST architecture:
    - Single PaliGemma VLM for vision, language, and action generation
    - Autoregressive token-based action decoding
    - Action tokenizer for discretized representation
    - Faster inference compared to flow matching

    Example:
        >>> from lerobot.policies.mot_pi0_fast import MoTPI0FastConfig, MoTPI0FastPolicy
        >>> config = MoTPI0FastConfig(max_action_dim=7)
        >>> policy = MoTPI0FastPolicy(config)
    """

    config_class = MoTPI0FastConfig
    name = "mot_pi0_fast"

    def __init__(self, config: MoTPI0FastConfig | None = None, **kwargs):
        if config is None:
            config = MoTPI0FastConfig(**kwargs)
        super().__init__(config=config, **kwargs)
