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
MoT-PI0 Model Implementation.

This module provides the MoTPI0Policy class, which is a specialized wrapper
around the base MoTPolicy for PI0-style flow matching models.

The actual model implementation is shared with the base MoT architecture.
This module mainly provides the policy class registration and any PI0-specific
customizations.
"""

from lerobot.policies.mot.modeling_mot import MoTPolicy
from lerobot.policies.mot_pi0.configuration_mot_pi0 import MoTPI0Config


class MoTPI0Policy(MoTPolicy):
    """
    Policy implementation for MoT-PI0 architecture.

    This is a thin wrapper around MoTPolicy that uses MoTPI0Config.
    The core model implementation is shared with the base MoT architecture.

    MoT-PI0 implements the PI0 architecture:
    - PaliGemma VLM for vision and language processing
    - Gemma expert for action generation
    - Flow matching for action decoding
    - Joint layer attention between VLM and expert

    Example:
        >>> from lerobot.policies.mot_pi0 import MoTPI0Config, MoTPI0Policy
        >>> config = MoTPI0Config(max_action_dim=7)
        >>> policy = MoTPI0Policy(config)
    """

    config_class = MoTPI0Config
    name = "mot_pi0"

    def __init__(self, config: MoTPI0Config | None = None, **kwargs):
        if config is None:
            config = MoTPI0Config(**kwargs)
        super().__init__(config=config, **kwargs)
