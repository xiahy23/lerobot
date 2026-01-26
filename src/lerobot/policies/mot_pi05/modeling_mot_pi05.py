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
MoT-PI05 Model Implementation.

This module provides the MoTPI05Policy class, which is a specialized wrapper
around the base MoTPolicy for PI0.5-style flow matching models with AdaRMS.

The actual model implementation is shared with the base MoT architecture.
This module mainly provides the policy class registration and any PI0.5-specific
customizations.
"""

from lerobot.policies.mot.modeling_mot import MoTPolicy
from lerobot.policies.mot_pi05.configuration_mot_pi05 import MoTPI05Config


class MoTPI05Policy(MoTPolicy):
    """
    Policy implementation for MoT-PI05 architecture.

    This is a thin wrapper around MoTPolicy that uses MoTPI05Config.
    The core model implementation is shared with the base MoT architecture.

    MoT-PI05 implements the PI0.5 architecture:
    - PaliGemma VLM for vision and language processing
    - Gemma expert with AdaRMS conditioning for action generation
    - Flow matching for action decoding
    - Quantile normalization for better generalization

    Example:
        >>> from lerobot.policies.mot_pi05 import MoTPI05Config, MoTPI05Policy
        >>> config = MoTPI05Config(max_action_dim=7)
        >>> policy = MoTPI05Policy(config)
    """

    config_class = MoTPI05Config
    name = "mot_pi05"

    def __init__(self, config: MoTPI05Config | None = None, **kwargs):
        if config is None:
            config = MoTPI05Config(**kwargs)
        super().__init__(config=config, **kwargs)
