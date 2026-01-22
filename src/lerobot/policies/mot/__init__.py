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
- Swapping underlying transformer implementations (PaliGemma, Gemma, Bagel, etc.)
- Supporting pi0, pi0.5, pi0FAST architectures through configuration alone

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
"""

from .configuration_mot import (MoTBackboneConfig, MoTConfig, MoTFlowConfig,
                                MoTNodeConfig)
from .modeling_mot import MoTPolicy
from .processor_mot import make_mot_pre_post_processors

__all__ = [
    "MoTConfig",
    "MoTNodeConfig",
    "MoTFlowConfig",
    "MoTBackboneConfig",
    "MoTPolicy",
    "make_mot_pre_post_processors",
]
