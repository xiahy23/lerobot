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
MoT-Pi0 Configuration

This module defines the configuration for Pi0 implemented on top of the MoT framework.
It inherits from MoTConfig and pre-configures the two standard streams:
- vision_language_stream: Based on PaliGemma (HF)
- action_stream: Based on Gemma (HF)
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.mot.configuration_mot import (AttentionType, MoTConfig,
                                                    MoTJointAttentionConfig,
                                                    MoTStreamConfig,
                                                    StreamRole,
                                                    create_gemma_stream,
                                                    create_paligemma_stream)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


# Gemma configuration class for model variants
class GemmaModelConfig:
    """Configuration for Gemma model variants."""

    def __init__(self, width: int, depth: int, mlp_dim: int, num_heads: int, num_kv_heads: int, head_dim: int):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_model_config(variant: str) -> GemmaModelConfig:
    """Get Gemma configuration for a specific variant."""
    if variant == "gemma_300m":
        return GemmaModelConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaModelConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown Gemma variant: {variant}")


@PreTrainedConfig.register_subclass("mot_pi0")
@dataclass
class MoTPI0Config(MoTConfig):
    """
    Configuration for Pi0 implemented on the MoT (Multistream-VLA) framework.

    This configuration inherits from MoTConfig and pre-configures the standard
    Pi0 architecture with two streams:
    - vision_language: PaliGemma-based stream for image and text processing
    - action: Gemma-based stream for action prediction

    The Pi0-specific parameters include:
    - Flow matching parameters (num_inference_steps, time_sampling, etc.)
    - Action/state dimensions
    - Model variants (paligemma_variant, action_expert_variant)
    - AdaRMS settings
    """

    # Model variants
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # "bfloat16" or "float32"

    # Inference settings
    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict
    n_action_steps: int = 50  # Number of action steps to execute

    # Dimension padding
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Image settings
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    empty_cameras: int = 0  # Add empty cameras when no image features present

    # AdaRMS settings
    use_adarms_vlm: bool = False
    use_adarms_action: bool = False

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"

    # Finetuning settings
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # Optimizer settings
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # Tokenizer settings
    tokenizer_max_length: int = 48

    def __post_init__(self):
        # Build the stream configurations
        self._build_streams()

        # Set attention implementation for Pi0 (always joint)
        self.attn_implementation = AttentionType.JOINT

        # Configure joint attention
        self.joint_attention = MoTJointAttentionConfig(
            enabled=True,
            streams=["vision_language", "action"],
            layers=None,  # All layers
            bidirectional=False,  # Prefix-LM style
        )

        # Call parent __post_init__
        super().__post_init__()

        # Validate Pi0-specific configuration
        self._validate_pi0_config()

    def _build_streams(self):
        """Build the vision-language and action stream configurations."""
        vlm_config = get_gemma_model_config(self.paligemma_variant)
        action_config = get_gemma_model_config(self.action_expert_variant)

        # Vision-Language stream (PaliGemma)
        vision_language_stream = MoTStreamConfig(
            name="vision_language",
            role=StreamRole.VISION_LANGUAGE,
            backbone_type="hf",
            preset="paligemma_language",
            hidden_size=vlm_config.width,
            head_dim=vlm_config.head_dim,
            num_attention_heads=vlm_config.num_heads,
            num_key_value_heads=vlm_config.num_kv_heads,
            num_layers=vlm_config.depth,
            use_adarms=self.use_adarms_vlm,
            use_gradient_checkpointing=self.gradient_checkpointing,
            dtype=self.dtype,
        )

        # Action stream (Gemma expert)
        action_stream = MoTStreamConfig(
            name="action",
            role=StreamRole.ACTION,
            backbone_type="hf",
            preset="gemma_lm",
            hidden_size=action_config.width,
            head_dim=action_config.head_dim,
            num_attention_heads=action_config.num_heads,
            num_key_value_heads=action_config.num_kv_heads,
            num_layers=action_config.depth,
            use_adarms=self.use_adarms_action,
            use_gradient_checkpointing=self.gradient_checkpointing,
            dtype=self.dtype,
        )

        self.streams = [vision_language_stream, action_stream]

    def _validate_pi0_config(self):
        """Validate Pi0-specific configuration."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than "
                f"chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        # Ensure streams have matching layer counts for joint attention
        vlm_layers = self.streams[0].num_layers
        action_layers = self.streams[1].num_layers
        if vlm_layers != action_layers:
            raise ValueError(
                f"VLM stream has {vlm_layers} layers but action stream has {action_layers}. "
                f"Pi0 requires matching layer counts for joint attention."
            )

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        # Add empty cameras if configured
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

        # Ensure state feature exists
        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature

        # Ensure action feature exists
        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        """Return optimizer configuration."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        """Return scheduler configuration."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def vlm_stream_config(self) -> MoTStreamConfig:
        """Get the vision-language stream configuration."""
        return self.get_stream("vision_language")

    @property
    def action_stream_config(self) -> MoTStreamConfig:
        """Get the action stream configuration."""
        return self.get_stream("action")

    @property
    def vlm_hidden_size(self) -> int:
        """Get the VLM hidden size."""
        return self.vlm_stream_config.hidden_size

    @property
    def action_hidden_size(self) -> int:
        """Get the action expert hidden size."""
        return self.action_stream_config.hidden_size
