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
MoT (Mixture of Transformers) Processor.

This module provides pre-processing and post-processing pipelines for the MoT policy.
It handles:
- Input normalization and tokenization
- Image preprocessing
- Output unnormalization
- Device management
"""

from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.mot.configuration_mot import DecodingMode, MoTConfig
from lerobot.processor import (AddBatchDimensionProcessorStep,
                               ComplementaryDataProcessorStep,
                               DeviceProcessorStep, NormalizerProcessorStep,
                               PolicyAction, PolicyProcessorPipeline,
                               ProcessorStep, ProcessorStepRegistry,
                               RenameObservationsProcessorStep,
                               TokenizerProcessorStep,
                               UnnormalizerProcessorStep)
from lerobot.processor.converters import (policy_action_to_transition,
                                          transition_to_policy_action)
from lerobot.utils.constants import (POLICY_POSTPROCESSOR_DEFAULT_NAME,
                                     POLICY_PREPROCESSOR_DEFAULT_NAME)


@ProcessorStepRegistry.register(name="mot_new_line_processor")
class MoTNewLineProcessor(ComplementaryDataProcessorStep):
    """
    Ensures that the task description string ends with a newline character.

    This processing step is required for compatibility with tokenizers like PaliGemma
    that expect a newline at the end of the text prompt.
    """

    def complementary_data(self, complementary_data):
        """
        Adds a newline to the 'task' field if it doesn't already have one.

        Args:
            complementary_data: A dictionary that may contain a 'task' key.

        Returns:
            A new dictionary with the modified 'task' field.
        """
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)

        if isinstance(task, str):
            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """This step does not alter the feature definitions."""
        return features


@ProcessorStepRegistry.register(name="mot_image_preprocessor")
class MoTImagePreprocessor(ProcessorStep):
    """
    Preprocesses images for MoT models.

    Handles:
    - Channel ordering (CHW vs HWC)
    - Normalization to [-1, 1] range
    - Resizing with padding
    """

    def __init__(
        self,
        image_resolution: tuple[int, int] = (224, 224),
        normalize_to_minus_one: bool = True,
    ):
        self.image_resolution = image_resolution
        self.normalize_to_minus_one = normalize_to_minus_one

    def forward(self, transition: dict[str, Any]) -> dict[str, Any]:
        """Process images in the transition."""
        from lerobot.policies.mot.modeling_mot import resize_with_pad_torch

        # Find image keys in observation
        if "observation" not in transition:
            return transition

        observation = transition["observation"]
        new_observation = dict(observation)

        for key, value in observation.items():
            if not isinstance(value, torch.Tensor):
                continue

            # Check if this looks like an image (3 or 4 dims, channel dim is 3)
            if value.ndim < 3:
                continue

            is_image = False
            if value.ndim == 3 and value.shape[0] == 3:  # C, H, W
                is_image = True
            elif value.ndim == 4 and value.shape[1] == 3:  # B, C, H, W
                is_image = True

            if not is_image:
                continue

            img = value
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # Handle channel ordering for resizing
            if img.ndim == 3:
                img = img.unsqueeze(0)  # Add batch dim

            # Convert to HWC for resize_with_pad_torch
            img = img.permute(0, 2, 3, 1)  # B, C, H, W -> B, H, W, C

            # Resize if needed
            if img.shape[1:3] != self.image_resolution:
                img = resize_with_pad_torch(img, *self.image_resolution)

            # Normalize to [-1, 1] if requested
            if self.normalize_to_minus_one:
                img = img * 2.0 - 1.0

            # Convert back to CHW
            img = img.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W

            if value.ndim == 3:
                img = img.squeeze(0)

            new_observation[key] = img

        transition["observation"] = new_observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Update feature shapes for resized images."""
        return features


def make_mot_pre_post_processors(
    config: MoTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the MoT policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Adding a batch dimension.
    3. Appending a newline character to the task description for tokenizer compatibility.
    4. Tokenizing the text prompt.
    5. Moving all data to the specified device.
    6. Normalizing input and output features based on dataset statistics.

    The post-processing pipeline handles the model's output by:
    1. Unnormalizing the output features to their original scale.
    2. Moving data to the CPU.

    Args:
        config: The configuration object for the MoT policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    # Determine tokenizer based on config
    tokenizer_name = config.text_tokenizer_name
    max_length = config.tokenizer_max_length

    # Build input processing steps
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        MoTNewLineProcessor(),
        TokenizerProcessorStep(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    # Add action tokenizer for autoregressive mode
    if config.decoding_mode == DecodingMode.AUTOREGRESSIVE and config.action_tokenizer_name:
        from lerobot.processor import ActionTokenizerProcessorStep

        input_steps.insert(
            -1,  # Before normalizer
            ActionTokenizerProcessorStep(
                tokenizer_name=config.action_tokenizer_name,
                max_length=config.max_decoding_steps,
            ),
        )

    # Build output processing steps
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def make_mot_flow_matching_processors(
    config: MoTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Create processors specifically for flow matching mode (pi0, pi0.5 style).

    This is a convenience function that ensures the config is set up for
    flow matching decoding.
    """
    if config.decoding_mode != DecodingMode.FLOW_MATCHING:
        config.decoding_mode = DecodingMode.FLOW_MATCHING

    return make_mot_pre_post_processors(config, dataset_stats)


def make_mot_autoregressive_processors(
    config: MoTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Create processors specifically for autoregressive mode (pi0FAST style).

    This is a convenience function that ensures the config is set up for
    autoregressive decoding with action tokenization.
    """
    if config.decoding_mode != DecodingMode.AUTOREGRESSIVE:
        config.decoding_mode = DecodingMode.AUTOREGRESSIVE

    return make_mot_pre_post_processors(config, dataset_stats)
