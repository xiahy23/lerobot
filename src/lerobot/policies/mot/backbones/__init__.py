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
MoT Backbone Adapters - Registry and Factory.

This module provides a unified interface for creating and managing backbone adapters
in the MoT (Mixture of Transformers) architecture.

The registry pattern allows adding new backbone types without modifying the core
MoT model code. To add a new backbone:

1. Create a new adapter file (e.g., `backbones/deepseek.py`)
2. Implement your adapter class extending `BackboneAdapter`
3. Register it using `@BackboneRegistry.register("deepseek")`

Example usage:
    from lerobot.policies.mot.backbones import build_backbone, BackboneRegistry

    # Build a backbone from config
    backbone = build_backbone(backbone_config)

    # Or register a custom backbone
    @BackboneRegistry.register("my_custom_backbone")
    class MyCustomAdapter(BackboneAdapter):
        ...
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Type

from .base import BackboneAdapter

if TYPE_CHECKING:
    from lerobot.policies.mot.configuration_mot import MoTBackboneConfig

logger = logging.getLogger(__name__)


class BackboneRegistry:
    """
    Registry for backbone adapter classes.

    This class implements a type-safe registry pattern that allows:
    - Registration of new backbone types via decorator
    - Runtime lookup of adapter classes by backbone type
    - Easy extensibility without modifying core code
    """

    _registry: dict[str, Type[BackboneAdapter]] = {}

    @classmethod
    def register(cls, backbone_type: str):
        """
        Decorator to register a backbone adapter class.

        Args:
            backbone_type: The backbone type string (e.g., "gemma", "paligemma")

        Returns:
            Decorator function that registers the class

        Example:
            @BackboneRegistry.register("my_backbone")
            class MyBackboneAdapter(BackboneAdapter):
                ...
        """

        def decorator(adapter_cls: Type[BackboneAdapter]) -> Type[BackboneAdapter]:
            if backbone_type in cls._registry:
                logger.warning(
                    f"Overwriting existing backbone registration for '{backbone_type}'"
                )
            cls._registry[backbone_type] = adapter_cls
            logger.debug(f"Registered backbone adapter: {backbone_type} -> {adapter_cls.__name__}")
            return adapter_cls

        return decorator

    @classmethod
    def get(cls, backbone_type: str) -> Type[BackboneAdapter]:
        """
        Get the adapter class for a backbone type.

        Args:
            backbone_type: The backbone type string

        Returns:
            The registered adapter class

        Raises:
            ValueError: If backbone type is not registered
        """
        # Normalize the backbone type (handle enum values)
        type_str = str(backbone_type).lower()
        if "." in type_str:  # Handle enum like "BackboneType.GEMMA"
            type_str = type_str.split(".")[-1]

        if type_str not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown backbone type: '{backbone_type}'. "
                f"Available types: {available}"
            )
        return cls._registry[type_str]

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return a list of all registered backbone types."""
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, backbone_type: str) -> bool:
        """Check if a backbone type is registered."""
        type_str = str(backbone_type).lower()
        if "." in type_str:
            type_str = type_str.split(".")[-1]
        return type_str in cls._registry


def build_backbone(config: MoTBackboneConfig) -> BackboneAdapter:
    """
    Factory function to build a backbone adapter from configuration.

    This is the main entry point for creating backbone instances. It:
    1. Looks up the appropriate adapter class from the registry
    2. Creates an instance with the provided config
    3. Initializes the underlying HuggingFace model

    Args:
        config: Configuration for the backbone (MoTBackboneConfig)

    Returns:
        Initialized backbone adapter ready for use

    Raises:
        ValueError: If the backbone type is not registered

    Example:
        from lerobot.policies.mot.backbones import build_backbone
        from lerobot.policies.mot.configuration_mot import MoTBackboneConfig

        config = MoTBackboneConfig(
            name="my_gemma",
            backbone_type="gemma",
            hidden_size=2048,
            ...
        )
        backbone = build_backbone(config)
    """
    adapter_cls = BackboneRegistry.get(config.backbone_type)
    adapter = adapter_cls(config)
    adapter.setup_model()

    logger.info(
        f"Built backbone '{config.name}' of type '{config.backbone_type}'"
    )

    return adapter


def build_and_load_backbone(
    config: MoTBackboneConfig,
    **load_kwargs,
) -> BackboneAdapter:
    """
    Build a backbone and load pretrained weights.

    This is a convenience function that combines building and loading.

    Args:
        config: Configuration for the backbone
        **load_kwargs: Additional arguments for weight loading

    Returns:
        Initialized backbone adapter with loaded weights

    Raises:
        ValueError: If backbone_type is not registered or pretrained_path is missing
    """
    if not config.pretrained_path:
        raise ValueError(
            f"Cannot load pretrained weights: pretrained_path is not set in config for '{config.name}'"
        )

    backbone = build_backbone(config)
    backbone.load_pretrained(config.pretrained_path, **load_kwargs)

    return backbone


# ==================== Register Built-in Adapters ====================
# Import adapters to trigger registration

from .gemma import GemmaAdapter
from .llama import LlamaAdapter
from .paligemma import PaliGemmaAdapter
from .qwen import QwenAdapter

# Register adapters
BackboneRegistry.register("gemma")(GemmaAdapter)
BackboneRegistry.register("paligemma")(PaliGemmaAdapter)
BackboneRegistry.register("llama")(LlamaAdapter)
BackboneRegistry.register("qwen")(QwenAdapter)

# Also register Bagel as a Gemma fallback (as in original code)
BackboneRegistry.register("bagel")(GemmaAdapter)


# ==================== Preset Utilities ====================

def get_backbone_preset(variant: str) -> dict:
    """
    Get preset configuration for a known model variant.

    This function looks up presets across all registered backbone types.

    Args:
        variant: Variant name (e.g., "gemma_2b", "paligemma_2b", "llama3_8b")

    Returns:
        Dictionary of configuration values

    Raises:
        ValueError: If variant is not recognized
    """
    from .gemma import GEMMA_PRESETS
    from .llama import LLAMA_PRESETS
    from .paligemma import PALIGEMMA_PRESETS
    from .qwen import QWEN_PRESETS

    all_presets = {
        **GEMMA_PRESETS,
        **PALIGEMMA_PRESETS,
        **LLAMA_PRESETS,
        **QWEN_PRESETS,
    }

    if variant not in all_presets:
        available = list(all_presets.keys())
        raise ValueError(f"Unknown variant: {variant}. Available: {available}")

    return all_presets[variant].copy()


# ==================== Public API ====================

__all__ = [
    # Base class
    "BackboneAdapter",
    # Registry
    "BackboneRegistry",
    # Factory functions
    "build_backbone",
    "build_and_load_backbone",
    # Preset utility
    "get_backbone_preset",
    # Concrete adapters (for direct use if needed)
    "GemmaAdapter",
    "PaliGemmaAdapter",
    "LlamaAdapter",
    "QwenAdapter",
]
