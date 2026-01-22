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
MoT (Mixture of Transformers) Configuration.

This module defines the configuration classes for the MoT architecture, which allows
for flexible definition of multi-transformer architectures through configuration.

The key abstractions are:
- MoTNodeConfig: Defines a single modality/input node (vision, state, action, etc.)
- MoTFlowConfig: Defines attention flow between nodes (who attends to whom)
- MoTBackboneConfig: Defines a transformer backbone (PaliGemma, Gemma, Bagel, etc.)
- MoTConfig: The main configuration that combines nodes, flows, and backbones

This design allows expressing architectures like pi0, pi0.5, pi0FAST, and novel
architectures purely through configuration changes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


class NodeType(str, Enum):
    """Type of node in the MoT architecture."""
    VISION = "vision"           # Processes image inputs (ViT, SigLIP, etc.)
    STATE = "state"             # Processes robot state inputs
    ACTION = "action"           # Processes/generates action outputs
    LANGUAGE = "language"       # Processes language tokens
    VLM = "vlm"                 # Vision-Language Model (PaliGemma, etc.)
    LM = "lm"                   # Language Model (Gemma, Bagel, etc.)
    CUSTOM = "custom"           # Custom node type


class AttentionType(str, Enum):
    """Type of attention pattern between nodes."""
    FULL = "full"               # Full bidirectional attention
    CAUSAL = "causal"           # Causal (autoregressive) attention
    PREFIX = "prefix"           # Prefix-LM style (bidirectional then causal)
    CROSS = "cross"             # Cross attention only (no self-attention)
    NONE = "none"               # No attention (blocked)


class BackboneType(str, Enum):
    """Type of transformer backbone."""
    PALIGEMMA = "paligemma"     # PaliGemma VLM
    GEMMA = "gemma"             # Gemma language model
    LLAMA = "llama"             # LLaMA family models
    BAGEL = "bagel"             # Bagel model
    QWEN = "qwen"               # Qwen models
    CUSTOM = "custom"           # Custom transformer backbone


class DecodingMode(str, Enum):
    """Decoding/generation mode for action outputs."""
    FLOW_MATCHING = "flow_matching"     # Flow matching denoising (pi0, pi0.5)
    AUTOREGRESSIVE = "autoregressive"   # Autoregressive token generation (pi0FAST)
    DIRECT = "direct"                   # Direct MLP prediction (ACT-like)


@dataclass
class MoTNodeConfig:
    """
    Configuration for a single node in the MoT architecture.

    A node represents a modality or processing unit, such as vision encoder,
    state embedder, or action decoder.

    Attributes:
        name: Unique identifier for this node (e.g., "vision_front", "arm_state")
        node_type: Type of node (vision, state, action, language, vlm, lm, custom)
        backbone_name: Name of the backbone to use (references MoTBackboneConfig)
        input_dim: Dimension of raw input features (before projection)
        token_dim: Dimension after projection to shared token space
        max_tokens: Maximum number of tokens this node produces
        position_encoding: Type of positional encoding ("learnable", "sinusoidal", "rotary", "none")
        is_input: Whether this node receives external inputs
        is_output: Whether this node produces outputs (actions, predictions)
        output_head: Type of output head ("mlp", "linear", "tokenizer", "none")
        output_dim: Dimension of output (for action nodes)

        # Vision-specific options
        patch_size: Patch size for vision transformers
        image_size: Expected input image size

        # Projection options
        proj_type: Type of input projection ("linear", "mlp", "cnn", "none")
        proj_hidden_dim: Hidden dimension for MLP projection

        # Embedding options
        embed_vocab_size: Vocabulary size for token embedding (language nodes)
        embed_scale: Scale factor for embeddings
    """
    name: str
    node_type: NodeType | str = NodeType.CUSTOM
    backbone_name: str | None = None  # Which backbone processes this node's tokens

    # Dimension settings
    input_dim: int = 512
    token_dim: int = 512  # Projected embedding dimension
    max_tokens: int = 1   # Number of tokens this node produces

    # Position encoding
    position_encoding: Literal["learnable", "sinusoidal", "rotary", "none"] = "learnable"

    # Input/Output flags
    is_input: bool = True
    is_output: bool = False
    output_head: Literal["mlp", "linear", "tokenizer", "none"] = "none"
    output_dim: int | None = None

    # Vision-specific
    patch_size: int = 14
    image_size: int = 224

    # Projection settings
    proj_type: Literal["linear", "mlp", "cnn", "none"] = "linear"
    proj_hidden_dim: int | None = None

    # Embedding settings (for language nodes)
    embed_vocab_size: int | None = None
    embed_scale: float = 1.0

    # Additional config passed to the underlying encoder/decoder
    extra_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.node_type, str):
            self.node_type = NodeType(self.node_type)

        # Set sensible defaults based on node type
        if self.node_type == NodeType.VISION and self.max_tokens == 1:
            # Default to patch-based tokens for vision
            num_patches = (self.image_size // self.patch_size) ** 2
            self.max_tokens = num_patches

        if self.is_output and self.output_dim is None:
            self.output_dim = self.input_dim


@dataclass
class MoTFlowConfig:
    """
    Configuration for attention flow between nodes.

    Defines how tokens from one node can attend to tokens from another node.
    This is the key abstraction that enables flexible attention patterns.

    Attributes:
        source: Name of the source node (provides Key/Value)
        target: Name of the target node (provides Query)
        attention_type: Type of attention ("full", "causal", "prefix", "cross", "none")
        layer_range: Range of layers where this flow applies (None = all layers)
        attention_scale: Scale factor for attention weights
        is_bidirectional: If True, also creates reverse flow
    """
    source: str              # K/V source (the node being attended to)
    target: str              # Query source (the node doing the attending)
    attention_type: AttentionType | str = AttentionType.FULL
    layer_range: tuple[int, int] | None = None  # (start, end) or None for all
    attention_scale: float = 1.0
    is_bidirectional: bool = False  # Create reverse flow automatically

    def __post_init__(self):
        if isinstance(self.attention_type, str):
            self.attention_type = AttentionType(self.attention_type)


@dataclass
class MoTBackboneConfig:
    """
    Configuration for a transformer backbone.

    This defines the architecture of a transformer model that can be used
    to process tokens from one or more nodes.

    Attributes:
        name: Unique identifier for this backbone
        backbone_type: Type of backbone (paligemma, gemma, llama, bagel, custom)
        variant: Specific variant (e.g., "gemma_2b", "gemma_300m")
        hidden_size: Hidden dimension of the transformer
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of KV heads (for GQA)
        head_dim: Dimension of each attention head
        intermediate_size: FFN intermediate dimension
        hidden_activation: Activation function for FFN

        # Model loading
        pretrained_path: Path or HF repo for pretrained weights
        freeze: Whether to freeze this backbone

        # Vision-specific (for VLM backbones)
        vision_config: Nested config for vision encoder

        # Additional HF config kwargs
        hf_config_kwargs: Extra arguments passed to HF config
    """
    name: str
    backbone_type: BackboneType | str = BackboneType.GEMMA
    variant: str = "gemma_2b"

    # Architecture settings
    hidden_size: int = 2048
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    intermediate_size: int = 16384
    hidden_activation: str = "gelu_pytorch_tanh"
    vocab_size: int = 257152

    # Loading settings
    pretrained_path: str | None = None
    freeze: bool = False
    train_only_projections: bool = False

    # AdaRMS settings (for pi0.5-style conditioning)
    use_adarms: bool = False
    adarms_cond_dim: int | None = None

    # Vision config (for VLM backbones)
    vision_hidden_size: int | None = None
    vision_num_layers: int | None = None
    vision_patch_size: int = 14
    vision_image_size: int = 224

    # Precision
    dtype: str = "float32"

    # Extra HF config kwargs
    hf_config_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.backbone_type, str):
            self.backbone_type = BackboneType(self.backbone_type)

        if self.use_adarms and self.adarms_cond_dim is None:
            self.adarms_cond_dim = self.hidden_size


# Preset configurations for common model variants
GEMMA_300M_CONFIG = {
    "hidden_size": 1024,
    "num_hidden_layers": 18,
    "intermediate_size": 4096,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
}

GEMMA_2B_CONFIG = {
    "hidden_size": 2048,
    "num_hidden_layers": 18,
    "intermediate_size": 16384,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
}

PALIGEMMA_2B_VISION_CONFIG = {
    "vision_hidden_size": 1152,
    "vision_num_layers": 27,
    "vision_patch_size": 14,
    "vision_image_size": 224,
}


def get_backbone_preset(variant: str) -> dict[str, Any]:
    """Get preset configuration for a known backbone variant."""
    presets = {
        "gemma_300m": GEMMA_300M_CONFIG,
        "gemma_2b": GEMMA_2B_CONFIG,
        "paligemma_2b": {**GEMMA_2B_CONFIG, **PALIGEMMA_2B_VISION_CONFIG},
    }
    if variant not in presets:
        raise ValueError(f"Unknown backbone variant: {variant}. Available: {list(presets.keys())}")
    return presets[variant]


@PreTrainedConfig.register_subclass("mot")
@dataclass
class MoTConfig(PreTrainedConfig):
    """
    Main configuration for the MoT (Mixture of Transformers) architecture.

    This configuration allows defining arbitrary multi-transformer architectures
    through a graph-like specification of nodes and flows.

    Attributes:
        nodes: List of node configurations (vision, state, action, etc.)
        flows: List of attention flow configurations (who attends to whom)
        backbones: List of backbone configurations (transformers used)

        decoding_mode: How actions are decoded (flow_matching, autoregressive, direct)

        # Standard policy parameters
        n_obs_steps: Number of observation steps
        chunk_size: Number of action steps to predict
        n_action_steps: Number of action steps to execute

        # Dimension settings
        max_state_dim: Maximum state dimension (padding)
        max_action_dim: Maximum action dimension (padding)

        # Flow matching parameters
        num_inference_steps: Denoising steps for flow matching
        time_sampling_beta_alpha: Beta distribution alpha for time sampling
        time_sampling_beta_beta: Beta distribution beta for time sampling

        # Autoregressive parameters
        max_decoding_steps: Maximum tokens to generate
        temperature: Sampling temperature

        # Image settings
        image_resolution: Expected image resolution
        empty_cameras: Number of empty camera slots

        # Training settings
        gradient_checkpointing: Enable gradient checkpointing
        compile_model: Enable torch.compile

        # Optimizer/scheduler settings
        optimizer_*: Optimizer configuration
        scheduler_*: Scheduler configuration
    """

    # ==================== Core MoT Architecture ====================
    nodes: list[MoTNodeConfig] = field(default_factory=list)
    flows: list[MoTFlowConfig] = field(default_factory=list)
    backbones: list[MoTBackboneConfig] = field(default_factory=list)

    # Decoding mode
    decoding_mode: DecodingMode | str = DecodingMode.FLOW_MATCHING

    # ==================== Standard Policy Parameters ====================
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # Dimension padding
    max_state_dim: int = 32
    max_action_dim: int = 32

    # ==================== Flow Matching Parameters ====================
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # ==================== Autoregressive Parameters ====================
    max_decoding_steps: int = 256
    temperature: float = 0.0
    use_kv_cache: bool = True

    # Tokenizer settings
    text_tokenizer_name: str = "google/paligemma-3b-pt-224"
    action_tokenizer_name: str | None = None  # For FAST-style
    tokenizer_max_length: int = 48

    # ==================== Image Settings ====================
    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 0

    # ==================== Normalization ====================
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ==================== Training Settings ====================
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    dtype: str = "float32"
    device: str | None = None

    # Finetuning
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # ==================== Optimizer Settings ====================
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # ==================== Scheduler Settings ====================
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 2.5e-6

    # ==================== RTC Configuration ====================
    # rtc_config can be imported and used if needed

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.decoding_mode, str):
            self.decoding_mode = DecodingMode(self.decoding_mode)

        # Convert dict configs to dataclass instances
        self.nodes = [
            MoTNodeConfig(**n) if isinstance(n, dict) else n
            for n in self.nodes
        ]
        self.flows = [
            MoTFlowConfig(**f) if isinstance(f, dict) else f
            for f in self.flows
        ]
        self.backbones = [
            MoTBackboneConfig(**b) if isinstance(b, dict) else b
            for b in self.backbones
        ]

        # Validation
        self._validate_config()

    def _validate_config(self):
        """Validate that the configuration is consistent."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than "
                f"chunk_size ({self.chunk_size})"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        # Validate node names are unique
        node_names = [n.name for n in self.nodes]
        if len(node_names) != len(set(node_names)):
            raise ValueError("Node names must be unique")

        # Validate flows reference valid nodes
        for flow in self.flows:
            if flow.source not in node_names:
                raise ValueError(f"Flow source '{flow.source}' not found in nodes")
            if flow.target not in node_names:
                raise ValueError(f"Flow target '{flow.target}' not found in nodes")

        # Validate backbones are referenced by nodes
        backbone_names = {b.name for b in self.backbones}
        for node in self.nodes:
            if node.backbone_name and node.backbone_name not in backbone_names:
                raise ValueError(
                    f"Node '{node.name}' references unknown backbone '{node.backbone_name}'"
                )

    def get_node(self, name: str) -> MoTNodeConfig | None:
        """Get a node configuration by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_backbone(self, name: str) -> MoTBackboneConfig | None:
        """Get a backbone configuration by name."""
        for backbone in self.backbones:
            if backbone.name == name:
                return backbone
        return None

    def get_output_nodes(self) -> list[MoTNodeConfig]:
        """Get all nodes that produce outputs."""
        return [n for n in self.nodes if n.is_output]

    def get_input_nodes(self) -> list[MoTNodeConfig]:
        """Get all nodes that receive inputs."""
        return [n for n in self.nodes if n.is_input]

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        # Add empty camera features
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
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
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
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


# ==================== Preset Configurations ====================

def make_pi0_config(**kwargs) -> MoTConfig:
    """
    Create a configuration equivalent to the original pi0 architecture.

    pi0 uses:
    - PaliGemma 2B as VLM (vision + language)
    - Gemma 300M as action expert
    - Joint layer attention between VLM and expert
    - Flow matching for action decoding
    """
    nodes = [
        MoTNodeConfig(
            name="vlm",
            node_type=NodeType.VLM,
            backbone_name="paligemma",
            input_dim=2048,
            token_dim=2048,
            max_tokens=256 + 48,  # image patches + language tokens
            is_input=True,
            is_output=False,
        ),
        MoTNodeConfig(
            name="action_expert",
            node_type=NodeType.LM,
            backbone_name="gemma_expert",
            input_dim=1024,
            token_dim=1024,
            max_tokens=51,  # state + action tokens
            is_input=True,
            is_output=True,
            output_head="linear",
            output_dim=kwargs.get("max_action_dim", 32),
        ),
    ]

    flows = [
        # VLM self-attention (bidirectional for images/language)
        MoTFlowConfig(source="vlm", target="vlm", attention_type=AttentionType.FULL),
        # Expert can attend to VLM (prefix-style)
        MoTFlowConfig(source="vlm", target="action_expert", attention_type=AttentionType.FULL),
        # Expert self-attention (causal for actions)
        MoTFlowConfig(source="action_expert", target="action_expert", attention_type=AttentionType.CAUSAL),
    ]

    backbones = [
        MoTBackboneConfig(
            name="paligemma",
            backbone_type=BackboneType.PALIGEMMA,
            variant="gemma_2b",
            **GEMMA_2B_CONFIG,
            **PALIGEMMA_2B_VISION_CONFIG,
        ),
        MoTBackboneConfig(
            name="gemma_expert",
            backbone_type=BackboneType.GEMMA,
            variant="gemma_300m",
            **GEMMA_300M_CONFIG,
        ),
    ]

    defaults = {
        "nodes": nodes,
        "flows": flows,
        "backbones": backbones,
        "decoding_mode": DecodingMode.FLOW_MATCHING,
        "chunk_size": 50,
        "n_action_steps": 50,
        "num_inference_steps": 10,
    }
    defaults.update(kwargs)

    return MoTConfig(**defaults)


def make_pi05_config(**kwargs) -> MoTConfig:
    """
    Create a configuration equivalent to the pi0.5 architecture.

    Similar to pi0 but with AdaRMS conditioning and different normalization.
    """
    config = make_pi0_config(**kwargs)

    # Update normalization for pi0.5
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.QUANTILES,
        "ACTION": NormalizationMode.QUANTILES,
    }

    # Enable AdaRMS for the expert backbone
    for backbone in config.backbones:
        if backbone.name == "gemma_expert":
            backbone.use_adarms = True

    return config


def make_pi0fast_config(**kwargs) -> MoTConfig:
    """
    Create a configuration equivalent to the pi0FAST architecture.

    Uses autoregressive token generation instead of flow matching.
    """
    nodes = [
        MoTNodeConfig(
            name="vlm",
            node_type=NodeType.VLM,
            backbone_name="paligemma",
            input_dim=2048,
            token_dim=2048,
            max_tokens=256 + 200,  # image patches + longer language
            is_input=True,
            is_output=True,
            output_head="tokenizer",
        ),
    ]

    flows = [
        # VLM self-attention (causal for autoregressive generation)
        MoTFlowConfig(source="vlm", target="vlm", attention_type=AttentionType.PREFIX),
    ]

    backbones = [
        MoTBackboneConfig(
            name="paligemma",
            backbone_type=BackboneType.PALIGEMMA,
            variant="gemma_2b",
            **GEMMA_2B_CONFIG,
            **PALIGEMMA_2B_VISION_CONFIG,
        ),
    ]

    defaults = {
        "nodes": nodes,
        "flows": flows,
        "backbones": backbones,
        "decoding_mode": DecodingMode.AUTOREGRESSIVE,
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_decoding_steps": 256,
        "temperature": 0.0,
        "action_tokenizer_name": "physical-intelligence/fast",
        "tokenizer_max_length": 200,
    }
    defaults.update(kwargs)

    return MoTConfig(**defaults)


def make_custom_mot_config(
    num_transformers: int = 2,
    transformer_configs: list[dict] | None = None,
    attention_matrix: list[list[str]] | None = None,
    **kwargs
) -> MoTConfig:
    """
    Create a custom MoT configuration with N transformers.

    Args:
        num_transformers: Number of transformer nodes
        transformer_configs: List of dicts with transformer settings per node
        attention_matrix: NxN matrix of attention types between nodes
            attention_matrix[i][j] = attention type from node j to node i
        **kwargs: Additional MoTConfig parameters

    Example:
        config = make_custom_mot_config(
            num_transformers=3,
            transformer_configs=[
                {"name": "vision", "backbone_type": "paligemma"},
                {"name": "language", "backbone_type": "gemma"},
                {"name": "action", "backbone_type": "gemma"},
            ],
            attention_matrix=[
                ["full", "none", "none"],   # vision self-attends
                ["full", "full", "none"],   # language attends to vision + self
                ["full", "full", "causal"], # action attends to all, causal self
            ],
        )
    """
    if transformer_configs is None:
        transformer_configs = [
            {"name": f"transformer_{i}", "backbone_type": "gemma"}
            for i in range(num_transformers)
        ]

    if attention_matrix is None:
        # Default: each transformer attends to all previous + self
        attention_matrix = []
        for i in range(num_transformers):
            row = []
            for j in range(num_transformers):
                if j < i:
                    row.append("full")
                elif j == i:
                    row.append("causal" if i == num_transformers - 1 else "full")
                else:
                    row.append("none")
            attention_matrix.append(row)

    nodes = []
    backbones = []

    for i, cfg in enumerate(transformer_configs):
        name = cfg.get("name", f"transformer_{i}")
        backbone_type = cfg.get("backbone_type", "gemma")
        variant = cfg.get("variant", "gemma_300m")

        # Create backbone
        preset = get_backbone_preset(variant) if variant in ["gemma_300m", "gemma_2b", "paligemma_2b"] else {}
        backbone = MoTBackboneConfig(
            name=f"{name}_backbone",
            backbone_type=backbone_type,
            variant=variant,
            **preset,
            **cfg.get("backbone_kwargs", {}),
        )
        backbones.append(backbone)

        # Create node
        node_type = NodeType.VLM if backbone_type == "paligemma" else NodeType.LM
        is_output = i == num_transformers - 1  # Last transformer is output

        node = MoTNodeConfig(
            name=name,
            node_type=node_type,
            backbone_name=backbone.name,
            input_dim=preset.get("hidden_size", 1024),
            token_dim=preset.get("hidden_size", 1024),
            is_output=is_output,
            output_head="linear" if is_output else "none",
            **cfg.get("node_kwargs", {}),
        )
        nodes.append(node)

    # Create flows from attention matrix
    flows = []
    for i, row in enumerate(attention_matrix):
        for j, att_type in enumerate(row):
            if att_type != "none":
                flows.append(MoTFlowConfig(
                    source=nodes[j].name,
                    target=nodes[i].name,
                    attention_type=att_type,
                ))

    defaults = {
        "nodes": nodes,
        "flows": flows,
        "backbones": backbones,
    }
    defaults.update(kwargs)

    return MoTConfig(**defaults)
