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
MoT-Pi0 Policy Implementation

This module implements the Pi0 policy using the MoT (Multistream-VLA) framework.
It follows the "mechanism vs policy separation" philosophy:

- MoT Core (MoTModel): Handles synchronized layer execution and joint attention
- MoTPI0Policy: Handles Pi0-specific business logic:
  - Image embedding via PaliGemma vision tower
  - Text tokenization and embedding
  - Action/state projection
  - Attention mask construction (Prefix-LM + Causal Action)
  - Flow matching loss computation

The policy maintains the same functionality as the original Pi0 implementation
while leveraging the modular MoT architecture.
"""

from __future__ import annotations

import builtins
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.mot import MoTModel, MoTPolicy
from lerobot.policies.mot.backbones import HuggingFaceBackboneWrapper
from lerobot.policies.mot.modeling_mot import ActionSelectKwargs
from lerobot.policies.mot_pi0.configuration_mot_pi0 import (
    DEFAULT_IMAGE_SIZE, GemmaModelConfig, MoTPI0Config, get_gemma_model_config)
from lerobot.utils.constants import (ACTION, OBS_LANGUAGE_ATTENTION_MASK,
                                     OBS_LANGUAGE_TOKENS, OBS_STATE,
                                     OPENPI_ATTENTION_MASK_VALUE)

if TYPE_CHECKING:
    from lerobot.policies.pretrained import T

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions (adapted from original Pi0 implementation)
# ============================================================================

def get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device = None,
) -> Tensor:
    """Compute sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape (batch_size,)")

    if device is None:
        device = time.device

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> Tensor:
    """Sample from Beta distribution."""
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """
    Create 2D attention masks from padding and attention masks.

    This creates the prefix-LM style attention pattern:
    - Tokens with mask_ar=0 can attend to all previous tokens with mask_ar=0
    - Tokens with mask_ar=1 create attention boundaries

    Args:
        pad_masks: bool[B, N] - True if token is valid, False if padding
        att_masks: int/bool[B, N] - Attention boundary markers

    Returns:
        bool[B, N, N] - 2D attention mask
    """
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2D, got {att_masks.ndim}D")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2D, got {pad_masks.ndim}D")

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Pad the last dimension of a vector to new_dim with zeros."""
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(
    images: Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> Tensor:
    """Resize an image to target size without distortion by padding with black."""
    # Check format: [*b, h, w, c] or [*b, c, h, w]
    if images.shape[-1] <= 4:  # Channels-last
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    batch_size, channels, cur_height, cur_width = images.shape

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)

    return padded_images


# ============================================================================
# PaliGemma + Gemma Expert Model Wrapper
# ============================================================================

class PaliGemmaWithExpertWrapper(nn.Module):
    """
    Wrapper for PaliGemma and Gemma Expert models.

    This wraps the HuggingFace models and provides a unified interface for:
    - Image embedding via vision tower
    - Token embedding
    - Precision conversion

    Note: Unlike the original implementation, this wrapper does NOT handle
    the forward pass through transformer layers - that's delegated to MoTModel.
    """

    def __init__(
        self,
        vlm_config: GemmaModelConfig,
        action_expert_config: GemmaModelConfig,
        use_adarms: list[bool] | None = None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.precision = precision

        # Import HuggingFace components
        from transformers.models.auto import CONFIG_MAPPING
        from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
        from transformers.models.paligemma.modeling_paligemma import \
            PaliGemmaForConditionalGeneration

        # Build PaliGemma config
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        # Build Gemma expert config
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        # Initialize models
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        # Apply precision and requires_grad settings
        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        """Convert model to bfloat16 while keeping selected params in float32."""
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep certain params in float32 for numerical stability
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        """Set requires_grad based on freeze settings."""
        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for param in self.paligemma.vision_tower.parameters():
                param.requires_grad = False

        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: Tensor) -> Tensor:
        """Embed images using the PaliGemma vision tower."""
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        """Embed language tokens using the embedding layer."""
        return self.paligemma.language_model.embed_tokens(tokens)


# ============================================================================
# MoT-Pi0 Policy
# ============================================================================

class MoTPI0Policy(MoTPolicy):
    """
    Pi0 Policy implemented on the MoT (Multistream-VLA) framework.

    This policy implements the Pi0 algorithm using the modular MoT architecture.
    The key components are:

    1. PaliGemmaWithExpertWrapper: Handles image/text embedding
    2. Projection layers: action_in_proj, action_out_proj, state_proj, time_mlp
    3. MoTModel: Handles synchronized layer execution (from parent class)

    The forward pass flow:
    1. get_input_embeddings: Compute embeddings for vision_language and action streams
    2. _make_attention_mask: Build the Prefix-LM + Causal attention mask
    3. MoTModel.forward: Execute synchronized layers with joint attention
    4. Compute flow matching loss

    Attributes:
        config: MoTPI0Config with model configuration.
        paligemma_with_expert: Wrapper for VLM and action expert.
        action_in_proj: Projects noisy actions to hidden space.
        action_out_proj: Projects hidden states to action space.
        state_proj: Projects robot state to hidden space.
        action_time_mlp_in: First layer of time-action MLP.
        action_time_mlp_out: Second layer of time-action MLP.
    """

    config_class = MoTPI0Config
    name = "mot_pi0"

    def __init__(self, config: MoTPI0Config, **kwargs):
        """
        Initialize the MoT-Pi0 policy.

        Args:
            config: MoTPI0Config with model configuration.
            **kwargs: Additional arguments.
        """
        super().__init__(config, **kwargs)

        config.validate_features()
        self.config = config

        # Get model configurations
        vlm_config = get_gemma_model_config(config.paligemma_variant)
        action_expert_config = get_gemma_model_config(config.action_expert_variant)

        # Initialize the combined VLM + Expert wrapper
        self.paligemma_with_expert = PaliGemmaWithExpertWrapper(
            vlm_config=vlm_config,
            action_expert_config=action_expert_config,
            use_adarms=[config.use_adarms_vlm, config.use_adarms_action],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        # Projection layers
        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        self.state_proj = nn.Linear(config.max_state_dim, action_expert_config.width)

        # Time-action MLP
        self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
        self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Build backbone adapters for MoTModel
        self._init_mot_model()

        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()

        # Move to configured device
        if config.device:
            self.to(config.device)

        # Compile model if requested (for faster inference)
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.predict_action_chunk = torch.compile(
                self.predict_action_chunk, mode=config.compile_mode
            )
            # Optionally compile forward for faster training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

        # Initialize action queue for inference
        self.reset()

    def _init_mot_model(self):
        """Initialize the MoTModel with backbone adapters."""
        # Create adapters for each stream
        # Note: paligemma.language_model is a GemmaModel (layers at .layers directly)
        # while gemma_expert is GemmaForCausalLM (layers at .model.layers)
        vlm_adapter = HuggingFaceBackboneWrapper(
            model=self.paligemma_with_expert.paligemma.language_model,
            layers_attr="layers",  # GemmaModel has layers directly
            norm_attr="norm",      # GemmaModel has norm directly
            embed_tokens_attr="embed_tokens",
            rotary_emb_attr="rotary_emb",
            use_adarms=self.config.use_adarms_vlm,
        )

        action_adapter = HuggingFaceBackboneWrapper(
            model=self.paligemma_with_expert.gemma_expert,
            layers_attr="model.layers",  # GemmaForCausalLM has layers at .model.layers
            norm_attr="model.norm",
            embed_tokens_attr=None,  # Action expert doesn't use token embeddings
            rotary_emb_attr="model.rotary_emb",
            use_adarms=self.config.use_adarms_action,
        )

        # Initialize parent's MoT model
        self._init_model({
            "vision_language": vlm_adapter,
            "action": action_adapter,
        })

    # ========================================================================
    # Sampling Methods
    # ========================================================================

    def sample_noise(self, shape: tuple, device: torch.device) -> Tensor:
        """Sample Gaussian noise for flow matching."""
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize: int, device: torch.device) -> Tensor:
        """Sample time values from Beta distribution for flow matching."""
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize,
            device,
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    # ========================================================================
    # Embedding Methods
    # ========================================================================

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Embed images and language tokens for the vision-language stream.

        Args:
            images: List of image tensors.
            img_masks: List of image mask tensors.
            lang_tokens: Language token IDs.
            lang_masks: Language attention masks.

        Returns:
            tuple: (embeddings, padding_masks, attention_masks)
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        # Convert lang_masks to bool if it's not already
        if lang_masks.dtype != torch.bool:
            lang_masks = lang_masks.bool()
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(
        self,
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """
        Embed state, noisy actions, and timestep for the action stream.

        Args:
            state: Robot state tensor.
            noisy_actions: Noisy action trajectory.
            timestep: Current diffusion timestep.

        Returns:
            tuple: (embeddings, padding_masks, attention_masks, adarms_cond)
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Convert state to match projection dtype if needed
        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        # Embed state
        state_emb = self.state_proj(state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]  # State creates attention boundary

        # Embed timestep using sinusoidal encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Embed noisy actions
        action_emb = self.action_in_proj(noisy_actions)

        # Fuse timestep + action using MLP
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        adarms_cond = None  # AdaRMS conditioning (not used in base Pi0)

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Attention masks: state=1 (boundary), first action=1 (boundary), rest=0 (causal)
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def get_input_embeddings(
        self, batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """
        Compute input embeddings for each stream.

        This implements the abstract method from MoTPolicy.

        Args:
            batch: Input batch containing images, text, state, actions.

        Returns:
            Dictionary mapping stream names to embeddings.
        """
        # Preprocess images
        images, img_masks = self._preprocess_images(batch)

        # Get language tokens
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        # Get state
        state = batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)

        # For training, we need actions
        actions = batch.get(ACTION)
        if actions is not None:
            actions = pad_vector(actions, self.config.max_action_dim)

            # Sample noise and time
            noise = self.sample_noise(actions.shape, actions.device)
            time = self.sample_time(actions.shape[0], actions.device)

            # Compute noisy actions
            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
        else:
            # Inference: use provided noise or sample
            noise = batch.get("noise")
            if noise is None:
                actions_shape = (state.shape[0], self.config.chunk_size, self.config.max_action_dim)
                noise = self.sample_noise(actions_shape, state.device)
            x_t = noise
            time = batch.get("time")
            if time is None:
                time = torch.ones(state.shape[0], dtype=torch.float32, device=state.device)

        # Embed prefix (vision-language stream)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        # Embed suffix (action stream)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time
        )

        # Convert to appropriate dtype
        if self.paligemma_with_expert.precision == "bfloat16":
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # Store masks for later use in _make_attention_mask
        self._prefix_pad_masks = prefix_pad_masks
        self._prefix_att_masks = prefix_att_masks
        self._suffix_pad_masks = suffix_pad_masks
        self._suffix_att_masks = suffix_att_masks
        self._adarms_cond = adarms_cond

        return {
            "vision_language": prefix_embs,
            "action": suffix_embs,
        }

    def _make_attention_mask(
        self, batch: dict[str, Tensor]
    ) -> Tensor:
        """
        Construct the joint attention mask for all streams.

        This creates the Prefix-LM + Causal Action attention pattern:
        - Vision and language tokens can attend to each other (prefix)
        - Action tokens use causal attention
        - Image/language/state do not attend to action tokens

        Args:
            batch: Input batch (used for batch size/device).

        Returns:
            4D attention mask tensor.
        """
        # Combine padding and attention masks
        pad_masks = torch.cat([self._prefix_pad_masks, self._suffix_pad_masks], dim=1)
        att_masks = torch.cat([self._prefix_att_masks, self._suffix_att_masks], dim=1)

        # Create 2D attention mask
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

        # Convert to 4D mask for transformer
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        attention_mask = torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

        # Compute position IDs
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Store for forward pass
        self._position_ids = position_ids

        return attention_mask

    # ========================================================================
    # Forward Pass
    # ========================================================================

    def forward(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict | None]:
        """
        Forward pass with flow matching loss computation.

        Args:
            batch: Input batch containing images, text, state, actions.

        Returns:
            tuple: (loss, info_dict)
        """
        # Get target actions and compute flow target
        actions = batch[ACTION]
        actions = pad_vector(actions, self.config.max_action_dim)

        noise = self.sample_noise(actions.shape, actions.device)
        time = self.sample_time(actions.shape[0], actions.device)

        # Store for embedding
        batch["noise"] = noise
        batch["time"] = time

        # Compute flow target: u_t = noise - actions
        u_t = noise - actions

        # Get embeddings
        inputs_embeds_dict = self.get_input_embeddings(batch)

        # Get attention mask
        attention_mask = self._make_attention_mask(batch)

        # Forward through MoT model
        outputs = self.model(
            inputs_embeds_dict=inputs_embeds_dict,
            attention_mask=attention_mask,
            position_ids=self._position_ids,
            adarms_cond_dict={
                "vision_language": None,
                "action": self._adarms_cond,
            } if self._adarms_cond is not None else None,
        )

        # Get action stream output
        suffix_out = outputs["hidden_states"]["action"]
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Project to action space
        v_t = self.action_out_proj(suffix_out)

        # Compute MSE loss
        loss = F.mse_loss(u_t, v_t, reduction="mean")

        return loss, {"mse_loss": loss.item()}

    # ========================================================================
    # Inference
    # ========================================================================

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """
        Predict an action chunk using flow matching denoising with Static Prefix Caching.

        Flow Matching differs from autoregressive generation:
        - The Prefix (images + language) is STATIC across all denoising steps
        - The Suffix (actions) CHANGES at each step (refinement, not generation)

        Therefore, we use "Static Prefix Cache" strategy:
        1. Pre-compute: Run only Vision-Language stream, generate and freeze Prefix_KV
        2. Denoising Loop: Run Action stream with frozen Prefix_KV as context,
           but DON'T cache Suffix_KV (it changes every step)

        Args:
            batch: Observation batch.
            **kwargs: Additional inference arguments.

        Returns:
            Action chunk tensor of shape (batch, chunk_size, action_dim).
        """
        num_steps = self.config.num_inference_steps

        # 1. Prepare basic data
        state = batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        bsize = state.shape[0]
        device = state.device

        # 2. Sample initial noise
        noise = kwargs.get("noise")
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # 3. [KEY STEP] Pre-compute Static Prefix Cache
        # Only run Vision-Language stream to generate KV Cache
        images, img_masks = self._preprocess_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        if self.paligemma_with_expert.precision == "bfloat16":
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Build Prefix-only attention mask
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks_4d = prefix_att_2d_masks[:, None, :, :]
        prefix_attention_mask = torch.where(prefix_att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Run Model: only input vision_language to get prefix cache
        prefix_outputs = self.model(
            inputs_embeds_dict={"vision_language": prefix_embs},  # Only prefix
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            use_cache=True,  # Enable caching
        )

        # Get static cache [Prefix_KV]
        # IMPORTANT: Do NOT overwrite this variable in the loop!
        static_prefix_cache = prefix_outputs.past_key_values

        # Prepare prefix length info for the loop
        prefix_len = prefix_pad_masks.shape[1]
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]

        # 4. Denoising Loop with Static Prefix Cache
        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            # Embed Suffix (Action) with current x_t
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                state, x_t, time_tensor
            )

            if self.paligemma_with_expert.precision == "bfloat16":
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

            suffix_len = suffix_pad_masks.shape[1]

            # Build Joint Attention Mask (Suffix attends to Prefix + Suffix)
            # Shape: [B, 1, suffix_len, prefix_len + suffix_len]
            # - First prefix_len columns: Suffix can attend to all Prefix positions (0.0 = visible)
            # - Last suffix_len columns: Suffix follows causal/boundary attention pattern
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

            combined_att_mask = torch.zeros(
                bsize, 1, suffix_len, prefix_len + suffix_len,
                device=device, dtype=suffix_embs.dtype
            )

            # Part A: Suffix attends to Prefix (indices 0 to prefix_len)
            # Set to 0.0 meaning visible (full attention to prefix)
            combined_att_mask[:, :, :, :prefix_len] = 0.0

            # Part B: Suffix attends to Suffix (indices prefix_len to end)
            # Apply causal/boundary mask from suffix_att_2d_masks
            suffix_causal_mask = torch.where(
                suffix_att_2d_masks[:, None, :, :], 0.0, OPENPI_ATTENTION_MASK_VALUE
            )
            combined_att_mask[:, :, :, prefix_len:] = suffix_causal_mask

            # Position IDs: Suffix positions continue after Prefix
            suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            # Run Model: only input action, use static_prefix_cache
            # use_cache=False because we don't need to accumulate suffix KV
            # (each step's suffix is different, so caching it is useless)
            outputs = self.model(
                inputs_embeds_dict={
                    "vision_language": None,  # Don't recompute prefix, use cache
                    "action": suffix_embs,
                },
                attention_mask=combined_att_mask,
                position_ids=suffix_position_ids,
                past_key_values=static_prefix_cache,  # Use static prefix cache
                use_cache=False,  # Don't cache suffix KV (it changes every step)
                adarms_cond_dict={
                    "vision_language": None,
                    "action": adarms_cond,
                } if adarms_cond is not None else None,
            )

            # Get action output and predict velocity
            suffix_out = outputs.hidden_states["action"]
            suffix_out = suffix_out[:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=torch.float32)

            v_t = self.action_out_proj(suffix_out)

            # Euler step
            x_t = x_t + dt * v_t

        return x_t

    # ========================================================================
    # Image Preprocessing
    # ========================================================================

    def _preprocess_images(
        self, batch: dict[str, Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]:
        """
        Preprocess images for the model.

        Args:
            batch: Input batch containing image tensors.

        Returns:
            tuple: (list of image tensors, list of image masks)
        """
        images = []
        img_masks = []

        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. "
                f"batch keys: {batch.keys()}, image_features: {self.config.image_features}"
            )

        for key in present_img_keys:
            img = batch[key]

            if img.device != device:
                img = img.to(device)

            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # Handle format: [B, C, H, W] vs [B, H, W, C]
            is_channels_first = img.shape[1] == 3

            if is_channels_first:
                img = img.permute(0, 2, 3, 1)

            # Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1]
            img = img * 2.0 - 1.0

            if is_channels_first:
                img = img.permute(0, 3, 1, 2)

            images.append(img)
            img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=device))

        return images, img_masks

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_optim_params(self) -> dict:
        """Return optimization parameters."""
        return self.parameters()

    def reset(self):
        """Reset internal state."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        super().gradient_checkpointing_enable()
        # Also enable for the wrapper components
        if hasattr(self.paligemma_with_expert.paligemma, 'gradient_checkpointing_enable'):
            self.paligemma_with_expert.paligemma.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        super().gradient_checkpointing_disable()
        if hasattr(self.paligemma_with_expert.paligemma, 'gradient_checkpointing_disable'):
            self.paligemma_with_expert.paligemma.gradient_checkpointing_disable()

    # ========================================================================
    # Checkpoint Loading
    # ========================================================================

    @classmethod
    def from_pretrained(
        cls: builtins.type["T"],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> "MoTPI0Policy":
        """
        Load a pretrained MoT-PI0 model from a checkpoint.

        This method handles loading checkpoints from:
        1. HuggingFace Hub (e.g., "Physical-Intelligence/openpi")
        2. Local directory path
        3. Local file path (safetensors or bin)

        The method also handles key remapping for compatibility with the original
        OpenPI checkpoint format.

        Args:
            pretrained_name_or_path: HuggingFace model ID or local path.
            config: Optional config override.
            force_download: Force re-download even if cached.
            resume_download: Resume interrupted downloads.
            proxies: Proxy configuration.
            token: HuggingFace token for private models.
            cache_dir: Cache directory for downloads.
            local_files_only: Only use local files, don't download.
            revision: Model revision/branch.
            strict: Whether to strictly enforce state dict matching.
            **kwargs: Additional arguments passed to model constructor.

        Returns:
            Loaded MoTPI0Policy instance.
        """
        import re

        logger.info(
            "Loading MoT-PI0 model, compatible with OpenPI checkpoint format.\n"
            "MoT-PI0 uses the Multistream-VLA architecture for efficient inference.\n"
            "Original OpenPI: https://github.com/Physical-Intelligence/openpi"
        )

        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Load config if not provided
        if config is None:
            try:
                config = PreTrainedConfig.from_pretrained(
                    pretrained_name_or_path=pretrained_name_or_path,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    revision=revision,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(f"Could not load config from {pretrained_name_or_path}: {e}")
                logger.info("Using default MoTPI0Config")
                from lerobot.policies.mot_pi0.configuration_mot_pi0 import \
                    MoTPI0Config
                config = MoTPI0Config(**kwargs)

        # Initialize model without loading weights
        model = cls(config, **kwargs)

        # Load state dict
        try:
            logger.info(f"Loading checkpoint from: {pretrained_name_or_path}")
            original_state_dict = cls._load_state_dict_from_path(
                pretrained_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
                revision=revision,
            )

            if original_state_dict is None:
                logger.warning("Could not load state dict. Returning model with random weights.")
                return model

            # Fix state dict keys for compatibility
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict)

            # Remap keys: add proper prefixes for MoT structure
            remapped_state_dict = model._remap_state_dict_keys(fixed_state_dict)

            # Load state dict
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            # Report loading results
            if missing_keys:
                logger.warning(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                for key in missing_keys[:5]:
                    logger.warning(f"  - {key}")
                if len(missing_keys) > 5:
                    logger.warning(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                for key in unexpected_keys[:5]:
                    logger.warning(f"  - {key}")
                if len(unexpected_keys) > 5:
                    logger.warning(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                logger.info("✓ All checkpoint keys loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise

        return model

    @staticmethod
    def _load_state_dict_from_path(
        pretrained_name_or_path: str | Path,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> dict | None:
        """
        Load state dict from a path (local or HuggingFace Hub).

        Args:
            pretrained_name_or_path: Path or HF model ID.
            **kwargs: Arguments for cached_file.

        Returns:
            State dict or None if loading fails.
        """
        from pathlib import Path

        path = Path(pretrained_name_or_path)

        # Try local file first
        if path.exists():
            if path.is_file():
                # Direct file path
                if str(path).endswith(".safetensors"):
                    from safetensors.torch import load_file
                    logger.info(f"✓ Loading from local safetensors: {path}")
                    return load_file(str(path))
                else:
                    logger.info(f"✓ Loading from local bin: {path}")
                    return torch.load(str(path), map_location="cpu")
            elif path.is_dir():
                # Directory - look for model files
                for filename in ["model.safetensors", "pytorch_model.bin"]:
                    file_path = path / filename
                    if file_path.exists():
                        if filename.endswith(".safetensors"):
                            from safetensors.torch import load_file
                            logger.info(f"✓ Loading from local directory: {file_path}")
                            return load_file(str(file_path))
                        else:
                            logger.info(f"✓ Loading from local directory: {file_path}")
                            return torch.load(str(file_path), map_location="cpu")

        # Try HuggingFace Hub
        try:
            from transformers.utils import cached_file

            # Try safetensors first
            for filename in ["model.safetensors", "pytorch_model.bin"]:
                try:
                    resolved_file = cached_file(
                        pretrained_name_or_path,
                        filename,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        token=token,
                        revision=revision,
                        local_files_only=local_files_only,
                    )

                    if filename.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        logger.info(f"✓ Loaded from HuggingFace Hub: {filename}")
                        return load_file(resolved_file)
                    else:
                        logger.info(f"✓ Loaded from HuggingFace Hub: {filename}")
                        return torch.load(resolved_file, map_location="cpu")

                except Exception:
                    continue

        except Exception as e:
            logger.warning(f"Could not load from HuggingFace Hub: {e}")

        return None

    def _fix_pytorch_state_dict_keys(self, state_dict: dict) -> dict:
        """
        Fix state dict keys to match current model architecture.

        This handles differences between OpenPI checkpoint format and MoT-PI0 format:
        - Layer norm structure changes (standard vs AdaRMS)
        - MLP naming changes (time_mlp_* vs action_time_mlp_*)
        - Other architecture-specific differences

        Args:
            state_dict: Original state dict from checkpoint.

        Returns:
            Fixed state dict with compatible keys.
        """
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes for AdaRMS
            # Gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\."
                r"(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                expert_uses_adarms = getattr(self.config, "use_adarms_action", False)
                if expert_uses_adarms:
                    logger.debug(f"Skipping layer norm key (AdaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                expert_uses_adarms = getattr(self.config, "use_adarms_action", False)
                if expert_uses_adarms:
                    logger.debug(f"Skipping norm key (AdaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes
            # OpenPI uses time_mlp_*, MoT-PI0 uses action_time_mlp_*
            if key.startswith("time_mlp_in."):
                new_key = key.replace("time_mlp_in.", "action_time_mlp_in.")
            elif key.startswith("time_mlp_out."):
                new_key = key.replace("time_mlp_out.", "action_time_mlp_out.")

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def _remap_state_dict_keys(self, state_dict: dict) -> dict:
        """
        Remap state dict keys to match MoT-PI0 model structure.

        The MoT-PI0 model has a different structure from the original PI0:
        - paligemma_with_expert.paligemma.* -> paligemma_with_expert.paligemma.*
        - paligemma_with_expert.gemma_expert.* -> paligemma_with_expert.gemma_expert.*
        - action_in_proj.* -> action_in_proj.*
        - action_out_proj.* -> action_out_proj.*
        - state_proj.* -> state_proj.*
        - action_time_mlp_in.* -> action_time_mlp_in.*
        - action_time_mlp_out.* -> action_time_mlp_out.*

        OpenPI checkpoints may have different prefixes that need remapping.

        Args:
            state_dict: Fixed state dict.

        Returns:
            Remapped state dict with correct prefixes.
        """
        remapped_state_dict = {}
        remap_count = 0

        for key, value in state_dict.items():
            new_key = key

            # If the checkpoint has "model." prefix (from LeRobot training), keep it
            # If not, we need to check if it matches our expected structure

            # Handle OpenPI checkpoint format -> MoT-PI0 format
            # OpenPI: paligemma_with_expert.* -> MoT-PI0: paligemma_with_expert.*
            # This is already compatible, no change needed

            # Handle LeRobot checkpoint format (has "model." prefix for PI0Pytorch)
            # LeRobot PI0: model.paligemma_with_expert.* -> MoT-PI0: paligemma_with_expert.*
            if key.startswith("model."):
                # Remove the "model." prefix as MoT-PI0 doesn't have this wrapper
                new_key = key[6:]  # Remove "model."
                remap_count += 1

            remapped_state_dict[new_key] = value

        if remap_count > 0:
            logger.info(f"Remapped {remap_count} state dict keys (removed 'model.' prefix)")

        return remapped_state_dict
