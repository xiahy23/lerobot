import builtins
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    from transformers.models.paligemma.modeling_paligemma import \
        PaliGemmaForConditionalGeneration
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    GemmaForCausalLM = None
    PaliGemmaForConditionalGeneration = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.mot import MoTModel, MoTPolicy
from lerobot.policies.mot.backbones import HuggingFaceBackboneWrapper
from lerobot.policies.mot.modeling_mot import ActionSelectKwargs
from lerobot.policies.mot_pi05.configuration_mot_pi05 import (
    DEFAULT_IMAGE_SIZE, GemmaModelConfig, MoTPI05Config,
    get_gemma_model_config)
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (ACTION, OBS_LANGUAGE_ATTENTION_MASK,
                                     OBS_LANGUAGE_TOKENS,
                                     OPENPI_ATTENTION_MASK_VALUE)

logger = logging.getLogger(__name__)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images

class PaliGemmaWithExpertWrapper(nn.Module):

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

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
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

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

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

class MoTPI05Policy(MoTPolicy):

    config_class = MoTPI05Config
    name = "mot_pi05"

    def __init__(self, config: MoTPI05Config, rtc_processor: RTCProcessor | None = None, **kwargs):
        super().__init__(config, **kwargs)

        config.validate_features()
        self.config = config
        self.rtc_processor = rtc_processor

        vlm_config = get_gemma_model_config(config.paligemma_variant)
        action_expert_config = get_gemma_model_config(config.action_expert_variant)

        # Initialize the combined VLM + Expert wrapper
        self.paligemma_with_expert = PaliGemmaWithExpertWrapper(
            vlm_config=vlm_config,
            action_expert_config=action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        # Projection layers
        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self._init_mot_model()
        self.gradient_checkpointing_enabled = False

        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()

        # Move to configured device
        if config.device:
            self.to(config.device)

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


    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
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

        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

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
                actions_shape = (
                    bsize,
                    self.config.chunk_size,
                    self.config.max_action_dim,
                )
                noise = self.sample_noise(actions_shape, device)
            x_t = noise

        # Embed prefix (vision-language stream)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        # Embed suffix (action stream)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, time
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

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

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

        # 1. Prepare basic data - get batch size and device from language tokens
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        # 2. Sample initial noise
        noise = kwargs.get("noise")
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # 3. [KEY STEP] Pre-compute Static Prefix Cache
        # Only run Vision-Language stream to generate KV Cache
        images, img_masks = self._preprocess_images(batch)
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

            # Create partial function for RTC support
            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    static_prefix_cache=static_prefix_cache,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            # Apply RTC if enabled, otherwise use standard denoising
            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            # Euler step
            x_t = x_t + dt * v_t

            # Track denoising data if RTC processor is available and debug enabled
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

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

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def get_optim_params(self) -> dict:
        """Return optimization parameters."""
        return self.parameters()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Select a single action given environment observations.

        Uses action queue logic to execute n_action_steps from each chunk prediction.

        Args:
            batch: Observation batch.

        Returns:
            Single action tensor of shape (batch, action_dim).
        """
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # Unpad actions to actual action dimension
            original_action_dim = self.config.output_features[ACTION].shape[0]
            actions = actions[:, :, :original_action_dim]

            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def denoise_step(
        self,
        prefix_pad_masks: Tensor,
        static_prefix_cache: dict,
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """
        Apply one denoising step of the noise `x_t` at a given timestep.

        This is separated out to support RTC (Real-Time Chunking).

        Args:
            prefix_pad_masks: Padding masks for prefix.
            static_prefix_cache: Cached KV from prefix computation.
            x_t: Current noisy actions.
            timestep: Current diffusion timestep.

        Returns:
            Predicted velocity v_t.
        """
        bsize = x_t.shape[0]
        device = x_t.device
        prefix_len = prefix_pad_masks.shape[1]

        # Embed Suffix (Action) with current x_t
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, timestep
        )

        if self.paligemma_with_expert.precision == "bfloat16":
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        suffix_len = suffix_pad_masks.shape[1]

        # Build Joint Attention Mask (Suffix attends to Prefix + Suffix)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        combined_att_mask = torch.zeros(
            bsize, 1, suffix_len, prefix_len + suffix_len,
            device=device, dtype=suffix_embs.dtype
        )

        # Part A: Suffix attends to Prefix (full attention)
        combined_att_mask[:, :, :, :prefix_len] = 0.0

        # Part B: Suffix attends to Suffix (causal)
        suffix_causal_mask = torch.where(
            suffix_att_2d_masks[:, None, :, :], 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        combined_att_mask[:, :, :, prefix_len:] = suffix_causal_mask

        # Position IDs: Suffix positions continue after Prefix
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Run Model with static prefix cache
        outputs = self.model(
            inputs_embeds_dict={
                "vision_language": None,
                "action": suffix_embs,
            },
            attention_mask=combined_att_mask,
            position_ids=suffix_position_ids,
            past_key_values=static_prefix_cache,
            use_cache=False,
            adarms_cond_dict={
                "vision_language": None,
                "action": adarms_cond,
            } if adarms_cond is not None else None,
        )

        # Get action output and predict velocity
        suffix_out = outputs.hidden_states["action"]
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)

        return self.action_out_proj(suffix_out)

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