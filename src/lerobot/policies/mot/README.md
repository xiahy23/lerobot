# MoT (Mixture of Transformers) Policy

The MoT (Mixture of Transformers) policy is a highly configurable multi-transformer architecture designed to enable flexible attention patterns between different modalities and processing stages.

## Key Features

- **Configurable Architecture**: Define arbitrary numbers of transformer nodes (vision, state, action, language, etc.) through configuration alone
- **Flexible Attention Flows**: Specify which nodes can attend to which other nodes using flow configurations
- **Heterogeneous Hidden Sizes**: Support for different hidden dimensions across backbones (e.g., PaliGemma 2B with 2048 hidden_size + Gemma 300M expert with 1024 hidden_size) through unified head-space attention
- **Swappable Backbones**: Easily swap underlying transformer implementations (PaliGemma, Gemma, LLaMA, Bagel, Qwen, etc.)
- **Multiple Decoding Modes**: Support for flow matching (pi0, pi0.5), autoregressive (pi0FAST), and direct prediction
- **Preset Configurations**: Built-in presets for pi0, pi0.5, and pi0FAST architectures

## Heterogeneous Joint Attention Architecture

The MoT framework implements **explicit heterogeneous joint layer attention** to support architectures like pi0 where different backbones have different hidden sizes but share the same head structure:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HETEROGENEOUS JOINT LAYER ATTENTION                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VLM (hidden_size=2048)          Expert (hidden_size=1024)                  │
│  ┌─────────────────────┐         ┌─────────────────────┐                    │
│  │  Input LayerNorm    │         │  Input LayerNorm    │                    │
│  │  (+ AdaRMS cond)    │         │  (+ AdaRMS cond)    │                    │
│  └─────────┬───────────┘         └─────────┬───────────┘                    │
│            ↓                               ↓                                │
│  ┌─────────────────────┐         ┌─────────────────────┐                    │
│  │     QKV Proj        │         │     QKV Proj        │                    │
│  │  2048 → 8×256       │         │  1024 → 8×256       │                    │
│  └─────────┬───────────┘         └─────────┬───────────┘                    │
│            ↓                               ↓                                │
│            └───────────────┬───────────────┘                                │
│                            ↓                                                │
│            ┌───────────────────────────────┐                                │
│            │   CONCATENATE IN HEAD SPACE   │                                │
│            │  Q: [B, 8, seq_total, 256]    │                                │
│            │  K: [B, 8, seq_total, 256]    │                                │
│            │  V: [B, 8, seq_total, 256]    │                                │
│            └───────────────┬───────────────┘                                │
│                            ↓                                                │
│            ┌───────────────────────────────┐                                │
│            │         APPLY RoPE            │                                │
│            │    (shared rotary emb)        │                                │
│            └───────────────┬───────────────┘                                │
│                            ↓                                                │
│            ┌───────────────────────────────┐                                │
│            │     GLOBAL ATTENTION          │                                │
│            │  with MoT attention mask      │                                │
│            │  [B, 8, seq_total, 256]       │                                │
│            └───────────────┬───────────────┘                                │
│                            ↓                                                │
│            ┌───────────────────────────────┐                                │
│            │          SPLIT                │                                │
│            │  back to per-backbone slices  │                                │
│            └───────────────┬───────────────┘                                │
│            ┌───────────────┴───────────────┐                                │
│            ↓                               ↓                                │
│  ┌─────────────────────┐         ┌─────────────────────┐                    │
│  │     O Projection    │         │     O Projection    │                    │
│  │   8×256 → 2048      │         │   8×256 → 1024      │                    │
│  └─────────┬───────────┘         └─────────┬───────────┘                    │
│            ↓                               ↓                                │
│  ┌─────────────────────┐         ┌─────────────────────┐                    │
│  │   Residual + FFN    │         │   Residual + FFN    │                    │
│  │   (independent)     │         │   (independent)     │                    │
│  └─────────────────────┘         └─────────────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Requirements for Heterogeneous Joint Attention:**

- All backbones must have the **same `num_attention_heads`** (e.g., 8)
- All backbones must have the **same `head_dim`** (e.g., 256)
- All backbones must have the **same `num_hidden_layers`** (e.g., 18)
- **Hidden sizes can differ** (e.g., 2048 vs 1024) since QKV/O projections handle the conversion

## Quick Start

### Using Presets

```python
from lerobot.policies.mot import MoTConfig, MoTPolicy
from lerobot.policies.mot.configuration_mot import (
    make_pi0_config,
    make_pi05_config,
    make_pi0fast_config,
)

# Create a pi0-like configuration
config = make_pi0_config()
policy = MoTPolicy(config)

# Create a pi0.5-like configuration
config = make_pi05_config()
policy = MoTPolicy(config)

# Create a pi0FAST-like configuration
config = make_pi0fast_config()
policy = MoTPolicy(config)
```

### Custom Configuration

```python
from lerobot.policies.mot import (
    MoTConfig,
    MoTNodeConfig,
    MoTFlowConfig,
    MoTBackboneConfig,
    MoTPolicy,
)
from lerobot.policies.mot.configuration_mot import (
    NodeType,
    AttentionType,
    BackboneType,
    DecodingMode,
)

# Define nodes
nodes = [
    MoTNodeConfig(
        name="vlm",
        node_type=NodeType.VLM,
        backbone_name="paligemma",
        input_dim=2048,
        token_dim=2048,
        max_tokens=300,
        is_input=True,
        is_output=False,
    ),
    MoTNodeConfig(
        name="action_expert",
        node_type=NodeType.LM,
        backbone_name="gemma_expert",
        input_dim=1024,
        token_dim=1024,
        max_tokens=51,
        is_input=True,
        is_output=True,
        output_head="linear",
        output_dim=32,
    ),
]

# Define attention flows
flows = [
    # VLM self-attention
    MoTFlowConfig(source="vlm", target="vlm", attention_type=AttentionType.FULL),
    # Expert can attend to VLM
    MoTFlowConfig(source="vlm", target="action_expert", attention_type=AttentionType.FULL),
    # Expert self-attention (causal)
    MoTFlowConfig(source="action_expert", target="action_expert", attention_type=AttentionType.CAUSAL),
]

# Define backbones
backbones = [
    MoTBackboneConfig(
        name="paligemma",
        backbone_type=BackboneType.PALIGEMMA,
        variant="gemma_2b",
        hidden_size=2048,
        num_hidden_layers=18,
    ),
    MoTBackboneConfig(
        name="gemma_expert",
        backbone_type=BackboneType.GEMMA,
        variant="gemma_300m",
        hidden_size=1024,
        num_hidden_layers=18,
    ),
]

# Create config
config = MoTConfig(
    nodes=nodes,
    flows=flows,
    backbones=backbones,
    decoding_mode=DecodingMode.FLOW_MATCHING,
    chunk_size=50,
    n_action_steps=50,
)

# Create policy
policy = MoTPolicy(config)
```

### Using Custom N-Transformer Architecture

```python
from lerobot.policies.mot.configuration_mot import make_custom_mot_config

# Create a 3-transformer architecture with custom attention pattern
config = make_custom_mot_config(
    num_transformers=3,
    transformer_configs=[
        {"name": "vision", "backbone_type": "paligemma", "variant": "paligemma_2b"},
        {"name": "language", "backbone_type": "gemma", "variant": "gemma_300m"},
        {"name": "action", "backbone_type": "gemma", "variant": "gemma_300m"},
    ],
    attention_matrix=[
        ["full", "none", "none"],   # vision: self-attend only
        ["full", "full", "none"],   # language: attend to vision + self
        ["full", "full", "causal"], # action: attend to all, causal for self
    ],
)

policy = MoTPolicy(config)
```

## Configuration Reference

### MoTNodeConfig

Defines a single node in the MoT architecture.

| Parameter           | Type     | Default     | Description                                                     |
| ------------------- | -------- | ----------- | --------------------------------------------------------------- |
| `name`              | str      | required    | Unique identifier for the node                                  |
| `node_type`         | NodeType | CUSTOM      | Type of node (VISION, STATE, ACTION, LANGUAGE, VLM, LM, CUSTOM) |
| `backbone_name`     | str      | None        | Name of the backbone to use                                     |
| `input_dim`         | int      | 512         | Dimension of raw input features                                 |
| `token_dim`         | int      | 512         | Dimension after projection                                      |
| `max_tokens`        | int      | 1           | Maximum number of tokens produced                               |
| `position_encoding` | str      | "learnable" | Position encoding type                                          |
| `is_input`          | bool     | True        | Whether node receives inputs                                    |
| `is_output`         | bool     | False       | Whether node produces outputs                                   |
| `output_head`       | str      | "none"      | Type of output head                                             |
| `output_dim`        | int      | None        | Output dimension                                                |

### MoTFlowConfig

Defines attention flow between nodes.

| Parameter          | Type          | Default  | Description                       |
| ------------------ | ------------- | -------- | --------------------------------- |
| `source`           | str           | required | Node providing Key/Value          |
| `target`           | str           | required | Node providing Query              |
| `attention_type`   | AttentionType | FULL     | Attention pattern type            |
| `layer_range`      | tuple         | None     | Layer range for this flow         |
| `attention_scale`  | float         | 1.0      | Scale factor for attention        |
| `is_bidirectional` | bool          | False    | Create reverse flow automatically |

### MoTBackboneConfig

Defines a transformer backbone.

| Parameter             | Type         | Default    | Description                 |
| --------------------- | ------------ | ---------- | --------------------------- |
| `name`                | str          | required   | Unique identifier           |
| `backbone_type`       | BackboneType | GEMMA      | Type of backbone            |
| `variant`             | str          | "gemma_2b" | Specific variant            |
| `hidden_size`         | int          | 2048       | Hidden dimension            |
| `num_hidden_layers`   | int          | 18         | Number of layers            |
| `num_attention_heads` | int          | 8          | Number of attention heads   |
| `pretrained_path`     | str          | None       | Path for pretrained weights |
| `freeze`              | bool         | False      | Whether to freeze backbone  |

### AttentionType

| Value    | Description                       |
| -------- | --------------------------------- |
| `FULL`   | Full bidirectional attention      |
| `CAUSAL` | Causal (autoregressive) attention |
| `PREFIX` | Prefix-LM style attention         |
| `CROSS`  | Cross attention only              |
| `NONE`   | No attention (blocked)            |

### BackboneType

| Value       | Description                     |
| ----------- | ------------------------------- |
| `PALIGEMMA` | PaliGemma Vision-Language Model |
| `GEMMA`     | Gemma language model            |
| `LLAMA`     | LLaMA family models             |
| `BAGEL`     | Bagel model (placeholder)       |
| `QWEN`      | Qwen models                     |
| `CUSTOM`    | Custom transformer backbone     |

### DecodingMode

| Value            | Description                               |
| ---------------- | ----------------------------------------- |
| `FLOW_MATCHING`  | Flow matching denoising (pi0, pi0.5)      |
| `AUTOREGRESSIVE` | Autoregressive token generation (pi0FAST) |
| `DIRECT`         | Direct MLP prediction                     |

## Extending with New Backbones

To add support for a new transformer backbone (e.g., Bagel):

1. Add the backbone type to `BackboneType` enum in `configuration_mot.py`:

```python
class BackboneType(str, Enum):
    # ... existing types ...
    BAGEL = "bagel"
```

2. Implement the initialization method in `MoTBackboneWrapper`:

```python
def _init_bagel(self):
    from transformers import BagelForCausalLM

    cfg = self.backbone_config
    hf_config = CONFIG_MAPPING["bagel"](
        hidden_size=cfg.hidden_size,
        # ... other config params ...
    )

    self.model = BagelForCausalLM(config=hf_config)
    self.language_model = self.model.model
    self._apply_precision(cfg.dtype)
```

3. Add preset configurations if desired:

```python
BAGEL_CONFIG = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    # ... other params ...
}
```

## Architecture Overview

```
MoTPolicy
├── MoTModel
│   ├── MoTBackboneWrapper (per backbone)
│   │   └── Transformer (PaliGemma/Gemma/LLaMA/etc.)
│   ├── MoTInputProjection (per input node)
│   ├── MoTOutputHead (per output node)
│   ├── MoTJointLayerAttention
│   │   └── MoTAttentionMaskBuilder
│   └── Flow Matching Components (if applicable)
└── Processors
    ├── Preprocessor (tokenization, normalization, etc.)
    └── Postprocessor (unnormalization, etc.)
```

## Training

Train a MoT policy using the standard LeRobot training pipeline:

```bash
python lerobot/scripts/train.py \
    --policy.type=mot \
    --policy.nodes='[{name: vlm, ...}, {name: expert, ...}]' \
    --policy.flows='[{source: vlm, target: expert, ...}]' \
    --policy.backbones='[{name: paligemma, ...}, ...]' \
    --dataset.repo_id=lerobot/pusht
```

Or use a preset:

```bash
python lerobot/scripts/train.py \
    --policy.type=mot \
    --dataset.repo_id=lerobot/pusht
```

## Citation

If you use MoT in your research, please cite:

```bibtex
@misc{mot2025,
  title={MoT: Mixture of Transformers for Flexible Multi-Modal Robot Learning},
  year={2025},
  publisher={LeRobot},
}
```
