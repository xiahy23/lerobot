# MoT (Multistream-VLA) 开发者扩展指南

## 1. 架构设计

MoT 架构的设计核心是 **“机制 (Mechanism) 与策略 (Policy) 分离”**。

- **机制层 (The Core):** 位于 `lerobot/policies/mot/`。
- 负责 **形式 (Form)**：多流同步执行、Joint Attention 计算、KV Cache 管理。
- **不关心** 内容：它不知道什么是“图像”，什么是“动作”，只认 Tensor 和 Embeddings。
- **原则**：除非你要修改 Transformer 的底层执行逻辑（如修改 Attention 算子），否则 **不要修改这一层**。

- **策略层 (The Implementation):** 位于 `lerobot/policies/mot_<model_name>/` (如 `mot_pi0`)。
- 负责 **语义 (Semantics)**：数据预处理、Embedding 生成、Mask 构建、Loss 计算。
- **原则**：所有的业务逻辑、复杂的模态对齐、Tokenizer 调用都写在这里。

---

## 2. 快速扩展步骤：添加一个新的 VLA 模型

假设我们要实现一个名为 **`MoT-X`** 的新模型。

### 第一步：创建目录结构

在 `lerobot/policies/` 下创建一个新文件夹 `mot_x`：

```text
lerobot/policies/mot_x/
├── __init__.py
├── configuration_mot_x.py  # 定义流的配置
├── modeling_mot_x.py       # 定义具体的策略逻辑
└── processor_mot_x.py      # 定义数据预处理流程

```

### 第二步：定义配置 (`configuration_mot_x.py`)

你需要继承 `MoTConfig`，并在 `__post_init__` 中定义你的流（Streams）。

**关键点**：

1. 如果使用 Joint Attention，确保所有流的 `num_layers` 一致。
2. 利用 `MoTStreamConfig` 的 `backbone_type="hf"` 来自动加载 HuggingFace 模型。

```python
from dataclasses import dataclass
from lerobot.policies.mot.configuration_mot import MoTConfig, MoTStreamConfig, StreamRole, AttentionType
from lerobot.configs.policies import PreTrainedConfig

@PreTrainedConfig.register_subclass("mot_x")
@dataclass
class MoTXConfig(MoTConfig):
    # 定义你的模型特有参数
    vision_backbone: str = "meta-llama/Llama-3.2-11B-Vision"
    action_backbone: str = "meta-llama/Llama-3.2-1B"

    def __post_init__(self):
        # 1. 定义 Vision Stream
        vision_stream = MoTStreamConfig(
            name="vision",
            role=StreamRole.VISION_LANGUAGE,
            backbone_type="hf",
            model_path=self.vision_backbone,
            # 指定 HF 模型中 layer 的路径 (通过查看 model.modules() 确定)
            layers_attr="model.layers",
            norm_attr="model.norm",
            use_gradient_checkpointing=True
        )

        # 2. 定义 Action Stream
        action_stream = MoTStreamConfig(
            name="action",
            role=StreamRole.ACTION,
            backbone_type="hf",
            model_path=self.action_backbone,
            layers_attr="model.layers",
            norm_attr="model.norm"
        )

        # 3. 注册流
        self.streams = [vision_stream, action_stream]

        # 4. 设置 Joint Attention
        self.attn_implementation = AttentionType.JOINT

        super().__post_init__()

```

### 第三步：实现策略 (`modeling_mot_x.py`)

你需要继承 `MoTPolicy`。建议参考 `mot_pi0`，创建一个内部 Wrapper 来处理具体的 Tokenizer 和 Embedding 逻辑。

**必须实现的方法**：

1. **`get_input_embeddings(batch)`**:

- 输入：Raw Batch (images, text, actions)。
- 输出：`Dict[str, Tensor]`，对应配置中的流名字（如 `{"vision": ..., "action": ...}`）。
- 职责：调用 Tokenizer，Projector，Vision Encoder。

2. **`_make_attention_mask(batch)`**:

- 输出：Joint Attention Mask (4D Tensor)。
- 职责：定义流之间的可见性（例如：Action 能看 Vision，Vision 不能看 Action）。

3. **`forward(batch)`**:

- 职责：计算 Loss。
- 流程：Embed -> Mask -> `self.model(...)` -> Loss Function。

4. **`predict_action_chunk(batch)`**:

- 职责：实现推理循环（如 Flow Matching 或 Autoregressive）。
- **技巧**：利用 MoT 的 `past_key_values` 实现 Prefix Caching。

```python
from lerobot.policies.mot import MoTPolicy
from lerobot.policies.mot.backbones import HuggingFaceBackboneWrapper

class MoTXPolicy(MoTPolicy):
    config_class = MoTXConfig
    name = "mot_x"

    def __init__(self, config: MoTXConfig, **kwargs):
        super().__init__(config, **kwargs)

        # 1. 初始化你的具体组件 (Projectors, Wrappers)
        self.vision_encoder = ...
        self.action_proj = ...

        # 2. 初始化 MoT Core Model
        # 这里将具体的 HF 模型层注册给 MoT Core
        self._init_mot_model()

    def _init_mot_model(self):
        # 创建 Adapter，连接 HF 模型和 MoT Core
        vision_adapter = HuggingFaceBackboneWrapper(
            model=self.vision_encoder,
            layers_attr="model.layers",
            norm_attr="model.norm"
        )
        # ... action_adapter ...

        # 注入 Core
        self._init_model({
            "vision": vision_adapter,
            "action": action_adapter
        })

    def get_input_embeddings(self, batch):
        # 实现你的 Embedding 逻辑
        return {"vision": ..., "action": ...}

    def _make_attention_mask(self, batch):
        # 实现你的 Mask 逻辑
        pass

    def forward(self, batch):
        embeddings = self.get_input_embeddings(batch)
        mask = self._make_attention_mask(batch)

        # 调用 MoT Core
        outputs = self.model(
            inputs_embeds_dict=embeddings,
            attention_mask=mask
        )

        # 计算 Loss
        return loss, {}

```

### 第四步：注册模型

为了让 LeRobot 识别你的新 Policy，需要修改以下文件：

1. **`lerobot/policies/__init__.py`**:

```python
from .mot_x.configuration_mot_x import MoTXConfig

```

2. **`lerobot/policies/factory.py`**:

- 在 `get_policy_class` 中添加：

```python
elif name == "mot_x":
    from lerobot.policies.mot_x.modeling_mot_x import MoTXPolicy
    return MoTXPolicy

```

- 在 `make_policy_config` 中添加：

```python
elif policy_type == "mot_x":
    return MoTXConfig(**kwargs)

```

- 在 `make_pre_post_processors` 中添加对应的处理器逻辑。

---

## 3. 进阶：如何适配“非主流”模型？

如果你的模型 **不是 HuggingFace 标准格式**（例如你自己手写的 PyTorch Module，或者来自 JAX 转换的模型），MoT 依然支持。

### 使用 `GenericPyTorchWrapper`

在 `configuration_mot_x.py` 中设置 `backbone_type="custom"`，然后在 Policy 初始化时手动构建 Wrapper：

```python
from lerobot.policies.mot.backbones.custom import GenericPyTorchWrapper

# 在 modeling_mot_x.py 中
def _init_mot_model(self):
    my_custom_module = MyExperimentalTransformer()

    adapter = GenericPyTorchWrapper(
        layers=my_custom_module.blocks,  # 必须是 nn.ModuleList
        norm=my_custom_module.ln_f,      # 必须是 nn.Module
        hidden_size=512,
        head_dim=64,
        num_attention_heads=8
    )

    self._init_model({"custom_stream": adapter})

```

---

## 4. 常见问题 (FAQ)

**Q: Joint Attention 报错 "Dimension mismatch"？**

- **A:** 检查所有流的 `hidden_size` 是否一致？如果不一致，你需要在 Policy 层面的 Projector 里先把它们投影到相同的维度，然后再传给 `MoTModel`。MoT Core 假设输入 Embedding 的 `hidden_size` 等于 Backbone 的 `hidden_size`。如果 Backbone 宽度不同，目前的 Joint Attention 实现需要修改 `compute_joint_layer` 中的拼接逻辑（当前实现支持不同宽度的拼接，但要求 Attention 算子能处理。通常建议通过 Linear Layer 投影对齐维度）。
- **A (更正):** MoT 的 `compute_joint_layer` **支持** 不同宽度的流进行拼接，只要 Transformer 层的 `head_dim` 相同即可。请检查 `head_dim` 是否对齐。

**Q: 推理时 KV Cache 怎么用？**

- **A:** \* 如果你是 **Autoregressive (Next token prediction)**: 像标准 LLM 一样，每一步把 `past_key_values` 传回去，并只传入最新的 token embedding。
- 如果你是 **Flow Matching (Denoising)**: 采用 `Static Prefix Caching`。先跑一次 Prefix 获取 Cache，然后在去噪循环中反复使用这个 Cache，但不要更新它（设置 `use_cache=False` 并手动传入 Cache）。参考 `mot_pi0` 的实现。

**Q: 我想用 Cross-Attention 而不是 Joint-Attention？**

- **A:** 修改 Config 中的 `attn_implementation = AttentionType.CROSS`。目前 MoT Core 主要实现了 `JOINT` 和 `INDEPENDENT`。如果需要 Cross-Attention，你需要在 `MoTModel._forward_cross_attention` 中实现具体的逻辑（例如：Stream A 查询 Stream B 的 KV）。

---

## 5. 调试建议

1. **先跑通 INDEPENDENT 模式**：先把 `attn_implementation` 设为 `INDEPENDENT`，把 `attention_mask` 设为 `None`。确保每个流单独都能跑通，Loss 能下降。
2. **检查 Mask 形状**：Joint Attention 最容易出错的地方是 Mask 的形状。在 `modeling_mot_x.py` 里打印 `attention_mask.shape`，确保它是 `(Batch, 1, Total_Len, Total_Len)`。
3. **使用 `custom.py` 的 SimpleTransformerBlock**：如果你怀疑是 HF 模型适配的问题，先用 `custom.py` 里的简单 Transformer 替换掉，排除干扰。
