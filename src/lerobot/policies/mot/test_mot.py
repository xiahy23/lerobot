import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.mot.configuration_mot import make_pi0_config
from lerobot.policies.mot.modeling_mot import MoTPolicy
from lerobot.utils.constants import (OBS_LANGUAGE_ATTENTION_MASK,
                                     OBS_LANGUAGE_TOKENS)


def verify_pi0_reproduction():
    # 1. 创建配置
    config = make_pi0_config()
    # 强制把维度改小以便在 CPU 上快速测试，或者保留默认测试显存
    # config.backbones[0].hidden_size = 2048 (PaliGemma)
    # config.backbones[1].hidden_size = 1024 (Gemma Expert)

    config.input_features["observation.images.camera_0"] = PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 224, 224)
    )
    # =================================

    print("Initializing MoT-pi0 Policy...")
    policy = MoTPolicy(config)
    policy.eval()
    # 2. 构造 Dummy Data
    batch_size = 2
    device = policy.device

    # PaliGemma Image Input (SigLIP style normalized)
    images = torch.randn(batch_size, 3, 224, 224, device=device)

    # Language Input
    lang_tokens = torch.randint(0, 1000, (batch_size, 10), device=device)
    lang_mask = torch.ones(batch_size, 10, dtype=torch.bool, device=device)

    # State Input
    state = torch.randn(batch_size, 32, device=device) # max_state_dim

    # Action Input (Chunk)
    actions = torch.randn(batch_size, 50, 32, device=device) # chunk_size=50, max_action_dim=32

    batch = {
        "observation.images.camera_0": images,
        OBS_LANGUAGE_TOKENS: lang_tokens,
        OBS_LANGUAGE_ATTENTION_MASK: lang_mask,
        "observation.state": state,
        "action": actions
    }

    # 3. 运行 Forward (Training Loss)
    print("Running Forward Pass (Flow Matching Loss)...")
    loss, _ = policy(batch)
    print(f"✅ Loss computed: {loss.item()}")

    # 4. 运行 Inference (Action Sampling)
    print("Running Inference (Action Generation)...")
    # 减少步数以加快测试
    with torch.no_grad():
        generated_actions = policy.select_action(batch)
    print(f"✅ Action generated. Shape: {generated_actions.shape}")

if __name__ == "__main__":
    verify_pi0_reproduction()