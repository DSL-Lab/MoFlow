from .mtr_encoder import MTREncoder
from .eth_encoder import ETHEncoder

# 注册表（registry）
__all__ = {
    'MTREncoder': MTREncoder,
    'ETHEncoder': ETHEncoder,
}


def build_context_encoder(config, use_pre_norm, device):
    # 从注册表中取出类名对应的类，然后实例化
    model = __all__[config.NAME](
        config=config,
        use_pre_norm=use_pre_norm,
        device=device
    ).to(device=device)

    return model