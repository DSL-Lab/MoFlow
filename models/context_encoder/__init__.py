from .mtr_encoder import MTREncoder
from .eth_encoder import ETHEncoder

__all__ = {
    'MTREncoder': MTREncoder,
    'ETHEncoder': ETHEncoder,
}


def build_context_encoder(config, use_pre_norm):
    model = __all__[config.NAME](
        config=config,
        use_pre_norm=use_pre_norm
    )

    return model
