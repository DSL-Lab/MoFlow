from .mtr_decoder import MTRDecoder

__all__ = {
    'MTRDecoder': MTRDecoder,
}


def build_decoder(config, use_pre_norm, **kwargs):
    model = __all__[config.NAME](
        config=config,
        use_pre_norm=use_pre_norm,
        **kwargs
    )

    return model


