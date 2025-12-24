# models/registry.py

MODEL_REGISTRY = {}


def register_model(name: str):
    """
    Usage:
        @register_model("moflow")
        class MoFlowModel(nn.Module):
            ...
    """
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model {name} already registered!")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(cfg, logger):
    model_name = cfg.model.name
    print("model name = {}".format(model_name))
    
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    logger.info(f"Building model: {model_name}")
    return MODEL_REGISTRY[model_name](cfg, logger)
