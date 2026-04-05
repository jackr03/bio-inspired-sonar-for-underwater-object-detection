from src.models.snn_direct import SNNDirect
from src.types.model_type import ModelType


class SNNDirectLT(SNNDirect):
    def __init__(self, **kwargs):
        super().__init__(learn_threshold=True, **kwargs)

    @property
    def name(self) -> ModelType:
        return ModelType.SNN_DIRECT_LT