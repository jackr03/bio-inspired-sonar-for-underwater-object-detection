from enum import Enum
from typing import Callable

from src.preprocessing import get_cnn_pipeline, get_snn_pipeline


class ModelType(str, Enum):
    CNN = 'cnn'
    SNN_DIRECT = 'snn_direct'
    SNN = 'snn'

    @property
    def pipeline(self) -> Callable:
        match self:
            case ModelType.CNN:
                return get_cnn_pipeline
            case ModelType.SNN_DIRECT | ModelType.SNN:
                return get_snn_pipeline
