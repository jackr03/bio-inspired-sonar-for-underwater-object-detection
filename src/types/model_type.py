from enum import Enum
from typing import Callable

from src.utils.preprocessing_utils import get_cnn_pipeline, get_snn_pipeline


class ModelType(str, Enum):
    CNN = 'cnn'
    SNN_DIRECT = 'snn_direct'
    SNN_DIRECT_LT = 'snn_direct_lt'
    SNN = 'snn'

    @property
    def pipeline(self) -> Callable:
        match self:
            case ModelType.CNN:
                return get_cnn_pipeline
            case ModelType.SNN_DIRECT | ModelType.SNN_DIRECT_LT | ModelType.SNN:
                return get_snn_pipeline
