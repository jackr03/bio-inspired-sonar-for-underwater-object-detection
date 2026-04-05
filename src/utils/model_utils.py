import json
from pathlib import Path

from src.config import CONFIG
from src.models.cnn import CNN
from src.models.snn import SNN
from src.models.snn_direct import SNNDirect
from src.models.snn_direct_lt import SNNDirectLT
from src.types.dataset_type import DatasetType
from src.types.filterbank_type import FilterbankType
from src.types.model_type import ModelType

MODEL_CLASSES = {
    ModelType.CNN: CNN,
    ModelType.SNN_DIRECT: SNNDirect,
    ModelType.SNN_DIRECT_LT: SNNDirectLT,
    ModelType.SNN: SNN,
}


def get_model_components(model_type: ModelType, dataset_type: DatasetType, filterbank_type: FilterbankType) -> dict:
    file_name = _format_file_name(model_type, dataset_type, filterbank_type)
    # SNN_DIRECT_LT shares hyperparameters with SNN_DIRECT (same architecture, only learn_threshold differs)
    hyperparameters_name = _format_file_name(
        ModelType.SNN_DIRECT if model_type == ModelType.SNN_DIRECT_LT else model_type,
        dataset_type,
        filterbank_type,
    )
    return {
        'model_class': MODEL_CLASSES[model_type],
        'model_path': CONFIG.project_root / 'models' / f'{file_name}.pth',
        'hyperparameters_path': CONFIG.project_root / 'hyperparameters' / f'{hyperparameters_name}.json',
    }


def load_model_hyperparameters(hyperparameters_path: Path) -> dict:
    with open(hyperparameters_path, 'r') as f:
        data = json.load(f)

    return data


def _format_file_name(model_type: ModelType, dataset_type: DatasetType, filterbank_type: FilterbankType) -> str:
    return f'{model_type.value}-{dataset_type.value}-{filterbank_type.value}'
