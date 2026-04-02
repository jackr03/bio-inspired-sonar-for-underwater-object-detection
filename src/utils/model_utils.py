import json
from pathlib import Path

from src.config import CONFIG
from src.models.cnn import CNN
from src.models.snn import SNN
from src.models.snn_direct import SNNDirect
from src.types.dataset_type import DatasetType
from src.types.filterbank_type import FilterbankType
from src.types.model_type import ModelType

MODEL_CLASSES = {
    ModelType.CNN: CNN,
    ModelType.SNN_DIRECT: SNNDirect,
    ModelType.SNN: SNN,
}


def get_model_components(model_type: ModelType, dataset_type: DatasetType, filterbank_type: FilterbankType) -> dict:
    file_name = _format_file_name(model_type, dataset_type, filterbank_type)
    return {
        'model_class': MODEL_CLASSES[model_type],
        'model_path': CONFIG.project_root / 'models' / f'{file_name}.pth',
        'hyperparameters_path': CONFIG.project_root / 'hyperparameters' / f'{file_name}.json',
    }


def load_model_hyperparameters(model_type: ModelType, hyperparameters_path: Path) -> dict:
    with open(hyperparameters_path, 'r') as f:
        data = json.load(f)

    match model_type:
        case ModelType.CNN | ModelType.SNN:
            return data
        case ModelType.SNN_DIRECT:
            # Choose lowest timesteps from Pareto front
            return data[1]['params']


def _format_file_name(model_type: ModelType, dataset_type: DatasetType, filterbank_type: FilterbankType) -> str:
    return f'{model_type.value}-{dataset_type.value}-{filterbank_type.value}'
