import json
from pathlib import Path

import torch
from matplotlib import pyplot as plt

from src.config import CONFIG
from src.engine import train_one_epoch_cnn, validate_cnn, benchmark_cnn, train_one_epoch_snn, validate_snn, \
    benchmark_snn
from src.models.cnn import CNN
from src.models.snn import SNN
from src.models.snn_direct import SNNDirect
from src.types.dataset_type import DatasetType
from src.types.filterbank_type import FilterbankType
from src.types.model_type import ModelType
from src.utils.dataset_utils import get_dataset, get_split_dataloaders
from src.utils.plotting_utils import plot_accuracy_comparison, plot_energy_comparison


def get_model_components(model_type: ModelType, dataset_type: DatasetType, filterbank_type: FilterbankType):
    file_name = _format_file_name(model_type, dataset_type, filterbank_type)
    components = {
        'model_path': CONFIG.project_root / 'models'/ f'{file_name}.pth',
        'hyperparameters_path': CONFIG.project_root / f'hyperparameters' / f'{file_name}.json'
    }

    match model_type:
        case ModelType.CNN:
            model_specific = {
                'model_class': CNN,
                'train_fn': train_one_epoch_cnn,
                'val_fn': validate_cnn,
                'benchmark_fn': benchmark_cnn,
            }
        case ModelType.SNN_DIRECT:
            model_specific = {
                'model_class': SNNDirect,
                'train_fn': train_one_epoch_snn,
                'val_fn': validate_snn,
                'benchmark_fn': benchmark_snn,
            }
        case ModelType.SNN:
            model_specific = {
                'model_class': SNN,
                'train_fn': train_one_epoch_snn,
                'val_fn': validate_snn,
                'benchmark_fn': benchmark_snn,
            }

    components.update(model_specific)
    return components


def load_model_hyperparameters(model_type: ModelType, hyperparameters_path: Path) -> dict:
    MODEL_INIT_KEYS = {'beta_init', 'slope', 'timesteps'}

    with open(hyperparameters_path, 'r') as f:
        data = json.load(f)

    match model_type:
        case ModelType.CNN | ModelType.SNN:
            flat_params = data
        case ModelType.SNN_DIRECT:
            # Choose lowest timesteps
            flat_params = data[1]['params']

    model_init = {k: v for k, v in flat_params.items() if k in MODEL_INIT_KEYS}

    return {
        'lr': flat_params['lr'],
        'model_init': model_init
    }


def compare_models(device, model1_type: ModelType, model2_type: ModelType, dataset_type: DatasetType, filterbank1_type: FilterbankType, filterbank2_type: FilterbankType):
    components1 = get_model_components(model1_type, dataset_type, filterbank1_type)
    components2 = get_model_components(model2_type, dataset_type, filterbank2_type)

    dataset1 = get_dataset(model1_type, dataset_type, filterbank1_type)
    dataset2 = get_dataset(model2_type, dataset_type, filterbank2_type)

    # Use the same seed for split so that models are tested on exact same things
    _, _, dataloader1 = get_split_dataloaders(dataset1)
    _, _, dataloader2 = get_split_dataloaders(dataset2)

    _, sample_labels1 = next(iter(dataloader1))
    _, sample_labels2 = next(iter(dataloader2))
    assert torch.equal(sample_labels1, sample_labels2) # Should fail here immediately, otherwise not a fair test

    # Load models
    hyperparameters1 = load_model_hyperparameters(model1_type, components1['hyperparameters_path'])
    model1 = components1['model_class'](**dataset_type.get_model_config(model1_type), **hyperparameters1['model_init']).to(device)
    model1.load_state_dict(torch.load(components1['model_path'], map_location=device))

    hyperparameters2 = load_model_hyperparameters(model2_type, components2['hyperparameters_path'])
    model2 = components2['model_class'](**dataset_type.get_model_config(model2_type), **hyperparameters2['model_init']).to(device)
    model2.load_state_dict(torch.load(components2['model_path'], map_location=device))

    # Benchmark
    model1_acc, model1_macs, model1_acs = components1['benchmark_fn'](device, model1, dataloader1)
    model2_acc, model2_macs, model2_acs = components2['benchmark_fn'](device, model2, dataloader2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_accuracy_comparison(ax1, model1_type.value, model1_acc, model2_type.value, model2_acc)
    plot_energy_comparison(ax2, model1_type.value, model1_macs, model1_acs, model2_type.value, model2_macs, model2_acs)
    plt.tight_layout()
    plt.show()


def _format_file_name(model_type: ModelType, dataset_type: DatasetType, filterbank_type: FilterbankType) -> str:
    return f'{model_type.value}-{dataset_type.value}-{filterbank_type.value}'