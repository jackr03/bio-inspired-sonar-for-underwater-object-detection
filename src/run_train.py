import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import torch
from matplotlib import pyplot as plt

from src.config import CONFIG
from src.engine import train, train_kfold, load_and_benchmark
from src.types.dataset_type import DatasetType
from src.types.filterbank_type import FilterbankType
from src.types.model_type import ModelType
from src.utils.hyperparameter_sweep_utils import run_hyperparameter_sweep
from src.utils.plotting_utils import plot_training_history

warnings.filterwarnings('ignore', category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training Entry Point')
    parser.add_argument('--model', type=ModelType, choices=list(ModelType), required=True)
    parser.add_argument('--dataset', type=DatasetType, choices=list(DatasetType), required=True)
    parser.add_argument('--filterbank', type=FilterbankType, choices=list(FilterbankType), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / 'results.json'
    training_history_path = run_dir / 'training_history.png'

    # Device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    )

    # Load
    model_type = args.model
    dataset_type = args.dataset
    filterbank_type = args.filterbank

    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # Hyperparameter Sweep
    if CONFIG.hyperparameter_tuning.should_run:
        run_hyperparameter_sweep(device, model_type, dataset_type, filterbank_type)
    else:
        print('Skipping hyperparameter sweep.')

    # Train and benchmark
    if dataset_type == DatasetType.SHIPSEAR:
        results = train_kfold(device, model_type, dataset_type, filterbank_type)
    else:
        train_result = train(device, model_type, dataset_type, filterbank_type)
        benchmark_result = load_and_benchmark(device, model_type, dataset_type, filterbank_type)
        results = {
            'model': model_type.value,
            'dataset': dataset_type.value,
            'filterbank': filterbank_type.value,
            'device': str(device),
            **train_result,
            'benchmark': benchmark_result['benchmark'],
        }

    # Save results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {results_path}')

    matplotlib.use('Agg')
    if dataset_type == DatasetType.SHIPSEAR:
        for i, fold in enumerate(results['folds']):
            plot_training_history(**fold['epoch_history'])
            plt.savefig(
                run_dir / f'training_history_fold_{i + 1}.png',
                dpi=150,
                bbox_inches='tight',
            )
            plt.close()
    else:
        plot_training_history(**results['epoch_history'])
        plt.savefig(training_history_path, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()
