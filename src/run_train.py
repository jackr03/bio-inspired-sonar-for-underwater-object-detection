import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
import torch
from matplotlib import pyplot as plt
from torch import nn

from src.config import CONFIG
from src.engine import train_one_epoch, validate, benchmark
from src.types.dataset_type import DatasetType
from src.types.filterbank_type import FilterbankType
from src.types.model_type import ModelType
from src.utils.dataset_utils import get_dataset, get_split_dataloaders
from src.utils.hyperparameter_sweep_utils import run_hyperparameter_sweep
from src.utils.model_utils import get_model_components, load_model_hyperparameters
from src.utils.plotting_utils import plot_training_history

warnings.filterwarnings('ignore', category=UserWarning)

def train_and_benchmark(device, model_type: ModelType, dataset_type: DatasetType, filterbank_type: FilterbankType) -> dict:
    components = get_model_components(model_type, dataset_type, filterbank_type)

    # Load dataset and dataloaders
    dataset = get_dataset(model_type, dataset_type, filterbank_type)
    train_dataloader, val_dataloader, test_dataloader = get_split_dataloaders(dataset, dataset_type)

    # Load model parameters
    model_config = dataset_type.get_model_config(model_type)
    hyperparameters = load_model_hyperparameters(model_type, components['hyperparameters_path'])
    model_params = {**model_config, **hyperparameters['model_init']} # Combine into one for easier use

    print(f'Model Config: {model_config}')
    print(f'Hyperparameters used: {hyperparameters}')
    print()

    model = components['model_class'](**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr'])
    criterion = nn.CrossEntropyLoss(weight=dataset_type.class_weights.to(device))

    print(f'Training {model_type.name}...')
    best_macro_f1 = 0.0
    epochs_without_improvement = 0
    start_time = time.time()
    history = {
        'train_losses': [], 'train_accs': [], 'train_macro_f1s': [], 'train_weighted_f1s': [],
        'val_losses': [], 'val_accs': [], 'val_macro_f1s': [], 'val_weighted_f1s': [],
    }
    for epoch in range(CONFIG.epochs):
        print(f'[Epoch {epoch + 1}/{CONFIG.epochs}]')
        epoch_start = time.time()

        train_metrics = train_one_epoch(device, model, criterion, optimizer, train_dataloader)
        history['train_losses'].append(train_metrics['loss'])
        history['train_accs'].append(train_metrics['accuracy'])
        history['train_macro_f1s'].append(train_metrics['macro_f1'])
        history['train_weighted_f1s'].append(train_metrics['weighted_f1'])

        val_metrics = validate(device, model, criterion, val_dataloader)
        history['val_losses'].append(val_metrics['loss'])
        history['val_accs'].append(val_metrics['accuracy'])
        history['val_macro_f1s'].append(val_metrics['macro_f1'])
        history['val_weighted_f1s'].append(val_metrics['weighted_f1'])

        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            torch.save(model.state_dict(), components['model_path'])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement == CONFIG.patience:
            print(f'Stopping early at epoch {epoch + 1} (no improvement for {CONFIG.patience} epochs)')
            break

        epoch_duration = time.time() - epoch_start
        total_elapsed = time.time() - start_time

        print(f'[Train] Loss: {train_metrics["loss"]:.2f} | Acc: {train_metrics["accuracy"]:.2f}% | Macro-F1: {train_metrics["macro_f1"]:.4f} | Weighted-F1: {train_metrics['weighted_f1']:.4f}')
        print(f'[Val] Loss: {val_metrics["loss"]:.2f} | Acc: {val_metrics["accuracy"]:.2f}% | Macro-F1: {val_metrics["macro_f1"]:.4f} | Weighted-F1: {val_metrics['weighted_f1']:.4f}')
        print(f'Epoch Duration: {epoch_duration:.0f}s | Total Time Elapsed: {total_elapsed:.0f}s')
        print()

    total_time = time.time() - start_time
    print(f'Training completed in {total_time:.0f}s.')
    print(f'Best model had a macro-F1 of {best_macro_f1:.4f}.')

    # Benchmark
    print(f'Running benchmark on test set...')
    model.load_state_dict(torch.load(components['model_path'], map_location=device))
    test_metrics = benchmark(device, model, test_dataloader)
    print(f'[Benchmark Results]')
    print(f' Accuracy: {test_metrics["accuracy"]:.2f}%')
    print(f'  Macro    — F1: {test_metrics["macro_f1"]:.4f} | Precision: {test_metrics["macro_precision"]:.4f} | Recall: {test_metrics["macro_recall"]:.4f}')
    print(f'  Weighted — F1: {test_metrics["weighted_f1"]:.4f} | Precision: {test_metrics["weighted_precision"]:.4f} | Recall: {test_metrics["weighted_recall"]:.4f}')
    print(f'  MACs: {test_metrics["macs"]:,} | ACs: {test_metrics["acs"]:,}')

    return {
        'model': model_type.value,
        'dataset': dataset_type.value,
        'filterbank': filterbank_type.value,
        'device': str(device),
        'training_time': round(total_time, 1),
        'num_epochs': len(history['train_losses']),
        'model_config': model_config,
        'hyperparameters': hyperparameters,
        'best_val_macro_f1': best_macro_f1,
        'benchmark': test_metrics,
        'epoch_history': history,
    }


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
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
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
    results = train_and_benchmark(device, model_type, dataset_type, filterbank_type)

    # Save results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {results_path}')

    matplotlib.use('Agg')
    plot_training_history(**results['epoch_history'])
    plt.savefig(training_history_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
