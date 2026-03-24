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
    train_dataloader, val_dataloader, test_dataloader = get_split_dataloaders(dataset)

    # Load hyperparameters
    hyperparameters = load_model_hyperparameters(model_type, components['hyperparameters_path'])
    print(f'Hyperparameters used: {hyperparameters}')
    print()

    model = components['model_class'](num_classes=dataset_type.num_classes, **hyperparameters['model_init']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr'])
    criterion = nn.CrossEntropyLoss()

    print(f'Training {model_type.name}...')
    best_acc = 0.0
    epochs_without_improvement = 0
    start_time = time.time()
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    for epoch in range(CONFIG.epochs):
        print(f'[Epoch {epoch + 1}/{CONFIG.epochs}]')
        epoch_start = time.time()

        train_loss, train_acc = components['train_fn'](device, model, criterion, optimizer, train_dataloader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = components['val_fn'](device, model, criterion, val_dataloader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), components['model_path'])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement == CONFIG.patience:
            print(f'Stopping early at epoch {epoch + 1} (no improvement for {CONFIG.patience} epochs)')
            break

        epoch_duration = time.time() - epoch_start
        total_elapsed = time.time() - start_time

        print(f'Train Loss: {train_loss:.2f} | Train Accuracy: {train_acc:.2f}% | Val Loss: {val_loss:.2f} | Val Accuracy: {val_acc:.2f}%')
        print(f'Epoch Duration: {epoch_duration:.0f}s | Total Time Elapsed: {total_elapsed:.0f}s')
        print()

    total_time = time.time() - start_time
    print(f'Training completed in {total_time:.0f}s.')
    print(f'Best model had an accuracy of {best_acc:.2f}%.')

    # Benchmark
    print(f'Running benchmark on test set...')
    model.load_state_dict(torch.load(components['model_path'], map_location=device))
    test_acc, macs, acs = components['benchmark_fn'](device, model, test_dataloader)
    print(f'Test Accuracy: {test_acc:.2f}% | Total MACs: {macs:,} | Total ACs: {acs:,}')

    return {
        'model': model_type.value,
        'dataset': dataset_type.value,
        'filterbank': filterbank_type.value,
        'device': str(device),
        'training_time': round(total_time, 1),
        'num_epochs': len(train_losses), # In case we finished early
        'hyperparameters': hyperparameters,
        'best_val_accuracy': best_acc,
        'benchmark_accuracy': test_acc,
        'macs': macs,
        'acs': acs,
        'epoch_history': {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Unified Training Entry Point')
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
