import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import optuna
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from src.config import CONFIG
from src.datasets.audio_dataset import AudioDataset
from src.engine import train_one_epoch_cnn, validate_cnn, get_split_dataloaders, benchmark_cnn
from src.models.cnn import CNN
from src.preprocessing import get_cnn_pipeline
from src.utils import plot_training_history, run_sweep

warnings.filterwarnings('ignore', category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train CNN model')
    parser.add_argument('--run-dir', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    hyperparameters_path = run_dir / f'hyperparameters.json'
    model_path = run_dir / f'model.pth'
    results_path = run_dir / 'results.json'

    # Device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    pipeline = get_cnn_pipeline(CONFIG.audio_pipeline, filterbank='mel')

    # --- Hyperparameter Sweep ---
    if CONFIG.hyperparameter_tuning.should_run:
        print('Starting hyperparameter sweep...')

        def objective(trial) -> float:
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

            dataset = AudioDataset(args.data_dir, pipeline)
            train_dataloader, val_dataloader, _ = get_split_dataloaders(dataset)

            model = CNN().to(device)
            optimiser = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            val_acc = 0.0
            for epoch in range(CONFIG.hyperparameter_tuning.epochs):
                train_one_epoch_cnn(device, model, criterion, optimiser, train_dl, leave=False)
                _, val_acc = validate_cnn(device, model, criterion, val_dl, leave=False)

                trial.report(val_acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return val_acc

        run_sweep(objective, hyperparameters_path)
    else:
        print('Skipping hyperparameter sweep.')

    # --- Training ---
    dataset = AudioDataset(args.data_dir, pipeline)
    train_dl, val_dl, test_dl = get_split_dataloaders(dataset)

    hyperparameters = json.load(open(hyperparameters_path, 'r'))
    print(f'Hyperparameters: {hyperparameters}')

    model = CNN().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr'])
    criterion = nn.CrossEntropyLoss()

    print(f'Training {model.NAME}...')
    best_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(args.num_epochs):
        print(f'[Epoch {epoch + 1}/{args.num_epochs}]')

        train_loss, train_acc = train_one_epoch_cnn(device, model, criterion, optimiser, train_dl)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = validate_cnn(device, model, criterion, val_dl)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)

        print(f'Train Loss: {train_loss:.2f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc:.2f}%')

    print(f'Best validation accuracy: {best_acc:.2f}%')

    # --- Benchmark ---
    print('Running benchmark on test set...')
    model.load_state_dict(torch.load(model_path))
    test_accuracy, macs = benchmark_cnn(device, model, test_dl)
    print(f'Test Accuracy: {test_accuracy:.2f}% | Total MACs: {macs}')

    # --- Save results ---
    results = {
        'model': model.NAME,
        'test_accuracy': test_accuracy,
        'macs': macs,
        'best_val_accuracy': best_acc,
        'num_epochs': args.num_epochs,
        'hyperparameters': hyperparameters,
        'device': str(device),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {results_path}')

    # --- Plot ---
    matplotlib.use('Agg')
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    plt.savefig(run_dir / 'plots' / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {run_dir / 'plots' / 'training_history.png'}')


if __name__ == '__main__':
    main()
