import shutil
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from tqdm.auto import tqdm

from src.config import CONFIG
from src.types.dataset_type import DatasetType
from src.types.filterbank_type import FilterbankType
from src.types.model_type import ModelType
from src.utils.dataset_utils import (
    get_dataset,
    get_split_dataloaders,
    get_kfold_dataloaders,
)
from src.utils.model_utils import get_model_components, load_model_hyperparameters
from src.utils.plotting_utils import plot_accuracy_comparison, plot_accuracy_comparison_grouped, plot_energy_comparison, plot_energy_comparison_grouped
from src.utils.snn_ac_monitor import SNNACMonitor


def train(
    device,
    model_type: ModelType,
    dataset_type: DatasetType,
    filterbank_type: FilterbankType,
    fold: int = None,
    dataloaders: tuple[DataLoader, DataLoader, DataLoader] = None,
) -> dict:
    components = get_model_components(model_type, dataset_type, filterbank_type)
    if fold is not None:
        standard_path = components['model_path']
        tmp_dir = standard_path.parent / f'tmp-{standard_path.stem}'
        tmp_dir.mkdir(exist_ok=True)
        components['model_path'] = tmp_dir / f'fold_{fold}.pth'

    if dataloaders is None:
        dataset = get_dataset(model_type, dataset_type, filterbank_type)
        train_dataloader, val_dataloader, _ = get_split_dataloaders(dataset)
    else:
        train_dataloader, val_dataloader, _ = dataloaders

    # Load model parameters
    model_config = dataset_type.get_model_config(model_type)
    hyperparameters = load_model_hyperparameters(components['hyperparameters_path'])

    # Combine into one for easier use
    model_params = {
        **model_config,
        **hyperparameters['model_init'],
    }

    print(f'Model Config: {model_config}')
    print(f'Hyperparameters used: {hyperparameters}')

    model = components['model_class'](**model_params).to(device)
    criterion = nn.CrossEntropyLoss(weight=dataset_type.class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters['lr'],
        weight_decay=hyperparameters['weight_decay'],
    )
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=5)

    print(f'Training {model_type.name}...')
    best_macro_f1 = 0.0
    epochs_without_improvement = 0
    history = {
        'train_losses': [],
        'train_accs': [],
        'train_macro_f1s': [],
        'train_weighted_f1s': [],
        'val_losses': [],
        'val_accs': [],
        'val_macro_f1s': [],
        'val_weighted_f1s': [],
    }
    start_time = time.time()
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

        scheduler.step(val_metrics['macro_f1'])

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

        print(
            f'[Train] Loss: {train_metrics["loss"]:.2f} | Acc: {train_metrics["accuracy"]:.2f}% | Macro-F1: {train_metrics["macro_f1"]:.4f} | Weighted-F1: {train_metrics["weighted_f1"]:.4f}'
        )
        print(
            f'[Val] Loss: {val_metrics["loss"]:.2f} | Acc: {val_metrics["accuracy"]:.2f}% | Macro-F1: {val_metrics["macro_f1"]:.4f} | Weighted-F1: {val_metrics["weighted_f1"]:.4f}'
        )
        print(f'Epoch Duration: {epoch_duration:.0f}s | Total Time Elapsed: {total_elapsed:.0f}s')
        print()

    total_time = time.time() - start_time
    print(f'Training completed in {total_time:.0f}s.')
    print(f'Best model had a macro-F1 of {best_macro_f1:.4f}.')

    return {
        'training_time': round(total_time, 1),
        'num_epochs': len(history['train_losses']),
        'model_config': model_config,
        'hyperparameters': hyperparameters,
        'best_val_macro_f1': best_macro_f1,
        'epoch_history': history,
    }


def train_kfold(
    device,
    model_type: ModelType,
    dataset_type: DatasetType,
    filterbank_type: FilterbankType,
) -> dict:
    dataset = get_dataset(model_type, dataset_type, filterbank_type)
    folds = get_kfold_dataloaders(dataset, dataset_type)

    components = get_model_components(model_type, dataset_type, filterbank_type)

    standard_path = components['model_path']
    tmp_dir = standard_path.parent / f'tmp-{standard_path.stem}'

    fold_results = []
    for i, dataloaders in enumerate(folds):
        print(f'\n{"=" * 60}')
        print(f'Fold {i + 1}/{len(folds)}')
        print(f'{"=" * 60}\n')
        train_result = train(device, model_type, dataset_type, filterbank_type, fold=i + 1, dataloaders=dataloaders)

        # Load best model from this fold and benchmark on fold's test set
        _, _, test_dataloader = dataloaders
        model_config = dataset_type.get_model_config(model_type)
        hyperparameters = train_result['hyperparameters']
        model = components['model_class'](**{**model_config, **hyperparameters['model_init']}).to(device)
        fold_path = tmp_dir / f'fold_{i + 1}.pth'
        model.load_state_dict(torch.load(fold_path, map_location=device, weights_only=True))

        print('Running benchmark on test set...')
        test_metrics = benchmark(device, model, test_dataloader)
        print('[Benchmark Results]')
        print(f' Accuracy: {test_metrics["accuracy"]:.2f}%')
        print(
            f'  Macro    — F1: {test_metrics["macro_f1"]:.4f} | Precision: {test_metrics["macro_precision"]:.4f} | Recall: {test_metrics["macro_recall"]:.4f}'
        )
        print(
            f'  Weighted — F1: {test_metrics["weighted_f1"]:.4f} | Precision: {test_metrics["weighted_precision"]:.4f} | Recall: {test_metrics["weighted_recall"]:.4f}'
        )
        print(f'  MACs: {test_metrics["macs"]:,} | ACs: {test_metrics["acs"]:,}')

        fold_results.append({**train_result, 'benchmark': test_metrics})

    # Save the best fold's model (by val macro F1) to the standard path, then clean up tmp dir
    best_fold_idx = max(range(len(fold_results)), key=lambda i: fold_results[i]['best_val_macro_f1'])
    shutil.copy(tmp_dir / f'fold_{best_fold_idx + 1}.pth', standard_path)
    shutil.rmtree(tmp_dir)
    print(
        f'Best model: fold {best_fold_idx + 1} (val macro F1: {fold_results[best_fold_idx]["best_val_macro_f1"]:.4f}) → saved to {standard_path}'
    )

    # Aggregate benchmark metrics across folds
    metric_keys = [
        'accuracy',
        'macro_f1',
        'macro_precision',
        'macro_recall',
        'weighted_f1',
        'weighted_precision',
        'weighted_recall',
    ]
    aggregate = {}
    for key in metric_keys:
        values = [result['benchmark'][key] for result in fold_results]
        aggregate[key] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}

    print(f'\n{"=" * 60}')
    print(f'K-Fold Results (mean ± std across {len(folds)} folds)')
    print(f'{"=" * 60}')
    print(f' Accuracy: {aggregate["accuracy"]["mean"]:.2f}% ± {aggregate["accuracy"]["std"]:.2f}%')
    print(f'  Macro    — F1: {aggregate["macro_f1"]["mean"]:.4f} ± {aggregate["macro_f1"]["std"]:.4f}')
    print(f'  Weighted — F1: {aggregate["weighted_f1"]["mean"]:.4f} ± {aggregate["weighted_f1"]["std"]:.4f}')

    return {
        'model': model_type.value,
        'dataset': dataset_type.value,
        'filterbank': filterbank_type.value,
        'device': str(device),
        'n_folds': len(folds),
        'aggregate': aggregate,
        'folds': fold_results,
    }


def load_and_benchmark(
    device,
    model_type: ModelType,
    dataset_type: DatasetType,
    filterbank_type: FilterbankType
) -> dict:
    components = get_model_components(model_type, dataset_type, filterbank_type)
    dataset = get_dataset(model_type, dataset_type, filterbank_type)
    model_config = dataset_type.get_model_config(model_type)

    model = components['model_class'](**model_config).to(device)
    model.load_state_dict(torch.load(components['model_path'], map_location=device))

    if dataset_type == DatasetType.SHIPSEAR:
        return _load_and_benchmark_kfold(device, model, model_type, dataset_type, filterbank_type, dataset)
    else:
        return _load_and_benchmark_single(device, model, model_type, dataset_type, filterbank_type, dataset)


def _load_and_benchmark_single(
    device,
    model,
    model_type: ModelType,
    dataset_type: DatasetType,
    filterbank_type: FilterbankType,
    dataset: Dataset,
) -> dict:
    _, _, test_dataloader = get_split_dataloaders(dataset)

    print('Running benchmark on test set...')
    test_metrics = benchmark(device, model, test_dataloader)
    _print_benchmark_metrics(test_metrics)

    return {
        'model': model_type.value,
        'dataset': dataset_type.value,
        'filterbank': filterbank_type.value,
        'device': str(device),
        'benchmark': test_metrics,
    }


def _load_and_benchmark_kfold(
    device,
    model,
    model_type: ModelType,
    dataset_type: DatasetType,
    filterbank_type: FilterbankType,
    dataset: Dataset,
) -> dict:
    folds = get_kfold_dataloaders(dataset, dataset_type)
    fold_metrics = []

    for i, (_, _, test_dataloader) in enumerate(folds):
        print(f'Benchmarking fold {i + 1}/{len(folds)}...')
        test_metrics = benchmark(device, model, test_dataloader)
        _print_benchmark_metrics(test_metrics)
        fold_metrics.append(test_metrics)

    metric_keys = [
        'accuracy',
        'macro_f1',
        'macro_precision',
        'macro_recall',
        'weighted_f1',
        'weighted_precision',
        'weighted_recall',
    ]
    aggregate = {}
    for key in metric_keys:
        values = [m[key] for m in fold_metrics]
        aggregate[key] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}

    # MACs/ACs are deterministic — same across folds, take from first
    aggregate['macs'] = fold_metrics[0]['macs']
    aggregate['acs'] = fold_metrics[0]['acs']

    print(f'\n{"=" * 60}')
    print(f'K-Fold Benchmark Results (mean ± std across {len(folds)} folds)')
    print(f'{"=" * 60}')
    print(f' Accuracy: {aggregate["accuracy"]["mean"]:.2f}% ± {aggregate["accuracy"]["std"]:.2f}%')
    print(f'  Macro    — F1: {aggregate["macro_f1"]["mean"]:.4f} ± {aggregate["macro_f1"]["std"]:.4f}')
    print(f'  Weighted — F1: {aggregate["weighted_f1"]["mean"]:.4f} ± {aggregate["weighted_f1"]["std"]:.4f}')
    print(f'  MACs: {aggregate["macs"]:,} | ACs: {aggregate["acs"]:,}')

    return {
        'model': model_type.value,
        'dataset': dataset_type.value,
        'filterbank': filterbank_type.value,
        'device': str(device),
        'n_folds': len(folds),
        'aggregate': aggregate,
        'folds': fold_metrics,
    }


def _print_benchmark_metrics(metrics: dict) -> None:
    print(f' Accuracy: {metrics["accuracy"]:.2f}%')
    print(
        f'  Macro    — F1: {metrics["macro_f1"]:.4f} | Precision: {metrics["macro_precision"]:.4f} | Recall: {metrics["macro_recall"]:.4f}'
    )
    print(
        f'  Weighted — F1: {metrics["weighted_f1"]:.4f} | Precision: {metrics["weighted_precision"]:.4f} | Recall: {metrics["weighted_recall"]:.4f}'
    )
    print(f'  MACs: {metrics["macs"]:,} | ACs: {metrics["acs"]:,}')


def train_one_epoch(device, model, criterion, optimizer, train_dataloader: DataLoader) -> dict:
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    if CONFIG.show_progress:
        train_dataloader = tqdm(train_dataloader, desc='Training', unit='batches')

    for inputs, labels in train_dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = _forward(model, inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.append(predicted.cpu())
        all_labels.append(labels.cpu())

        optimizer.step()

    return _compute_metrics(torch.cat(all_labels), torch.cat(all_preds), total_loss, len(train_dataloader))


def validate(device, model, criterion, val_dataloader: DataLoader) -> dict:
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    if CONFIG.show_progress:
        val_dataloader = tqdm(val_dataloader, desc='Validating', unit='batches')

    with torch.inference_mode():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = _forward(model, inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

    return _compute_metrics(torch.cat(all_labels), torch.cat(all_preds), total_loss, len(val_dataloader))


def benchmark(device, model, test_dataloader: DataLoader) -> dict:
    model.eval()

    macs = 0
    snn_ac_monitor = None
    if model.name == ModelType.CNN:
        sample_input, _ = next(iter(test_dataloader))
        input_size = (1, *sample_input.shape[1:])
        model_stats = summary(model, input_size, device=device, verbose=0)
        macs = model_stats.total_mult_adds
    else:
        snn_ac_monitor = SNNACMonitor(model)
        snn_ac_monitor.attach()

    all_preds = []
    all_labels = []
    total = 0

    if CONFIG.show_progress:
        test_dataloader = tqdm(test_dataloader, desc='Benchmarking', unit='batches')

    with torch.inference_mode():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = _forward(model, inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
            total += labels.size(0)

    if model.name == ModelType.CNN:
        acs = 0
    else:
        snn_ac_monitor.remove()
        macs = (
            _calculate_conv2d_macs(model, next(iter(test_dataloader))[0]) if model.name in (ModelType.SNN_DIRECT, ModelType.SNN_DIRECT_LT) else 0
        )
        total_acs = snn_ac_monitor.get_total_acs()
        acs = int(total_acs / total)

    metrics = _compute_benchmark_metrics(torch.cat(all_labels), torch.cat(all_preds), macs, acs)
    return metrics


def compare_models(
    device,
    models: list[tuple[ModelType, FilterbankType]],
    dataset_type: DatasetType,
):
    results = []
    for model_type, filterbank_type in models:
        print(f'\n{model_type.value} | {filterbank_type.value}')
        result = load_and_benchmark(device, model_type, dataset_type, filterbank_type)
        results.append(result)

    is_kfold = 'aggregate' in results[0]
    filterbanks = list(dict.fromkeys(r['filterbank'] for r in results))
    groups = list(dict.fromkeys(r['model'] for r in results))

    if is_kfold:
        get_acc = lambda r: r['aggregate']['accuracy']['mean']
        get_std = lambda r: r['aggregate']['accuracy']['std']
        get_macs = lambda r: r['aggregate']['macs']
        get_acs = lambda r: r['aggregate']['acs']
    else:
        get_acc = lambda r: r['benchmark']['accuracy']
        get_std = None
        get_macs = lambda r: r['benchmark']['macs']
        get_acs = lambda r: r['benchmark']['acs']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    if len(filterbanks) > 1:
        results_by = {(r['model'], r['filterbank']): r for r in results}
        accuracies = {fb: [get_acc(results_by[(g, fb)]) for g in groups] for fb in filterbanks}
        stds = {fb: [get_std(results_by[(g, fb)]) for g in groups] for fb in filterbanks} if is_kfold else None
        macs = {fb: [get_macs(results_by[(g, fb)]) for g in groups] for fb in filterbanks}
        acs = {fb: [get_acs(results_by[(g, fb)]) for g in groups] for fb in filterbanks}
        plot_accuracy_comparison_grouped(ax1, groups, filterbanks, accuracies, stds)
        plot_energy_comparison_grouped(ax2, groups, filterbanks, macs, acs)
    else:
        accuracies = [get_acc(r) for r in results]
        stds = [get_std(r) for r in results] if is_kfold else None
        macs = [get_macs(r) for r in results]
        acs = [get_acs(r) for r in results]
        plot_accuracy_comparison(ax1, groups, accuracies, stds)
        plot_energy_comparison(ax2, groups, macs, acs)

    plt.tight_layout()
    plt.show()

def _forward(model, inputs: torch.Tensor) -> torch.Tensor:
    outputs = model(inputs)
    if model.name != ModelType.CNN:
        outputs = outputs.sum(dim=0)  # Need to sum if an SNN
    return outputs


def _compute_metrics(
    all_labels: torch.Tensor,
    all_preds: torch.Tensor,
    total_loss: float,
    num_batches: int,
) -> dict:
    """Returns a dict of loss, accuracy, macro_f1 and weighted_f1."""
    report = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)

    return {
        'loss': total_loss / num_batches,
        'accuracy': report['accuracy'] * 100,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
    }


def _compute_benchmark_metrics(all_labels: torch.Tensor, all_preds: torch.Tensor, macs: int, acs: int) -> dict:
    report = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)

    return {
        'accuracy': report['accuracy'] * 100,
        'macro_f1': report['macro avg']['f1-score'],
        'macro_precision': report['macro avg']['precision'],
        'macro_recall': report['macro avg']['recall'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'weighted_precision': report['weighted avg']['precision'],
        'weighted_recall': report['weighted avg']['recall'],
        'macs': macs,
        'acs': acs,
    }


def _calculate_conv2d_macs(model, sample_input: torch.Tensor) -> int:
    """
    Calculates the MACs for a Conv2d across all timesteps.
    For use with an SNN using direct coding, as the first layer receives continuous values and not spikes, which we need to account for.
    """
    conv2d = model.blocks[0].conv1
    h_in, w_in = sample_input.shape[2], sample_input.shape[3]
    h_k, w_k = conv2d.kernel_size
    h_s, w_s = conv2d.stride
    h_p, w_p = conv2d.padding

    h_out = ((h_in - h_k + 2 * h_p) // h_s) + 1
    w_out = ((w_in - w_k + 2 * w_p) // w_s) + 1

    macs_per_timestep = conv2d.in_channels * h_k * w_k * conv2d.out_channels * h_out * w_out
    return macs_per_timestep * model.timesteps
