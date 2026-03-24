import json
from pathlib import Path

import optuna
import torch
from torch import nn

from src.config import CONFIG
from src.types.dataset_type import DatasetType
from src.types.filterbank_type import FilterbankType
from src.types.model_type import ModelType
from src.utils.dataset_utils import get_dataset, get_split_dataloaders
from src.utils.model_utils import get_model_components


def run_hyperparameter_sweep(device, model_type: ModelType, dataset_type: DatasetType, filterbank_type: FilterbankType):
    components = get_model_components(model_type, dataset_type, filterbank_type)

    def objective(trial) -> float | tuple[float, int]:
        hyperparameters = get_hyperparameter_suggestions(trial, model_type)
        dataset = get_dataset(model_type, dataset_type, filterbank_type)
        train_dataloader, val_dataloader, _ = get_split_dataloaders(dataset)

        model = components['model_class'](**dataset_type.get_model_config(model_type), **hyperparameters['model_init']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr'])
        criterion = nn.CrossEntropyLoss()

        val_acc = 0.0
        for epoch in range(CONFIG.hyperparameter_tuning.epochs):
            components['train_fn'](device, model, criterion, optimizer, train_dataloader)
            _, val_acc = components['val_fn'](device, model, criterion, val_dataloader)

            if model_type != ModelType.SNN_DIRECT:
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        match model_type:
            case ModelType.CNN | ModelType.SNN:
                return val_acc
            case ModelType.SNN_DIRECT:
                return val_acc, hyperparameters['model_init']['timesteps']

    match model_type:
        case ModelType.CNN | ModelType.SNN:
            run_sweep(objective, components['hyperparameters_path'])
        case ModelType.SNN_DIRECT:
            run_sweep_pareto(objective, components['hyperparameters_path'])

def get_hyperparameter_suggestions(trial, model_type: ModelType) -> dict:
    # Common parameters
    hyperparameters = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    }

    match model_type:
        case ModelType.CNN:
            model_specific = {}
        case ModelType.SNN_DIRECT:
            model_specific = {
                'beta_init': trial.suggest_float('beta_init', 0.5, 0.99),
                'slope': trial.suggest_int('slope', 10, 50),
                'timesteps': trial.suggest_categorical('timesteps', [5, 10, 15])
            }
        case ModelType.SNN:
            model_specific = {
                'beta_init': trial.suggest_float('beta_init', 0.5, 0.99),
                'slope': trial.suggest_int('slope', 10, 50)
            }
        case _:
            raise ValueError

    hyperparameters.update({'model_init': model_specific})
    return hyperparameters


def run_sweep(objective, output_path: Path) -> None:
    print('Running hyperparameter sweep...')
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=CONFIG.hyperparameter_tuning.trials)

    print('Hyperparameter sweep completed.')
    print(f'Accuracy: {study.best_value:.2f}%')
    print(f'Parameters: {study.best_params}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)


def run_sweep_pareto(objective, output_path: Path) -> None:
    """A special hyperparameter sweep for finding the Pareto front, e.g. maximising SNN accuracy while minimising timesteps."""
    print('Running hyperparameter sweep (Pareto front)...')
    study = optuna.create_study(directions=['maximize', 'minimize'])
    study.optimize(objective, n_trials=CONFIG.hyperparameter_tuning.trials)

    print('Pareto front hyperparameter sweep completed.')
    pareto_results = []
    for trial in study.best_trials:
        pareto_results.append({
            'accuracy': trial.values[0],
            'timesteps': trial.values[1],
            'params': trial.params
        })
    pareto_results.sort(key=lambda x: x['accuracy'], reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(pareto_results, f, indent=2)
