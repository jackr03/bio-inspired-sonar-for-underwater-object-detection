import json
from pathlib import Path

import optuna
from matplotlib import pyplot as plt

MAC_ENERGY_PJ = 3.7 + 0.9
AC_ENERGY_PJ = 0.9


def run_sweep(objective, output_path: Path, n_trials: int) -> None:
    print('Running hyperparameter sweep...')
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)

    print('Hyperparameter sweep completed.')
    print(f'Accuracy: {study.best_value:.2f}%')
    print(f'Parameters: {study.best_params}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)

def run_sweep_pareto(objective, output_path: Path, n_trials: int) -> None:
    """A special hyperparameter sweep for finding the Pareto front, e.g. maximising SNN accuracy while minimising timesteps."""
    print('Running hyperparameter sweep (Pareto front)...')
    study = optuna.create_study(directions=['maximize', 'minimize'])
    study.optimize(objective, n_trials=n_trials)

    print('[REMEMBER TO MANUALLY SELECT BEST SET OF HYPERPARAMETERS]')
    print('Hyperparameter sweep completed.')
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


def plot_training_history(train_losses: list[float], train_accs: list[float], val_losses: list[float], val_accs: list[float]) -> None:
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharex=True)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax1.plot(epochs, train_losses, label='Train')
    ax1.plot(epochs, val_losses, label='Validation')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, label='Train')
    ax2.plot(epochs, val_accs, label='Validation')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def estimate_energy(macs: int, acs: int) -> dict:
    """Given MACs and ACs, estimate energy usage in uJ."""
    mac_energy = macs * MAC_ENERGY_PJ * 1e-6
    ac_energy = acs * AC_ENERGY_PJ * 1e-6
    total = mac_energy + ac_energy * 1e-6
    return {
        'mac_uJ': mac_energy,
        'ac_uJ': ac_energy,
        'total_uJ': total,
    }