import json
from pathlib import Path

import optuna
from matplotlib import pyplot as plt

from src.config import CONFIG

MAC_ENERGY_PJ = 3.7 + 0.9
AC_ENERGY_PJ = 0.9


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


def plot_accuracy_comparison(ax, model1_name: str, model1_acc: float, model2_name: str, model2_acc: float) -> None:
    models = [model1_name, model2_name]
    accuracies = [model1_acc, model2_acc]
    colours = ['#4C72B0', '#DD8452']

    bars = ax.bar(models, accuracies, color=colours, width=0.5)
    ax.bar_label(bars, fmt='%.2f%%', padding=4)
    ax.set_title('Model Accuracies')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)


def plot_energy_comparison(ax, model1_name: str, model1_macs: int, model1_acs: int, model2_name: str, model2_macs: int, model2_acs: int) -> None:
    energy1 = estimate_energy(model1_macs, model1_acs)
    energy2 = estimate_energy(model2_macs, model2_acs)

    models = [model1_name, model2_name]
    mac_energies = [energy1['mac_uJ'], energy2['mac_uJ']]
    ac_energies = [energy1['ac_uJ'], energy2['ac_uJ']]

    bars_mac = ax.bar(models, mac_energies, color='#6A4C93', width=0.5, label='MACs')
    bars_ac = ax.bar(models, ac_energies, color='#1982C4', width=0.5, bottom=mac_energies, label='ACs')

    totals = [m + a for m, a in zip(mac_energies, ac_energies)]
    ax.bar_label(bars_ac, labels=[f'{t:.2f} µJ' for t in totals], padding=4)

    ax.set_title('Estimated Energy per Inference')
    ax.set_ylabel('Energy (µJ)')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)  # Just so label at the top doesn't hit the border
    ax.legend()


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