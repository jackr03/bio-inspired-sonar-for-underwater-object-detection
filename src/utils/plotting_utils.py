import torch
from matplotlib import pyplot as plt

MAC_ENERGY_PJ = 3.7 + 0.9
AC_ENERGY_PJ = 0.9


def visualise_spikes(
    file_name: str,
    spike_tensor: torch.Tensor,
) -> None:
    spikes = spike_tensor.squeeze(1).detach().cpu().numpy()
    time_steps, bins = spikes.shape

    on_times, on_bins = (spikes == 1).nonzero()
    off_times, off_bins = (spikes == -1).nonzero()

    plt.scatter(on_times, on_bins, color='green', marker='|', s=50, label='On-spikes (+)')
    plt.scatter(off_times, off_bins, color='red', marker='|', s=50, label='Off-spikes (-)')
    plt.title(f'Spike Raster plot for {file_name}')
    plt.xlabel('Time Steps')
    plt.ylabel('Bins')
    plt.ylim(-0.5, bins - 0.5)
    plt.xlim(-0.5, time_steps - 0.5)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float],
    train_macro_f1s: list[float],
    val_macro_f1s: list[float],
    **_kwargs,
) -> None:
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

    ax2.plot(epochs, train_macro_f1s, label='Train')
    ax2.plot(epochs, val_macro_f1s, label='Validation')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Macro F1')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_accuracy_comparison(
    ax,
    names: list[str],
    accuracies: list[float],
    accuracy_stds: list[float] | None = None,
) -> None:
    colours = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

    bars = ax.bar(
        names,
        accuracies,
        color=colours[: len(names)],
        width=0.5,
        yerr=accuracy_stds,
        capsize=5,
    )
    if accuracy_stds:
        labels = [f'{a:.2f}%±{s:.2f}' for a, s in zip(accuracies, accuracy_stds)]
    else:
        labels = [f'{a:.2f}%' for a in accuracies]
    ax.bar_label(bars, labels=labels, padding=4)
    ax.set_title('Model Accuracies')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)


def plot_energy_comparison(
    ax,
    names: list[str],
    macs: list[int],
    acs: list[int],
) -> None:
    energies = [_estimate_energy(m, a) for m, a in zip(macs, acs)]

    mac_energies = [e['mac_uJ'] for e in energies]
    ac_energies = [e['ac_uJ'] for e in energies]

    ax.bar(names, mac_energies, color='#6A4C93', width=0.5, label='MACs')
    bars_ac = ax.bar(
        names,
        ac_energies,
        color='#1982C4',
        width=0.5,
        bottom=mac_energies,
        label='ACs',
    )

    totals = [m + a for m, a in zip(mac_energies, ac_energies)]
    ax.bar_label(bars_ac, labels=[f'{t:.2f} µJ' for t in totals], padding=4)

    ax.set_title('Estimated Energy per Inference')
    ax.set_ylabel('Energy (µJ)')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend()


def _estimate_energy(macs: int, acs: int) -> dict:
    """Given MACs and ACs, estimate energy usage in uJ."""
    mac_energy = macs * MAC_ENERGY_PJ * 1e-6
    ac_energy = acs * AC_ENERGY_PJ * 1e-6
    total = mac_energy + ac_energy
    return {
        'mac_uJ': mac_energy,
        'ac_uJ': ac_energy,
        'total_uJ': total,
    }
