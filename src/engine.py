import torch
from torchinfo import summary
from tqdm.auto import tqdm

from src.config import CONFIG
from src.snn_ac_monitor import SNNACMonitor
from src.types.model_type import ModelType


def train_one_epoch_cnn(device, model, criterion, optimizer, train_dataloader) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    if CONFIG.show_progress:
        train_dataloader = tqdm(train_dataloader, desc='Training', unit='batches')

    for inputs, labels in train_dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    avg_accuracy = 100 * correct / total

    return avg_loss, avg_accuracy


def validate_cnn(device, model, criterion, val_dataloader) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    if CONFIG.show_progress:
        val_dataloader = tqdm(val_dataloader, desc='Validating', unit='batches')

    with torch.inference_mode():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_dataloader)
    avg_accuracy = 100 * correct / total

    return avg_loss, avg_accuracy


def benchmark_cnn(device, model, test_dataloader) -> tuple[float, int, int]:
    """Returns a tuple of (accuracy, macs, acs)."""
    model.eval()

    sample_input, _ = next(iter(test_dataloader))
    input_size = (1, *sample_input.shape[1:])
    model_stats = summary(model, input_size, device=device)
    macs = model_stats.total_mult_adds

    correct = 0
    total = 0

    if CONFIG.show_progress:
        test_dataloader = tqdm(test_dataloader, desc='Testing', unit='batches')

    with torch.inference_mode():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy, macs, 0


def train_one_epoch_snn(device, model, criterion, optimizer, train_dataloader) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    if CONFIG.show_progress:
        train_dataloader = tqdm(train_dataloader, desc='Training', unit='batches')

    for inputs, labels in train_dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        spk_rec = model(inputs)
        spk_count = spk_rec.sum(dim=0)
        loss = criterion(spk_count, labels)
        loss.backward()
        total_loss += loss.item()
        _, predicted = torch.max(spk_count, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    avg_accuracy = 100 * correct / total

    return avg_loss, avg_accuracy


def validate_snn(device, model, criterion, val_dataloader) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    if CONFIG.show_progress:
        val_dataloader = tqdm(val_dataloader, desc='Validation', unit='batches')

    with torch.inference_mode():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            spk_rec = model(inputs)
            spk_count = spk_rec.sum(dim=0)
            loss = criterion(spk_count, labels)
            total_loss += loss.item()
            _, predicted = torch.max(spk_count, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_dataloader)
    avg_accuracy = 100 * correct / total

    return avg_loss, avg_accuracy


def benchmark_snn(device, model, test_dataloader) -> tuple[float, int, int]:
    """Returns a tuple of (accuracy, first_layer_macs, avg_acs)."""
    model.eval()

    snn_ac_monitor = SNNACMonitor(model)
    snn_ac_monitor.attach()

    correct = 0
    total = 0

    if CONFIG.show_progress:
        test_dataloader = tqdm(test_dataloader, desc='Testing', unit='batches')

    with torch.inference_mode():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            spk_rec = model(inputs)
            spk_count = spk_rec.sum(dim=0)
            _, predicted = torch.max(spk_count, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    snn_ac_monitor.remove()

    accuracy = 100 * correct / total

    # Calculate MACs if direct encoded, skip otherwise
    total_macs = _calculate_conv2d_macs(model, next(iter(test_dataloader))[0]) if model.name == ModelType.SNN_DIRECT else 0
    total_acs = snn_ac_monitor.get_total_acs()

    # Divide by number of samples to get the per inference AC
    avg_acs_per_inference = int(total_acs / total)

    return accuracy, total_macs, avg_acs_per_inference


def _calculate_conv2d_macs(model, sample_input: torch.Tensor) -> int:
    """
    Calculates the MACs for a Conv2d across all timesteps.
    For use with an SNN using direct coding, as the first layer receives continuous values and not spikes, which we need to account for.
    """
    conv2d = model.block1.conv
    h_in, w_in = sample_input.shape[2], sample_input.shape[3]
    h_k, w_k = conv2d.kernel_size
    h_s, w_s = conv2d.stride
    h_p, w_p = conv2d.padding

    h_out = ((h_in - h_k + 2 * h_p) // h_s) + 1
    w_out = ((w_in - w_k + 2 * w_p) // w_s) + 1

    macs_per_timestep = conv2d.in_channels * h_k * w_k * conv2d.out_channels * h_out * w_out
    return macs_per_timestep * model.timesteps
