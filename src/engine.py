import torch
from sklearn.metrics import f1_score
from torchinfo import summary
from tqdm.auto import tqdm

from src.config import CONFIG
from src.types.model_type import ModelType
from src.utils.snn_ac_monitor import SNNACMonitor


def train_one_epoch(device, model, criterion, optimizer, train_dataloader) -> dict:
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

    return _compute_metrics(torch.cat(all_preds), torch.cat(all_labels), total_loss)


def validate(device, model, criterion, val_dataloader) -> dict:
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

    return _compute_metrics(torch.cat(all_preds), torch.cat(all_labels), total_loss)


def benchmark(device, model, test_dataloader) -> dict:
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
        macs = _calculate_conv2d_macs(model, next(iter(test_dataloader))[0]) if model.name == ModelType.SNN_DIRECT else 0
        total_acs = snn_ac_monitor.get_total_acs()
        acs = int(total_acs / total)

    metrics = _compute_metrics(torch.cat(all_preds), torch.cat(all_labels))
    metrics['macs'] = macs
    metrics['acs'] = acs
    return metrics


def _forward(model, inputs) -> torch.Tensor:
    outputs = model(inputs)
    if model.name != ModelType.CNN:
        outputs = outputs.sum(dim=0) # Need to sum if an SNN
    return outputs


def _compute_metrics(all_preds: torch.Tensor, all_labels: torch.Tensor, total_loss: float = None) -> dict:
    """Returns a dict of loss, accuracy, macro_f1 and weighted_f1."""
    accuracy = 100 * (all_preds == all_labels).sum().item() / len(all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
    }

    if total_loss is not None:
        metrics['loss'] = total_loss

    return metrics


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