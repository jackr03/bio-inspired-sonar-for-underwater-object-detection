import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary
from tqdm.auto import tqdm

from src.config import CONFIG
from src.snn_ac_monitor import SNNACMonitor


def get_split_dataloaders(dataset: Dataset, seed: int = CONFIG.seed, batch_size: int = 64) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns an 80/10/10 split of DataLoaders from the given dataset."""
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True, pin_memory=True, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


def train_one_epoch_cnn(device, model, criterion, optimizer, train_dataloader, leave: bool = True) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_dataloader, desc='Training', unit='batches', leave=leave):
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

def validate_cnn(device, model, criterion, val_dataloader, leave: bool = True) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in tqdm(val_dataloader, desc='Validating', unit='batches', leave=leave):
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

def benchmark_cnn(device, model, test_dataloader) -> tuple[float, float]:
    model.eval()

    sample_input, _ = next(iter(test_dataloader))
    input_size = (1, *sample_input.shape[1:])
    model_stats = summary(model, input_size, device=device)
    macs = model_stats.total_mult_adds

    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in tqdm(test_dataloader, desc='Benchmarking', unit='batches'):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy, macs

def train_one_epoch_snn(device, model, criterion, optimizer, train_dataloader, leave: bool = True) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_dataloader, desc='Training', unit='batches', leave=leave):
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

def validate_snn(device, model, criterion, val_dataloader, leave: bool = True) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in tqdm(val_dataloader, desc='Validating', unit='batches', leave=leave):
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

def benchmark_snn(device, model, test_dataloader) -> tuple[float, float]:
    model.eval()

    snn_ac_monitor = SNNACMonitor(model)
    snn_ac_monitor.attach()

    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in tqdm(test_dataloader, desc='Benchmarking', unit='batches'):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            spk_rec = model(inputs)
            spk_count = spk_rec.sum(dim=0)
            _, predicted = torch.max(spk_count, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    snn_ac_monitor.remove()

    accuracy = 100 * correct / total
    total_acs = snn_ac_monitor.get_total_acs()

    # Divide by number of samples to get the per inference AC
    avg_acs_per_inference = total_acs / len(test_dataloader.dataset)

    return accuracy, avg_acs_per_inference