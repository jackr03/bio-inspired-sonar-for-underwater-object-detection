import torch
from tqdm import tqdm

def train_one_epoch_cnn(device, model, criterion, optimizer, train_dataloader) -> tuple[float, float]:
    """
    Trains the given CNN for one epoch.
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_dataloader, desc='Training', unit='batches'):
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
    """
    Validates the given CNN.
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in tqdm(val_dataloader, desc='Validating', unit='batches'):
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

# TODO: Benchmark latency and energy cost
def benchmark_cnn(device, model, test_dataloader) -> float:
    """
    Benchmarks the given CNN.
    """
    model.eval()

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

    return 100 * correct / total

def train_one_epoch_snn(device, model, criterion, optimizer, train_dataloader) -> tuple[float, float]:
    """
    Trains the given SNN for one epoch.
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_dataloader, desc='Training', unit='batches'):
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
    """
    Validates the given SNN.
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in tqdm(val_dataloader, desc='Validating', unit='batches'):
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

# TODO: Benchmark latency and energy cost
def benchmark_snn(device, model, test_dataloader) -> float:
    """
    Benchmarks the given CNN.
    """
    model.eval()

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

    return 100 * correct / total