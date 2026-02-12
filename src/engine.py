import torch
from tqdm import tqdm

# CNN stuff
def train_one_epoch_cnn(device, model, criterion, optimizer, train_dataloader) -> tuple[float, float]:
    """
    Trains the given CNN model for one epoch.
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    for data in tqdm(train_dataloader, desc='Training', unit='batches'):
        inputs, labels = data[0].to(device), data[1].to(device)

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
    Validates the given CNN model.
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for data in tqdm(val_dataloader, desc='Validating', unit='batches'):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_dataloader)
    avg_accuracy = 100 * correct / total

    return avg_loss, avg_accuracy

# TODO: Need to benchmark latency and energy cost
def benchmark_cnn(device, model, test_dataloader) -> float:
    """
    Benchmarks the given CNN model.
    """
    model.eval()

    correct = 0
    total = 0
    with torch.inference_mode():
        for data in tqdm(test_dataloader, desc='Benchmarking', unit='batches'):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy
