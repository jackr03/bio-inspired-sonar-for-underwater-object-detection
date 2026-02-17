import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split


def get_random_audio(base_dir: str) -> str:
    """Randomly select a digit, speaker and sample from the AudioMNIST dataset and return its path."""
    digit = np.random.randint(0, 10)
    speaker = np.random.randint(1, 61)
    sample = np.random.randint(0, 50)

    if speaker < 10:
        return f'{base_dir}/0{speaker}/{digit}_0{speaker}_{sample}.wav'
    else:
        return f'{base_dir}/{speaker}/{digit}_{speaker}_{sample}.wav'

# TODO: Add seed for reproducibility
# TODO: Check if 80/10/10 is the split
def get_split_dataloaders(dataset: Dataset, batch_size: int = 64) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns an 80/10/10 split of DataLoaders from the given dataset."""
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True, pin_memory=True, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader
