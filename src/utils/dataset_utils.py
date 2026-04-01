import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from src.config import CONFIG
from src.datasets.augmented_dataset import AugmentedSubset
from src.datasets.spectrogram_dataset import SpectrogramDataset
from src.datasets.spike_dataset import SpikeDataset
from src.types.dataset_type import DatasetType
from src.types.filterbank_type import FilterbankType
from src.types.model_type import ModelType


def get_dataset(model_type: ModelType, dataset_type: DatasetType, filterbank_type: FilterbankType) -> Dataset:
    match model_type:
        case ModelType.CNN | ModelType.SNN_DIRECT:
            return SpectrogramDataset(dataset_type, filterbank_type)
        case ModelType.SNN:
            return SpikeDataset(dataset_type, filterbank_type)


def get_split_dataloaders(dataset: Dataset, dataset_type: DatasetType) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns an 80/10/10 split of DataLoaders from the given dataset."""
    match dataset_type:
        case DatasetType.AUDIOMNIST:
            train_dataset, val_dataset, test_dataset = _simple_split(dataset)
        case DatasetType.OCEANSHIP: # Oceanship is imbalanced so we need to do a stratified split + SpecAug
            train_dataset, val_dataset, test_dataset = _stratified_split(dataset)
            train_dataset = AugmentedSubset(train_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, num_workers=4, persistent_workers=True, pin_memory=True, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG.batch_size, num_workers=4, persistent_workers=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=CONFIG.batch_size, num_workers=4, persistent_workers=True, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


def _simple_split(dataset: Dataset) -> tuple[Subset, Subset, Subset]:
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    return random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(CONFIG.seed)
    )


def _stratified_split(dataset: Dataset) -> tuple[Subset, Subset, Subset]:
    labels = np.array(dataset.labels)
    indices = np.arange(len(dataset))

    # First split: 80% train, 20% temp (val + test)
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=CONFIG.seed)
    train_idx, temp_idx = next(splitter1.split(indices, labels))

    # Second split: 50/50 on the 20% temp → 10% val, 10% test
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=CONFIG.seed)
    val_idx, test_idx = next(splitter2.split(temp_idx, labels[temp_idx]))

    # splitter2 indices are relative to temp_idx, so map back
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


