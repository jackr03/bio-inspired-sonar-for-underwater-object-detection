from pathlib import Path, PureWindowsPath

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset, random_split, WeightedRandomSampler

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
            return _build_audiomnist_dataloaders(dataset)
        case DatasetType.OCEANSHIP:
            # Oceanship is imbalanced so we need to do a stratified split + SpecAug
            return _build_oceanship_dataloaders(dataset)
        case DatasetType.SHIPSEAR:
            raise ValueError('Use get_kfold_dataloaders() for ShipsEar')


def _build_audiomnist_dataloaders(dataset: Dataset) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(CONFIG.seed)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, num_workers=4, persistent_workers=True, pin_memory=True, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG.batch_size, num_workers=4, persistent_workers=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=CONFIG.batch_size, num_workers=4, persistent_workers=True, pin_memory=True)
    return train_dataloader, val_dataloader, test_dataloader


def _build_oceanship_dataloaders(dataset: Dataset) -> tuple[DataLoader, DataLoader, DataLoader]:
    labels = np.array(dataset.labels)
    indices = np.arange(len(dataset))

    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=CONFIG.seed)
    train_idx, temp_idx = next(splitter1.split(indices, labels))

    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=CONFIG.seed)
    val_idx, test_idx = next(splitter2.split(temp_idx, labels[temp_idx]))

    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    train_dataset, val_dataset, test_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

    labels = [dataset.labels[i] for i in train_dataset.indices]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dataset = AugmentedSubset(train_dataset)
    train_dataset = AugmentedSubset(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, num_workers=4, persistent_workers=True, pin_memory=True, sampler=sampler, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG.batch_size, num_workers=4, persistent_workers=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=CONFIG.batch_size, num_workers=4, persistent_workers=True, pin_memory=True)
    return train_dataloader, val_dataloader, test_dataloader


def get_kfold_dataloaders(dataset: Dataset, dataset_type: DatasetType, n_folds: int = 5) -> list[tuple[DataLoader, DataLoader, DataLoader]]:
    """Returns a list of (train, val, test) DataLoader tuples, one per fold."""
    stem_to_idx = {stem: i for i, stem in enumerate(dataset.stems)}
    folds = []

    for fold in range(1, n_folds + 1):
        train_list = dataset_type.input_dir / f'train_list_{fold}.txt'
        test_list = dataset_type.input_dir / f'test_list_{fold}.txt'

        train_stems = _parse_split_file(train_list)
        test_stems = _parse_split_file(test_list)

        train_idx = np.array([stem_to_idx[s] for s in train_stems if s in stem_to_idx])
        test_idx = np.array([stem_to_idx[s] for s in test_stems if s in stem_to_idx])

        # Carve validation set from train (stratified 90/10)
        train_labels = np.array([dataset.labels[i] for i in train_idx])
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=CONFIG.seed)
        train_rel_idx, val_rel_idx = next(splitter.split(train_idx, train_labels))

        val_idx = train_idx[val_rel_idx]
        train_idx = train_idx[train_rel_idx]

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)

        train_dataloader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, num_workers=4, persistent_workers=True, pin_memory=True, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=CONFIG.batch_size, num_workers=4, persistent_workers=True, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=CONFIG.batch_size, num_workers=4, persistent_workers=True, pin_memory=True)
        folds.append((train_dataloader, val_dataloader, test_dataloader))

    return folds


def _parse_split_file(path: Path) -> list[str]:
    """Parses a ShipsEar split file, returning file stems."""
    stems = []
    with open(path) as f:
        for line in f:
            wav_path = line.strip().split('\t')[0].strip()
            stems.append(PureWindowsPath(wav_path).stem)
    return stems
