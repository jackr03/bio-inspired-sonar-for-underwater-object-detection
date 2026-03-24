import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.config import CONFIG
from src.datasets.audio_dataset import AudioDataset
from src.datasets.spike_dataset import SpikeDataset
from src.types.dataset_type import DatasetType
from src.types.filterbank_type import FilterbankType
from src.types.model_type import ModelType


def get_dataset(model_type: ModelType, dataset_type: DatasetType, filterbank_type: FilterbankType) -> Dataset:
    match model_type:
        case ModelType.CNN | ModelType.SNN_DIRECT:
            return AudioDataset(dataset_type.input_dir, dataset_type.label_map, model_type.pipeline(dataset_type.config, filterbank_type))
        case ModelType.SNN:
            return SpikeDataset(dataset_type.spike_dir, dataset_type.label_map)


def get_split_dataloaders(dataset: Dataset) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns an 80/10/10 split of DataLoaders from the given dataset."""
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
