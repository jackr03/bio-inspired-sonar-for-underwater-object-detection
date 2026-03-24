import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.types.dataset_type import DatasetType
from src.types.filterbank_type import FilterbankType


class SpikeDataset(Dataset):
    def __init__(self, dataset: DatasetType, filterbank: FilterbankType) -> None:
        spike_dir = dataset.get_spike_dir(filterbank)
        all_files = sorted(spike_dir.rglob('*.pt'), key=lambda x: x.stem)
        self.file_paths = [file for file in all_files if dataset.file_to_label[file.stem] not in dataset.config.excluded_classes]
        self.labels = [dataset.file_to_label_id[file.stem] for file in self.file_paths]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        file_path = self.file_paths[index]
        label = self.labels[index]

        try:
            spikes = torch.load(file_path)

            # Remove the initial channel dimension
            spikes = spikes.squeeze(1)

            # Stack two channels at index 0
            on_spikes = torch.clamp(spikes, min=0, max=1)
            off_spikes = torch.abs(torch.clamp(spikes, max=0))
            dual_channel_spikes = torch.stack((on_spikes, off_spikes), dim=0)

            return dual_channel_spikes, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            raise RuntimeError(f'Failed to load corrupt file at {file_path}.') from e

