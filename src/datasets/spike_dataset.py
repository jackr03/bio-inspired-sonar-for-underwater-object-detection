import time

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.types.dataset_type import DatasetType
from src.types.filterbank_type import FilterbankType


class SpikeDataset(Dataset):
    def __init__(self, dataset: DatasetType, filterbank: FilterbankType) -> None:
        spike_dir = dataset.get_spike_dir(filterbank)
        all_files = sorted(spike_dir.rglob('*.pt'), key=lambda x: x.stem)
        files = [file for file in all_files if file.stem in dataset.label_map]

        print(f'Loading {len(files)} spikes...')
        start_time = time.time()
        self.spikes = []
        for file in files:
            spikes = torch.load(file)

            # Remove the initial channel dimension
            spikes = spikes.squeeze(1)

            on_spikes = torch.clamp(spikes, min=0, max=1)
            off_spikes = torch.abs(torch.clamp(spikes, max=0))
            dual_channel_spikes = torch.stack((on_spikes, off_spikes), dim=0)

            self.spikes.append(dual_channel_spikes)
        print(f'Finished loading spikes.')
        print(f'Time taken: {time.time() - start_time:.0f} seconds')

        self.labels = [dataset.label_map[file.stem] for file in files]

    def __len__(self) -> int:
        return len(self.spikes)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        spikes = self.spikes[index]
        label = self.labels[index]
        return spikes, torch.tensor(label, dtype=torch.long)
