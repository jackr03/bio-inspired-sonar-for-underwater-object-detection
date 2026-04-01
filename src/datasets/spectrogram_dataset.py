import time

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.types.dataset_type import DatasetType
from src.types.filterbank_type import FilterbankType


class SpectrogramDataset(Dataset):
    def __init__(self, dataset: DatasetType, filterbank: FilterbankType):
        spectrogram_dir = dataset.get_spectrogram_dir(filterbank)
        files = sorted(spectrogram_dir.rglob('*.pt'), key=lambda x: x.stem)

        print(f'Loading {len(files)} spectrograms...')
        start_time = time.time()
        self.spectrograms = []
        for file in files:
            spectrogram = torch.load(file)
            self.spectrograms.append(spectrogram)
        print(f'Finished loading spectrograms.')
        print(f'Time taken: {time.time() - start_time:.0f} seconds')

        self.stems = [file.stem for file in files]
        self.labels = [dataset.label_map[stem] for stem in self.stems]

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        spectrogram = self.spectrograms[index]
        label = self.labels[index]
        return spectrogram, torch.tensor(label, dtype=torch.long)
