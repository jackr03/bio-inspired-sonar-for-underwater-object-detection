import torch
import torchaudio
from torch.utils.data import Dataset


class AugmentedSubset(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=10)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        spectrogram, label = self.subset[index]
        spectrogram = self.freq_mask(spectrogram)
        spectrogram = self.time_mask(spectrogram)
        return spectrogram, label

    def _augment(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.time_mask(self.freq_mask(spectrogram))
