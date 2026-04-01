import torch
import torchaudio
from torch.utils.data import Dataset


class AugmentedSubset(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=5)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=5)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        spectrogram, label = self.subset[index]
        
        # Avoid modifying the original
        spectrogram = spectrogram.clone()
        spectrogram = self.freq_mask(spectrogram)
        spectrogram = self.time_mask(spectrogram)
        return spectrogram, label

