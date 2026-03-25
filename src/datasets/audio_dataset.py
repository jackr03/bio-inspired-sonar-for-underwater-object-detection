import torch
import torchaudio
from torch import nn, Tensor
from torch.utils.data import Dataset

from src.types.dataset_type import DatasetType


class AudioDataset(Dataset):
    def __init__(self, dataset: DatasetType, pipeline: nn.Module):
        audio_files = sorted(dataset.input_dir.rglob('*.wav'), key=lambda x: x.stem)
        self.spectrograms = []
        for file in audio_files:
            waveform, _ = torchaudio.load(file)
            spectrogram = pipeline(waveform)
            self.spectrograms.append(spectrogram)

        self.labels = [dataset.label_map[file.stem] for file in audio_files]

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        spectrogram = self.spectrograms[index]
        label = self.labels[index]
        return spectrogram, torch.tensor(label, dtype=torch.long)
