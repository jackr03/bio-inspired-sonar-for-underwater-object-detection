import warnings
from pathlib import Path

import torch
import torchaudio
from torch import nn, Tensor
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, input_dir: Path, pipeline: nn.Module):
        self.pipeline = pipeline
        self.audio_files = sorted(input_dir.rglob('*.wav'), key=lambda x: int(x.stem))
        self.labels = [int(path.name.split('_')[0]) for path in self.audio_files]

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        audio_file = self.audio_files[index]
        label = self.labels[index]

        # Couldn't get torchcodec to work, so ignore these deprecation warnings
        warnings.filterwarnings('ignore', message='.*torchcodec.*')
        waveform, sample_rate = torchaudio.load(audio_file)
        mel_spectrogram = self.pipeline(waveform)
        return mel_spectrogram, torch.tensor(label, dtype=torch.long)
