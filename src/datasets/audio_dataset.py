import warnings
from pathlib import Path

import torch
import torchaudio
from torch import nn, Tensor
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, input_dir: Path, label_map: dict, pipeline: nn.Module):
        self.pipeline = pipeline

        all_files = sorted(input_dir.rglob('*.wav'), key=lambda x: x.stem)
        # Ignore those that don't have a mapping, i.e. we've decided to ignore
        self.audio_files = [f for f in all_files if f.stem in label_map]
        self.labels = [label_map[file.stem] for file in self.audio_files]

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
