import glob
import os
import warnings

import torch
import torchaudio
from torch import nn, Tensor
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """
    A custom DataSet for audio files.
    """
    def __init__(self, data_dir: str, pipeline: nn.Module):
        self.pipeline = pipeline
        self.audio_files = glob.glob(os.path.join(data_dir, '**/*.wav'), recursive=True)

        self.labels = []
        for file in self.audio_files:
            filename = os.path.basename(file)
            label = int(filename.split('_')[0])
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        audio_file = self.audio_files[index]
        label = self.labels[index]

        # Couldn't get torchcodec to work, so ignore these deprecation warnings
        warnings.filterwarnings('ignore', message='.*torchcodec.*')
        waveform, sample_rate = torchaudio.load(audio_file, normalize=True)
        mel_spectrogram = self.pipeline(waveform)
        return mel_spectrogram, torch.tensor(label, dtype=torch.long)