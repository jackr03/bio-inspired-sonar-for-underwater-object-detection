import warnings

import torch
import torchaudio
from torch import nn, Tensor
from torch.utils.data import Dataset

from src.types.dataset_type import DatasetType


class AudioDataset(Dataset):
    def __init__(self, dataset: DatasetType, pipeline: nn.Module):
        self.pipeline = pipeline

        all_files = sorted(dataset.input_dir.rglob('*.wav'), key=lambda x: x.stem)
        # Ignore those that excluded in the config
        self.audio_files = [file for file in all_files if dataset.file_to_label[file.stem] not in dataset.config.excluded_classes]
        self.labels = [dataset.file_to_label_id[file.stem] for file in self.audio_files]

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        audio_file = self.audio_files[index]
        label = self.labels[index]

        # Couldn't get torchcodec to work, so ignore these deprecation warnings
        warnings.filterwarnings('ignore', message='.*torchcodec.*')
        waveform, sample_rate = torchaudio.load(audio_file)
        spectrogram = self.pipeline(waveform)
        return spectrogram, torch.tensor(label, dtype=torch.long)
