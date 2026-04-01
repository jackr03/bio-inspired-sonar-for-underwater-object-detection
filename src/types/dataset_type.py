import os
from collections import Counter
from enum import Enum
from functools import cached_property
from pathlib import Path

import torch

from src.config import AudioConfig, CONFIG
from src.types.filterbank_type import FilterbankType
from src.types.model_type import ModelType


class DatasetType(str, Enum):
    AUDIOMNIST = 'audiomnist'
    SHIPSEAR = 'shipsear'

    @property
    def config(self) -> AudioConfig:
        match self:
            case DatasetType.AUDIOMNIST:
                return CONFIG.audiomnist
            case DatasetType.SHIPSEAR:
                return CONFIG.shipsear

    @property
    def input_dir(self) -> Path:
        return CONFIG.project_root / 'data' / self

    @property
    def _processed_data_dir(self) -> Path:
        """Determines the directory for processed data (for use with CSF3 clusters)."""
        return (
            Path(os.environ['PROCESSED_DATA_DIR'])
            if 'PROCESSED_DATA_DIR' in os.environ
            else CONFIG.project_root / 'processed'
        )

    def get_spectrogram_dir(self, filterbank_type: FilterbankType) -> Path:
        return self._processed_data_dir / f'{self.value}-{filterbank_type.value}-spectrograms'

    def get_spike_dir(self, filterbank_type: FilterbankType) -> Path:
        return self._processed_data_dir / f'{self.value}-{filterbank_type.value}-spikes'

    @cached_property
    def label_map(self) -> dict[str, int]:
        """Maps file stem to label ID."""
        match self:
            case DatasetType.AUDIOMNIST | DatasetType.SHIPSEAR:
                audio_files = list(self.input_dir.rglob('*.wav'))
                return {file.stem: int(file.stem.split('_')[0]) for file in audio_files}

    @property
    def num_classes(self) -> int:
        match self:
            case DatasetType.AUDIOMNIST:
                return 10
            case DatasetType.SHIPSEAR:
                return 5

    @cached_property
    def class_weights(self) -> torch.Tensor:
        counts = Counter(self.label_map.values())
        total = sum(counts.values())
        weights = torch.zeros(self.num_classes)
        for label_id, count in counts.items():
            weights[label_id] = total / (self.num_classes * count)
        return weights

    def get_model_config(self, model: ModelType) -> dict:
        match self:
            case DatasetType.AUDIOMNIST:
                channels = [8, 16]
            case DatasetType.SHIPSEAR:
                channels = [16, 32]

        match model:
            case ModelType.CNN | ModelType.SNN_DIRECT:
                in_channels = 1
            case ModelType.SNN:
                in_channels = 2

        return {
            'in_channels': in_channels,
            'num_classes': self.num_classes,
            'channels': channels,
        }
