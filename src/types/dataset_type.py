import csv
import random
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
    OCEANSHIP = 'oceanship'

    @property
    def config(self) -> AudioConfig:
        match self:
            case DatasetType.AUDIOMNIST:
                return CONFIG.audiomnist
            case DatasetType.OCEANSHIP:
                return CONFIG.oceanship

    @property
    def input_dir(self) -> Path:
        return CONFIG.project_root / 'data' / self

    def get_spectrogram_dir(self, filterbank_type: FilterbankType) -> Path:
        return CONFIG.project_root / 'processed' / f'{self.value}-{filterbank_type.value}-spectrograms'

    def get_spike_dir(self, filterbank_type: FilterbankType) -> Path:
        return CONFIG.project_root / 'processed' / f'{self.value}-{filterbank_type.value}-spikes'

    @cached_property
    def label_map(self) -> dict[str, int]:
        """Maps file stem to label ID, excluding classes specified in config."""
        match self:
            case DatasetType.AUDIOMNIST:
                audio_files = list(self.input_dir.rglob('*.wav'))
                return {file.stem: int(file.stem.split('_')[0]) for file in audio_files}
            case DatasetType.OCEANSHIP:
                train_csv = self.input_dir / 'oceanship_full_train.csv'
                test_csv = self.input_dir / 'oceanship_full_test.csv'
                excluded = self.config.excluded_classes

                file_to_label = {}
                for csv_path in [train_csv, test_csv]:
                    with open(csv_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            key = Path(row['wav_path']).stem
                            label = row['label']
                            if label not in excluded:
                                file_to_label[key] = label

                unique_classes = sorted(set(file_to_label.values()))
                label_to_id = {label: i for i, label in enumerate(unique_classes)}
                return {stem: label_to_id[label] for stem, label in file_to_label.items()}

    @property
    def num_classes(self) -> int:
        return len(set(self.label_map.values()))

    @cached_property
    def class_weights(self) -> torch.Tensor:
        counts = Counter(self.label_map.values())
        total = sum(counts.values())
        weights = torch.zeros(self.num_classes)
        for label_id, count in counts.items():
            weights[label_id] = total / (self.num_classes * count)
        return weights

    def get_random_encoded_file(self, filterbank: FilterbankType) -> Path:
        files = list(self.get_spike_dir(filterbank).rglob('*pt'))
        return random.choice(files)

    def get_model_config(self, model: ModelType) -> dict:
        match self:
            case DatasetType.AUDIOMNIST:
                channels = [8, 16]
            case DatasetType.OCEANSHIP:
                channels = [32, 64, 128]

        match model:
            case ModelType.CNN | ModelType.SNN_DIRECT:
                in_channels = 1
            case ModelType.SNN:
                in_channels = 2

        return {
            'in_channels': in_channels,
            'num_classes': self.num_classes,
            'channels': channels
        }
