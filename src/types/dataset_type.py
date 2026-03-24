import csv
import random
from enum import Enum
from functools import cached_property
from pathlib import Path

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

    def get_spike_dir(self, filterbank_type: FilterbankType) -> Path:
        return CONFIG.project_root / 'processed' / f'{self.value}-{filterbank_type.value}'

    @cached_property
    def _metadata(self) -> dict:
        match self:
            case DatasetType.AUDIOMNIST:
                audio_files = list(self.input_dir.rglob('*.wav'))
                file_to_label = {file.stem: file.stem.split('_')[0] for file in audio_files}
                file_to_label_id = {file.stem: int(file.stem.split('_')[0]) for file in audio_files}
                label_id_to_label = {label_id: str(label_id) for label_id in sorted(set(file_to_label_id.values()))}

                return {
                    'file_to_label': file_to_label,
                    'file_to_label_id': file_to_label_id,
                    'label_id_to_label': label_id_to_label
                }
            case DatasetType.OCEANSHIP:
                train_csv = self.input_dir / 'oceanship_full_train.csv'
                test_csv = self.input_dir / 'oceanship_full_test.csv'

                file_to_label = {}
                for csv_path in [train_csv, test_csv]:
                    with open(csv_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            key = Path(row['wav_path']).stem
                            file_to_label[key] = row['label']

                unique_classes = sorted(set(file_to_label.values()))
                label_to_label_id = {label: label_id for label_id, label in enumerate(unique_classes)}
                label_id_to_label = {label_id: label for label_id, label in enumerate(unique_classes)}
                file_to_label_id = {stem: label_to_label_id[label] for stem, label in file_to_label.items()}

                return {
                    'file_to_label': file_to_label,
                    'file_to_label_id': file_to_label_id,
                    'label_id_to_label': label_id_to_label
                }

    @property
    def file_to_label(self) -> dict:
        return self._metadata['file_to_label']

    @property
    def file_to_label_id(self) -> dict:
        return self._metadata['file_to_label_id']

    @property
    def label_id_to_label(self) -> dict:
        return self._metadata['label_id_to_label']

    @property
    def num_classes(self) -> int:
        return len(self.label_id_to_label)

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
