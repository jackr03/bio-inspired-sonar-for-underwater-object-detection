import csv
from enum import Enum
from pathlib import Path

from src.config import AudioConfig, CONFIG


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

    @property
    def spike_dir(self) -> Path:
        return CONFIG.project_root / 'processed' / self

    @property
    def label_map(self) -> dict:
        match self:
            case DatasetType.AUDIOMNIST:
                audio_files = self.input_dir.rglob('*.wav')
                return {file.stem: int(file.stem.split('_')[0]) for file in audio_files}
            case DatasetType.OCEANSHIP:
                train_csv = self.input_dir / 'oceanship_full_train.csv'
                test_csv = self.input_dir / 'oceanship_full_test.csv'
                excluded_classes = {'Military ship', 'Anti-pollution', 'Dredging', 'Search and Rescue'}

                raw_lookup = {}
                for csv_path in [train_csv, test_csv]:
                    with open(csv_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row['label'] in excluded_classes:
                                continue
                            key = Path(row['wav_path']).stem
                            raw_lookup[key] = row['label']

                label_to_int = {label: i for i, label in enumerate(sorted(set(raw_lookup.values())))}
                return {stem: label_to_int[label] for stem, label in raw_lookup.items()}

    @property
    def num_classes(self) -> int:
        return len(set(self.label_map.values()))