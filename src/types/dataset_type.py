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
