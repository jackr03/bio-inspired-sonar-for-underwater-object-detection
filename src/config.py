from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    path: Path
    original_sample_rate: int
    target_sample_rate: int
    target_duration: float
    n_fft: int
    n_bins: int
    delta_threshold: float

    @property
    def target_samples(self) -> int:
        return int(self.target_sample_rate * self.target_duration)

@dataclass(frozen=True)
class HyperparameterTuningConfig:
    should_run: bool = False
    epochs: int = 5
    trials: int = 20

@dataclass(frozen=True)
class Config:
    seed: int = 100
    epochs: int = 50
    patience: int = 5
    batch_size: int = 256
    show_progress: bool = True
    audiomnist: AudioConfig = AudioConfig(
        path=Path(f'data/audio-mnist'),
        original_sample_rate=48_000,
        target_sample_rate=16_000,
        target_duration=0.84,
        n_fft=1024,
        n_bins=64,
        delta_threshold=0.1
    )
    oceanship: AudioConfig = AudioConfig(
        path=Path(f'data/oceanship'),
        original_sample_rate=32_000,
        target_sample_rate=32_000,
        target_duration=5.0,
        n_fft=1024,
        n_bins=128,
        delta_threshold=0.1
    )
    hyperparameter_tuning: HyperparameterTuningConfig = HyperparameterTuningConfig()

CONFIG = Config()
