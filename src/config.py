from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    original_sample_rate: int
    target_sample_rate: int
    target_duration: float
    n_fft: int
    n_bins: int
    delta_threshold: float
    excluded_classes: set[str]

    @property
    def target_samples(self) -> int:
        return int(self.target_sample_rate * self.target_duration)


@dataclass(frozen=True)
class HyperparameterTuningConfig:
    should_run: bool = True
    epochs: int = 5
    trials: int = 20


# TODO: Specific batch for each
@dataclass(frozen=True)
class Config:
    project_root = Path(__file__).resolve().parent.parent
    seed: int = 100
    epochs: int = 100
    patience: int = 20
    batch_size: int = 64
    show_progress: bool = True
    audiomnist: AudioConfig = AudioConfig(
        original_sample_rate=48_000,
        target_sample_rate=16_000,
        target_duration=0.84,
        n_fft=1024,
        n_bins=64,
        delta_threshold=0.1,
        excluded_classes=set(),
    )
    oceanship: AudioConfig = AudioConfig(
        original_sample_rate=32_000,
        target_sample_rate=16_000,
        target_duration=2.0,
        n_fft=1024,
        n_bins=64,
        delta_threshold=0.1,
        excluded_classes={'Search and Rescue vessel', 'Military ship', 'Anti-pollution equipment', 'Dredging'}
    )
    shipsear: AudioConfig = AudioConfig(
        original_sample_rate=16_000,
        target_sample_rate=16_000,
        target_duration=5.0,
        n_fft=1024,
        n_bins=64,
        delta_threshold=0.1,
        excluded_classes=set()
    )
    hyperparameter_tuning: HyperparameterTuningConfig = HyperparameterTuningConfig()

CONFIG = Config()
