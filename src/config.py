from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    original_sample_rate: int
    target_sample_rate: int
    target_duration: float
    n_fft: int
    n_bins: int
    hop_length: int
    delta_threshold: float

    @property
    def target_samples(self) -> int:
        return int(self.target_sample_rate * self.target_duration)

    @property
    def timesteps(self) -> int:
        return int(self.target_samples / self.hop_length) + 1


@dataclass(frozen=True)
class HyperparameterTuningConfig:
    should_run: bool = False
    epochs: int = 5
    trials: int = 20


@dataclass(frozen=True)
class Config:
    project_root = Path(__file__).resolve().parent.parent
    seed: int = 100
    epochs: int = 100
    patience: int = 15
    batch_size: int = 64
    should_train: bool = True
    show_progress: bool = False
    audiomnist: AudioConfig = AudioConfig(
        original_sample_rate=48_000,
        target_sample_rate=16_000,
        target_duration=0.84,
        n_fft=1024,
        hop_length=512,
        n_bins=64,
        delta_threshold=0.15,
    )
    shipsear: AudioConfig = AudioConfig(
        original_sample_rate=16_000,
        target_sample_rate=16_000,
        target_duration=5.0,
        n_fft=1024,
        n_bins=64,
        hop_length=512,
        delta_threshold=0.1,
    )
    hyperparameter_tuning: HyperparameterTuningConfig = HyperparameterTuningConfig()


CONFIG = Config()
