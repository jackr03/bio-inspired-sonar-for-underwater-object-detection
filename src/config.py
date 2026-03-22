from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
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
    epochs: int = 5
    trials: int = 20
    should_run: bool = True

@dataclass(frozen=True)
class Config:
    seed: int = 100
    delta_threshold: float = 0.1
    epochs: int = 50
    audiomnist: AudioConfig = AudioConfig(
        original_sample_rate=48_000,
        target_sample_rate=16_000,
        target_duration=0.84,
        n_fft=1024,
        n_bins=64,
        delta_threshold=0.1
    )
    oceanship: AudioConfig = AudioConfig(
        original_sample_rate=32_000,
        target_sample_rate=32_000,
        target_duration=5.0,
        n_fft=1024,
        n_bins=128,
        delta_threshold=0.1
    )
    hyperparameter_tuning: HyperparameterTuningConfig = HyperparameterTuningConfig()

CONFIG = Config()
