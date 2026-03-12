from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    original_sample_rate: int = 48_000
    target_sample_rate: int = 16_000
    target_duration: float = 0.84
    n_mels: int = 64

    @property
    def target_samples(self) -> int:
        return int(self.target_sample_rate * self.target_duration)

@dataclass(frozen=True)
class HyperparameterTuningConfig:
    epochs: int = 3
    trials: int = 20
    should_run: bool = False

@dataclass(frozen=True)
class Config:
    seed: int = 100
    delta_threshold: float = 0.1
    audio: AudioConfig = AudioConfig()
    hyperparameter_tuning: HyperparameterTuningConfig = HyperparameterTuningConfig()

CONFIG = Config()
