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

AUDIO_CONFIG = AudioConfig()
