import torch
import torchaudio
from snntorch import spikegen
from torch import nn, Tensor

from src.config import AUDIO_CONFIG


class DurationNormaliser(nn.Module):
    """
    Normalises audio duration by truncating / padding to the desired length.
    """
    def __init__(self, target_samples: float):
        super().__init__()
        self.target_samples = target_samples

    # noinspection PyMethodMayBeStatic
    def forward(self, x: Tensor) -> Tensor:
        num_samples = x.shape[1]

        if num_samples > self.target_samples:
            x = x[:, :self.target_samples]
        elif num_samples < self.target_samples:
            padding_needed = self.target_samples - num_samples
            x = nn.functional.pad(x, (0, padding_needed))

        return x

class MinMaxScaler(nn.Module):
    """
    Clamps decibel values between the max and min values provided before normalising to return a value in [0, 1].
    """
    def __init__(self, min_val=-80.0, max_val=0.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: Tensor) -> Tensor:
        x = torch.clamp(x, self.min_val, self.max_val)
        return (x - self.min_val) / (self.max_val - self.min_val)

class RateEncoder(nn.Module):
    def __init__(self, num_steps: int):
        super().__init__()
        self.num_steps = num_steps

    def forward(self, x: Tensor) -> Tensor:
        return spikegen.rate(x, num_steps=self.num_steps)

def get_waveform_transformer() -> nn.Module:
    """
    A preprocessing pipeline that:
    1. Resamples the waveform to 16000Hz
    2. Normalises audio duration by truncating / padding
    """
    return nn.Sequential(
        torchaudio.transforms.Resample(
            orig_freq=AUDIO_CONFIG.original_sample_rate,
            new_freq=AUDIO_CONFIG.target_sample_rate
        ),
        DurationNormaliser(target_samples=AUDIO_CONFIG.target_samples),
    )

def get_spectrogram_transformer() -> nn.Module:
    """
    A preprocessing pipeline that:
    1. Converts the waveform to a Mel Spectrogram
    2. Performs log-scaling by dB
    """
    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=AUDIO_CONFIG.target_sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=AUDIO_CONFIG.n_mels,
            power=2.0,
            pad_mode='constant',
            norm='slaney',
            mel_scale='slaney'
        ),
        torchaudio.transforms.AmplitudeToDB()
    )

def get_cnn_pipeline() -> nn.Module:
    """
    Returns the preprocessing pipeline for CNNs.
    """
    return nn.Sequential(
        get_waveform_transformer(),
        get_spectrogram_transformer(),
    )

# TODO: Implement other encoding types
def get_snn_pipeline() -> nn.Module:
    """
    Returns the preprocessing pipeline for SNNs:
    1. Generates the base spectrogram
    2. Normalises dB to [0, 1]
    3. Encodes with the specified encoding type.
    """
    return nn.Sequential(
        get_waveform_transformer(),
        get_spectrogram_transformer(),
        MinMaxScaler(min_val=-80.0, max_val=0.0),
        RateEncoder(num_steps=100)
    )
