import torch
import torchaudio
from nnAudio.features import Gammatonegram
from torch import nn, Tensor

from src.config import CONFIG


class DurationNormaliser(nn.Module):
    """Normalises audio duration by truncating / padding to the desired length."""
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
    """Clamps decibel values between the max and min values provided before normalising to return a value in [0, 1]."""
    def __init__(self, min_val=-80.0, max_val=0.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: Tensor) -> Tensor:
        x = torch.clamp(x, self.min_val, self.max_val)
        return (x - self.min_val) / (self.max_val - self.min_val)

def get_waveform_transformer() -> nn.Module:
    """
    A preprocessing pipeline that:
    1. Resamples the waveform to 16000Hz
    2. Normalises audio duration by truncating / padding
    """
    return nn.Sequential(
        torchaudio.transforms.Resample(
            orig_freq=CONFIG.audio.original_sample_rate,
            new_freq=CONFIG.audio.target_sample_rate
        ),
        DurationNormaliser(target_samples=CONFIG.audio.target_samples),
    )

# TODO: Higher bins for Gammatone?
def get_gammatone_filterbank() -> nn.MOdule:
    return Gammatonegram(
        sr=CONFIG.audio.target_sample_rate,
        n_fft=CONFIG.audio.n_fft,
        n_bins=CONFIG.audio.n_mels,
    )

def get_mel_filterbank() -> nn.Module:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=CONFIG.audio.target_sample_rate,
        n_fft=CONFIG.audio.n_fft,
        n_mels=CONFIG.audio.n_mels,
        pad_mode='constant',
        norm='slaney',
        mel_scale='slaney'
    )

def get_spectrogram_transformer(filterbank: str) -> nn.Module:
    """
    A preprocessing pipeline that:
    1. Converts the waveform to a Mel Spectrogram / Gammatone Spectrogram
    2. Performs log-scaling by dB
    """
    filterbank = get_mel_filterbank() if filterbank == 'mel' else get_gammatone_filterbank()
    return nn.Sequential(
        filterbank,
        torchaudio.transforms.AmplitudeToDB()
    )

def get_cnn_pipeline(filterbank: str = 'mel') -> nn.Module:
    return nn.Sequential(
        get_waveform_transformer(),
        get_spectrogram_transformer(filterbank),
    )

def get_snn_pipeline(filterbank: str = 'mel') -> nn.Module:
    """
    Returns the preprocessing pipeline for SNNs:
    1. Generates the base spectrogram
    2. Normalises dB to [0, 1] in preparation for encoding
    """
    return nn.Sequential(
        get_waveform_transformer(),
        get_spectrogram_transformer(filterbank),
        MinMaxScaler()
    )
