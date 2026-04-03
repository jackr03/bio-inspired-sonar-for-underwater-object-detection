from pathlib import Path

import torch
import torchaudio
from nnAudio.features import Gammatonegram
from snntorch import spikegen
from torch import nn, Tensor

from src.config import AudioConfig
from src.types.filterbank_type import FilterbankType


class DurationNormaliser(nn.Module):
    """Normalises audio duration by truncating / padding to the desired length."""

    def __init__(self, target_samples: int):
        super().__init__()
        self.target_samples = target_samples

    # noinspection PyMethodMayBeStatic
    def forward(self, x: Tensor) -> Tensor:
        num_samples = x.shape[1]

        if num_samples > self.target_samples:
            x = x[:, : self.target_samples]
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


def get_filterbank(
    config: AudioConfig,
    filterbank_type: FilterbankType,
) -> nn.Module:
    match filterbank_type:
        case FilterbankType.MEL:
            return torchaudio.transforms.MelSpectrogram(
                sample_rate=config.target_sample_rate,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=config.n_bins,
                pad_mode='constant',
                norm='slaney',
                mel_scale='slaney',
            )
        case FilterbankType.GAMMATONE:
            return Gammatonegram(
                sr=config.target_sample_rate,
                hop_length=config.hop_length,
                n_fft=config.n_fft,
                n_bins=config.n_bins,
            )
        case _:
            raise NotImplementedError


def get_waveform_transformer(config: AudioConfig) -> nn.Module:
    """
    A preprocessing pipeline that:
    1. Resamples the waveform to the target sample rate
    2. Normalises audio duration by truncating / padding
    """
    return nn.Sequential(
        torchaudio.transforms.Resample(orig_freq=config.original_sample_rate, new_freq=config.target_sample_rate),
        DurationNormaliser(target_samples=config.target_samples),
    )


def get_spectrogram_transformer(config: AudioConfig, filterbank: FilterbankType) -> nn.Module:
    """
    A preprocessing pipeline that:
    1. Converts the waveform to a Mel Spectrogram / Gammatone Spectrogram
    2. Performs log-scaling by dB
    """
    return nn.Sequential(get_filterbank(config, filterbank), torchaudio.transforms.AmplitudeToDB())


def get_cnn_pipeline(config: AudioConfig, filterbank: FilterbankType) -> nn.Module:
    return nn.Sequential(
        get_waveform_transformer(config),
        get_spectrogram_transformer(config, filterbank),
    )


def get_snn_pipeline(config: AudioConfig, filterbank: FilterbankType) -> nn.Module:
    """
    Returns the preprocessing pipeline for SNNs:
    1. Generates the base spectrogram
    2. Normalises dB to [0, 1] in preparation for encoding
    """
    return nn.Sequential(
        get_waveform_transformer(config),
        get_spectrogram_transformer(config, filterbank),
        MinMaxScaler(),
    )


def process_file(file_path: Path, output_dir: Path, pipeline: nn.Module) -> None:
    try:
        waveform, _ = torchaudio.load(file_path)
        spectrogram = pipeline(waveform)
        save_path = output_dir / f'{file_path.stem}.pt'
        torch.save(spectrogram.to(torch.float32), save_path)
    except Exception as e:
        print(f'Error processing {file_path}: {e}')


def encode_file(file_path: Path, output_dir: Path, pipeline: nn.Module, delta_threshold: float) -> None:
    try:
        waveform, _ = torchaudio.load(file_path)
        spectrogram = pipeline(waveform)

        # Permute to (Time, Channels, n_mels) as spikegen requires time to be first
        transposed = spectrogram.permute(2, 0, 1)

        spikes = spikegen.delta(transposed, threshold=delta_threshold, off_spike=True)
        save_path = output_dir / f'{file_path.stem}.pt'
        torch.save(spikes.to(torch.float32), save_path)
    except Exception as e:
        print(f'Error processing {file_path}: {e}')
