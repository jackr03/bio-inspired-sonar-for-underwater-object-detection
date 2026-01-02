import torchaudio
from torch import nn, Tensor

# TODO: Move these constants out to a YAML file
# Constants
ORIGINAL_SAMPLE_RATE = 48_000
TARGET_SAMPLE_RATE = 16_000
TARGET_AUDIO_LENGTH = 0.84
TARGET_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_AUDIO_LENGTH)
NUM_MELS = 64

def get_transformation_pipeline() -> nn.Module:
    """
    A preprocessing pipeline for audio transformation that:
    1. Resamples the waveform to 16,000 Hz
    2. Fixes the audio duration by truncating / padding
    3. Converts to a Mel Spectrogram
    4. Performs log-scaling by dB
    """
    return nn.Sequential(
        torchaudio.transforms.Resample(
            orig_freq=ORIGINAL_SAMPLE_RATE,
            new_freq=TARGET_SAMPLE_RATE),
        FixAudioLength(),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=NUM_MELS,
            power=2.0,
            pad_mode='constant',
            norm='slaney',
            mel_scale='slaney'
        ),
        torchaudio.transforms.AmplitudeToDB()
    )

class FixAudioLength(nn.Module):
    """
    Fixes the audio duration by truncating / padding to the desired length.
    """
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, waveform: Tensor) -> Tensor:
        num_samples = waveform.shape[1]

        if num_samples > TARGET_SAMPLES:
            waveform = waveform[:, :TARGET_SAMPLES]
        elif num_samples < TARGET_SAMPLES:
            padding_needed = TARGET_SAMPLES - num_samples
            waveform = nn.functional.pad(waveform, (0, padding_needed))

        return waveform
