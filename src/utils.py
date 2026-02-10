import numpy as np


def get_random_audio(base_dir: str) -> str:
    """
    Randomly select a digit, speaker and sample and return its path.
    """
    digit = np.random.randint(0, 10)
    speaker = np.random.randint(1, 61)
    sample = np.random.randint(0, 50)

    if speaker < 10:
        return f'{base_dir}/0{speaker}/{digit}_0{speaker}_{sample}.wav'
    else:
        return f'{base_dir}/{speaker}/{digit}_{speaker}_{sample}.wav'
