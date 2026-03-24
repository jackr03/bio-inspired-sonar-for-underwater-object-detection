import argparse
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torchaudio
from snntorch import spikegen

from src.preprocessing import get_snn_pipeline
from src.types.dataset_type import DatasetType
from src.types.filterbank_type import FilterbankType

warnings.filterwarnings('ignore', category=UserWarning)

def encode(dataset: DatasetType, filterbank: FilterbankType):
    def process_file(file_path: Path, output_dir: Path) -> None:
        try:
            waveform, _ = torchaudio.load(file_path)
            processed = pipeline(waveform)

            # Permute to (Time, Channels, n_mels) as spikegen requires time to be first
            transposed = processed.permute(2, 0, 1)

            spikes = spikegen.delta(transposed, threshold=dataset.config.delta_threshold, off_spike=True)
            save_path = output_dir / f'{file_path.stem}.pt'
            torch.save(spikes.to(torch.float32), save_path)
        except Exception as e:
            print(f'Error processing {file_path}: {e}')

    print(f'Encoding {dataset.value} using {filterbank.value} filterbank...')

    wav_files = list(dataset.input_dir.rglob('*.wav'))
    print(f'Found {len(wav_files)} wav files.')

    pipeline = get_snn_pipeline(dataset.config, filterbank)
    output_dir = dataset.get_spike_dir(filterbank)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_workers = os.cpu_count()
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_file, f, output_dir): f for f in wav_files
        }
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 500 == 0:
                print(f'{completed}/{len(futures)} files processed...')

    print(f'Processing for {dataset.value} done.')
    print(f'Time taken: {time.time() - start_time:.0f}s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Encoding Entry Point')
    parser.add_argument('--dataset', type=DatasetType, choices=list(DatasetType), required=True)
    parser.add_argument('--filterbank', type=FilterbankType, choices=list(FilterbankType), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    encode(args.dataset, args.filterbank)


if __name__ == '__main__':
    main()
