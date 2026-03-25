import argparse
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.types.dataset_type import DatasetType
from src.types.filterbank_type import FilterbankType
from src.utils.preprocessing_utils import get_snn_pipeline, get_cnn_pipeline, encode_file, process_file

warnings.filterwarnings('ignore', category=UserWarning)


def run_process(dataset: DatasetType, filterbank: FilterbankType, encode: bool) -> None:
    config = dataset.config

    if encode:
        pipeline = get_snn_pipeline(config, filterbank)
        output_dir = dataset.get_spike_dir(filterbank)
        submit = lambda f: encode_file(f, output_dir, pipeline, config.delta_threshold)
    else:
        pipeline = get_cnn_pipeline(config, filterbank)
        output_dir = dataset.get_spectrogram_dir(filterbank)
        submit = lambda f: process_file(f, output_dir, pipeline)

    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = list(dataset.input_dir.rglob('*.wav'))
    print(f'{'Encoding' if encode else 'Processing'} {len(wav_files)} files for {dataset.value} using {filterbank.value}) filterbanks...')

    num_workers = os.cpu_count()
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(submit, f): f for f in wav_files}
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 500 == 0:
                print(f'{completed}/{len(futures)} files processed...')

    print(f'Done. Time taken: {time.time() - start_time:.0f}s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Preprocessing Entry Point')
    parser.add_argument('--dataset', type=DatasetType, choices=list(DatasetType), required=True)
    parser.add_argument('--filterbank', type=FilterbankType, choices=list(FilterbankType), required=True)
    parser.add_argument('--encode', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    run_process(args.dataset, args.filterbank, args.encode)


if __name__ == '__main__':
    main()