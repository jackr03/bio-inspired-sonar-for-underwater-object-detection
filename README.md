# Bio-Inspired Sonar for Underwater Object Detection

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

Final-year dissertation for the University of Manchester, evaluating **spiking neural networks (SNNs)** as a low-energy alternative to CNNs for **passive underwater vessel classification**. The core question: can spike-driven computation match CNN accuracy on passive sonar at a fraction of the theoretical energy cost?

## Key Findings

- **Direct-coded SNNs give a consistent ~2.4-2.6× energy reduction** over an equivalent CNN, replacing multiply-accumulate operations with spike-driven accumulate-only ones, with accuracy cost ranging from negligible on AudioMNIST to −5.2% on ShipsEar
- **Encoding must match the signal.** Delta modulation (which encodes frame-to-frame change) gives a 70× energy cut on speech but collapses to 52.8% accuracy on quasi-stationary ship noise, discarding the steady-state spectral content that distinguishes vessel classes
- **Learnable firing thresholds** buy accuracy and stability, not energy
- **A more biologically faithful gammatone filterbank made the SNNs worse** (−13.9 to −17.2%) while barely touching the CNN - bio-faithfulness per stage doesn't help if the representations between stages stop matching

## Method

All models share a compact two-block **VGG-style backbone** for a controlled, one-variable-at-a-time comparison, differing only in activation (ReLU vs LIF), normalisation, and input encoding. Variants are found in `src/models/`:

- **`cnn`** - ReLU + BatchNorm baseline, all MACs.
- **`snn_direct`** - LIF neurons, direct coding; first-layer MACs, rest AC.
- **`snn_direct_lt`** - as above with a learnable per-layer firing threshold.
- **`snn`** - delta-modulation encoding, fully AC from the first layer.

Energy is estimated theoretically (Horowitz 45nm CMOS: 4.6 pJ/MAC, 0.9 pJ/AC), with AC counts measured empirically via forward hooks on the LIF layers.

## Datasets

Download from source and place the `.wav` files under `data/<dataset>/`.

- **AudioMNIST**: spoken digits, used as a fast pipeline-validation task. [Source](https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist)
- **ShipsEar**: real hydrophone vessel recordings, five classes, five-fold CV using splits from [Qian et al.](https://www.mdpi.com/2072-4292/17/17/2961). [Source](https://huggingface.co/datasets/peng7554/DS3500)

## Repository Structure

```
src/
  run_process.py     # preprocessing (spectrograms / spike encoding)
  run_train.py       # training + benchmarking
  engine.py          # train / k-fold / benchmark loops
  models/            # cnn, snn_direct, snn_direct_lt, snn (delta)
  datasets/
  utils/
  types/
notebooks/           # per-phase exploration + comparison
hyperparameters/     # tuned configs
jobs/                # SLURM scripts (CSF3)
```

## Running

Requires Python 3.12+ and the dataset `.wav` files in place. GPU recommended.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/requirements.txt      # or requirements-cuda.txt for CUDA

# Preprocess (add --encode for delta-spike tensors instead of spectrograms)
python -m src.run_process --dataset shipsear --filterbank mel

# Train + benchmark
python -m src.run_train --model snn_direct --dataset shipsear --filterbank mel --run-dir runs/snn_direct
```
