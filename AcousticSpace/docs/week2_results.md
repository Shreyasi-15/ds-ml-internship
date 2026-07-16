# AcousticSpace — Week 2 Results

## 1. Week 2 Objective

The goal of Week 2 was to build and evaluate an initial deepfake-audio classification model and add an interactive waveform viewer to the frontend.

Week 2 focused on:

- Preparing ASVspoof 2019 dataset manifests.
- Converting audio recordings into log-mel spectrograms.
- Training a baseline CNN classifier.
- Evaluating the classifier on the official evaluation split.
- Displaying uploaded audio as an interactive waveform.

## 2. Dataset

The baseline model uses the **ASVspoof 2019 Logical Access (LA)** dataset.

The full dataset is stored outside the GitHub repository at:

```text
C:\Users\shrey\Datasets\ASVspoof2019\LA