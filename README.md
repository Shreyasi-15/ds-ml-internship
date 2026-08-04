# 🎙️ AcousticSpace — AI Deepfake Audio Detection

[![AcousticSpace CI](https://github.com/Shreyasi-15/ds-ml-internship/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Shreyasi-15/ds-ml-internship/actions/workflows/ci.yml)

AcousticSpace is an AI-powered system that checks whether an uploaded audio recording is **bonafide human speech** or **AI-generated spoof audio**.

It uses a fine-tuned **Audio Spectrogram Transformer (AST)** together with supporting acoustic evidence such as reverberation, breathing-band energy, waveform patterns, and suspicious four-second segments.

---

## 📌 Overview

Modern AI-generated voices can sound highly realistic. AcousticSpace helps an analyst examine a recording through an accessible dashboard.

The system provides:

- Fine-tuned Audio Spectrogram Transformer
- FastAPI prediction backend
- React and TypeScript dashboard
- Interactive audio waveform
- Recording-level confidence scores
- Suspicious segment detection
- Acoustic and breathing-cadence evidence
- Analysis history
- Docker deployment
- GitHub Actions CI validation

---

## 🧠 How It Works

1. The user uploads an audio recording.
2. The system validates and converts it to mono audio at 16 kHz.
3. The recording is divided into four-second segments.
4. The trained AST model checks every segment.
5. Segment results are combined into a final prediction.
6. The dashboard displays the result, confidence, waveform, and acoustic evidence.

```text
Audio Upload
     ↓
Audio Validation and Preprocessing
     ↓
Four-Second Segmentation
     ↓
Audio Spectrogram Transformer
     ↓
Prediction and Confidence
     ↓
React Analyst Dashboard
# ds-ml-internship
Data Science &amp; Machine Learning internship projects and assignments completed during the Infotact Solutions internship.
# 🎙️ AcousticSpace — AI Deepfake Audio Detection

[![AcousticSpace CI](https://github.com/Shreyasi-15/ds-ml-internship/actions/workflows/ci.yml/badge.svg?branch=week4)](https://github.com/Shreyasi-15/ds-ml-internship/actions/workflows/ci.yml)

AcousticSpace is an AI-powered system that checks whether an uploaded audio recording is **bonafide human speech** or **AI-generated spoof audio**.

It uses a fine-tuned **Audio Spectrogram Transformer (AST)** together with supporting acoustic evidence such as reverberation, breathing-band energy, waveform patterns, and suspicious four-second segments.

---

## 📌 Overview

Modern AI-generated voices can sound highly realistic. AcousticSpace helps an analyst examine a recording through an accessible dashboard.

The system provides:

- Fine-tuned Audio Spectrogram Transformer
- FastAPI prediction backend
- React and TypeScript dashboard
- Interactive audio waveform
- Recording-level confidence scores
- Suspicious segment detection
- Acoustic and breathing-cadence evidence
- Analysis history
- Docker deployment
- GitHub Actions CI validation

---

## 🧠 How It Works

1. The user uploads an audio recording.
2. The system validates and converts it to mono audio at 16 kHz.
3. The recording is divided into four-second segments.
4. The trained AST model checks every segment.
5. Segment results are combined into a final prediction.
6. The dashboard displays the result, confidence, waveform, and acoustic evidence.

```text
Audio Upload
     ↓
Audio Validation and Preprocessing
     ↓
Four-Second Segmentation
     ↓
Audio Spectrogram Transformer
     ↓
Prediction and Confidence
     ↓
React Analyst Dashboard
```