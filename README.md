# ASR-Aided-Multi-Speaker-Speech-Separation

## Overview
[cite_start]This repository contains the implementation and documentation for a speech separation pipeline that integrates classical beamforming theory (MVDR) with modern deep learning-based separation (SepFormer)[cite: 1, 4]. [cite_start]The project compares linear and spherical microphone array configurations in simulated reverberant environments[cite: 4, 206]. [cite_start]The pipeline includes signal simulation, source separation, and evaluation using Automatic Speech Recognition (ASR) metrics (WER) and signal quality metrics (SI-SNR, PESQ)[cite: 51, 59].

## Table of Contents
1. Theory and Signal Model
2. Installation and Requirements
3. Project Structure
4. Pipeline 1: Linear Array Separation
5. Pipeline 2: Spherical Array Separation
6. Evaluation Metrics
7. Suggested Improvements

## Theory and Signal Model

### MVDR Beamforming
[cite_start]The Minimum Variance Distortionless Response (MVDR) beamformer is used to minimize output power (interference and noise) while maintaining a distortionless response for the desired look direction[cite: 19, 32].

**Signal Model**
[cite_start]For an M-element microphone array, the narrowband snapshot is modeled as[cite: 11, 12]:
$$x(t) = a(\theta_0)s(t) + n(t)$$
Where:
* [cite_start]$a(\theta_0)$ is the steering vector for direction $\theta_0$[cite: 13].
* [cite_start]$s(t)$ is the desired source[cite: 13].
* [cite_start]$n(t)$ is the interference and noise[cite: 14].

**Optimization**
[cite_start]The weights $w$ are chosen to minimize $w^H R w$ subject to $w^H a(\theta_0) = 1$, where $R$ is the covariance matrix [cite: 20-23]. [cite_start]The closed-form solution is[cite: 26]:
$$w_{MVDR} = \frac{R^{-1}a(\theta_0)}{a^H(\theta_0)R^{-1}a(\theta_0)}$$

### SepFormer (Transformer-Based Separation)
[cite_start]SepFormer is a time-domain separation approach that utilizes a dual-path transformer architecture[cite: 34].
* [cite_start]**Encoder:** A 1-D convolution maps the raw waveform into encoded frames[cite: 38].
* [cite_start]**Dual-Path Processing:** The model splits the input into chunks and applies intra-chunk transformers (short-term dependencies) and inter-chunk transformers (long-term dependencies)[cite: 42, 43].
* [cite_start]**Decoder:** A transposed convolution reconstructs the time waveform from the masked encoder output[cite: 46, 47].

[cite_start]*Note: The SepFormer implementation in SpeechBrain typically expects an 8 kHz sampling rate, requiring resampling within the pipeline[cite: 50].*

## Installation and Requirements

[cite_start]The project relies on `numpy`, `scipy`, `pyroomacoustics`, `torch`, `torchaudio`, `speechbrain`, and `openai-whisper` [cite: 64-67].

To install the necessary dependencies:

```bash
# System requirements
sudo apt update && sudo apt install -y ffmpeg

# Python libraries
pip install numpy scipy soundfile pyroomacoustics
pip install torch torchaudio
pip install speechbrain
pip install q jiwer matplotlib
pip install -q git+[https://github.com/openai/whisper.git](https://github.com/openai/whisper.git)
