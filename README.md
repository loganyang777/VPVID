# VPVID: Variance-Preserving Velocity-Guided Interpolant Diffusion for Speech Enhancement and Dereverberation

## Overview

VPVID (Variance-Preserving Velocity-guided Interpolant Diffusion) is a novel framework for speech enhancement and dereverberation that achieves competitive enhancement performance while maintaining high computational efficiency. This repository contains the official implementation of the paper "VPVID: Variance-Preserving Velocity-Guided Interpolant Diffusion for Speech Enhancement and Dereverberation".

🎧 **[Demo Page](https://loganyang777.github.io/VPVID-demo/)** - Listen to audio samples and compare results

## Acknowledgments

This work is built upon the excellent [SGMSE repository](https://github.com/sp-uhh/sgmse). We thank the authors for their foundational work and open-source contributions to the speech enhancement community. The preprocessing scripts in this repository are adapted from their original implementation.

**Important Note**: The model currently supports only 16kHz audio as input.

## Key Features

- **Scalable Interpolant Framework**: Reconstructs the reverse diffusion process using velocity terms and state variables
- **Velocity-Based Loss Function**: Directly estimates the instantaneous rate of change for more stable training
- **Hybrid Sampling Strategy**: Combines stochastic diffusion sampling with probability flow ODEs
- **Adaptive Corrector Mechanism**: Provides flexible sampling that balances quality and efficiency
- **High Performance**: Achieves up to 4.7 dB SI-SIR improvement and 7× faster inference than existing diffusion-based methods

## Installation

### Prerequisites

- Python 3.11
- CUDA 11.8+ (for GPU support)
- PyTorch 2.3.1+

## Usage

### Training

**Note**: Training scripts and parameters will be released after the paper is accepted for publication.

```bash
# Training scripts will be available soon
```

### Inference

We recommend the following configurations for different inference modes:

#### 1. VPVID-SDE (Stochastic Differential Equation)
```bash
python enhancement.py --test_dir <noisy_data_dir> \
                     --enhanced_dir <output_dir> \
                     --ckpt <path_to_checkpoint> \
                     --corrector_steps=0 \
                     --t_eps=0.10 \
                     --N=10
```

#### 2. VPVID-SDEC (SDE with Corrector)
```bash
python enhancement.py --test_dir <noisy_data_dir> \
                     --enhanced_dir <output_dir> \
                     --ckpt <path_to_checkpoint> \
                     --corrector_steps=1 \
                     --snr=0.5 \
                     --t_eps=0.06 \
                     --N=15
```

#### 3. VPVID-ODE (Ordinary Differential Equation)
```bash
python enhancement.py --test_dir <noisy_data_dir> \
                     --enhanced_dir <output_dir> \
                     --ckpt <path_to_checkpoint> \
                     --corrector_steps=0 \
                     --probability_flow \
                     --t_eps=0.09 \
                     --N=8
```

### Pre-trained Models

We provide pre-trained checkpoints for three datasets:

📁 **[Download Pre-trained Models](https://drive.google.com/drive/folders/1wjwtiBk8CF3kO43LuBbhDsiis1_iQ0x0)**

- VoiceBank-DEMAND dataset checkpoint
- WSJ0-CHiME3 dataset checkpoint  
- WSJ0-REVERB dataset checkpoint

## Repository Structure

```
VPVID/
├── sgmse/                 # Core model implementation
│   ├── backbones/         # Neural network architectures
│   ├── sampling/          # Sampling algorithms (predictors & correctors)
│   └── util/              # Utility functions
├── preprocessing/         # Data preprocessing scripts
├── enhancement.py        # Inference script
├── calc_metrics.py       # Metrics calculation
├── evaluate.py           # Evaluation utilities
└── requirements.txt      # Python dependencies
```

## Contact

For questions or issues, please open an issue in this repository.