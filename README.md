# NoiseNER
# Noise-Aware Joint Training for Cybersecurity NER

This repository contains the official PyTorch implementation of the paper:
**"A noise-aware method for low source named entity recognition in cybersecurity"** *(Currently under review at Information Processing & Management, IP&M)*.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-orange.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

## 📌 Introduction
Named Entity Recognition (NER) in the cybersecurity domain suffers from severe label noise and scarce annotation resources. To address this, we propose a novel joint training framework that effectively mitigates the misguidance of label noise while maximizing the utility of noisy samples without relying on external resources.

**Key Features:**
* **Fine-Grained Sample Partitioning:** Employs Kernel Density Estimation (KDE) to accurately isolate Out-of-Distribution (OOD) noise and label-entity prototypes to identify In-Distribution (ID) noise.
* **Semantic Similarity-based LDL:** Recycles discarded ID noise via a novel Label Distribution Learning strategy constrained by contextual semantic similarity.
* **Plug-and-Play Resilience:** Demonstrates robust performance in low-resource settings, outperforming state-of-the-art noise-robust NER baselines.

## 📂 Repository Structure
```text
NoiseNER/
├── data/                    # Sample data for APTNER, CoNLL03, Webpage
├── source/                  # Core implementation of our proposed model
│   ├── main.py              # Entry point for training and evaluation
│   ├── models.py            # Dual-network encoder (ALBERT/ELECTRA)
│   ├── data_processor.py    # Data loading and augmentation module
│   └── loss.py              # Self-supervised, KDE, and LDL loss functions
├── baselines/               # Scripts and replicated code for baseline models
│   ├── NAF_reproduced/      # Our faithful reproduction of NAF using the Luke model
│   └── runner_scripts/      # Shell scripts to run other open-source baselines
├── requirements.txt         # Dependencies
└── README.md
