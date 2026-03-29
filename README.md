# NoiseNER

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-orange.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

## 📌 Introduction
Named Entity Recognition (NER) in the cybersecurity domain suffers from severe label noise and scarce annotation resources. To address this, we propose a novel joint training framework that effectively mitigates the misguidance of label noise while maximizing the utility of noisy samples without relying on external resources.

**Key Features:**
* **Fine-Grained Sample Partitioning:** Employs Kernel Density Estimation (KDE) to accurately isolate Out-of-Distribution (OOD) noise and label-entity prototypes to identify In-Distribution (ID) noise.
* **Semantic Similarity-based LDL:** Recycles discarded ID noise via a novel Label Distribution Learning strategy constrained by contextual semantic similarity.
* **Plug-and-Play Resilience:** Demonstrates robust performance in low-resource settings, outperforming state-of-the-art noise-robust NER baselines.

## 📊 Baselines & Reproducibility

To ensure a strictly fair and rigorous comparison as stated in our manuscript, we adopted a hybrid implementation strategy for the baselines:

### 1. Reproduced Baselines (Code provided in this repo)
Since the official source code for **NAF** was unavailable, we have faithfully reproduced it using the `Luke` model backbone as described in their original paper. The complete reproduced code is available in our repository:
* 📂 **NAF**: Available in `baselines/NAF_reproduced/`

### 2. Open-Source Baselines (Links to official repos)
For baselines with officially released code, we deliberately retained their original pre-trained backbones and optimal configurations. To run these models on our datasets, we provide the data-formatting scripts and hyperparameter runner scripts in `baselines/runner_scripts/`. You can find their official source code below:

| Baseline | Official Repository Link |
| :--- | :--- |
| **Co-teaching** | [https://github.com/bhanML/Co-teaching](https://github.com/bhanML/Co-teaching) |
| **RoSTER** | [https://github.com/yumeng5/RoSTER](https://github.com/yumeng5/RoSTER) |
| **BOND** | [https://github.com/cliang1453/BOND](https://github.com/cliang1453/BOND) |
| **STGN** | [https://github.com/wutong8023/STGN](https://github.com/wutong8023/STGN) |
| **MSR** | [https://github.com/z10is1an/Meta-Self-Refinement](https://github.com/z10is1an/Meta-Self-Refinement) |
| **CuPUL** | [https://github.com/liyp0095/CuPUL](https://github.com/liyp0095/CuPUL) |
| **CENSOR** | [https://github.com/PKUnlp-icler/CENSOR](https://github.com/PKUnlp-icler/CENSOR) |
| **DS-NER** |[https://github.com/ yumeng5/RoSTER.](https://github.com/yyDing1/DS-NER)|
| **DEER** | [https://github.com/bflashcp3f/deer](https://github.com/bflashcp3f/deer) |


## 📂 Repository Structure
```text
NoiseNER/
├── data/                    # Sample data for APTNER, CoNLL03, Webpage
├── source/                  # Core implementation of our proposed model
│   ├── main.py              # Entry point for training and evaluation
│   ├── models.py            # Dual-network encoder 
│   ├── data_processor.py    # Data loading and augmentation module
│   └── loss.py              # Self-supervised, KDE, and LDL loss functions
├── ├── NAF_reproduced/      # Our faithful reproduction of NAF using the Luke model
├── requirements.txt         # Dependencies
└── README.md
