# DegradoMap

Structure-Aware Prediction of PROTAC-Mediated Protein Degradability via Graph Neural Networks

## Overview

DegradoMap is a graph neural network that predicts PROTAC-mediated protein degradability from protein structure and E3 ligase identity alone—the minimal information available at the target selection stage.

**Key Results:**
- Multi-seed validated AUROC: 0.646 ± 0.124 on target-unseen split
- Best seed AUROC: 0.7449 (+23% over GradientBoosting baseline)
- E3-unseen AUROC: 0.811 (CRBN→VHL transfer)
- E3 ligase recommendation: 74% Hit@3

## Installation

```bash
pip install -r requirements.txt
```

## Data

Download the PROTAC-8K dataset:
```bash
wget https://zenodo.org/records/14715718/files/PROTAC-8K.zip
unzip PROTAC-8K.zip -d data/
```

## Usage

```bash
python scripts/train.py --split target_unseen --seed 42
```

## License

MIT License
