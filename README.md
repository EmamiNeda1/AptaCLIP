# AptaCLIP

A contrastive learning framework for aptamer–protein interaction prediction using pretrained biological language models.

## Overview

AptaCLIP is a deep learning framework designed to predict aptamer–protein interactions directly from raw sequences using pretrained DNABERT and ESM2 encoders within a contrastive co-embedding architecture.

The model learns a shared latent representation space in which interacting aptamers are positioned closer to their corresponding protein targets while non-binding pairs are separated through triplet contrastive learning.

## Features

- DNABERT-based aptamer encoding
- ESM2-based protein encoding
- Contrastive triplet learning
- Shared embedding space
- Protein-specific generalization
- Gradient-based interpretability analysis

---

## Repository Structure

``` id="uxl4li"
AptaCLIP/
│
├── data/
├── src/
│   ├── train.py
│   ├── model.py
│   └── dataset.py
│
├── requirements.txt
└── README.md