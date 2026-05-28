# BMI-DTI-main

BMI-DTI: Drug–Target Binding Affinity Prediction with Bidirectional Multi-Path Interaction Learning

This repository contains the source code and datasets for drug–target binding affinity prediction based on sequence modeling and bidirectional interaction learning.

---

## 1. Environment and Dependencies

The model is implemented in **PyTorch**.

### Dependencies
- Python 3.10+  
- PyTorch  
- numpy  
- scikit-learn  
- tqdm  
- tensorboardX  
- pandas  

### Hardware
All experiments were conducted on an **NVIDIA RTX 5090 GPU** with **CUDA 12.8**.

---

## 2. Datasets

The datasets used in this project are stored in the `datasets/` directory.

Supported datasets:
- `Davis.txt`
- `KIBA.txt`
- `Metz.txt`

Each sample contains the following fields:

- `Drug_ID`
- `Protein_ID`
- `Drug_SMILES`
- `Amino_acid_sequence`
- `affinity`

Example format:

```text
Drug_ID    Protein_ID    Drug_SMILES    Amino_acid_sequence    affinity
CHEMBL...  000141        COC1=...       MTVKTE...              11.1