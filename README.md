# BIM-DTI

**BIM-DTI: Drug–Target Binding Affinity Prediction with Bidirectional Multi-Path Interaction Learning**

This repository contains the source code and datasets for BIM-DTI, a drug–target interaction prediction model based on sequence modeling and bidirectional interaction learning.

## 1. Environment and Dependencies

The model is implemented in PyTorch.

- Python 3.10+
- PyTorch
- NumPy, pandas, scikit-learn
- tqdm, tensorboardX, TensorBoard
- RDKit, Transformers, prefetch_generator

**Hardware:** All experiments were conducted on an NVIDIA RTX 5090 GPU with CUDA 12.8.

### Installation

Create a Python environment and install the dependencies:

```bash
conda create -n bmi-dti python=3.10 -y
conda activate bmi-dti

python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy pandas scikit-learn tqdm tensorboardX tensorboard rdkit transformers prefetch_generator
```

## 2. Datasets

The datasets are stored in the `datasets/` directory.

- `Bank.txt` — DrugBank
- `Davis.txt` — Davis
- `KIBA.txt` — KIBA

Each sample contains five fields: drug ID, protein ID, drug SMILES, protein amino acid sequence, and binary label.

Example:

```text
Drug_ID  Protein_ID  Drug_SMILES  Amino_acid_sequence  label
CHEMBL... 000141     COC1=...     MTVKTE...            1
```

The example is illustrative. Use the prepared data files and ensure that the final field contains the appropriate class label (`0` or `1`). Do not add a header row if using the current data loader.

Drug sequences are limited to 100 characters, and protein sequences to 1,000 characters. Shorter sequences are padded, while longer sequences are truncated.

## 3. Training

Run the commands from the directory containing `DTI_main.py`.

```bash
# DrugBank
python DTI_main.py --dataset Bank --data_path ./datasets/

# Davis
python DTI_main.py --dataset Davis --data_path ./datasets/

# KIBA
python DTI_main.py --dataset KIBA --data_path ./datasets/
```

The script uses five-fold cross-validation by default. To view the available command-line arguments, run:

```bash
python DTI_main.py --help
```

## 4. Main Hyperparameters

| Parameter | Default | Description |
|---|---:|---|
| Learning rate | 0.0001 | Optimizer learning rate |
| Batch size | 64 | Training batch size |
| Epochs | 50 | Maximum training epochs |
| Patience | 25 | Early-stopping patience |
| Folds | 5 | Cross-validation folds |
| Random seed | 4321 | Random seed |
| Dropout | 0.1 | Dropout rate |

These values correspond to the defaults in the current training script. Modify the parameters when necessary to match the experimental configuration.

## 5. Outputs and Evaluation

Training logs, model checkpoints, and prediction results are saved under the `Results/` directory.

Model performance is evaluated using five metrics:

- Accuracy (Acc)
- Area Under the ROC Curve (AUC)
- Area Under the Precision–Recall Curve (AUPR)
- Precision
- Recall

The training logs contain the evaluation results for each fold. The mean and standard deviation across folds can be calculated for reporting.

