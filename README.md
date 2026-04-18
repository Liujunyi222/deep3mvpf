# Deep3MVPF

Code and data for **Deep3MVPF**, a deep learning framework for RNA degradation prediction and m6A-related sequence-structure analysis.

This repository currently contains two task families:

- **Regression task**: RNA / 3'UTR degradation prediction
- **Classification task**: m6A site prediction with **cell-line-specific 10-fold training**

## Repository structure

```text
Deep3MVPF/
├── data/
│   ├── GSE106677_sequences.fasta
│   ├── output_structure.txt
│   ├── degradation.xlsx
│   ├── m6a_final_data_3utr/              
│   │   ├── train/
│   │   │   ├── A549/
│   │   │   │   ├── A549_train_data_with_struct.tsv
│   │   │   │   ├── A549_train_fold0.fa
│   │   │   │   ├── ...
│   │   │   │   └── A549_train_fold9.fa
│   │   │   └── <other cell lines>/
│   │   ├── validation/
│   │   └── test/
│   └── README.md
├── src/
│   ├── data_utils.py
│   ├── model.py
│   └── plot_utils.py
├── scripts/
│   ├── train_ablation.py
│   ├── train_m6a.py
│   ├── train_m6a_ablation.py
│   ├── analyze_gradient.py
│   ├── analyze_structure.py
│   └── m6A_interpret.py
├── checkpoints/
├── results/
├── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

## Data organization

### 1) Regression / degradation task

The regression task uses:

- `data/GSE106677_sequences.fasta`
- `data/output_structure.txt`
- `data/degradation.xlsx`

### 2) m6A task

The m6A pipeline expects a **pre-split fold dataset**. For each split (`train`, `validation`, `test`), and for each of the 9 cell lines
`A549, CD8T, ESC, HCT116, HEK293, HEK293T, Hela, HepG2, MOLM13`, the folder should contain:

- one structure table: `CELL_split_data_with_struct.tsv`
- ten fold FASTA files: `CELL_split_fold0.fa` to `CELL_split_fold9.fa`

Example:

```text
data/m6a_final_data_3utr/test/A549/
├── A549_test_data_with_struct.tsv
├── A549_test_fold0.fa
├── A549_test_fold1.fa
├── ...
└── A549_test_fold9.fa
```

The current scripts will automatically look for the m6A dataset in the following order:

1. `data/m6a_final_data_3utr/`
2. `m6a_final_data_3utr/`
3. the legacy placeholder path

So if you put the folder under `data/m6a_final_data_3utr/`, you usually **do not need to modify the scripts again**.

## Python environment

Recommended: **Python 3.10 or 3.11**.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Notes on PyTorch / PyG

This project depends on **PyTorch** and **PyTorch Geometric**. They are now included in `requirements.txt`, but on some CUDA environments you may still prefer to install them from the official instructions first and then run the rest of the requirements.

The m6A pipeline may also call **RNAfold** when a structure is missing from the TSV files, so installing **ViennaRNA** is recommended.

## Quick start

### 1) Regression training

```bash
python scripts/train_ablation.py --exp_name Full_Model
```

Useful ablations:

```bash
python scripts/train_ablation.py --exp_name no_structure --no_struct
python scripts/train_ablation.py --exp_name no_debruijn --no_debruijn
python scripts/train_ablation.py --exp_name cnn_only --cnn_only
python scripts/train_ablation.py --exp_name single_scale --single_scale
```

### 2) Regression interpretation

```bash
python scripts/analyze_gradient.py --checkpoint checkpoints/ablation/best_model_Full_Model.pth
python scripts/analyze_structure.py --checkpoint checkpoints/ablation/best_model_Full_Model.pth
```

### 3) m6A 10-fold training

If your dataset is placed at `data/m6a_final_data_3utr/`, you can run:

```bash
python scripts/train_m6a.py
```

Run a single fold only:

```bash
python scripts/train_m6a.py --fold 0
```

If the dataset is stored elsewhere:

```bash
python scripts/train_m6a.py --root_path /path/to/m6a_final_data_3utr
```

### 4) m6A ablation experiments

```bash
python scripts/train_m6a_ablation.py --mode full
python scripts/train_m6a_ablation.py --mode no_debruijn
python scripts/train_m6a_ablation.py --mode no_structure
python scripts/train_m6a_ablation.py --mode cnn_only
```

### 5) m6A interpretability analysis

```bash
python scripts/m6A_interpret.py
```
