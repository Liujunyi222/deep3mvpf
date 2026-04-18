from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
M6A_DATA_DIR_CANDIDATES = [
    DATA_DIR / "m6a_final_data_3utr",
    REPO_ROOT / "m6a_final_data_3utr",
    REPO_ROOT / "m6a_data_placeholder",
]
M6A_DATA_DIR = next((path for path in M6A_DATA_DIR_CANDIDATES if path.exists()), M6A_DATA_DIR_CANDIDATES[0])


def ensure_dir(path: os.PathLike | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(log_path: os.PathLike | str, resume: bool = False) -> Callable[[str], None]:
    log_path = Path(log_path)
    ensure_dir(log_path.parent)
    if not resume and log_path.exists():
        log_path.unlink()

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return log


def load_regression_data(seq_path, struct_path, deg_path, logger=None):
    logger = logger or print
    logger(f"Loading sequences from {seq_path}")

    seq_dict = {}
    with open(seq_path, "r", encoding="utf-8") as f:
        current_seq, current_id = [], None
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    seq_dict[current_id] = "".join(current_seq)
                current_id = line[1:].strip().split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_id:
            seq_dict[current_id] = "".join(current_seq)

    struct_dict = {}
    with open(struct_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(">"):
            sid = line[1:].strip().split()[0]
            content = ""
            if i + 2 < len(lines) and re.search(r"[.()]+", lines[i + 2]):
                m = re.search(r"([.()]+)", lines[i + 2])
                content = m.group(1) if m else lines[i + 2]
                i += 3
            elif i + 1 < len(lines):
                content = lines[i + 1]
                i += 2
            else:
                i += 1
            struct_dict[sid] = content
        else:
            i += 1

    df = pd.read_excel(deg_path, dtype={0: str})
    ids = df.iloc[:, 0].astype(str).str.strip()
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    deg_dict = dict(zip(ids, vals))

    matched_sequences, matched_structures, matched_rates = [], [], []
    logger(f"Aligning {len(seq_dict)} sequences by ID")
    for seq_id in seq_dict:
        if seq_id in struct_dict and seq_id in deg_dict:
            rate = deg_dict[seq_id]
            if rate is None or pd.isna(rate):
                continue
            matched_sequences.append(seq_dict[seq_id])
            matched_structures.append(struct_dict[seq_id])
            matched_rates.append(float(rate))

    logger(f"Final matched samples: {len(matched_sequences)}")
    return matched_sequences, matched_structures, matched_rates


def load_regression_data_as_records(seq_path, struct_path, deg_path):
    seqs, structs, rates = load_regression_data(seq_path, struct_path, deg_path)
    return [
        {"sequence": s, "structure": st, "rate": r}
        for s, st, r in zip(seqs, structs, rates)
    ]


def get_rna_structure_fallback(sequence: str) -> str:
    try:
        process = subprocess.Popen(
            ["RNAfold", "--noPS"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, _ = process.communicate(input=sequence)
        if process.returncode != 0:
            return "." * len(sequence)
        parts = stdout.strip().split("\n")
        if len(parts) >= 2:
            return parts[1].split()[0]
        return "." * len(sequence)
    except Exception:
        return "." * len(sequence)


def load_all_structures_to_dict(root_path, cell_line):
    root_path = Path(root_path)
    struct_map = {}
    for split in ["train", "test", "validation"]:
        fname = f"{cell_line}_{split}_data_with_struct.tsv"
        fpath = root_path / split / cell_line / fname
        if fpath.exists():
            df = pd.read_csv(fpath, sep="\t")
            if "sequence" in df.columns and "structure" in df.columns:
                for seq, stru in zip(df["sequence"], df["structure"]):
                    if isinstance(seq, str) and isinstance(stru, str):
                        struct_map[seq.strip()] = stru.strip()
    return struct_map


def load_m6a_fold_data(root_path, cell_line, split_type, fold_idx, struct_map, dataset_cls, max_len=None):
    root_path = Path(root_path)
    folder = root_path / split_type / cell_line
    file_path = folder / f"{cell_line}_{split_type}_fold{fold_idx}.fa"
    if not file_path.exists():
        alt = folder / f"{cell_line}_{split_type}_fold{fold_idx}.fasta"
        file_path = alt if alt.exists() else file_path
    if not file_path.exists():
        return None

    sequences, structures, labels = [], [], []
    with file_path.open("r", encoding="utf-8") as f:
        header, seq_lines = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    seq = "".join(seq_lines)
                    structures.append(struct_map.get(seq, get_rna_structure_fallback(seq)))
                    sequences.append(seq)
                    parts = header.split()
                    label = float(parts[-1]) if len(parts) > 1 and parts[-1] in {"0", "1"} else 0.0
                    labels.append(label)
                header, seq_lines = line[1:], []
            else:
                seq_lines.append(line)
        if header:
            seq = "".join(seq_lines)
            structures.append(struct_map.get(seq, get_rna_structure_fallback(seq)))
            sequences.append(seq)
            parts = header.split()
            label = float(parts[-1]) if len(parts) > 1 and parts[-1] in {"0", "1"} else 0.0
            labels.append(label)

    return dataset_cls(sequences, structures, labels, max_seq_len=max_len)
