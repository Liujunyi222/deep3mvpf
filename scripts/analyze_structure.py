from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum.attr import IntegratedGradients
from torch_geometric.data import Batch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_utils import CHECKPOINT_DIR, DATA_DIR, FIGURES_DIR, set_seed
from src.model import UTRDataset, UTRDegradationPredictor
from src.plot_utils import setup_basic_plot_style


def load_data(seq_path, struct_path, deg_path):
    seq_dict = {}
    with open(seq_path, "r", encoding="utf-8") as f:
        current_seq, current_id = [], None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    seq_dict[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            seq_dict[current_id] = "".join(current_seq)

    struct_dict = {}
    with open(struct_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            sid = lines[i][1:].split()[0]
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

    aligned = []
    for sid, seq in seq_dict.items():
        if sid in struct_dict and sid in deg_dict and not np.isnan(deg_dict[sid]):
            aligned.append({"id": sid, "sequence": seq, "structure": struct_dict[sid], "rate": deg_dict[sid]})
    return aligned


def structure_forward_wrapper(structure_x_perturbed, sequence_batch, debruijn_batch, base_structure_batch, model):
    outputs = []
    for i in range(structure_x_perturbed.size(0)):
        x_step = structure_x_perturbed[i]
        step_structure_graph = base_structure_batch.clone()
        step_structure_graph.x = x_step
        outputs.append(model(sequence_batch, debruijn_batch, step_structure_graph))
    return torch.stack(outputs)


def compute_global_structure_attributions(model, dataset, indices, device):
    model.eval()
    ig = IntegratedGradients(structure_forward_wrapper)
    collected_data = []
    idx_to_nuc = {v: k for k, v in dataset.nucleotide_dict.items()}
    idx_to_struct = {v: k for k, v in dataset.structure_dict.items()}

    for idx in tqdm(indices, desc="Analyzing structure"):
        try:
            data_item = dataset[idx]
            seq_input = data_item["sequence"].unsqueeze(0).to(device)
            deb_input = Batch.from_data_list([data_item["debruijn_graph"]]).to(device)
            str_input = Batch.from_data_list([data_item["structure_graph"]]).to(device)
            input_x = str_input.x.unsqueeze(0).clone().requires_grad_()

            attributions = ig.attribute(
                inputs=input_x,
                additional_forward_args=(seq_input, deb_input, str_input, model),
                n_steps=15,
                internal_batch_size=1,
                return_convergence_delta=False,
            )
            attr = attributions.squeeze(0).cpu().detach()
            seq_scores = attr[:, :5].sum(dim=1).numpy()
            struct_scores = attr[:, 5:].sum(dim=1).numpy()
            node_features = str_input.x.cpu().numpy()
            seq_indices = np.argmax(node_features[:, :5], axis=1)
            struct_indices = np.argmax(node_features[:, 5:], axis=1)

            for i in range(len(seq_scores)):
                nuc_char = idx_to_nuc.get(seq_indices[i], "N")
                if nuc_char in ["N", 4]:
                    continue
                if nuc_char == "T":
                    nuc_char = "U"
                collected_data.append(
                    {
                        "Nucleotide": nuc_char,
                        "Structure_Type": idx_to_struct.get(struct_indices[i], "."),
                        "Seq_Contribution": seq_scores[i],
                        "Struct_Contribution": struct_scores[i],
                        "Total_Contribution": seq_scores[i] + struct_scores[i],
                        "Position": i,
                    }
                )
        except Exception:
            continue
    return pd.DataFrame(collected_data)


def visualize_global_analysis_combined(df, save_dir):
    setup_basic_plot_style()
    save_dir.mkdir(parents=True, exist_ok=True)

    df["Structure_Label"] = df["Structure_Type"].map({".": "Loop / unpaired", "(": "Stem left", ")": "Stem right"}).fillna("Other")
    structure_order = ["Loop / unpaired", "Stem left", "Stem right"]
    palette_struct = {"Loop / unpaired": "#5DADE2", "Stem left": "#F5B041", "Stem right": "#58D68D"}

    fig, (ax_pos, ax_struct) = plt.subplots(1, 2, figsize=(28, 12), gridspec_kw={"width_ratios": [1.1, 1.0]})
    df["Norm_Position"] = df["Position"] / max(df["Position"].max(), 1)
    sns.lineplot(x="Norm_Position", y="Total_Contribution", data=df, ax=ax_pos, linewidth=3.0, errorbar="sd", err_kws={"alpha": 0.3})
    ax_pos.axhline(0, color="#C0392B", linestyle="--", linewidth=2.0, alpha=0.8)
    ax_pos.set_xlabel("Normalized position")
    ax_pos.set_ylabel("Average contribution score")

    sns.violinplot(
        x="Structure_Label",
        y="Struct_Contribution",
        data=df,
        hue="Structure_Label",
        palette=palette_struct,
        order=structure_order,
        inner=None,
        linewidth=0,
        cut=0,
        ax=ax_struct,
        legend=False,
    )
    sns.boxplot(
        x="Structure_Label",
        y="Struct_Contribution",
        data=df,
        hue="Structure_Label",
        order=structure_order,
        width=0.22,
        ax=ax_struct,
        legend=False,
        boxprops=dict(facecolor="white", edgecolor="#95A5A6"),
        whiskerprops=dict(color="#95A5A6", linewidth=2.0),
        capprops=dict(color="#95A5A6", linewidth=2.0),
        medianprops=dict(color="#95A5A6", linewidth=2.5),
        flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="#95A5A6", markeredgewidth=1.5, markersize=5),
    )
    ax_struct.axhline(0, color="#34495E", linestyle="--", linewidth=2.0, alpha=0.8)
    ax_struct.set_xlabel("Structure class")
    ax_struct.set_ylabel("Contribution score")
    plt.savefig(save_dir / "combined_position_and_structure_contribution.svg", format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)

    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Nucleotide", y="Seq_Contribution", data=df, order=["A", "U", "C", "G"], showfliers=False)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
    plt.xlabel("Nucleotide")
    plt.ylabel("Sequence contribution score")
    plt.savefig(save_dir / "nucleotide_sequence_contribution.svg", format="svg", bbox_inches="tight", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Structure attribution analysis for the regression task")
    parser.add_argument("--seq_path", type=str, default=str(DATA_DIR / "GSE106677_sequences.fasta"))
    parser.add_argument("--struct_path", type=str, default=str(DATA_DIR / "output_structure.txt"))
    parser.add_argument("--deg_path", type=str, default=str(DATA_DIR / "degradation.xlsx"))
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "ablation" / "best_model_Full_Model.pth"))
    parser.add_argument("--save_dir", type=str, default=str(FIGURES_DIR / "structure_analysis"))
    parser.add_argument("--num_analyze", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)

    model = UTRDegradationPredictor().to(device)
    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)

    data_list = load_data(args.seq_path, args.struct_path, args.deg_path)
    seqs = [d["sequence"] for d in data_list]
    structs = [d["structure"] for d in data_list]
    rates = [d["rate"] for d in data_list]
    dataset = UTRDataset(seqs, structs, rates, max_seq_len=176)

    total_samples = len(dataset)
    num_analyze = min(args.num_analyze, total_samples)
    indices = np.random.choice(total_samples, num_analyze, replace=False)
    df_results = compute_global_structure_attributions(model, dataset, indices, device)
    if df_results.empty:
        raise RuntimeError("No attribution results were collected.")
    df_results.to_csv(save_dir / "global_attribution_clean.csv", index=False)
    visualize_global_analysis_combined(df_results, save_dir)
    print(f"Saved structure analysis to: {save_dir}")


if __name__ == "__main__":
    main()
