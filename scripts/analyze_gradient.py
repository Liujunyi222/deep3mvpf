from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from torch_geometric.data import Batch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_utils import CHECKPOINT_DIR, DATA_DIR, FIGURES_DIR, load_regression_data, set_seed
from src.model import UTRDataset, UTRDegradationPredictor
from src.plot_utils import setup_basic_plot_style


def model_forward_with_embeddings(seq_embeddings, debruijn_graph_batch, structure_graph_batch, model):
    x = seq_embeddings.transpose(1, 2)
    conv_outputs = []
    for conv, bn in zip(model.sequence_cnn.conv_layers, model.sequence_cnn.bn_layers):
        out = conv(x)
        out = bn(out)
        out = F.relu(out)
        out = F.max_pool1d(out, kernel_size=out.size(2)).squeeze(2)
        conv_outputs.append(out)
    seq_features = torch.cat(conv_outputs, dim=1)
    seq_features = model.sequence_cnn.dropout(seq_features)
    debruijn_features = model.debruijn_gnn(debruijn_graph_batch)
    structure_features = model.structure_gnn(structure_graph_batch)
    current_batch_size = seq_features.size(0)
    if debruijn_features.size(0) != current_batch_size:
        debruijn_features = debruijn_features.expand(current_batch_size, -1)
    if structure_features.size(0) != current_batch_size:
        structure_features = structure_features.expand(current_batch_size, -1)
    combined = torch.cat([seq_features, debruijn_features, structure_features], dim=1)
    combined = model.layer_norm(combined)
    return model.fusion_fc(combined).squeeze(1)


def compute_batch_attributions(model, dataset, indices, device):
    model.eval()
    ig = IntegratedGradients(model_forward_with_embeddings)
    results = []
    for idx in tqdm(indices, desc="Computing IG"):
        data_item = dataset[idx]
        seq_in = data_item["sequence"].unsqueeze(0).to(device)
        deb_in = Batch.from_data_list([data_item["debruijn_graph"]]).to(device)
        str_in = Batch.from_data_list([data_item["structure_graph"]]).to(device)
        with torch.no_grad():
            input_embeddings = model.sequence_cnn.embedding(seq_in)
        attr = ig.attribute(inputs=input_embeddings, additional_forward_args=(deb_in, str_in, model), n_steps=20)
        attr_score = attr.sum(dim=2).squeeze(0).cpu().detach().numpy()
        results.append((data_item["sequence"].cpu().numpy(), attr_score))
    return results


def visualize_all_results(df, results, nuc_dict, save_dir, top_n=20):
    setup_basic_plot_style()
    save_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(28, 10))
    gs = fig.add_gridspec(1, 2, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    top_unstab = df.head(top_n).copy()
    top_unstab["Type"] = "Unstable motifs"
    top_stab = df.tail(top_n).copy()
    top_stab["Type"] = "Stable motifs"
    combined = pd.concat([top_unstab, top_stab])
    sns.barplot(x="Score", y="Motif", data=combined, hue="Type", dodge=False, width=0.5, ax=ax1)
    ax1.set_xlabel("Average contribution score", fontsize=18)
    ax1.set_ylabel("Motif", fontsize=18)
    ax1.axvline(0, color="black", linestyle="--", linewidth=1)

    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(df["Score"], bins=100, kde=True, alpha=0.6, ax=ax2)
    ax2.set_xlabel("Contribution score", fontsize=18)
    ax2.set_ylabel("Motif count", fontsize=18)
    plt.savefig(save_dir / "combined_motif_distribution.svg", format="svg", bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(14, 10))
    idx_to_nuc = {v: k for k, v in nuc_dict.items()}
    data_nuc = {"Nucleotide": [], "Score": []}
    for seq_idxs, scores in results:
        for i in range(len(seq_idxs)):
            nt = idx_to_nuc.get(seq_idxs[i], "N").replace("T", "U")
            if nt in ["A", "U", "C", "G"]:
                data_nuc["Nucleotide"].append(nt)
                data_nuc["Score"].append(scores[i])
    df_nuc = pd.DataFrame(data_nuc)
    ax_n = sns.boxplot(x="Nucleotide", y="Score", data=df_nuc, order=["A", "U", "C", "G"], showfliers=False, linewidth=1.5)
    plt.axhline(0, color="black", linestyle="--", linewidth=2)
    ax_n.set_xlabel("Nucleotide type", fontsize=18)
    ax_n.set_ylabel("Contribution score", fontsize=18)
    plt.savefig(save_dir / "nucleotide_contribution.svg", format="svg", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(14, 10))
    ax_s = sns.scatterplot(x="Score", y="Count", data=df, alpha=0.5, s=100, edgecolor=None)
    plt.yscale("log")
    ax_s.set_xlabel("Contribution score", fontsize=18)
    ax_s.set_ylabel("Motif frequency (log)", fontsize=18)
    plt.savefig(save_dir / "motif_frequency_scatter.svg", format="svg", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Gradient-based motif analysis for the regression task")
    parser.add_argument("--seq_path", type=str, default=str(DATA_DIR / "GSE106677_sequences.fasta"))
    parser.add_argument("--struct_path", type=str, default=str(DATA_DIR / "output_structure.txt"))
    parser.add_argument("--deg_path", type=str, default=str(DATA_DIR / "degradation.xlsx"))
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "ablation" / "best_model_Full_Model.pth"))
    parser.add_argument("--save_dir", type=str, default=str(FIGURES_DIR / "motif_analysis"))
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)

    model = UTRDegradationPredictor().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)

    seqs, structs, rates = load_regression_data(args.seq_path, args.struct_path, args.deg_path)
    dataset = UTRDataset(seqs, structs, rates)

    indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    results = compute_batch_attributions(model, dataset, indices, device)

    idx_to_nuc = {0: "A", 1: "C", 2: "G", 3: "U"}
    motif_dict = defaultdict(list)
    k = 6
    for seq_idxs, scores in results:
        seq = "".join([idx_to_nuc.get(int(i), "N") for i in seq_idxs])
        for i in range(len(seq) - k + 1):
            motif = seq[i : i + k]
            if "N" not in motif:
                motif_dict[motif].append(np.mean(scores[i : i + k]))

    df_motifs = pd.DataFrame(
        [{"Motif": m, "Score": np.mean(s), "Count": len(s)} for m, s in motif_dict.items() if len(s) >= 10]
    ).sort_values(by="Score")
    visualize_all_results(df_motifs, results, dataset.nucleotide_dict, save_dir)
    print(f"Saved motif analysis to: {save_dir}")


if __name__ == "__main__":
    main()
