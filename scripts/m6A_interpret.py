from __future__ import annotations

import argparse
import sys
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum.attr import LayerIntegratedGradients
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_utils import CHECKPOINT_DIR, FIGURES_DIR, M6A_DATA_DIR, load_all_structures_to_dict, load_m6a_fold_data
from src.model import UTRDataset, UTRDegradationPredictor, collate_fn
from src.plot_utils import setup_basic_plot_style

ALL_CELLS = ["A549", "CD8T", "ESC", "HCT116", "HEK293", "HEK293T", "Hela", "HepG2", "MOLM13"]
DIST_CELLS = ["A549", "CD8T", "HEK293T", "HepG2"]


def load_model_and_data(cell_name, checkpoint_dir, root_path, fold, device, batch_size=32):
    model_path = checkpoint_dir / f"best_model_{cell_name}_fold{fold}.pth"
    if not model_path.exists():
        return None, None
    model = UTRDegradationPredictor(use_cnn=True, use_debruijn=True, use_structure=True, cnn_multiscale=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    struct_map = load_all_structures_to_dict(root_path, cell_name)
    ds = load_m6a_fold_data(root_path, cell_name, "test", fold, struct_map, UTRDataset)
    if not ds or len(ds) == 0:
        return None, None
    return model, DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def extract_features_manual(model, seq, db, st):
    features_list = []
    if model.use_cnn:
        seq_feat = model.sequence_encoder(seq) if hasattr(model, "sequence_encoder") else model.sequence_cnn(seq)
        features_list.append(seq_feat * model.w_cnn)
    if model.use_debruijn:
        features_list.append(model.debruijn_gnn(db) * model.w_deb)
    if model.use_structure:
        features_list.append(model.structure_gnn(st) * model.w_stru)
    return model.layer_norm(torch.cat(features_list, dim=1))


def run_motif_logos_refined(save_dir, checkpoint_dir, root_path, fold, device):
    results = {}
    for cell in tqdm(ALL_CELLS, desc="Computing motif logos"):
        model, loader = load_model_and_data(cell, checkpoint_dir, root_path, fold, device, batch_size=1)
        if not model:
            continue
        all_probs, sample_data = [], []
        for i, batch in enumerate(loader):
            with torch.no_grad():
                p = torch.sigmoid(model(batch["sequence"].to(device), batch["debruijn_graph"].to(device), batch["structure_graph"].to(device)))
            all_probs.append(p[batch["degradation_rate"] == 1].cpu().numpy())
            sample_data.append(batch)
            if i > 40:
                break
        if len(all_probs) == 0 or np.concatenate(all_probs).size == 0:
            continue
        threshold = np.quantile(np.concatenate(all_probs), 0.75)
        enc = model.sequence_encoder if hasattr(model, "sequence_encoder") else model.sequence_cnn
        target_layer = enc.embedding if hasattr(enc, "embedding") else enc[0]
        lig = LayerIntegratedGradients(model, target_layer)

        acc_scores, acc_seqs, count = [], [], 0
        for batch in sample_data:
            seq = batch["sequence"].to(device)
            db = batch["debruijn_graph"].to(device)
            st = batch["structure_graph"].to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(seq, db, st))
            mask = (batch["degradation_rate"].to(device) == 1) & (probs >= threshold)
            if mask.sum() == 0:
                continue
            try:
                attrs = lig.attribute(inputs=seq, additional_forward_args=(db, st), n_steps=25, internal_batch_size=1)
                acc_scores.append(attrs.sum(dim=-1).detach().cpu().numpy())
                acc_seqs.append(seq.detach().cpu().numpy())
                count += 1
            except Exception:
                continue
            if count > 150:
                break

        if count > 0:
            all_scores, all_seqs = np.concatenate(acc_scores), np.concatenate(acc_seqs)
            start, end = (all_scores.shape[1] // 2) - 7, (all_scores.shape[1] // 2) + 8
            mat = np.zeros((15, 4))
            for i in range(len(all_scores)):
                for p_idx, p_orig in enumerate(range(start, end)):
                    idx = int(all_seqs[i, p_orig])
                    if idx < 4:
                        mat[p_idx, idx] += max(0, all_scores[i, p_orig])
            df = pd.DataFrame(mat / len(all_scores), columns=["A", "C", "G", "U"])
            df.index = np.arange(-7, 8)
            results[cell] = df

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for i, cell in enumerate(ALL_CELLS):
        ax = axes.flatten()[i]
        if results.get(cell) is not None:
            logomaker.Logo(results[cell], ax=ax, color_scheme="classic", vpad=0.1)
            ax.set_title(cell, fontsize=20, weight="bold")
            ax.set_xlim(-7.5, 7.5)
            ax.set_xticks(range(-7, 8))
        else:
            ax.text(0.5, 0.5, "No signal", ha="center", fontsize=16)
    fig.text(0.5, 0.04, "Relative position to center site", ha="center", fontsize=20)
    fig.text(0.04, 0.5, "Importance score", va="center", rotation="vertical", fontsize=20)
    plt.subplots_adjust(hspace=0.4, wspace=0.25, bottom=0.12, left=0.1)
    plt.savefig(save_dir / "m6a_motif_logos.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def run_umap_3x3_refined(save_dir, checkpoint_dir, root_path, fold, device):
    import umap

    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    for i, cell in enumerate(tqdm(ALL_CELLS, desc="Computing UMAP")):
        ax = axes.flatten()[i]
        model, loader = load_model_and_data(cell, checkpoint_dir, root_path, fold, device, batch_size=32)
        if not model:
            continue
        feats, labels = [], []
        with torch.no_grad():
            for batch in loader:
                feat = extract_features_manual(model, batch["sequence"].to(device), batch["debruijn_graph"].to(device), batch["structure_graph"].to(device))
                feats.append(feat.cpu().numpy())
                labels.append(batch["degradation_rate"].numpy())
                if len(np.concatenate(labels)) >= 1000:
                    break
        X, y = np.concatenate(feats)[:1000], np.concatenate(labels)[:1000]
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)
        ax.scatter(emb[y == 0, 0], emb[y == 0, 1], c="#d1d5db", s=12, alpha=0.5, label="Negative")
        ax.scatter(emb[y == 1, 0], emb[y == 1, 1], c="#ef4444", s=12, alpha=0.7, label="Positive")
        ax.set_title(cell, fontsize=22, weight="bold", pad=15)
        ax.axis("off")
        ax.legend(loc="lower right", markerscale=3.0, frameon=True)
    plt.tight_layout(pad=4.0)
    plt.savefig(save_dir / "umap_3x3.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def run_prediction_distribution_selected(save_dir, checkpoint_dir, root_path, fold, device):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for i, cell in enumerate(DIST_CELLS):
        ax = axes.flatten()[i]
        model, loader = load_model_and_data(cell, checkpoint_dir, root_path, fold, device)
        if not model:
            continue
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                p = torch.sigmoid(model(batch["sequence"].to(device), batch["debruijn_graph"].to(device), batch["structure_graph"].to(device)))
                preds.extend(p.cpu().numpy())
                targets.extend(batch["degradation_rate"].numpy())
                if len(preds) > 2000:
                    break
        preds, targets = np.array(preds), np.array(targets)
        sns.kdeplot(preds[targets == 0], fill=True, color="gray", label="Negative", ax=ax, clip=(0, 1), common_norm=False)
        sns.kdeplot(preds[targets == 1], fill=True, color="red", label="Positive", ax=ax, clip=(0, 1), common_norm=False)
        ax.set_title(cell, fontsize=20, weight="bold")
        ax.set_xlabel("Prediction score")
        ax.set_ylabel("Density")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        ax.legend(loc="upper right", frameon=True)
    plt.tight_layout(pad=4.0)
    plt.savefig(save_dir / "prediction_dist.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Interpretability visualizations for the m6A task")
    parser.add_argument("--root_path", type=str, default=str(M6A_DATA_DIR))
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR / "m6a" / "model_cnn"))
    parser.add_argument("--save_dir", type=str, default=str(FIGURES_DIR / "m6a_interpretation"))
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    root_path = Path(args.root_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if not root_path.exists():
        raise FileNotFoundError(f"m6A dataset directory not found: {root_path}")

    setup_basic_plot_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_motif_logos_refined(save_dir, checkpoint_dir, root_path, args.fold, device)
    run_umap_3x3_refined(save_dir, checkpoint_dir, root_path, args.fold, device)
    run_prediction_distribution_selected(save_dir, checkpoint_dir, root_path, args.fold, device)
    print(f"Saved m6A interpretation outputs to: {save_dir}")


if __name__ == "__main__":
    main()
