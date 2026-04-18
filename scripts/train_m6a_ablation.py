from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_utils import CHECKPOINT_DIR, M6A_DATA_DIR, load_all_structures_to_dict, load_m6a_fold_data, set_seed
from src.model import UTRDataset, UTRDegradationPredictor, collate_fn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["sequence"].to(device), batch["debruijn_graph"].to(device), batch["structure_graph"].to(device))
            all_preds.extend(torch.sigmoid(out).cpu().numpy())
            all_labels.extend(batch["degradation_rate"].numpy())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
    except Exception:
        auc, auprc = 0.5, 0.0
    return auc, auprc


def main():
    parser = argparse.ArgumentParser(description="m6A ablation experiments")
    parser.add_argument("--mode", type=str, required=True, choices=["full", "no_debruijn", "no_structure", "cnn_only"])
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cell_lines", nargs="+", default=["A549", "HEK293"])
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--root_path", type=str, default=str(M6A_DATA_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    root_path = Path(args.root_path)
    if not root_path.exists():
        raise FileNotFoundError(
            f"m6A dataset directory not found: {root_path}. Please provide the dataset with --root_path."
        )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    config = {
        "full": {"use_cnn": True, "use_debruijn": True, "use_structure": True},
        "no_debruijn": {"use_cnn": True, "use_debruijn": False, "use_structure": True},
        "no_structure": {"use_cnn": True, "use_debruijn": True, "use_structure": False},
        "cnn_only": {"use_cnn": True, "use_debruijn": False, "use_structure": False},
    }[args.mode]

    save_base = CHECKPOINT_DIR / "m6a_ablation" / args.mode
    save_base.mkdir(parents=True, exist_ok=True)

    for cell in args.cell_lines:
        print(f"\n{'#' * 40}\nCELL: {cell} | MODE: {args.mode}\n{'#' * 40}")
        struct_map = load_all_structures_to_dict(root_path, cell)
        fold_results = []

        for fold_idx in range(10):
            model = UTRDegradationPredictor(**config).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
            criterion = FocalLoss().to(device)

            train_ds = load_m6a_fold_data(root_path, cell, "train", fold_idx, struct_map, UTRDataset)
            val_ds = load_m6a_fold_data(root_path, cell, "validation", fold_idx, struct_map, UTRDataset)
            test_ds = load_m6a_fold_data(root_path, cell, "test", fold_idx, struct_map, UTRDataset)
            if not train_ds:
                continue

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)

            best_val_auc, patience, counter = 0.0, 15, 0
            best_model_path = save_base / f"{cell}_f{fold_idx}_best.pth"

            for epoch in range(1, 101):
                model.train()
                pbar = tqdm(train_loader, desc=f"Fold{fold_idx} Ep{epoch}", leave=False, disable=args.no_progress)
                epoch_loss = 0.0
                for batch in pbar:
                    optimizer.zero_grad()
                    out = model(batch["sequence"].to(device), batch["debruijn_graph"].to(device), batch["structure_graph"].to(device))
                    loss = criterion(out, batch["degradation_rate"].to(device))
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                val_auc, val_auprc = evaluate(model, val_loader, device)
                print(
                    f"Cell:{cell} F{fold_idx} E{epoch:02d} | Loss:{epoch_loss / max(len(train_loader), 1):.4f} | "
                    f"ValAUC:{val_auc:.4f} | ValAUPRC:{val_auprc:.4f}"
                )
                if val_auc > best_val_auc:
                    best_val_auc, counter = val_auc, 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    counter += 1
                if counter >= patience:
                    break

            model.load_state_dict(torch.load(best_model_path, map_location=device))
            t_auc, t_auprc = evaluate(model, test_loader, device)
            fold_results.append((t_auc, t_auprc))
            print(f">>> Fold {fold_idx} Final: AUC {t_auc:.4f}, AUPRC {t_auprc:.4f}")

        if fold_results:
            aucs = [r[0] for r in fold_results]
            auprcs = [r[1] for r in fold_results]
            summary = f"{cell} Overall: AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f}, AUPRC {np.mean(auprcs):.4f}±{np.std(auprcs):.4f}"
            print(f"\n{summary}")
            with (save_base / "summary.txt").open("a", encoding="utf-8") as f:
                f.write(summary + "\n")


if __name__ == "__main__":
    main()
