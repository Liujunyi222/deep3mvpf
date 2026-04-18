from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_utils import CHECKPOINT_DIR, M6A_DATA_DIR, RESULTS_DIR, load_all_structures_to_dict, load_m6a_fold_data, set_seed
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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        seq = batch["sequence"].to(device)
        db_graph = batch["debruijn_graph"].to(device)
        st_graph = batch["structure_graph"].to(device)
        labels = batch["degradation_rate"].to(device)

        optimizer.zero_grad()
        outputs = model(seq, db_graph, st_graph)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(device)
            db_graph = batch["debruijn_graph"].to(device)
            st_graph = batch["structure_graph"].to(device)
            labels = batch["degradation_rate"].cpu().numpy()
            outputs = model(seq, db_graph, st_graph)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    except ValueError:
        auc, auprc, f1, acc = 0.0, 0.0, 0.0, 0.0
    return auc, auprc, f1, acc


def main():
    parser = argparse.ArgumentParser(description="m6A Site Prediction 10-Fold")
    parser.add_argument("--encoder_type", type=str, default="cnn", help="Kept for backward compatibility")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fold", type=int, default=None, help="Specific fold to run (0-9)")
    parser.add_argument("--root_path", type=str, default=str(M6A_DATA_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    root_path = Path(args.root_path)
    if not root_path.exists():
        raise FileNotFoundError(
            f"m6A dataset directory not found: {root_path}. "
            "Please put the fold-structured m6A data there or pass --root_path."
        )

    checkpoint_base = CHECKPOINT_DIR / "m6a"
    results_dir = checkpoint_base / f"results_{args.encoder_type}"
    model_dir = checkpoint_base / f"model_{args.encoder_type}"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    cell_lines = ["A549", "CD8T", "ESC", "HCT116", "HEK293", "HEK293T", "Hela", "HepG2", "MOLM13"]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Running experiment: {args.encoder_type.upper()} on {device}")

    for cell in cell_lines:
        print(f"\n{'=' * 40}\nProcessing cell line: {cell}\n{'=' * 40}")
        struct_map = load_all_structures_to_dict(root_path, cell)
        fold_aucs, fold_auprcs = [], []
        folds_to_run = range(10) if args.fold is None else [args.fold]

        for fold_idx in folds_to_run:
            print(f"\n>>> Running fold {fold_idx} / 9 ...")
            log_filename = results_dir / f"results_{cell}_fold{fold_idx}.txt"
            best_model_path = model_dir / f"best_model_{cell}_fold{fold_idx}.pth"

            model = UTRDegradationPredictor(encoder_type=args.encoder_type).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
            criterion = FocalLoss(alpha=0.5, gamma=2.0).to(device)

            train_ds = load_m6a_fold_data(root_path, cell, "train", fold_idx, struct_map, UTRDataset)
            val_ds = load_m6a_fold_data(root_path, cell, "validation", fold_idx, struct_map, UTRDataset)
            test_ds = load_m6a_fold_data(root_path, cell, "test", fold_idx, struct_map, UTRDataset)
            if not train_ds:
                print(f"Skipping fold {fold_idx} due to missing data.")
                continue

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

            with log_filename.open("w", encoding="utf-8") as f:
                f.write("Epoch\tTrain_Loss\tVal_AUC\tVal_AUPRC\tTest_AUC\tTest_AUPRC\tNote\n")

            best_fold_auc = 0.0
            patience_counter = 0
            for epoch in range(1, args.epochs + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                scheduler.step()
                val_auc, val_auprc, _, _ = evaluate(model, val_loader, device)

                note = ""
                test_auc_display = 0.0
                test_auprc_display = 0.0
                if val_auc > best_fold_auc:
                    best_fold_auc = val_auc
                    patience_counter = 0
                    note = "Best"
                    torch.save(model.state_dict(), best_model_path)
                    test_auc_display, test_auprc_display, _, _ = evaluate(model, test_loader, device)
                    print(f"  [Fold {fold_idx}] Epoch {epoch} | Val AUC: {val_auc:.4f} | Val AUPRC: {val_auprc:.4f}")
                else:
                    patience_counter += 1

                with log_filename.open("a", encoding="utf-8") as f:
                    f.write(
                        f"{epoch}\t{train_loss:.6f}\t{val_auc:.6f}\t{val_auprc:.6f}\t"
                        f"{test_auc_display:.6f}\t{test_auprc_display:.6f}\t{note}\n"
                    )

                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

            model.load_state_dict(torch.load(best_model_path, map_location=device))
            final_test_auc, final_test_auprc, _, _ = evaluate(model, test_loader, device)
            print(f">>> Fold {fold_idx} Final Test AUC: {final_test_auc:.4f} | AUPRC: {final_test_auprc:.4f}")
            fold_aucs.append(final_test_auc)
            fold_auprcs.append(final_test_auprc)
            del model, optimizer, train_loader, val_loader, test_loader
            torch.cuda.empty_cache()

        if fold_aucs:
            avg_auc, std_auc = np.mean(fold_aucs), np.std(fold_aucs)
            avg_auprc, std_auprc = np.mean(fold_auprcs), np.std(fold_auprcs)
            print(f"\n{'#' * 40}\nCell line {cell} 10-fold result:\nAUC:   {avg_auc:.4f} ± {std_auc:.4f}\nAUPRC: {avg_auprc:.4f} ± {std_auprc:.4f}\n{'#' * 40}\n")
            summary_path = results_dir / f"summary_{cell}.txt"
            with summary_path.open("w", encoding="utf-8") as f:
                f.write(f"Mean AUC: {avg_auc:.4f}\nStd AUC: {std_auc:.4f}\nValues AUC: {fold_aucs}\n")
                f.write(f"Mean AUPRC: {avg_auprc:.4f}\nStd AUPRC: {std_auprc:.4f}\nValues AUPRC: {fold_auprcs}\n")


if __name__ == "__main__":
    main()
