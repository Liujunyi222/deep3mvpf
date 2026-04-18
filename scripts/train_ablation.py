from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_utils import CHECKPOINT_DIR, DATA_DIR, get_logger, load_regression_data, set_seed
from src.model import UTRDataset, UTRDegradationPredictor, collate_fn


def concordance_index(y_true, y_pred):
    concordant = 0
    discordant = 0
    n = len(y_true)
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or (
                    y_true[i] > y_true[j] and y_pred[i] > y_pred[j]
                ):
                    concordant += 1
                elif (y_true[i] < y_true[j] and y_pred[i] > y_pred[j]) or (
                    y_true[i] > y_true[j] and y_pred[i] < y_pred[j]
                ):
                    discordant += 1
    total = concordant + discordant
    return 0.5 if total == 0 else concordant / total


def pairwise_ranking_loss(y_true, y_pred, margin=1.0):
    y_true = y_true.view(-1, 1)
    y_pred = y_pred.view(-1, 1)
    diff_true = y_true - y_true.t()
    diff_pred = y_pred - y_pred.t()
    mask = (diff_true > 0).float()
    loss = F.binary_cross_entropy_with_logits(diff_pred * margin, mask, reduction="none")
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def build_dataloader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def train_ablation(args):
    set_seed(args.seed)

    save_dir = CHECKPOINT_DIR / "ablation"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / f"log_{args.exp_name}.txt"
    logger = get_logger(log_file, resume=(args.resume is not None))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger(f"\n=== Experiment: {args.exp_name} ===")
    logger(f"Time: {datetime.datetime.now()}")
    logger(f"Device: {device}")

    seq_path = Path(args.seq_path)
    struct_path = Path(args.struct_path)
    deg_path = Path(args.deg_path)

    sequences, structures, degradation_rates = load_regression_data(seq_path, struct_path, deg_path, logger)

    train_seqs, temp_seqs, train_stru, temp_stru, train_rate, temp_rate = train_test_split(
        sequences, structures, degradation_rates, test_size=0.2, random_state=args.seed
    )
    val_seqs, test_seqs, val_stru, test_stru, val_rate, test_rate = train_test_split(
        temp_seqs, temp_stru, temp_rate, test_size=0.5, random_state=args.seed
    )

    logger(f"Dataset Split -> Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")

    train_dataset = UTRDataset(train_seqs, train_stru, train_rate, max_seq_len=args.max_seq_len)
    val_dataset = UTRDataset(val_seqs, val_stru, val_rate, max_seq_len=args.max_seq_len)
    test_dataset = UTRDataset(test_seqs, test_stru, test_rate, max_seq_len=args.max_seq_len)

    train_loader = build_dataloader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = build_dataloader(val_dataset, args.batch_size, False, args.num_workers)
    test_loader = build_dataloader(test_dataset, args.batch_size, False, args.num_workers)

    model = UTRDegradationPredictor(
        use_cnn=True,
        use_debruijn=not args.cnn_only and not args.no_debruijn,
        use_structure=not args.cnn_only and not args.no_struct,
        cnn_multiscale=not args.single_scale,
    ).to(device)
    logger(
        f"Modules -> CNN:True, DeBruijn:{not args.cnn_only and not args.no_debruijn}, "
        f"Structure:{not args.cnn_only and not args.no_struct}, MultiScale:{not args.single_scale}"
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    start_epoch = 0
    best_val_ci = -float("inf")
    best_model_path = save_dir / f"best_model_{args.exp_name}.pth"
    checkpoint_path = save_dir / f"last_checkpoint_{args.exp_name}.pth"

    if args.resume and Path(args.resume).is_file():
        logger(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_ci = checkpoint.get("best_val_ci", -float("inf"))
        logger(f"=> Loaded checkpoint (epoch {checkpoint['epoch']}, best CI {best_val_ci:.4f})")

    patience_counter = 0
    logger(f"\nStarting training from epoch {start_epoch + 1} to {args.epochs}.")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_total = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False):
            seqs = batch["sequence"].to(device)
            deb = batch["debruijn_graph"].to(device)
            stru = batch["structure_graph"].to(device)
            targets = batch["degradation_rate"].to(device).view(-1)

            optimizer.zero_grad()
            preds = model(seqs, deb, stru).view(-1)
            loss = criterion(preds, targets) + 0.1 * pairwise_ranking_loss(targets, preds, margin=0.2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_total += loss.item() * seqs.size(0)

        avg_train_loss = train_loss_total / len(train_dataset)

        model.eval()
        all_preds, all_trues = [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                seqs = batch["sequence"].to(device)
                deb = batch["debruijn_graph"].to(device)
                stru = batch["structure_graph"].to(device)
                targets = batch["degradation_rate"].to(device).view(-1)
                preds = model(seqs, deb, stru).view(-1)
                loss = criterion(preds, targets) + 0.1 * pairwise_ranking_loss(targets, preds, margin=0.2)
                val_loss_total += loss.item() * seqs.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_trues.extend(targets.cpu().numpy().tolist())

        avg_val_loss = val_loss_total / len(val_dataset)
        trues_np = np.array(all_trues)
        preds_np = np.array(all_preds)
        if len(preds_np) > 1:
            val_mse = mean_squared_error(trues_np, preds_np)
            val_r2 = r2_score(trues_np, preds_np)
            val_ci = concordance_index(trues_np, preds_np)
            val_pearson, _ = pearsonr(trues_np, preds_np)
            val_spearman, _ = spearmanr(trues_np, preds_np)
        else:
            val_mse = val_r2 = val_ci = val_pearson = val_spearman = 0.0

        scheduler.step(val_ci)
        logger(
            f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f} | "
            f"MSE={val_mse:.4f}, R2={val_r2:.4f}, CI={val_ci:.4f}, "
            f"Pearson={val_pearson:.4f}, Spearman={val_spearman:.4f}"
        )

        if val_ci > best_val_ci:
            best_val_ci = val_ci
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger(f"  >>> Best Model Saved (CI: {best_val_ci:.4f})")
        else:
            patience_counter += 1
            logger(f"  . No improvement for {patience_counter}/{args.patience} epochs.")

        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_ci": best_val_ci,
            },
            checkpoint_path,
        )

        if patience_counter >= args.patience:
            logger(f"\nEarly stopping triggered after {args.patience} epochs without improvement.")
            break

    logger(f"\nTraining Finished. Loading best model from {best_model_path} for testing.")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            seqs = batch["sequence"].to(device)
            deb = batch["debruijn_graph"].to(device)
            stru = batch["structure_graph"].to(device)
            targets = batch["degradation_rate"].to(device).view(-1)
            preds = model(seqs, deb, stru).view(-1)
            test_preds.extend(preds.cpu().numpy().tolist())
            test_trues.extend(targets.cpu().numpy().tolist())

    t_trues = np.array(test_trues)
    t_preds = np.array(test_preds)
    logger(
        f"\n=== Final Test Results for {args.exp_name} ===\n"
        f"MSE:        {mean_squared_error(t_trues, t_preds):.4f}\n"
        f"R2:         {r2_score(t_trues, t_preds):.4f}\n"
        f"CI:         {concordance_index(t_trues, t_preds):.4f}\n"
        f"Pearson R:  {pearsonr(t_trues, t_preds)[0]:.4f}\n"
        f"Spearman R: {spearmanr(t_trues, t_preds)[0]:.4f}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression training / ablation for UTR degradation prediction")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_struct", action="store_true")
    parser.add_argument("--no_debruijn", action="store_true")
    parser.add_argument("--cnn_only", action="store_true")
    parser.add_argument("--single_scale", action="store_true")
    parser.add_argument("--seq_path", type=str, default=str(DATA_DIR / "GSE106677_sequences.fasta"))
    parser.add_argument("--struct_path", type=str, default=str(DATA_DIR / "output_structure.txt"))
    parser.add_argument("--deg_path", type=str, default=str(DATA_DIR / "degradation.xlsx"))
    args = parser.parse_args()
    train_ablation(args)
