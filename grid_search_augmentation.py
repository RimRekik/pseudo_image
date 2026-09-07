#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid search sur les hyperparamètres d'augmentation (RT shift et bruit), pour
répondre à la question : "les valeurs prises du rapport de Léo Calmettes
(prob=0.2 pour RT shift, prob=0.8 pour le bruit) sont-elles adaptées à NOTRE
dataset, ou une autre combinaison ferait-elle mieux (ou moins pire) ?"

Méthodologie (comme dans le rapport de référence) :
- Chaque augmentation (RT shift, bruit) est testée SÉPARÉMENT, jamais combinée,
  pour isoler son effet propre.
- Un SEUL split train/val/test (pas de validation croisée à 5 folds) est
  utilisé pour chaque combinaison -- une vraie grid search en 5-fold CV serait
  bien trop coûteuse en calcul (des dizaines de combinaisons x 5 entraînements
  chacune). C'est un criblage ("screening"), pas un chiffre définitif : une
  fois la meilleure zone identifiée, il faut confirmer avec cross_validate.py.
- Moins d'epochs que l'entraînement final (configurable, 15 par défaut au lieu
  de 30) pour accélérer le criblage.
- Le critère retenu est val_loss (comme Calmettes) ET test accuracy (plus
  parlant), les deux sont rapportés.

Usage :
    # Grid search sur le RT shift uniquement
    python3 grid_search_augmentation.py \
        --dataset_dir ../ms1_processed_11classes --which rt \
        --rt_shift_probs 0.2,0.4,0.6,0.8,1.0 --rt_shift_stds 2.5,5,7.5,10 \
        --grid_epochs 15 --output_dir grid_search_rt

    # Grid search sur le bruit uniquement
    python3 grid_search_augmentation.py \
        --dataset_dir ../ms1_processed_11classes --which noise \
        --noise_probs 0.2,0.4,0.6,0.8,1.0 --noise_maxs 1.25,1.5,1.75,2,2.25,2.5 \
        --grid_epochs 15 --output_dir grid_search_noise
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset.dataset import (
    scan_dataset, stratified_split, MS1NpyDataset, build_train_transform, DEFAULT_TARGET_SIZE,
)
from model.model import build_model
from torch.utils.data import DataLoader

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_float_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def compute_class_weights(train_samples, num_classes, device):
    from collections import Counter
    counts = Counter(s["label"] for s in train_samples)
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_one_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _protocols in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_n += images.size(0)
    return total_loss / total_n, total_correct / total_n


def train_one_config(args, samples, class_to_idx, train_idx, val_idx, test_idx, device,
                      rt_shift_prob=0.0, rt_shift_std=0.0, noise_prob=0.0, noise_max=1.0):
    num_classes = len(class_to_idx)
    target_size = (args.target_h, args.target_w)

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    train_transform = build_train_transform(
        rt_shift_prob=rt_shift_prob, rt_shift_mean=0.0, rt_shift_std=rt_shift_std,
        noise_prob=noise_prob, noise_max=noise_max,
    )
    train_ds = MS1NpyDataset(train_samples, target_size=target_size, normalize="minmax", transform=train_transform)
    val_ds = MS1NpyDataset(val_samples, target_size=target_size, normalize="minmax")
    test_ds = MS1NpyDataset(test_samples, target_size=target_size, normalize="minmax")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = build_model(num_classes, backbone=args.model, pretrained=True).to(device)

    if args.weighted_entropy:
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_samples, num_classes, device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc, best_val_loss, best_state = -1.0, None, None
    for epoch in range(1, args.grid_epochs + 1):
        run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc, best_val_loss = val_acc, val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    test_loss, test_acc = run_one_epoch(model, test_loader, criterion, optimizer, device, train=False)

    return best_val_loss, best_val_acc, test_loss, test_acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--which", choices=["rt", "noise"], required=True,
                     help="Quelle augmentation cribler : 'rt' (Random_shift_rt) ou 'noise' (Random_int_noise)")
    ap.add_argument("--rt_shift_probs", type=parse_float_list, default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--rt_shift_stds", type=parse_float_list, default="2.5,5,7.5,10")
    ap.add_argument("--noise_probs", type=parse_float_list, default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--noise_maxs", type=parse_float_list, default="1.25,1.5,1.75,2,2.25,2.5")
    ap.add_argument("--model", type=str, default="resnet18")
    ap.add_argument("--grid_epochs", type=int, default=15, help="Epochs par combinaison (moins que l'entraînement final, pour accélérer le criblage)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--beta1", type=float, default=0.938)
    ap.add_argument("--beta2", type=float, default=0.9928)
    ap.add_argument("--weighted_entropy", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True)
    ap.add_argument("--target_h", type=int, default=224)
    ap.add_argument("--target_w", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default="grid_search_output")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[grid] device = {device}  |  which = {args.which}  |  epochs/combo = {args.grid_epochs}")

    samples, class_to_idx = scan_dataset(args.dataset_dir)
    train_idx, val_idx, test_idx = stratified_split(samples, seed=args.random_state)
    print(f"[grid] Split -> train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} (split UNIQUE, pas de CV)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Baseline (aucune augmentation), pour référence ---
    print("\n[grid] === Baseline (sans augmentation) ===")
    base_val_loss, base_val_acc, base_test_loss, base_test_acc = train_one_config(
        args, samples, class_to_idx, train_idx, val_idx, test_idx, device,
    )
    print(f"[grid] Baseline -> val_loss={base_val_loss:.4f} val_acc={base_val_acc:.4f}  "
          f"test_loss={base_test_loss:.4f} test_acc={base_test_acc:.4f}")

    # --- 2) Grille ---
    results = []
    if args.which == "rt":
        probs, others = args.rt_shift_probs, args.rt_shift_stds
        other_name = "std"
    else:
        probs, others = args.noise_probs, args.noise_maxs
        other_name = "max"

    total_combos = len(probs) * len(others)
    i = 0
    for prob in probs:
        for other in others:
            i += 1
            print(f"\n[grid] === Combo {i}/{total_combos} : prob={prob}, {other_name}={other} ===")
            if args.which == "rt":
                val_loss, val_acc, test_loss, test_acc = train_one_config(
                    args, samples, class_to_idx, train_idx, val_idx, test_idx, device,
                    rt_shift_prob=prob, rt_shift_std=other,
                )
            else:
                val_loss, val_acc, test_loss, test_acc = train_one_config(
                    args, samples, class_to_idx, train_idx, val_idx, test_idx, device,
                    noise_prob=prob, noise_max=other,
                )
            print(f"[grid] -> val_loss={val_loss:.4f} val_acc={val_acc:.4f}  test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
            results.append({"prob": prob, other_name: other, "val_loss": val_loss, "val_acc": val_acc,
                             "test_loss": test_loss, "test_acc": test_acc})

    # --- 3) Sauvegarde CSV ---
    csv_path = output_dir / f"grid_search_{args.which}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prob", other_name, "val_loss", "val_acc", "test_loss", "test_acc"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[grid] Résultats sauvegardés : {csv_path}")

    # --- 4) Meilleure combinaison ---
    best = max(results, key=lambda r: r["val_acc"])
    print(f"\n[grid] === Meilleure combinaison (val_acc) : prob={best['prob']}, {other_name}={best[other_name]} "
          f"-> val_acc={best['val_acc']:.4f}, test_acc={best['test_acc']:.4f} ===")
    print(f"[grid] Comparaison à la baseline (sans augmentation) : val_acc={base_val_acc:.4f}, test_acc={base_test_acc:.4f}")

    summary = {
        "which": args.which, "grid_epochs": args.grid_epochs,
        "baseline": {"val_loss": base_val_loss, "val_acc": base_val_acc, "test_loss": base_test_loss, "test_acc": base_test_acc},
        "best": best,
        "all_results": results,
    }
    with open(output_dir / f"grid_search_{args.which}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- 5) Heatmap (val_acc) ---
    if HAS_MATPLOTLIB:
        prob_vals = sorted(set(r["prob"] for r in results))
        other_vals = sorted(set(r[other_name] for r in results))
        grid = np.full((len(prob_vals), len(other_vals)), np.nan)
        for r in results:
            pi = prob_vals.index(r["prob"])
            oi = other_vals.index(r[other_name])
            grid[pi, oi] = r["val_acc"]

        fig, ax = plt.subplots(figsize=(1.2 * len(other_vals) + 2, 1.0 * len(prob_vals) + 2))
        im = ax.imshow(grid, cmap="RdYlGn", vmin=np.nanmin(grid), vmax=max(np.nanmax(grid), base_val_acc))
        ax.set_xticks(range(len(other_vals))); ax.set_xticklabels(other_vals)
        ax.set_yticks(range(len(prob_vals))); ax.set_yticklabels(prob_vals)
        ax.set_xlabel(other_name); ax.set_ylabel("prob")
        ax.set_title(f"val_acc -- grid search {args.which} (baseline sans augmentation = {base_val_acc:.3f})")
        for pi in range(len(prob_vals)):
            for oi in range(len(other_vals)):
                v = grid[pi, oi]
                if not np.isnan(v):
                    ax.text(oi, pi, f"{v:.3f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / f"grid_search_{args.which}_heatmap.png", dpi=150)
        print(f"[grid] Heatmap sauvegardée : {output_dir / f'grid_search_{args.which}_heatmap.png'}")


if __name__ == "__main__":
    main()