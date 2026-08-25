#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation croisée à K folds (par défaut 5), pour obtenir une performance
FIABLE (moyenne ± écart-type) plutôt qu'un seul chiffre qui dépend du hasard
du split train/val/test (on a vu une variance de ±2.3 points entre seeds sur
un split unique -- ce script règle ce problème).

Principe :
- Les échantillons sont groupés par 'group_key' (même logique que
  stratified_split dans dataset.py : AER/ANA/réplicats d'un même échantillon
  biologique restent TOUJOURS dans le même fold, jamais séparés).
- K folds stratifiés par classe sont construits une seule fois (reproductible
  via --random_state).
- Pour chaque fold i (i=1..K) : ce fold sert de TEST, les K-1 autres sont
  combinés puis re-découpés en train/val (85/15) pour l'early stopping.
- Un modèle est entraîné et évalué par fold. A la fin : accuracy moyenne
  ± écart-type sur les K folds, et matrice de confusion agrégée (tous les
  folds concaténés = équivalent à avoir testé sur 100% des données une fois
  chacune, sans jamais tester sur des données vues en train).

Usage :
    python3 cross_validate.py --dataset_dir ../ms1_processed --cv_folds 5 \
        --weighted_entropy True --epoches 30 --batch_size 32 \
        --output_dir output_cv
"""

import json
import random
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import load_args
from dataset.dataset import (
    scan_dataset, MS1NpyDataset, build_train_transform, DEFAULT_TARGET_SIZE,
)
from model.model import build_model
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def make_k_folds(samples, k, seed):
    """Découpe les samples (groupés par group_key, stratifiés par classe) en k folds."""
    groups_by_class = defaultdict(lambda: defaultdict(list))
    for i, s in enumerate(samples):
        groups_by_class[s["class_name"]][s["group_key"]].append(i)

    rng = random.Random(seed)
    folds = [[] for _ in range(k)]

    for class_name, groups in groups_by_class.items():
        group_items = list(groups.items())
        rng.shuffle(group_items)
        # Répartit les groupes de cette classe sur les k folds de façon cyclique
        for i, (_key, indices) in enumerate(group_items):
            folds[i % k].extend(indices)

    for f in folds:
        rng.shuffle(f)
    return folds


def split_train_val(indices, val_frac, seed):
    rng = random.Random(seed)
    idx = indices.copy()
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * val_frac)))
    return idx[n_val:], idx[:n_val]


def compute_class_weights_from_samples(samples, indices, num_classes, device):
    counts = Counter(samples[i]["label"] for i in indices)
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


def train_one_fold(args, samples, class_to_idx, train_idx, val_idx, test_idx, fold_output_dir, device):
    num_classes = len(class_to_idx)
    target_size = (args.target_h, args.target_w)

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    train_transform = build_train_transform(
        rt_shift_prob=args.rt_shift_prob, rt_shift_mean=args.rt_shift_mean,
        rt_shift_std=args.rt_shift_std, noise_prob=args.noise_prob, noise_max=args.noise_max,
    )
    train_ds = MS1NpyDataset(train_samples, target_size=target_size, normalize="minmax",
                              noise_threshold=args.noise_threshold, transform=train_transform)
    val_ds = MS1NpyDataset(val_samples, target_size=target_size, normalize="minmax",
                            noise_threshold=args.noise_threshold)
    test_ds = MS1NpyDataset(test_samples, target_size=target_size, normalize="minmax",
                             noise_threshold=args.noise_threshold)

    if args.oversample:
        counts = Counter(s["label"] for s in train_samples)
        w = [1.0 / counts[s["label"]] for s in train_samples]
        sampler = WeightedRandomSampler(w, num_samples=len(train_samples), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                   num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = build_model(num_classes, backbone=args.model, pretrained=True,
                         freeze_backbone=args.freeze_backbone).to(device)

    if args.weighted_entropy:
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights_from_samples(
            samples, train_idx, num_classes, device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = (optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
                 if args.optim == "Adam" else
                 optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc, best_state, evals_no_improve = -1.0, None, 0
    for epoch in range(1, args.epoches + 1):
        run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc, evals_no_improve = val_acc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            evals_no_improve += 1
            if evals_no_improve >= args.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _protocols in test_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    return acc, all_labels, all_preds


def main():
    args = load_args()
    k = getattr(args, "cv_folds", 5)
    output_dir = Path(getattr(args, "cv_output_dir", "output_cv"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[cv] device = {device}")

    samples, class_to_idx = scan_dataset(args.dataset_dir)
    idx_to_class = {v: k_ for k_, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    folds = make_k_folds(samples, k, args.random_state)
    print(f"[cv] {k} folds -> tailles: {[len(f) for f in folds]}")

    fold_accs = []
    all_labels_agg, all_preds_agg = [], []

    for i in range(k):
        test_idx = folds[i]
        remaining = [idx for j in range(k) if j != i for idx in folds[j]]
        train_idx, val_idx = split_train_val(remaining, val_frac=0.15, seed=args.random_state + i)

        print(f"\n[cv] === Fold {i+1}/{k} === train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
        acc, labels, preds = train_one_fold(
            args, samples, class_to_idx, train_idx, val_idx, test_idx, output_dir / f"fold{i+1}", device
        )
        print(f"[cv] Fold {i+1} test accuracy = {acc:.4f}")
        fold_accs.append(acc)
        all_labels_agg.extend(labels)
        all_preds_agg.extend(preds)

    mean_acc, std_acc = float(np.mean(fold_accs)), float(np.std(fold_accs))
    print(f"\n[cv] === Résultat final : accuracy = {mean_acc:.4f} ± {std_acc:.4f} sur {k} folds ===")
    print(f"[cv] Accuracies par fold : {[round(a, 4) for a in fold_accs]}")

    report_lines = [
        f"Validation croisée à {k} folds",
        f"Accuracy moyenne : {mean_acc:.4f} +/- {std_acc:.4f}",
        f"Accuracies par fold : {[round(a, 4) for a in fold_accs]}",
        "",
    ]
    target_names = [idx_to_class[i] for i in range(num_classes)]
    if HAS_SKLEARN:
        report_lines.append("Rapport agrégé (tous les folds concaténés = 100% des données testées) :")
        report_lines.append(classification_report(all_labels_agg, all_preds_agg,
                                                    target_names=target_names, digits=3))
        cm = confusion_matrix(all_labels_agg, all_preds_agg)
        report_lines.append(str(target_names))
        report_lines.append(str(cm))
        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(6, 5.5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(target_names))); ax.set_yticks(range(len(target_names)))
            ax.set_xticklabels(target_names, rotation=45, ha="right"); ax.set_yticklabels(target_names)
            ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
            ax.set_title(f"Matrice de confusion agrégée ({k}-fold CV)")
            thresh = cm.max() / 2.0
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                            color="white" if cm[r, c] > thresh else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(output_dir / "confusion_matrix_cv.png", dpi=150)
            plt.close(fig)

    with open(output_dir / "cv_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n[cv] Rapport complet sauvegardé dans {output_dir / 'cv_report.txt'}")

    if HAS_SKLEARN:
        analyze_count_vs_accuracy(samples, all_labels_agg, all_preds_agg, class_to_idx, output_dir)


def analyze_count_vs_accuracy(samples, all_labels_agg, all_preds_agg, class_to_idx, output_dir):
    """
    Quantifie le lien entre le nombre d'échantillons par classe (dans
    l'ensemble du dataset) et la précision (recall) du modèle sur cette
    classe -- même type d'analyse que dans Etude_data_scarcity_impact.pdf.
    Calcule aussi le coefficient de corrélation de Pearson entre les deux.
    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # Nombre total d'échantillons par classe dans le dataset (avant split)
    n_samples_per_class = Counter(s["label"] for s in samples)

    # Recall par classe sur les prédictions agrégées (tous les folds = 100% des données testées)
    all_labels_agg = np.array(all_labels_agg)
    all_preds_agg = np.array(all_preds_agg)
    recall_per_class = {}
    for c in range(num_classes):
        mask = all_labels_agg == c
        recall_per_class[c] = float((all_preds_agg[mask] == c).mean()) if mask.sum() > 0 else float("nan")

    rows = []
    for c in range(num_classes):
        rows.append({
            "classe": idx_to_class[c],
            "nb_echantillons_total": n_samples_per_class.get(c, 0),
            "recall": recall_per_class[c],
        })
    rows.sort(key=lambda r: r["nb_echantillons_total"])

    x = np.array([r["nb_echantillons_total"] for r in rows], dtype=float)
    y = np.array([r["recall"] for r in rows], dtype=float)
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        pearson_r = float(np.corrcoef(x, y)[0, 1])
    else:
        pearson_r = float("nan")

    print("\n[cv] === Lien nb échantillons / précision par classe ===")
    print(f"{'Classe':<28} {'Nb échantillons':>16} {'Recall':>10}")
    for r in rows:
        print(f"{r['classe']:<28} {r['nb_echantillons_total']:>16} {r['recall']:>10.3f}")
    print(f"\n[cv] Corrélation de Pearson (nb échantillons vs recall) : r = {pearson_r:.3f}")
    if not np.isnan(pearson_r):
        if pearson_r > 0.5:
            interp = "corrélation positive marquée -> plus de données = meilleure précision sur cette classe"
        elif pearson_r > 0.2:
            interp = "corrélation positive faible/modérée"
        elif pearson_r > -0.2:
            interp = "pas de lien clair (le modèle est déjà bon même sur les classes peu représentées)"
        else:
            interp = "corrélation négative (surprenant -- à examiner, souvent lié à une classe difficile indépendamment de sa taille)"
        print(f"[cv] Interprétation : {interp}")

    with open(output_dir / "count_vs_accuracy.csv", "w") as f:
        f.write("classe,nb_echantillons_total,recall\n")
        for r in rows:
            f.write(f"{r['classe']},{r['nb_echantillons_total']},{r['recall']:.4f}\n")

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(x, y, s=80, color="#2b6cb0")
        for r in rows:
            ax.annotate(r["classe"], (r["nb_echantillons_total"], r["recall"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("Nombre d'échantillons dans le dataset")
        ax.set_ylabel("Recall (précision) sur cette classe")
        ax.set_title(f"Nb échantillons vs précision par classe (r = {pearson_r:.2f})")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "count_vs_accuracy.png", dpi=150)
        plt.close(fig)
        print(f"[cv] Graphique sauvegardé : {output_dir / 'count_vs_accuracy.png'}")


if __name__ == "__main__":
    main()