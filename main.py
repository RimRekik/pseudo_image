#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraînement et évaluation du modèle de classification des 5 espèces à partir
des pseudo-images MS1 (.npy). S'appuie sur dataset.py (chargement des données)
et model.py (architecture).

Usage entraînement :
    python3 main.py train \
        --root /chemin/vers/ms1_processed \
        --backbone resnet18 \
        --epochs 30 \
        --batch-size 16 \
        --output-dir ./runs/exp1

Usage évaluation seule (à partir d'un checkpoint) :
    python3 main.py evaluate \
        --root /chemin/vers/ms1_processed \
        --checkpoint ./runs/exp1/best_model.pt

Ce que ça produit dans --output-dir :
    best_model.pt           (poids du meilleur modèle selon la val accuracy)
    class_to_idx.json       (mapping classe -> index, indispensable pour l'inférence)
    history.csv             (loss/accuracy par epoch, train + val)
    test_report.txt         (precision/recall/f1 par classe + matrice de confusion, sur le test set)
    splits/                 (train/val/test_split.csv, pour audit -- via dataset.py)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter

from dataset.dataset import create_dataloaders, DEFAULT_TARGET_SIZE
from model.model import build_model

try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")  # pas d'affichage interactif requis, juste sauvegarde en fichier
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Boucle d'une epoch (train ou eval selon le flag)
# ---------------------------------------------------------------------------

def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
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


# ---------------------------------------------------------------------------
# Courbes d'entraînement (loss + accuracy) et détection de surapprentissage
# ---------------------------------------------------------------------------

def plot_training_curves(history, output_dir):
    if not HAS_MATPLOTLIB:
        print("[main] matplotlib non installé -> pas de courbes générées "
              "(`pip install matplotlib`). history.csv reste disponible.")
        return

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train_loss", marker="o")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val_loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss (train vs val)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train_acc", marker="o")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val_acc", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy (train vs val)")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = Path(output_dir) / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[main] Courbes d'entraînement sauvegardées : {out_path}")


def analyze_convergence(history):
    """
    Diagnostic simple, basé sur l'écart train/val en fin d'entraînement :
    - surapprentissage si train_acc >> val_acc (le modèle a mémorisé le train)
    - sous-apprentissage si les deux accuracies restent faibles
    - stagnation si val_acc n'a quasi plus bougé sur la 2e moitié de l'entraînement
    """
    if len(history) < 2:
        return "Pas assez d'epochs pour analyser la convergence."

    last = history[-1]
    gap = last["train_acc"] - last["val_acc"]
    lines = []

    if gap > 0.20:
        lines.append(f"⚠️  Surapprentissage probable : train_acc={last['train_acc']:.3f} "
                      f"vs val_acc={last['val_acc']:.3f} (écart={gap:.3f}). "
                      f"Le modèle mémorise le train set sans généraliser. "
                      f"Pistes : plus de données, augmentation de données, dropout/régularisation, "
                      f"réduire la capacité du modèle (resnet18 au lieu de resnet50), "
                      f"ou --freeze-backbone pour limiter le nombre de poids entraînés.")
    elif last["train_acc"] < 0.5 and last["val_acc"] < 0.5:
        lines.append(f"⚠️  Sous-apprentissage probable : train_acc={last['train_acc']:.3f} "
                      f"reste faible. Pistes : plus d'epochs, learning rate plus élevé, "
                      f"vérifier que les labels/images sont corrects, backbone plus gros.")
    else:
        lines.append(f"✅ Pas de signe évident de sur/sous-apprentissage "
                      f"(train_acc={last['train_acc']:.3f}, val_acc={last['val_acc']:.3f}).")

    half = len(history) // 2
    if half >= 1:
        val_acc_first_half = max(h["val_acc"] for h in history[:half])
        val_acc_second_half = max(h["val_acc"] for h in history[half:])
        if val_acc_second_half - val_acc_first_half < 0.01:
            lines.append("ℹ️  La val_acc a peu progressé sur la 2e moitié de l'entraînement "
                          "-> le modèle a probablement convergé (ou stagné). "
                          "Vérifie training_curves.png pour confirmer visuellement.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pondération des classes pour compenser le déséquilibre (ex: 554 E. coli vs
# 118 Proteus mirabilis) -- pondération inverse de la fréquence.
# ---------------------------------------------------------------------------

def compute_class_weights(train_loader, num_classes, device):
    counts = Counter(s["label"] for s in train_loader.dataset.samples)
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
    print(f"[main] Poids de classe (weighted_entropy) : "
          f"{dict((c, round(w, 3)) for c, w in enumerate(weights))}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Entraînement complet
# ---------------------------------------------------------------------------

def train_model(root_dir, output_dir, backbone="resnet18", pretrained=True, freeze_backbone=False,
                 epochs=30, batch_size=16, lr=1e-4, target_size=DEFAULT_TARGET_SIZE,
                 num_workers=2, patience=8, weighted_entropy=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device = {device}")

    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        root_dir=root_dir, batch_size=batch_size, target_size=target_size, num_workers=num_workers,
        save_splits_to=str(output_dir / "splits"),
    )
    with open(output_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

    num_classes = len(class_to_idx)
    model = build_model(num_classes, backbone=backbone, pretrained=pretrained,
                         freeze_backbone=freeze_backbone).to(device)

    if weighted_entropy:
        class_weights = compute_class_weights(train_loader, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = []
    best_val_acc = -1.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_acc)
        dt = time.time() - t0

        print(f"[epoch {epoch:03d}/{epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({dt:.1f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": backbone,
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "target_size": target_size,
            }, output_dir / "best_model.pt")
            print(f"  -> nouveau meilleur modèle sauvegardé (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[main] Early stopping : pas d'amélioration depuis {patience} epochs.")
                break

    with open(output_dir / "history.csv", "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for row in history:
            f.write(f"{row['epoch']},{row['train_loss']:.6f},{row['train_acc']:.6f},"
                    f"{row['val_loss']:.6f},{row['val_acc']:.6f}\n")

    plot_training_curves(history, output_dir)
    convergence_report = analyze_convergence(history)
    print(f"\n[main] Analyse de convergence :\n{convergence_report}")
    with open(output_dir / "convergence_analysis.txt", "w") as f:
        f.write(convergence_report)

    print(f"\n[main] Meilleure val_acc = {best_val_acc:.4f}")
    print("[main] Évaluation finale sur le test set avec le meilleur modèle...")
    evaluate_model(root_dir, output_dir / "best_model.pt", output_dir=output_dir,
                    target_size=target_size, batch_size=batch_size, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Évaluation à partir d'un checkpoint
# ---------------------------------------------------------------------------

def evaluate_model(root_dir, checkpoint_path, output_dir=None, target_size=DEFAULT_TARGET_SIZE,
                    batch_size=16, num_workers=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    _, _, test_loader, _ = create_dataloaders(
        root_dir=root_dir, batch_size=batch_size, target_size=ckpt.get("target_size", target_size),
        num_workers=num_workers,
    )
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(ckpt["num_classes"], backbone=ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _protocols in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    print(f"[eval] Test accuracy = {acc:.4f}")

    report_lines = [f"Test accuracy: {acc:.4f}", ""]
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    if HAS_SKLEARN:
        report_lines.append(classification_report(all_labels, all_preds, target_names=target_names, digits=3))
        report_lines.append("Matrice de confusion (lignes=vrai, colonnes=prédit) :")
        report_lines.append(str(target_names))
        report_lines.append(str(confusion_matrix(all_labels, all_preds)))
    else:
        report_lines.append("(scikit-learn non installé -> rapport détaillé indisponible, "
                             "seule l'accuracy globale est fournie. `pip install scikit-learn`)")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    if output_dir is not None:
        out_path = Path(output_dir) / "test_report.txt"
        with open(out_path, "w") as f:
            f.write(report_text)
        print(f"\n[eval] Rapport sauvegardé dans {out_path}")

        if HAS_SKLEARN and HAS_MATPLOTLIB:
            cm = confusion_matrix(all_labels, all_preds)
            plot_confusion_matrix(cm, target_names, output_dir)

    return acc


def plot_confusion_matrix(cm, class_names, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Vrai")
    ax.set_title("Matrice de confusion (test set)")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = Path(output_dir) / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eval] Matrice de confusion sauvegardée : {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Entraîne un modèle")
    p_train.add_argument("--root", required=True, help="Dossier ms1_processed")
    p_train.add_argument("--output-dir", default="./runs/exp1")
    p_train.add_argument("--backbone", default="resnet18",
                          choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"])
    p_train.add_argument("--no-pretrained", action="store_true", help="Ne pas charger les poids ImageNet")
    p_train.add_argument("--freeze-backbone", action="store_true",
                          help="Gèle le backbone, n'entraîne que la dernière couche")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--patience", type=int, default=8, help="Early stopping (epochs sans amélioration)")
    p_train.add_argument("--target-h", type=int, default=DEFAULT_TARGET_SIZE[0])
    p_train.add_argument("--target-w", type=int, default=DEFAULT_TARGET_SIZE[1])
    p_train.add_argument("--num-workers", type=int, default=2)
    p_train.add_argument("--weighted-entropy", dest="weighted_entropy", action="store_true", default=True,
                          help="Pondère le loss par l'inverse de la fréquence des classes (activé par défaut)")
    p_train.add_argument("--no-weighted-entropy", dest="weighted_entropy", action="store_false",
                          help="Désactive la pondération des classes (loss standard non pondéré)")

    p_eval = sub.add_parser("evaluate", help="Évalue un modèle déjà entraîné sur le test set")
    p_eval.add_argument("--root", required=True, help="Dossier ms1_processed")
    p_eval.add_argument("--checkpoint", required=True, help="Chemin vers best_model.pt")
    p_eval.add_argument("--batch-size", type=int, default=16)
    p_eval.add_argument("--num-workers", type=int, default=2)

    args = ap.parse_args()

    if args.command == "train":
        train_model(
            root_dir=args.root,
            output_dir=args.output_dir,
            backbone=args.backbone,
            pretrained=not args.no_pretrained,
            freeze_backbone=args.freeze_backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            target_size=(args.target_h, args.target_w),
            num_workers=args.num_workers,
            patience=args.patience,
            weighted_entropy=args.weighted_entropy,
        )
    elif args.command == "evaluate":
        evaluate_model(
            root_dir=args.root,
            checkpoint_path=args.checkpoint,
            output_dir=Path(args.checkpoint).parent,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )