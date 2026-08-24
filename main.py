#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entraînement et évaluation du modèle de classification des 5 espèces à partir
des pseudo-images MS1 (.npy). Tous les hyperparamètres viennent de config.py
(load_args) -- voir ce fichier pour la liste complète et des exemples d'usage.

    python3 main.py --dataset_dir /chemin/vers/ms1_processed --epoches 30
    python3 main.py --dataset_dir /chemin/vers/ms1_processed --test output/best_model.pt

Fichiers produits dans le dossier de --save_path :
    best_model.pt              poids du meilleur modèle (val_acc la plus haute)
    class_to_idx.json          mapping classe -> index
    history.csv                loss/accuracy par epoch (train + val)
    training_curves.png        courbes loss/accuracy train vs val
    convergence_analysis.txt   diagnostic auto (sur/sous-apprentissage, stagnation)
    test_report.txt            precision/recall/f1 par classe + matrice de confusion (texte)
    confusion_matrix.png       la même matrice, en image
    splits/                    train/val/test_split.csv (traçabilité)
Et, au chemin indiqué par --output :
    test_predictions.csv       1 ligne par fichier de test : path, vraie classe, classe prédite, correct
"""

import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import load_args
from dataset.dataset import create_dataloaders, DEFAULT_TARGET_SIZE
from model.model import build_model

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


# ---------------------------------------------------------------------------
# Courbes d'entraînement + diagnostic de convergence
# ---------------------------------------------------------------------------

def plot_training_curves(history, output_dir):
    if not HAS_MATPLOTLIB:
        print("[main] matplotlib non installé -> pas de courbes générées (`pip install matplotlib`).")
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train_loss", marker="o")
    axes[0].plot(epochs, [h["val_loss"] for h in history if h["val_loss"] is not None],
                  label="val_loss", marker="o")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss (train vs val)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train_acc", marker="o")
    val_epochs = [h["epoch"] for h in history if h["val_acc"] is not None]
    val_accs = [h["val_acc"] for h in history if h["val_acc"] is not None]
    axes[1].plot(val_epochs, val_accs, label="val_acc", marker="o")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Accuracy (train vs val)")
    axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = Path(output_dir) / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[main] Courbes d'entraînement sauvegardées : {out_path}")


def analyze_convergence(history):
    evaluated = [h for h in history if h["val_acc"] is not None]
    if len(evaluated) < 2:
        return "Pas assez d'évaluations pour analyser la convergence."

    last = evaluated[-1]
    gap = last["train_acc"] - last["val_acc"]
    lines = []

    if gap > 0.20:
        lines.append(f"⚠️  Surapprentissage probable : train_acc={last['train_acc']:.3f} "
                      f"vs val_acc={last['val_acc']:.3f} (écart={gap:.3f}). "
                      f"Pistes : plus de données, augmentation de données, dropout/régularisation, "
                      f"réduire la capacité du modèle, ou --freeze_backbone True.")
    elif last["train_acc"] < 0.5 and last["val_acc"] < 0.5:
        lines.append(f"⚠️  Sous-apprentissage probable : train_acc={last['train_acc']:.3f} reste faible. "
                      f"Pistes : plus d'epochs, --lr plus élevé, vérifier labels/images, backbone plus gros.")
    else:
        lines.append(f"✅ Pas de signe évident de sur/sous-apprentissage "
                      f"(train_acc={last['train_acc']:.3f}, val_acc={last['val_acc']:.3f}).")

    half = len(evaluated) // 2
    if half >= 1:
        first_half_best = max(h["val_acc"] for h in evaluated[:half])
        second_half_best = max(h["val_acc"] for h in evaluated[half:])
        if second_half_best - first_half_best < 0.01:
            lines.append("ℹ️  La val_acc a peu progressé sur la 2e moitié de l'entraînement "
                          "-> convergence probable (ou stagnation). Vérifie training_curves.png.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pondération des classes (loss pondérée)
# ---------------------------------------------------------------------------

def compute_class_weights(train_loader, num_classes, device):
    counts = Counter(s["label"] for s in train_loader.dataset.samples)
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
    print(f"[main] Poids de classe : {dict((c, round(w, 3)) for c, w in enumerate(weights))}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_optimizer(args, trainable_params):
    if args.optim == "Adam":
        return optim.Adam(trainable_params, lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optim == "SGD":
        return optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum)
    raise ValueError(f"--optim inconnu : {args.optim} (choix: Adam, SGD)")


# ---------------------------------------------------------------------------
# Une epoch (train ou eval)
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
# Entraînement complet
# ---------------------------------------------------------------------------

def train_model(args):
    output_dir = Path(args.save_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # accélère si la taille d'image est fixe (224x224 ici)
    print(f"[main] device = {device}")

    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        root_dir=args.dataset_dir,
        batch_size=args.batch_size,
        target_size=(args.target_h, args.target_w),
        num_workers=args.num_workers,
        noise_threshold=args.noise_threshold,
        seed=args.random_state,
        save_splits_to=str(output_dir / "splits"),
    )
    with open(output_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)

    num_classes = len(class_to_idx)
    model = build_model(num_classes, backbone=args.model,
                         pretrained=(args.pretrain_path is None),
                         freeze_backbone=args.freeze_backbone).to(device)

    if args.pretrain_path:
        print(f"[main] Chargement des poids depuis --pretrain_path={args.pretrain_path}")
        ckpt = torch.load(args.pretrain_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    if args.weighted_entropy:
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_loader, num_classes, device))
    else:
        print("[main] Loss standard (non pondérée) -- --weighted_entropy False")
        criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(args, trainable_params)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = []
    best_val_acc = -1.0
    evals_no_improve = 0

    for epoch in range(1, args.epoches + 1):
        t0 = time.time()
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)

        do_eval = (epoch % args.eval_inter == 0) or (epoch == args.epoches)
        val_loss, val_acc = (None, None)
        if do_eval:
            val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
            scheduler.step(val_acc)

        dt = time.time() - t0
        val_str = f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}" if do_eval else "(pas d'évaluation cette epoch)"
        print(f"[epoch {epoch:03d}/{args.epoches}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"{val_str}  ({dt:.1f}s)")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if do_eval:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                evals_no_improve = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "backbone": args.model,
                    "num_classes": num_classes,
                    "class_to_idx": class_to_idx,
                    "target_size": (args.target_h, args.target_w),
                }, args.save_path)
                print(f"  -> nouveau meilleur modèle sauvegardé (val_acc={val_acc:.4f}) -> {args.save_path}")
            else:
                evals_no_improve += 1
                if evals_no_improve >= args.patience:
                    print(f"[main] Early stopping : pas d'amélioration depuis {args.patience} évaluations.")
                    break

    with open(output_dir / "history.csv", "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for row in history:
            vl = f"{row['val_loss']:.6f}" if row["val_loss"] is not None else ""
            va = f"{row['val_acc']:.6f}" if row["val_acc"] is not None else ""
            f.write(f"{row['epoch']},{row['train_loss']:.6f},{row['train_acc']:.6f},{vl},{va}\n")

    plot_training_curves(history, output_dir)
    convergence_report = analyze_convergence(history)
    print(f"\n[main] Analyse de convergence :\n{convergence_report}")
    with open(output_dir / "convergence_analysis.txt", "w") as f:
        f.write(convergence_report)

    print(f"\n[main] Meilleure val_acc = {best_val_acc:.4f}")
    print("[main] Évaluation finale sur le test set avec le meilleur modèle...")
    evaluate_model(args, checkpoint_path=args.save_path, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Évaluation (test set) à partir d'un checkpoint
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, class_names, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai"); ax.set_title("Matrice de confusion (test set)")
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


def evaluate_model(args, checkpoint_path=None, output_dir=None):
    checkpoint_path = checkpoint_path or args.test
    output_dir = output_dir or Path(checkpoint_path).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    _, _, test_loader, _ = create_dataloaders(
        root_dir=args.dataset_dir, batch_size=args.batch_size,
        target_size=ckpt.get("target_size", (args.target_h, args.target_w)),
        num_workers=args.num_workers, noise_threshold=args.noise_threshold, seed=args.random_state,
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
            preds = model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    # récupère les chemins dans le même ordre (test_loader n'est pas shuffle)
    all_paths = [s["path"] for s in test_loader.dataset.samples]

    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    print(f"[eval] Test accuracy = {acc:.4f}")

    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    report_lines = [f"Test accuracy: {acc:.4f}", ""]
    if HAS_SKLEARN:
        report_lines.append(classification_report(all_labels, all_preds, target_names=target_names, digits=3))
        report_lines.append("Matrice de confusion (lignes=vrai, colonnes=prédit) :")
        report_lines.append(str(target_names))
        cm = confusion_matrix(all_labels, all_preds)
        report_lines.append(str(cm))
        if HAS_MATPLOTLIB:
            plot_confusion_matrix(cm, target_names, output_dir)
    else:
        report_lines.append("(scikit-learn non installé -> `pip install scikit-learn`)")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    with open(Path(output_dir) / "test_report.txt", "w") as f:
        f.write(report_text)

    # CSV détaillé par fichier -> args.output
    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_class", "pred_class", "correct"])
        for path, y_true, y_pred in zip(all_paths, all_labels, all_preds):
            writer.writerow([path, idx_to_class[y_true], idx_to_class[y_pred], int(y_true == y_pred)])
    print(f"[eval] Prédictions détaillées sauvegardées : {out_csv}")

    return acc


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = load_args()
    if args.test:
        evaluate_model(args, checkpoint_path=args.test)
    else:
        train_model(args)