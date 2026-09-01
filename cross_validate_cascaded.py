#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification en CASCADE : d'abord le genre, puis l'espèce -- deux modèles
distincts et séquentiels (pas un multi-tâche à backbone partagé).

Pipeline d'inférence :
    Image -> Modèle GENRE (ex: 6 classes) -> genre prédit
           -> si ce genre n'a qu'UNE espèce dans le dataset : réponse directe
           -> sinon : Modèle ESPÈCE dédié à ce genre (ex: Klebsiella -> 5
              espèces) -> espèce finale prédite

Sur tes 11 espèces / 6 genres, seuls 2 genres ont plusieurs espèces à
distinguer (Klebsiella: 5, Citrobacter: 2) -- les 4 autres n'ont qu'une seule
espèce, la réponse est donc automatique dès que le genre est identifié.

Une erreur de genre entraîne automatiquement une erreur d'espèce (cascade
d'erreur) -- c'est le comportement réel d'un système en cascade, contrairement
à un multi-tâche où les deux têtes sont indépendantes.

Usage :
    python3 cross_validate_cascaded.py \
        --dataset_dir ../ms1_processed_11classes --cv_folds 5 \
        --epoches 30 --batch_size 32 \
        --cv_output_dir output_cv_cascaded
"""

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
from model.model import build_model, species_to_genus
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


# --- Réutilisation de la logique déjà validée dans cross_validate.py ---

def make_k_folds(samples, k, seed):
    groups_by_class = defaultdict(lambda: defaultdict(list))
    for i, s in enumerate(samples):
        groups_by_class[s["class_name"]][s["group_key"]].append(i)
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for class_name, groups in groups_by_class.items():
        group_items = list(groups.items())
        rng.shuffle(group_items)
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


def class_weights_tensor(labels, num_classes, device):
    counts = Counter(labels)
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


def make_relabeled_samples(samples_subset, label_map):
    """Copie les dicts d'échantillons avec un nouveau champ 'label' remappé
    (ex: espèce globale -> genre, ou espèce globale -> index local au genre)."""
    out = []
    for s in samples_subset:
        s2 = dict(s)
        s2["label"] = label_map(s)
        out.append(s2)
    return out


def build_loader(samples_subset, target_size, noise_threshold, batch_size, num_workers,
                  transform=None, oversample=False, shuffle=True):
    ds = MS1NpyDataset(samples_subset, target_size=target_size, normalize="minmax",
                        noise_threshold=noise_threshold, transform=transform)
    if oversample:
        counts = Counter(s["label"] for s in samples_subset)
        w = [1.0 / counts[s["label"]] for s in samples_subset]
        sampler = WeightedRandomSampler(w, num_samples=len(samples_subset), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                           num_workers=num_workers, pin_memory=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, pin_memory=True)


def train_classifier(args, train_subset, val_subset, num_classes, device, target_size):
    """Entraîne un classifieur ResNet18 standard (genre OU espèce-dans-un-genre)."""
    train_transform = build_train_transform(
        rt_shift_prob=args.rt_shift_prob, rt_shift_mean=args.rt_shift_mean,
        rt_shift_std=args.rt_shift_std, noise_prob=args.noise_prob, noise_max=args.noise_max,
    )
    train_loader = build_loader(train_subset, target_size, args.noise_threshold, args.batch_size,
                                 args.num_workers, transform=train_transform, oversample=args.oversample)
    val_loader = build_loader(val_subset, target_size, args.noise_threshold, args.batch_size,
                               args.num_workers, shuffle=False)

    model = build_model(num_classes, backbone=args.model, pretrained=True,
                         freeze_backbone=args.freeze_backbone).to(device)

    if args.weighted_entropy:
        labels = [s["label"] for s in train_subset]
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor(labels, num_classes, device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = (optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
                 if args.optim == "Adam" else
                 optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc, best_state, evals_no_improve = -1.0, None, 0
    for epoch in range(1, args.epoches + 1):
        run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        _, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)
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
    return model


def train_one_fold_cascaded(args, samples, class_to_idx, train_idx, val_idx, test_idx, device):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_size = (args.target_h, args.target_w)

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    # --- Structure genre / espèces ---
    genera = sorted(set(species_to_genus(c) for c in class_to_idx))
    genus_to_idx = {g: i for i, g in enumerate(genera)}
    species_by_genus = defaultdict(list)  # genre -> [espèces (noms complets)]
    for name in class_to_idx:
        species_by_genus[species_to_genus(name)].append(name)

    # --- 1) Entraîne le modèle GENRE sur tout le train set ---
    genus_train = make_relabeled_samples(
        train_samples, lambda s: genus_to_idx[species_to_genus(s["class_name"])])
    genus_val = make_relabeled_samples(
        val_samples, lambda s: genus_to_idx[species_to_genus(s["class_name"])])
    genus_model = train_classifier(args, genus_train, genus_val, len(genera), device, target_size)

    # --- 2) Entraîne un modèle ESPÈCE dédié pour chaque genre à >1 espèce ---
    species_models = {}       # genre -> (model, [espèces dans l'ordre des indices locaux])
    for genus, species_list in species_by_genus.items():
        if len(species_list) <= 1:
            continue  # une seule espèce dans ce genre -> pas besoin de modèle
        local_idx = {sp: i for i, sp in enumerate(sorted(species_list))}
        sub_train = [s for s in train_samples if species_to_genus(s["class_name"]) == genus]
        sub_val = [s for s in val_samples if species_to_genus(s["class_name"]) == genus]
        sub_train = make_relabeled_samples(sub_train, lambda s: local_idx[s["class_name"]])
        sub_val = make_relabeled_samples(sub_val, lambda s: local_idx[s["class_name"]])
        sub_model = train_classifier(args, sub_train, sub_val, len(species_list), device, target_size)
        species_models[genus] = (sub_model, sorted(species_list))

    # --- 3) Évaluation en cascade sur le test set (image par image, car le
    #         routage vers le bon sous-modèle dépend de la prédiction genre) ---
    test_ds = MS1NpyDataset(test_samples, target_size=target_size, normalize="minmax",
                             noise_threshold=args.noise_threshold)

    all_true_species, all_pred_species = [], []
    all_true_genus, all_pred_genus = [], []

    with torch.no_grad():
        for i in range(len(test_ds)):
            tensor, true_species_idx, _protocol = test_ds[i]
            true_species_name = idx_to_class[true_species_idx]
            true_genus_name = species_to_genus(true_species_name)

            x = tensor.unsqueeze(0).to(device)
            genus_logits = genus_model(x)
            pred_genus_idx = genus_logits.argmax(dim=1).item()
            pred_genus_name = genera[pred_genus_idx]

            if pred_genus_name in species_models:
                sub_model, species_order = species_models[pred_genus_name]
                species_logits = sub_model(x)
                local_pred = species_logits.argmax(dim=1).item()
                pred_species_name = species_order[local_pred]
            else:
                # genre prédit n'a qu'une seule espèce (ou genre inconnu du train,
                # cas limite improbable) -> réponse directe
                candidates = species_by_genus.get(pred_genus_name, [])
                pred_species_name = candidates[0] if candidates else true_species_name

            all_true_species.append(class_to_idx[true_species_name])
            all_pred_species.append(class_to_idx[pred_species_name])
            all_true_genus.append(genus_to_idx[true_genus_name])
            all_pred_genus.append(genus_to_idx[pred_genus_name])

    species_acc = float(np.mean(np.array(all_pred_species) == np.array(all_true_species)))
    genus_acc = float(np.mean(np.array(all_pred_genus) == np.array(all_true_genus)))
    return species_acc, genus_acc, all_true_species, all_pred_species


def main():
    args = load_args()
    k = getattr(args, "cv_folds", 5)
    output_dir = Path(getattr(args, "cv_output_dir", "output_cv_cascaded"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[cv-cascade] device = {device}")

    samples, class_to_idx = scan_dataset(args.dataset_dir)
    idx_to_class = {v: k_ for k_, v in class_to_idx.items()}
    num_species = len(class_to_idx)

    genera = sorted(set(species_to_genus(c) for c in class_to_idx))
    species_by_genus = defaultdict(list)
    for name in class_to_idx:
        species_by_genus[species_to_genus(name)].append(name)
    print(f"[cv-cascade] {len(genera)} genres : {genera}")
    for g, sp in species_by_genus.items():
        tag = "-> modèle espèce dédié" if len(sp) > 1 else "-> réponse directe (1 seule espèce)"
        print(f"    {g} ({len(sp)} espèce(s)) {tag} : {sp}")

    folds = make_k_folds(samples, k, args.random_state)
    print(f"[cv-cascade] {k} folds -> tailles: {[len(f) for f in folds]}")

    species_accs, genus_accs = [], []
    all_labels_agg, all_preds_agg = [], []

    for i in range(k):
        test_idx = folds[i]
        remaining = [idx for j in range(k) if j != i for idx in folds[j]]
        train_idx, val_idx = split_train_val(remaining, val_frac=0.15, seed=args.random_state + i)

        print(f"\n[cv-cascade] === Fold {i+1}/{k} === train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
        species_acc, genus_acc, labels, preds = train_one_fold_cascaded(
            args, samples, class_to_idx, train_idx, val_idx, test_idx, device
        )
        print(f"[cv-cascade] Fold {i+1} -> espèce (cascade) = {species_acc:.4f}  |  genre = {genus_acc:.4f}")
        species_accs.append(species_acc)
        genus_accs.append(genus_acc)
        all_labels_agg.extend(labels)
        all_preds_agg.extend(preds)

    mean_sp, std_sp = float(np.mean(species_accs)), float(np.std(species_accs))
    mean_ge, std_ge = float(np.mean(genus_accs)), float(np.std(genus_accs))
    print(f"\n[cv-cascade] === Résultat final ESPÈCE (cascade) : {mean_sp:.4f} ± {std_sp:.4f} ===")
    print(f"[cv-cascade] === Résultat final GENRE              : {mean_ge:.4f} ± {std_ge:.4f} ===")

    report_lines = [
        f"Validation croisée CASCADÉE (genre -> espèce) à {k} folds",
        f"Accuracy ESPÈCE finale (après cascade) : {mean_sp:.4f} +/- {std_sp:.4f}  "
        f"(comparable à cross_validate.py)",
        f"Accuracy GENRE (étape 1)                : {mean_ge:.4f} +/- {std_ge:.4f}",
        f"Accuracies espèce par fold : {[round(a, 4) for a in species_accs]}",
        f"Accuracies genre par fold  : {[round(a, 4) for a in genus_accs]}",
        "",
    ]
    target_names = [idx_to_class[i] for i in range(num_species)]
    if HAS_SKLEARN:
        report_lines.append("Rapport agrégé ESPÈCE (tous les folds concaténés, après cascade) :")
        report_lines.append(classification_report(all_labels_agg, all_preds_agg,
                                                    target_names=target_names, digits=3))
        cm = confusion_matrix(all_labels_agg, all_preds_agg)
        report_lines.append(str(target_names))
        report_lines.append(str(cm))
        if HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=(7, 6.5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(target_names))); ax.set_yticks(range(len(target_names)))
            ax.set_xticklabels(target_names, rotation=45, ha="right"); ax.set_yticklabels(target_names)
            ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
            ax.set_title(f"Matrice de confusion ESPÈCE - cascade genre->espèce ({k}-fold CV)")
            thresh = cm.max() / 2.0
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                            color="white" if cm[r, c] > thresh else "black", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(output_dir / "confusion_matrix_cascade.png", dpi=150)
            plt.close(fig)

    with open(output_dir / "cv_cascade_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n[cv-cascade] Rapport sauvegardé dans {output_dir / 'cv_cascade_report.txt'}")


if __name__ == "__main__":
    main()