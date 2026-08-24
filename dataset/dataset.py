#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset PyTorch pour les pseudo-images MS1 (.npy) triées par classe d'espèce.

Arborescence attendue (celle produite par sort_by_species.py) :
    ms1_processed/
    ├── Escherichia coli/*.npy
    ├── Klebsiella pneumoniae/*.npy
    ├── Enterobacter hormaechei/*.npy
    ├── Proteus mirabilis/*.npy
    └── Citrobacter freundii/*.npy

Ce qui est géré automatiquement :
1. Les classes sont déduites des noms de sous-dossiers (pas besoin de les coder en dur).
2. Les protocoles non retenus sont exclus, conformément à la doc fournie :
   'zPRM' (pas du DIA) et '300SPD' (acquisition trop courte).
3. Les images ont des tailles (H, W) variables (constaté : 901 ou 851 de large
   selon le protocole, hauteur variable). Chaque image est donc redimensionnée
   (interpolation bilinéaire) vers TARGET_SIZE au moment du chargement.
4. Split train/val/test stratifié par classe, reproductible (seed fixe),
   sauvegardé en CSV pour audit.

Usage rapide (test) :
    python3 dataset.py --root /chemin/vers/ms1_processed

Usage dans un script d'entraînement :
    from dataset import create_dataloaders
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        root_dir="/chemin/vers/ms1_processed",
        batch_size=16,
    )
"""

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Config par défaut -- à ajuster si besoin
# ---------------------------------------------------------------------------

# Protocoles à exclure du dataset, d'après le guide fourni :
# "300spd => trop court, pas retenu / zPRM => pas du DIA"
EXCLUDED_PROTOCOL_TOKENS = ["zprm", "300spd"]

# Taille cible (Hauteur, Largeur) vers laquelle chaque image est redimensionnée.
# 224x224 = taille standard attendue par les backbones ImageNet (ResNet/EfficientNet),
# la mieux adaptée au transfer learning et la plus rapide à entraîner.
DEFAULT_TARGET_SIZE = (224, 224)

VALID_EXTENSIONS = (".npy",)


def detect_protocol(filename: str) -> str:
    """Déduit le protocole d'acquisition à partir du nom de fichier (best-effort)."""
    lower = filename.lower()
    if "zprm" in lower:
        return "zPRM"
    if "300spd" in lower:
        return "300SPD"
    if "d200" in lower:
        return "d200"
    if "100vw" in lower or "100spd" in lower:
        return "100vW_100SPD"
    return "unknown"  # pas de tag explicite dans le nom -> pipeline par défaut


def is_excluded(filename: str) -> bool:
    lower = filename.lower()
    return any(tok in lower for tok in EXCLUDED_PROTOCOL_TOKENS)


# ---------------------------------------------------------------------------
# Construction de la liste des fichiers + labels
# ---------------------------------------------------------------------------

def extract_sample_group_key(class_name: str, filename: str):
    """
    Identifie l'échantillon biologique réel derrière un nom de fichier, en
    ignorant le milieu (AER/ANA) et le numéro de réplicat éventuel. On prend le
    premier segment purement numérique du nom (après avoir uniformisé espaces
    et tirets) : c'est le sample_nb, quel que soit le nombre de segments de
    code qui précèdent (COLI-217, CIT-FRE-18, EC 107, ENTHOR-38-AER-2 -> 217,
    18, 107, 38 respectivement). Retourne (class_name, sample_nb) : la clé de
    groupe utilisée pour que TOUTES les variantes (AER, ANA, réplicats) d'un
    même échantillon finissent dans le même split train/val/test.
    """
    stem = Path(filename).stem
    normalized = re.sub(r"\s+", "-", stem.strip())
    parts = normalized.split("-")
    for part in parts:
        if part.isdigit():
            return (class_name, part)
    # Aucun segment numérique trouvé (format inattendu) : on ne peut pas
    # grouper de façon fiable, chaque fichier constitue son propre groupe.
    return (class_name, f"__nogroup__:{stem}")


def scan_dataset(root_dir: str):
    """
    Parcourt root_dir/<classe>/**/*.npy et retourne :
      - samples : liste de dicts {path, label, class_name, protocol}
      - class_to_idx : {nom_de_classe: index}
    Les fichiers dont le protocole est exclu (zPRM, 300SPD) sont ignorés.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {root_dir}")

    class_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"Aucun sous-dossier (classe) trouvé dans {root_dir}")
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples = []
    n_excluded = 0
    for class_name in class_names:
        class_dir = root / class_name
        for p in class_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in VALID_EXTENSIONS:
                continue
            if is_excluded(p.name):
                n_excluded += 1
                continue
            samples.append({
                "path": str(p),
                "label": class_to_idx[class_name],
                "class_name": class_name,
                "protocol": detect_protocol(p.name),
                "group_key": extract_sample_group_key(class_name, p.name),
            })

    print(f"[dataset] {len(samples)} fichiers retenus, {n_excluded} exclus (zPRM/300SPD).")
    print(f"[dataset] Classes ({len(class_names)}) : {class_names}")
    return samples, class_to_idx


# ---------------------------------------------------------------------------
# Split train/val/test stratifié (sans dépendance à sklearn)
# ---------------------------------------------------------------------------

def stratified_split(samples, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Split stratifié par classe, MAIS en gardant ensemble tous les fichiers d'un
    même échantillon biologique (AER + ANA + réplicats -> même split), grâce à
    'group_key' (voir extract_sample_group_key). On assigne des GROUPES entiers
    à train/val/test, de façon gloutonne, pour approcher au mieux les fractions
    demandées au niveau du nombre de FICHIERS (pas juste du nombre de groupes).
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Les fractions doivent sommer à 1.0"

    # Regroupe les indices par (classe, group_key)
    groups_by_class = defaultdict(lambda: defaultdict(list))
    for i, s in enumerate(samples):
        groups_by_class[s["class_name"]][s["group_key"]].append(i)

    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []

    for class_name, groups in groups_by_class.items():
        group_items = list(groups.items())  # [(group_key, [indices]), ...]
        rng.shuffle(group_items)

        n_total = sum(len(idx) for _, idx in group_items)
        target_train = round(n_total * train_frac)
        target_val = round(n_total * val_frac)

        n_train, n_val = 0, 0
        for _group_key, indices in group_items:
            if n_train < target_train:
                train_idx.extend(indices)
                n_train += len(indices)
            elif n_val < target_val:
                val_idx.extend(indices)
                n_val += len(indices)
            else:
                test_idx.extend(indices)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def save_split_csv(samples, indices, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "class_name", "label", "protocol", "group_key"])
        writer.writeheader()
        for i in indices:
            writer.writerow(samples[i])


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

class MS1NpyDataset(Dataset):
    """
    Dataset PyTorch pour les pseudo-images MS1 (.npy, 2D, float32).
    Chaque item retourné : (image_tensor [1,H,W] float32, label:int, protocol:str)
    """

    def __init__(self, samples, target_size=DEFAULT_TARGET_SIZE, normalize="minmax", transform=None,
                 noise_threshold=0):
        """
        samples      : liste de dicts (voir scan_dataset), typiquement un sous-ensemble
                       (train/val/test) obtenu via stratified_split.
        target_size  : (H, W) — taille de sortie après redimensionnement (224x224
                       par défaut, standard pour les backbones ImageNet).
        normalize    : 'minmax' (par défaut -- adapté à des données DÉJÀ en
                       échelle log10, comme les tiennes : on ne fait qu'un
                       min-max par image, sans reprendre un log dessus),
                       'log1p_minmax' (log1p + min-max -- à utiliser SEULEMENT
                       si les .npy sont en intensité brute/linéaire, PAS déjà
                       log-transformés, sous peine de double-log),
                       'zscore', ou None (aucune normalisation).
        transform    : fonction optionnelle appliquée APRES normalisation
                       (ex: augmentation), signature: tensor[1,H,W] -> tensor[1,H,W].
        noise_threshold : seuil (sur la valeur BRUTE lue dans le .npy, avant
                       normalisation) sous lequel les intensités sont ramenées à
                       la valeur min de l'image -- filtrage simple du bruit de
                       fond. 0 (défaut) = désactivé, aucun effet.
        """
        self.samples = samples
        self.target_size = target_size
        self.normalize = normalize
        self.transform = transform
        self.noise_threshold = noise_threshold

    def __len__(self):
        return len(self.samples)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if self.noise_threshold:
            arr = np.where(arr < self.noise_threshold, arr.min(), arr)
        if self.normalize is None:
            return arr
        if self.normalize == "log1p_minmax":
            arr = np.log1p(np.clip(arr, a_min=0, a_max=None))
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-8)
        if self.normalize == "minmax":
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-8)
        if self.normalize == "zscore":
            mu, sigma = arr.mean(), arr.std()
            return (arr - mu) / (sigma + 1e-8)
        raise ValueError(f"normalize inconnu : {self.normalize}")

    def __getitem__(self, idx):
        entry = self.samples[idx]
        arr = np.load(entry["path"], allow_pickle=False).astype(np.float32)
        arr = self._normalize(arr)

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W] pour interpolate
        tensor = F.interpolate(tensor, size=self.target_size, mode="bilinear", align_corners=False)
        tensor = tensor.squeeze(0)  # -> [1,H,W]

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, entry["label"], entry["protocol"]


# ---------------------------------------------------------------------------
# Fonction "tout-en-un" pour obtenir les 3 DataLoaders
# ---------------------------------------------------------------------------

def create_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    target_size=DEFAULT_TARGET_SIZE,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    num_workers: int = 2,
    normalize: str = "minmax",
    noise_threshold: int = 0,
    save_splits_to: str = None,
):
    samples, class_to_idx = scan_dataset(root_dir)
    train_idx, val_idx, test_idx = stratified_split(samples, train_frac, val_frac, test_frac, seed)

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    print(f"[dataset] Split -> train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}")

    if save_splits_to:
        out = Path(save_splits_to)
        out.mkdir(parents=True, exist_ok=True)
        save_split_csv(samples, train_idx, out / "train_split.csv")
        save_split_csv(samples, val_idx, out / "val_split.csv")
        save_split_csv(samples, test_idx, out / "test_split.csv")
        print(f"[dataset] Splits sauvegardés dans {out}")

    train_ds = MS1NpyDataset(train_samples, target_size=target_size, normalize=normalize,
                              noise_threshold=noise_threshold)
    val_ds = MS1NpyDataset(val_samples, target_size=target_size, normalize=normalize,
                            noise_threshold=noise_threshold)
    test_ds = MS1NpyDataset(test_samples, target_size=target_size, normalize=normalize,
                             noise_threshold=noise_threshold)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_to_idx


# ---------------------------------------------------------------------------
# Test rapide en ligne de commande
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Dossier ms1_processed (contenant les 5 sous-dossiers de classe)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--target-h", type=int, default=DEFAULT_TARGET_SIZE[0])
    ap.add_argument("--target-w", type=int, default=DEFAULT_TARGET_SIZE[1])
    ap.add_argument("--save-splits-to", default=None, help="Dossier où sauvegarder train/val/test_split.csv")
    args = ap.parse_args()

    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        root_dir=args.root,
        batch_size=args.batch_size,
        target_size=(args.target_h, args.target_w),
        save_splits_to=args.save_splits_to,
    )

    print("\nclass_to_idx :", class_to_idx)

    images, labels, protocols = next(iter(train_loader))
    print(f"\nUn batch d'entraînement :")
    print(f"  images   : shape={tuple(images.shape)} dtype={images.dtype}")
    print(f"  labels   : {labels.tolist()}")
    print(f"  protocols: {list(protocols)}")
    print(f"  min={images.min().item():.4f}  max={images.max().item():.4f}  mean={images.mean().item():.4f}")