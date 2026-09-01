#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse a posteriori du lien entre nombre d'échantillons par classe et
précision du modèle -- à utiliser si tu as DÉJÀ un job d'entraînement (ou de
validation croisée) terminé, sans avoir besoin de relancer un entraînement.

Usage :
    python3 analyze_count_vs_accuracy.py \
        --dataset_dir ../ms1_processed \
        --report path/vers/test_report.txt (ou cv_report.txt)

Ça compte automatiquement le nombre de fichiers .npy par classe dans
--dataset_dir, extrait le recall par classe depuis le rapport texte
(peu importe qu'il vienne de main.py ou cross_validate.py, le format
classification_report de sklearn est le même), et produit le graphique +
la corrélation.
"""

import argparse
import re
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

VALID_EXTENSIONS = (".npy",)
EXCLUDED_PROTOCOL_TOKENS = ["zprm", "300spd"]


def count_samples_per_class(dataset_dir):
    """Recompte le nombre de fichiers .npy retenus par classe (même logique
    d'exclusion zPRM/300SPD que dataset.py, sans dépendre du reste du code)."""
    root = Path(dataset_dir)
    counts = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        n = 0
        for p in class_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in VALID_EXTENSIONS:
                continue
            if any(tok in p.name.lower() for tok in EXCLUDED_PROTOCOL_TOKENS):
                continue
            n += 1
        counts[class_dir.name] = n
    return counts


def parse_classification_report(report_text):
    """
    Extrait {nom_de_classe: recall} depuis un bloc classification_report de
    sklearn tel que sauvegardé dans test_report.txt / cv_report.txt.
    Repère les lignes du type :
        Nom De Classe      0.912   0.969   0.939   32
    (precision, recall, f1-score, support), en ignorant les lignes
    'accuracy', 'macro avg', 'weighted avg'.
    """
    recall_per_class = {}
    for line in report_text.splitlines():
        line = line.strip()
        if not line or line.startswith("precision") or "avg" in line or line.startswith("accuracy"):
            continue
        m = re.match(r"^(.*?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", line)
        if m:
            class_name = m.group(1).strip()
            recall = float(m.group(3))
            recall_per_class[class_name] = recall
    return recall_per_class


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True, help="Dossier ms1_processed (pour compter les échantillons)")
    ap.add_argument("--report", required=True, help="Chemin vers test_report.txt ou cv_report.txt déjà généré")
    ap.add_argument("--output_dir", default=".", help="Où sauvegarder le graphique/csv")
    args = ap.parse_args()

    counts = count_samples_per_class(args.dataset_dir)
    report_text = Path(args.report).read_text()
    recalls = parse_classification_report(report_text)

    print("Classes trouvées dans le dataset :", list(counts.keys()))
    print("Classes trouvées dans le rapport  :", list(recalls.keys()))

    rows = []
    for class_name, n in counts.items():
        if class_name not in recalls:
            print(f"⚠️  '{class_name}' non trouvé dans le rapport -- vérifie l'orthographe exacte, ignoré.")
            continue
        rows.append({"classe": class_name, "nb_echantillons": n, "recall": recalls[class_name]})

    if len(rows) < 2:
        print("Pas assez de classes appariées pour calculer une corrélation. Vérifie --report.")
        return

    rows.sort(key=lambda r: r["nb_echantillons"])
    x = np.array([r["nb_echantillons"] for r in rows], dtype=float)
    y = np.array([r["recall"] for r in rows], dtype=float)
    pearson_r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")

    print(f"\n{'Classe':<28} {'Nb échantillons':>16} {'Recall':>10}")
    for r in rows:
        print(f"{r['classe']:<28} {r['nb_echantillons']:>16} {r['recall']:>10.3f}")
    print(f"\nCorrélation de Pearson (nb échantillons vs recall) : r = {pearson_r:.3f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "count_vs_accuracy.csv", "w") as f:
        f.write("classe,nb_echantillons,recall\n")
        for r in rows:
            f.write(f"{r['classe']},{r['nb_echantillons']},{r['recall']:.4f}\n")

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(x, y, s=80, color="#2b6cb0")
        for r in rows:
            ax.annotate(r["classe"], (r["nb_echantillons"], r["recall"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("Nombre d'échantillons dans le dataset")
        ax.set_ylabel("Recall (précision) sur cette classe")
        ax.set_title(f"Nb échantillons vs précision par classe (r = {pearson_r:.2f})")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "count_vs_accuracy.png", dpi=150)
        plt.close(fig)
        print(f"Graphique sauvegardé : {out_dir / 'count_vs_accuracy.png'}")


if __name__ == "__main__":
    main()