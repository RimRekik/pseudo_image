#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration centralisée des hyperparamètres d'entraînement/évaluation.
Toutes les valeurs par défaut peuvent être surchargées en ligne de commande.

Exemples :
    # Entraînement normal, loss pondérée (par défaut)
    python3 main.py --dataset_dir /chemin/ms1_processed --epoches 30

    # Loss standard (non pondérée)
    python3 main.py --dataset_dir /chemin/ms1_processed --weighted_entropy False

    # SGD au lieu d'Adam
    python3 main.py --dataset_dir /chemin/ms1_processed --optim SGD --momentum 0.9 --lr 0.01

    # Reprendre l'entraînement depuis un checkpoint existant
    python3 main.py --dataset_dir /chemin/ms1_processed --pretrain_path output/best_model.pt

    # Évaluation SEULE (pas d'entraînement) sur un modèle déjà entraîné
    python3 main.py --dataset_dir /chemin/ms1_processed --test output/best_model.pt
"""

import argparse


def _str2bool(v):
    """Permet --weighted_entropy True / False / 1 / 0 en ligne de commande
    (contrairement à type=bool qui, en argparse, considère TOUT string non vide
    comme True -- y compris la chaîne 'False' ! Ce correctif évite ce piège)."""
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "1"):
        return True
    if str(v).lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Valeur booléenne attendue (True/False).")


def load_args():
    parser = argparse.ArgumentParser(description="Classification des 5 espèces à partir des pseudo-images MS1")

    # --- Mode ---
    parser.add_argument('--test', type=str, default=None,
                         help="Chemin vers un checkpoint (.pt) pour ÉVALUER UNIQUEMENT (pas d'entraînement). "
                              "Si non fourni (défaut), un nouveau modèle est entraîné.")

    # --- Entraînement ---
    parser.add_argument('--epoches', type=int, default=30, help="Nombre d'epochs maximum")
    parser.add_argument('--eval_inter', type=int, default=1,
                         help="Fréquence (en epochs) à laquelle la validation est calculée")
    parser.add_argument('--patience', type=int, default=8,
                         help="Early stopping : arrêt si pas d'amélioration de val_acc depuis N évaluations")

    # --- Données ---
    parser.add_argument('--dataset_dir', type=str, default='ms1_processed',
                         help="Dossier contenant les 5 sous-dossiers de classe (.npy)")
    parser.add_argument('--noise_threshold', type=int, default=0,
                         help="Seuil de filtrage du bruit de fond sur l'intensité brute (0 = désactivé)")
    parser.add_argument('--target_h', type=int, default=224, help="Hauteur cible après redimensionnement")
    parser.add_argument('--target_w', type=int, default=224, help="Largeur cible après redimensionnement")
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--random_state', type=int, default=42, help="Seed pour le split train/val/test")

    # --- Optimisation ---
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optim', type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument('--beta1', type=float, default=0.938, help="Adam uniquement")
    parser.add_argument('--beta2', type=float, default=0.9928, help="Adam uniquement")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD uniquement")
    parser.add_argument('--weighted_entropy', type=_str2bool, default=True,
                         help="True : pondère la loss par l'inverse de la fréquence des classes "
                              "(recommandé si classes déséquilibrées). False : CrossEntropyLoss standard.")
    parser.add_argument('--batch_size', type=int, default=16)

    # --- Modèle ---
    parser.add_argument('--model', type=str, default='resnet18',
                         choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'])
    parser.add_argument('--freeze_backbone', type=_str2bool, default=False,
                         help="True : gèle le backbone, n'entraîne que la dernière couche")
    parser.add_argument('--pretrain_path', type=str, default=None,
                         help="Chemin vers un checkpoint existant pour initialiser les poids "
                              "(au lieu des poids ImageNet par défaut)")

    # --- Sorties ---
    parser.add_argument('--save_path', type=str, default='output/best_model.pt',
                         help="Où sauvegarder le meilleur modèle. Les autres artefacts "
                              "(history.csv, courbes, rapport de test...) sont sauvegardés "
                              "dans le même dossier.")
    parser.add_argument('--output', type=str, default='output/test_predictions.csv',
                         help="Chemin du CSV listant les prédictions détaillées du test set "
                              "(1 ligne par fichier : chemin, vraie classe, classe prédite, correct)")

    args = parser.parse_args()
    return args