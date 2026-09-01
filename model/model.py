#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Architecture du modèle de classification (transfer learning ResNet / EfficientNet,
adapté en entrée 1 canal pour les pseudo-images MS1).

Ce fichier ne contient QUE la définition du modèle. L'entraînement et
l'évaluation se trouvent dans main.py.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def _adapt_first_conv_to_1_channel(conv: nn.Conv2d, pretrained: bool) -> nn.Conv2d:
    """
    Remplace une conv d'entrée 3 canaux (RGB, format ImageNet) par une conv
    1 canal (nos images MS1 sont en niveaux de gris / intensité scalaire).
    Si pretrained=True, les nouveaux poids sont la moyenne des 3 canaux du
    poids pré-entraîné, pour conserver l'information apprise sur ImageNet
    plutôt que de repartir de zéro sur cette première couche.
    """
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None),
    )
    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
    return new_conv

def species_to_genus(species_name: str) -> str:
    """Extrait le genre (1er mot) d'un nom d'espèce complet, ex:
    'Klebsiella pneumoniae' -> 'Klebsiella'."""
    return species_name.strip().split(" ")[0]

def build_model(num_classes: int, backbone: str = "resnet18", pretrained: bool = True,
                 freeze_backbone: bool = False) -> nn.Module:
    """
    backbone        : 'resnet18', 'resnet34', 'resnet50', ou 'efficientnet_b0'
    pretrained      : charge les poids ImageNet (téléchargés automatiquement par
                      torchvision au premier lancement -- nécessite internet ce jour-là)
    freeze_backbone : si True, gèle tout le réseau sauf la dernière couche
                      (fine-tuning rapide, utile si peu de données) ; si False,
                      tout le réseau est entraîné (recommandé si assez de données)
    """
    backbone = backbone.lower()

    if backbone in ("resnet18", "resnet34", "resnet50"):
        weights_arg = "IMAGENET1K_V1" if pretrained else None
        ctor = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50}[backbone]
        net = ctor(weights=weights_arg)
        net.conv1 = _adapt_first_conv_to_1_channel(net.conv1, pretrained)
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)  # nouvelle couche -> toujours entraînable

    elif backbone == "efficientnet_b0":
        weights_arg = "IMAGENET1K_V1" if pretrained else None
        net = models.efficientnet_b0(weights=weights_arg)
        old_conv = net.features[0][0]
        net.features[0][0] = _adapt_first_conv_to_1_channel(old_conv, pretrained)
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"backbone inconnu : {backbone} (choix: resnet18, resnet34, resnet50, efficientnet_b0)")

    return net