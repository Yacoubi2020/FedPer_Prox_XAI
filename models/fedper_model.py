"""Modèle FedPer-Prox : backbone partagé + head personnalisée par client."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Backbone(nn.Module):
    """Backbone partagé entre tous les clients (agrégé par le serveur)."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


class PersonalizedHead(nn.Module):
    """Head personnalisée (reste locale au client)."""

    def __init__(self, in_dim: int, hidden_dims: List[int], num_classes: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FedPerProxModel(nn.Module):
    """Modèle FedPer-Prox = Backbone(partagé) + Head(personnalisée)."""

    def __init__(self, input_dim: int, num_classes: int,
                 backbone_hidden: List[int] = [256, 128, 64],
                 head_hidden: List[int] = [32],
                 dropout: float = 0.3):
        super().__init__()
        self.backbone = Backbone(input_dim, backbone_hidden, dropout)
        self.head = PersonalizedHead(self.backbone.out_dim, head_hidden, num_classes, dropout)

    def forward(self, x):
        return self.head(self.backbone(x))

    # ── Gestion des paramètres backbone (échangés via Flower) ──────────────

    def get_backbone_params(self):
        """Retourne les paramètres backbone sous forme de liste de np.ndarray."""
        return [p.detach().cpu().numpy() for p in self.backbone.parameters()]

    def set_backbone_params(self, params):
        """Met à jour le backbone à partir de np.ndarray ou Tensor."""
        for p_model, p_new in zip(self.backbone.parameters(), params):
            if isinstance(p_new, np.ndarray):
                p_new = torch.from_numpy(p_new.copy())
            p_model.data.copy_(p_new)

    def get_backbone_state(self):
        """Snapshot des paramètres backbone (pour terme proximal FedProx)."""
        return [p.detach().clone() for p in self.backbone.parameters()]


def compute_proximal_loss(model, global_backbone_params, mu: float):
    """Terme proximal FedProx : ||w - w_global||² × μ/2."""
    if mu <= 0 or global_backbone_params is None:
        return torch.tensor(0.0)
    loss = torch.tensor(0.0)
    for p_local, p_global in zip(model.backbone.parameters(), global_backbone_params):
        if isinstance(p_global, np.ndarray):
            p_global = torch.from_numpy(p_global.copy())
        p_global = p_global.to(p_local.device)
        loss = loss + ((p_local - p_global) ** 2).sum()
    return (mu / 2.0) * loss


class FocalLoss(nn.Module):
    """Focal Loss pour datasets déséquilibrés."""

    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()
