"""R3.6 — SOTA baseline comparison for FedPer-Prox-XAI.

Implements and evaluates the following baselines to compare against
FedPer-Prox-XAI (Reviewer 3, Comment 6: compare with latest SOTA):

Classic / foundational:
  1. Centralized  — Standard MLP trained on all data pooled together
  2. Local-only   — Each client trains independently (no federation)
  3. FedAvg       — McMahan et al. (2017) — full model aggregation
  4. FedProx      — Li et al. (2020) — FedAvg + proximal term, full model
  5. FedPer       — Arivazhagan et al. (2019) — split backbone/head, no prox

Recent state-of-the-art (added to address Reviewer 3, Comment 6 — the
previous baseline set stopped at 2017-2019 methods and did not reflect
"latest" published SOTA in federated non-IID / personalized learning):
  6. SCAFFOLD     — Karimireddy et al. (ICML 2020) — control-variate drift correction
  7. FedNova      — Wang et al. (NeurIPS 2020) — normalized-averaging aggregation
  8. MOON         — Li et al. (CVPR 2021) — model-contrastive representation learning
  9. Ditto        — Li et al. (ICML 2021) — personalization via dual local/global model

Ours:
  10. FedPer-Prox-XAI — backbone+head split, proximal, SHAP (this paper)

All baselines use the SAME architecture, hyperparameters, and data splits
for fair comparison, only changing what is aggregated and how.

Usage:
  python -I scripts/run_sota_baselines.py
  python -I scripts/run_sota_baselines.py --seeds 0 1 2 3 42 --data-dir data/client_datasets
"""
import os, sys, argparse, json, time, copy
# See scripts/csv_to_npy.py for the FEDPER_DATA_ROOT convention.
DATA_ROOT = os.environ.get('FEDPER_DATA_ROOT', 'data')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import FLConfig, ALL_CLASSES
from models import FedPerProxModel, FocalLoss, compute_proximal_loss
from monitoring.advanced_metrics import compute_classification_metrics


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def set_all_seeds(seed: int):
    """Full deterministic seeding."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_client_data(data_dir: str, client_id: int, test_size: float = 0.2,
                      seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one client's partition and produce a stratified 80/20 train/test split.

    Each client folder holds the FULL local partition as X.npy / y.npy
    (plus stats.pkl, unused here) rather than a pre-made train/test split.
    This mirrors the preprocessing described in Section 4.2 of the paper:
    an 80/20 class-stratified split performed independently at each client.
    """
    from sklearn.model_selection import train_test_split

    path = Path(data_dir) / f'client_{client_id}'
    if not path.exists():
        raise FileNotFoundError(f"Client data not found: {path}")

    X_path, y_path = path / 'X.npy', path / 'y.npy'
    if not (X_path.exists() and y_path.exists()):
        raise FileNotFoundError(
            f"Expected {X_path} and {y_path} — got: {sorted(p.name for p in path.iterdir())}")

    X = np.load(X_path)
    y = np.load(y_path)

    # Stratify when every class has at least 2 samples (sklearn's requirement);
    # fall back to a plain random split otherwise (can happen for a rare class
    # under heavy label/quantity skew).
    counts = np.bincount(y.astype(np.int64))
    can_stratify = counts[counts > 0].min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y if can_stratify else None,
    )
    return (X_train.astype(np.float32), y_train.astype(np.int64),
            X_test.astype(np.float32), y_test.astype(np.int64))


def generate_synthetic_data(n_clients: int = 4, n_features: int = 23,
                            n_classes: int = 7, samples_per_client: int = 5000,
                            non_iid: bool = True, seed: int = 42):
    """Generate synthetic Non-IID data for baselines when real data unavailable."""
    rng = np.random.default_rng(seed)
    clients = []

    # Create class centers for more realistic separation
    class_centers = rng.standard_normal((n_classes, n_features)) * 2.0

    for c in range(n_clients):
        # Non-IID: each client sees only a subset of classes (label skew)
        if non_iid:
            n_local_classes = max(3, n_classes - c)
            local_classes = sorted(rng.choice(n_classes, n_local_classes, replace=False))
        else:
            local_classes = list(range(n_classes))

        # Generate samples per class with quantity skew
        X_parts, y_parts = [], []
        for cls in local_classes:
            n_cls = rng.integers(samples_per_client // (2 * len(local_classes)),
                                samples_per_client // len(local_classes))
            X_cls = class_centers[cls] + rng.standard_normal((n_cls, n_features)) * 0.5
            y_cls = np.full(n_cls, cls)
            X_parts.append(X_cls)
            y_parts.append(y_cls)

        X = np.vstack(X_parts).astype(np.float32)
        y = np.concatenate(y_parts).astype(np.int64)

        # Shuffle
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]

        # Split 80/20
        split = int(0.8 * len(X))
        clients.append({
            'X_train': X[:split], 'y_train': y[:split],
            'X_test': X[split:], 'y_test': y[split:],
        })

    return clients


def make_dataloader(X, y, batch_size, shuffle=True):
    """Create PyTorch DataLoader."""
    ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def compute_class_weights(y, n_classes):
    """Compute inverse-frequency class weights."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.clip(counts, 1, None)
    weights = len(y) / (n_classes * counts)
    return torch.FloatTensor(weights)


def evaluate_model(model, X_test, y_test, n_classes, class_names=None, device='cpu'):
    """Evaluate model and return metrics dict."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    # class_names (e.g. ALL_CLASSES from config.py) is written for a specific
    # dataset's label set. If the data actually loaded has a different number
    # of classes (different dataset/partition than the one ALL_CLASSES was
    # authored for), fall back to generic names instead of letting a length
    # mismatch crash the metrics computation.
    if class_names is not None and len(class_names) != n_classes:
        class_names = [f'class_{i}' for i in range(n_classes)]

    metrics = compute_classification_metrics(
        y_true=y_test, y_pred=preds, y_prob=probs,
        class_names=class_names, num_classes=n_classes,
    )
    return metrics


def _new_model(config, device):
    return FedPerProxModel(
        input_dim=config.model.input_dim,
        num_classes=config.model.num_classes,
        backbone_hidden=config.model.backbone_hidden,
        head_hidden=config.model.head_hidden,
        dropout=config.model.dropout,
    ).to(device)


def _flat_params(model):
    """Flatten all model parameters into a single 1-D tensor (for SCAFFOLD/FedNova deltas)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _unflatten_into(model, flat):
    """Write a flat parameter tensor back into a model's parameters (in place)."""
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p))
            offset += n


# ─────────────────────────────────────────────────────────────────────────
# Baseline 1: Centralized
# ─────────────────────────────────────────────────────────────────────────

def run_centralized(clients_data, config, device='cpu'):
    """Train a single model on pooled data from all clients."""
    print("\n  [Centralized] Pooling all client data...")

    # Pool all data
    X_train = np.vstack([c['X_train'] for c in clients_data])
    y_train = np.concatenate([c['y_train'] for c in clients_data])
    X_test = np.vstack([c['X_test'] for c in clients_data])
    y_test = np.concatenate([c['y_test'] for c in clients_data])

    model = _new_model(config, device)

    weights = compute_class_weights(y_train, config.model.num_classes).to(device)
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr,
                                   weight_decay=config.training.weight_decay)

    loader = make_dataloader(X_train, y_train, config.training.batch_size)

    total_epochs = config.training.num_rounds * config.training.local_epochs
    model.train()
    t0 = time.time()
    for epoch in range(total_epochs):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
    train_time = time.time() - t0

    metrics = evaluate_model(model, X_test, y_test, config.model.num_classes,
                             ALL_CLASSES, device)
    metrics['train_time_s'] = train_time
    metrics['total_epochs'] = total_epochs
    return metrics


# ─────────────────────────────────────────────────────────────────────────
# Baseline 2: Local-only (no federation)
# ─────────────────────────────────────────────────────────────────────────

def run_local_only(clients_data, config, device='cpu'):
    """Each client trains independently — no federation."""
    print("\n  [Local-only] Training each client independently...")
    all_metrics = []
    total_epochs = config.training.num_rounds * config.training.local_epochs

    for cid, cdata in enumerate(clients_data):
        model = _new_model(config, device)

        weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
        criterion = FocalLoss(alpha=weights, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr,
                                       weight_decay=config.training.weight_decay)

        loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

        model.train()
        for epoch in range(total_epochs):
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()

        m = evaluate_model(model, cdata['X_test'], cdata['y_test'],
                           config.model.num_classes, ALL_CLASSES, device)
        all_metrics.append(m)
        print(f"    Client {cid}: acc={m['accuracy']:.4f} F1={m['f1_macro']:.4f}")

    return _aggregate_client_metrics(all_metrics)


# ─────────────────────────────────────────────────────────────────────────
# Baseline 3: FedAvg
# ─────────────────────────────────────────────────────────────────────────

def _fedavg_aggregate(models, weights):
    """Weighted average of ALL parameters (backbone + head)."""
    global_state = copy.deepcopy(models[0].state_dict())
    for key in global_state:
        global_state[key] = sum(
            w * models[i].state_dict()[key].float()
            for i, w in enumerate(weights)
        )
    return global_state


def run_fedavg(clients_data, config, device='cpu'):
    """FedAvg — full model aggregation (McMahan et al., 2017)."""
    print("\n  [FedAvg] Federated averaging (full model)...")
    n_clients = len(clients_data)

    models = [_new_model(config, device) for _ in range(n_clients)]
    global_state = copy.deepcopy(models[0].state_dict())

    for rnd in range(config.training.num_rounds):
        for m in models:
            m.load_state_dict(copy.deepcopy(global_state))

        for cid in range(n_clients):
            cdata = clients_data[cid]
            weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
            criterion = FocalLoss(alpha=weights, gamma=2.0)
            optimizer = torch.optim.AdamW(models[cid].parameters(), lr=config.training.lr,
                                           weight_decay=config.training.weight_decay)
            loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

            models[cid].train()
            for _ in range(config.training.local_epochs):
                for X_b, y_b in loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    loss = criterion(models[cid](X_b), y_b)
                    loss.backward()
                    optimizer.step()

        n_samples = [len(c['y_train']) for c in clients_data]
        total = sum(n_samples)
        w = [n / total for n in n_samples]
        global_state = _fedavg_aggregate(models, w)

    all_metrics = []
    for cid in range(n_clients):
        models[cid].load_state_dict(copy.deepcopy(global_state))
        m = evaluate_model(models[cid], clients_data[cid]['X_test'],
                           clients_data[cid]['y_test'],
                           config.model.num_classes, ALL_CLASSES, device)
        all_metrics.append(m)

    return _aggregate_client_metrics(all_metrics)


# ─────────────────────────────────────────────────────────────────────────
# Baseline 4: FedProx (full model + proximal term)
# ─────────────────────────────────────────────────────────────────────────

def run_fedprox(clients_data, config, device='cpu'):
    """FedProx — FedAvg + proximal regularization (Li et al., 2020)."""
    print(f"\n  [FedProx] FedAvg + proximal term (mu={config.training.mu_prox})...")
    n_clients = len(clients_data)

    models = [_new_model(config, device) for _ in range(n_clients)]
    global_state = copy.deepcopy(models[0].state_dict())

    for rnd in range(config.training.num_rounds):
        for m in models:
            m.load_state_dict(copy.deepcopy(global_state))

        for cid in range(n_clients):
            cdata = clients_data[cid]
            weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
            criterion = FocalLoss(alpha=weights, gamma=2.0)
            optimizer = torch.optim.AdamW(models[cid].parameters(), lr=config.training.lr,
                                           weight_decay=config.training.weight_decay)
            loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

            global_params = [p.detach().clone() for p in models[cid].parameters()]

            models[cid].train()
            for _ in range(config.training.local_epochs):
                for X_b, y_b in loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    loss = criterion(models[cid](X_b), y_b)
                    prox = torch.tensor(0.0, device=device)
                    for p_local, p_global in zip(models[cid].parameters(), global_params):
                        prox = prox + ((p_local - p_global) ** 2).sum()
                    loss = loss + (config.training.mu_prox / 2.0) * prox
                    loss.backward()
                    optimizer.step()

        n_samples = [len(c['y_train']) for c in clients_data]
        total = sum(n_samples)
        w = [n / total for n in n_samples]
        global_state = _fedavg_aggregate(models, w)

    all_metrics = []
    for cid in range(n_clients):
        models[cid].load_state_dict(copy.deepcopy(global_state))
        m = evaluate_model(models[cid], clients_data[cid]['X_test'],
                           clients_data[cid]['y_test'],
                           config.model.num_classes, ALL_CLASSES, device)
        all_metrics.append(m)

    return _aggregate_client_metrics(all_metrics)


# ─────────────────────────────────────────────────────────────────────────
# Baseline 5: FedPer (backbone-only aggregation, no prox)
# ─────────────────────────────────────────────────────────────────────────

def _fedper_aggregate(models, weights):
    """Weighted average of BACKBONE parameters only."""
    global_backbone = {}
    for key in models[0].state_dict():
        if key.startswith('backbone.'):
            global_backbone[key] = sum(
                w * models[i].state_dict()[key].float()
                for i, w in enumerate(weights)
            )
    return global_backbone


def run_fedper(clients_data, config, device='cpu'):
    """FedPer — partial model sharing (Arivazhagan et al., 2019)."""
    print("\n  [FedPer] Backbone-only aggregation (no proximal)...")
    n_clients = len(clients_data)

    models = [_new_model(config, device) for _ in range(n_clients)]
    global_backbone = {k: v.clone() for k, v in models[0].state_dict().items()
                       if k.startswith('backbone.')}

    for rnd in range(config.training.num_rounds):
        for m in models:
            sd = m.state_dict()
            sd.update(copy.deepcopy(global_backbone))
            m.load_state_dict(sd)

        for cid in range(n_clients):
            cdata = clients_data[cid]
            weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
            criterion = FocalLoss(alpha=weights, gamma=2.0)
            optimizer = torch.optim.AdamW(models[cid].parameters(), lr=config.training.lr,
                                           weight_decay=config.training.weight_decay)
            loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

            models[cid].train()
            for _ in range(config.training.local_epochs):
                for X_b, y_b in loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    loss = criterion(models[cid](X_b), y_b)
                    loss.backward()
                    optimizer.step()

        n_samples = [len(c['y_train']) for c in clients_data]
        tau = config.training.aggregation_temp
        w_raw = [n ** tau for n in n_samples]
        w_sum = sum(w_raw)
        w = [wr / w_sum for wr in w_raw]
        global_backbone = _fedper_aggregate(models, w)

    all_metrics = []
    for cid in range(n_clients):
        m = evaluate_model(models[cid], clients_data[cid]['X_test'],
                           clients_data[cid]['y_test'],
                           config.model.num_classes, ALL_CLASSES, device)
        all_metrics.append(m)

    return _aggregate_client_metrics(all_metrics)


# ─────────────────────────────────────────────────────────────────────────
# Baseline 6: SCAFFOLD (Karimireddy et al., ICML 2020)
#   "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
#   Corrects client drift with control variates c (server) and c_i (client):
#   local gradient is adjusted as g - c_i + c, and c_i is updated after
#   each round from the observed local progress.
# ─────────────────────────────────────────────────────────────────────────

def run_scaffold(clients_data, config, device='cpu'):
    """SCAFFOLD — full-model aggregation with control-variate drift correction."""
    print("\n  [SCAFFOLD] Control-variate drift correction (Karimireddy et al., 2020)...")
    n_clients = len(clients_data)

    models = [_new_model(config, device) for _ in range(n_clients)]
    global_state = copy.deepcopy(models[0].state_dict())

    # Control variates, flattened over all parameters.
    n_params = sum(p.numel() for p in models[0].parameters())
    c_global = torch.zeros(n_params, device=device)
    c_local = [torch.zeros(n_params, device=device) for _ in range(n_clients)]

    for rnd in range(config.training.num_rounds):
        for m in models:
            m.load_state_dict(copy.deepcopy(global_state))

        delta_c_sum = torch.zeros(n_params, device=device)

        for cid in range(n_clients):
            cdata = clients_data[cid]
            weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
            criterion = FocalLoss(alpha=weights, gamma=2.0)
            optimizer = torch.optim.AdamW(models[cid].parameters(), lr=config.training.lr,
                                           weight_decay=config.training.weight_decay)
            loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

            theta_global_flat = _flat_params(models[cid]).clone()
            local_steps = 0

            models[cid].train()
            for _ in range(config.training.local_epochs):
                for X_b, y_b in loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    loss = criterion(models[cid](X_b), y_b)
                    loss.backward()

                    # SCAFFOLD correction: g <- g - c_i + c  (applied per-parameter,
                    # matched to the flat control-variate layout).
                    offset = 0
                    with torch.no_grad():
                        for p in models[cid].parameters():
                            n = p.numel()
                            if p.grad is not None:
                                correction = (c_global[offset:offset + n]
                                              - c_local[cid][offset:offset + n]).view_as(p)
                                p.grad.add_(-correction)
                            offset += n

                    optimizer.step()
                    local_steps += 1

            # Update client control variate using the standard "option II" rule:
            # c_i^+ = c_i - c + (theta_global - theta_local) / (local_steps * lr)
            theta_local_flat = _flat_params(models[cid])
            eff_lr = config.training.lr * max(local_steps, 1)
            c_local_new = (c_local[cid] - c_global
                           + (theta_global_flat - theta_local_flat) / eff_lr)
            delta_c_sum += (c_local_new - c_local[cid])
            c_local[cid] = c_local_new

        # Server-side update of global control variate and model.
        c_global = c_global + delta_c_sum / n_clients

        n_samples = [len(c['y_train']) for c in clients_data]
        total = sum(n_samples)
        w = [n / total for n in n_samples]
        global_state = _fedavg_aggregate(models, w)

    all_metrics = []
    for cid in range(n_clients):
        models[cid].load_state_dict(copy.deepcopy(global_state))
        m = evaluate_model(models[cid], clients_data[cid]['X_test'],
                           clients_data[cid]['y_test'],
                           config.model.num_classes, ALL_CLASSES, device)
        all_metrics.append(m)

    return _aggregate_client_metrics(all_metrics)


# ─────────────────────────────────────────────────────────────────────────
# Baseline 7: FedNova (Wang et al., NeurIPS 2020)
#   "Tackling the Objective Inconsistency Problem in Heterogeneous
#   Federated Optimization." Normalizes each client's cumulative update by
#   its number of local steps before averaging, correcting the implicit
#   bias FedAvg introduces when clients run different numbers of steps
#   (here steps differ because client dataset sizes differ under quantity
#   skew, even though local_epochs is fixed).
# ─────────────────────────────────────────────────────────────────────────

def run_fednova(clients_data, config, device='cpu'):
    """FedNova — normalized-averaging aggregation (Wang et al., 2020)."""
    print("\n  [FedNova] Normalized averaging (Wang et al., 2020)...")
    n_clients = len(clients_data)

    models = [_new_model(config, device) for _ in range(n_clients)]
    global_state = copy.deepcopy(models[0].state_dict())

    for rnd in range(config.training.num_rounds):
        for m in models:
            m.load_state_dict(copy.deepcopy(global_state))

        normalized_deltas = []
        tau_list = []
        n_list = []

        for cid in range(n_clients):
            cdata = clients_data[cid]
            weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
            criterion = FocalLoss(alpha=weights, gamma=2.0)
            optimizer = torch.optim.AdamW(models[cid].parameters(), lr=config.training.lr,
                                           weight_decay=config.training.weight_decay)
            loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

            theta_start = _flat_params(models[cid]).clone()
            tau_i = 0

            models[cid].train()
            for _ in range(config.training.local_epochs):
                for X_b, y_b in loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    loss = criterion(models[cid](X_b), y_b)
                    loss.backward()
                    optimizer.step()
                    tau_i += 1

            theta_end = _flat_params(models[cid])
            delta_i = theta_start - theta_end  # cumulative local "gradient" direction
            normalized_deltas.append(delta_i / max(tau_i, 1))
            tau_list.append(tau_i)
            n_list.append(len(cdata['y_train']))

        total_n = sum(n_list)
        p = [n / total_n for n in n_list]
        tau_eff = sum(pi * ti for pi, ti in zip(p, tau_list))

        agg_delta = torch.zeros_like(normalized_deltas[0])
        for pi, d_i in zip(p, normalized_deltas):
            agg_delta += pi * d_i

        # Reconstruct new global parameters from the (shared) pre-round state.
        global_model_tmp = _new_model(config, device)
        global_model_tmp.load_state_dict(copy.deepcopy(global_state))
        theta_old_flat = _flat_params(global_model_tmp)
        theta_new_flat = theta_old_flat - tau_eff * agg_delta
        _unflatten_into(global_model_tmp, theta_new_flat)
        global_state = copy.deepcopy(global_model_tmp.state_dict())

    all_metrics = []
    for cid in range(n_clients):
        models[cid].load_state_dict(copy.deepcopy(global_state))
        m = evaluate_model(models[cid], clients_data[cid]['X_test'],
                           clients_data[cid]['y_test'],
                           config.model.num_classes, ALL_CLASSES, device)
        all_metrics.append(m)

    return _aggregate_client_metrics(all_metrics)


# ─────────────────────────────────────────────────────────────────────────
# Baseline 8: MOON (Li et al., CVPR 2021)
#   "Model-Contrastive Federated Learning." Adds a contrastive loss that
#   pulls the local representation toward the global model's representation
#   and pushes it away from the previous round's local representation,
#   using the shared backbone's output as the representation.
# ─────────────────────────────────────────────────────────────────────────

def run_moon(clients_data, config, device='cpu', temperature: float = 0.5, mu_moon: float = 1.0):
    """MOON — model-contrastive federated learning (Li et al., 2021)."""
    print("\n  [MOON] Model-contrastive representation learning (Li et al., 2021)...")
    n_clients = len(clients_data)

    models = [_new_model(config, device) for _ in range(n_clients)]
    prev_models = [copy.deepcopy(m) for m in models]
    global_state = copy.deepcopy(models[0].state_dict())

    def representation(model, X):
        # Reuses the shared backbone (same component aggregated in FedPer /
        # FedPer-Prox-XAI) as the contrastive representation.
        return model.backbone(X)

    for rnd in range(config.training.num_rounds):
        global_model = _new_model(config, device)
        global_model.load_state_dict(copy.deepcopy(global_state))
        global_model.eval()
        for p in global_model.parameters():
            p.requires_grad_(False)

        for m in models:
            m.load_state_dict(copy.deepcopy(global_state))

        for cid in range(n_clients):
            cdata = clients_data[cid]
            weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
            criterion = FocalLoss(alpha=weights, gamma=2.0)
            optimizer = torch.optim.AdamW(models[cid].parameters(), lr=config.training.lr,
                                           weight_decay=config.training.weight_decay)
            loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

            prev_models[cid].eval()
            for p in prev_models[cid].parameters():
                p.requires_grad_(False)

            models[cid].train()
            for _ in range(config.training.local_epochs):
                for X_b, y_b in loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    optimizer.zero_grad()

                    logits = models[cid](X_b)
                    task_loss = criterion(logits, y_b)

                    z_local = representation(models[cid], X_b)
                    with torch.no_grad():
                        z_global = representation(global_model, X_b)
                        z_prev = representation(prev_models[cid], X_b)

                    z_local_n = F.normalize(z_local, dim=1)
                    z_global_n = F.normalize(z_global, dim=1)
                    z_prev_n = F.normalize(z_prev, dim=1)

                    sim_pos = (z_local_n * z_global_n).sum(dim=1) / temperature
                    sim_neg = (z_local_n * z_prev_n).sum(dim=1) / temperature
                    logits_con = torch.stack([sim_pos, sim_neg], dim=1)
                    labels_con = torch.zeros(logits_con.size(0), dtype=torch.long, device=device)
                    contrastive_loss = F.cross_entropy(logits_con, labels_con)

                    loss = task_loss + mu_moon * contrastive_loss
                    loss.backward()
                    optimizer.step()

            prev_models[cid] = copy.deepcopy(models[cid])

        n_samples = [len(c['y_train']) for c in clients_data]
        total = sum(n_samples)
        w = [n / total for n in n_samples]
        global_state = _fedavg_aggregate(models, w)

    all_metrics = []
    for cid in range(n_clients):
        models[cid].load_state_dict(copy.deepcopy(global_state))
        m = evaluate_model(models[cid], clients_data[cid]['X_test'],
                           clients_data[cid]['y_test'],
                           config.model.num_classes, ALL_CLASSES, device)
        all_metrics.append(m)

    return _aggregate_client_metrics(all_metrics)


# ─────────────────────────────────────────────────────────────────────────
# Baseline 9: Ditto (Li et al., ICML 2021)
#   "Ditto: Fair and Robust Federated Learning Through Personalization."
#   Trains a shared global model via FedAvg, and in parallel a personalized
#   model per client regularized toward the (periodically updated) global
#   model: min_v  F_k(v) + (lambda/2)||v - w||^2. Evaluation uses each
#   client's personalized model v_k, which is the quantity Ditto reports.
# ─────────────────────────────────────────────────────────────────────────

def run_ditto(clients_data, config, device='cpu', lam_ditto: float = 0.1):
    """Ditto — personalization via dual local/global model (Li et al., 2021)."""
    print(f"\n  [Ditto] Dual global/personalized model (lambda={lam_ditto})...")
    n_clients = len(clients_data)

    # Global model trained with plain FedAvg.
    global_models = [_new_model(config, device) for _ in range(n_clients)]
    global_state = copy.deepcopy(global_models[0].state_dict())

    # Personalized models v_k, initialized from the same point, updated locally only.
    personal_models = [_new_model(config, device) for _ in range(n_clients)]
    for pm in personal_models:
        pm.load_state_dict(copy.deepcopy(global_state))

    for rnd in range(config.training.num_rounds):
        for m in global_models:
            m.load_state_dict(copy.deepcopy(global_state))

        for cid in range(n_clients):
            cdata = clients_data[cid]
            weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
            criterion = FocalLoss(alpha=weights, gamma=2.0)
            loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

            # (1) Update the global copy with a standard local FedAvg step.
            optimizer_g = torch.optim.AdamW(global_models[cid].parameters(), lr=config.training.lr,
                                             weight_decay=config.training.weight_decay)
            global_models[cid].train()
            for _ in range(config.training.local_epochs):
                for X_b, y_b in loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    optimizer_g.zero_grad()
                    loss = criterion(global_models[cid](X_b), y_b)
                    loss.backward()
                    optimizer_g.step()

            # (2) Update the personalized model v_k against its own objective
            #     plus a proximal pull toward the CURRENT global model w_k.
            global_snapshot = [p.detach().clone() for p in global_models[cid].parameters()]
            optimizer_v = torch.optim.AdamW(personal_models[cid].parameters(), lr=config.training.lr,
                                             weight_decay=config.training.weight_decay)
            personal_models[cid].train()
            for _ in range(config.training.local_epochs):
                for X_b, y_b in loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    optimizer_v.zero_grad()
                    loss = criterion(personal_models[cid](X_b), y_b)
                    prox = torch.tensor(0.0, device=device)
                    for p_v, p_w in zip(personal_models[cid].parameters(), global_snapshot):
                        prox = prox + ((p_v - p_w) ** 2).sum()
                    loss = loss + (lam_ditto / 2.0) * prox
                    loss.backward()
                    optimizer_v.step()

        n_samples = [len(c['y_train']) for c in clients_data]
        total = sum(n_samples)
        w = [n / total for n in n_samples]
        global_state = _fedavg_aggregate(global_models, w)

    # Ditto is evaluated on each client's PERSONALIZED model, as in the
    # original paper — this is the quantity comparable to FedPer-Prox-XAI's
    # per-client head.
    all_metrics = []
    for cid in range(n_clients):
        m = evaluate_model(personal_models[cid], clients_data[cid]['X_test'],
                           clients_data[cid]['y_test'],
                           config.model.num_classes, ALL_CLASSES, device)
        all_metrics.append(m)

    return _aggregate_client_metrics(all_metrics)


# ─────────────────────────────────────────────────────────────────────────
# Our Method: FedPer-Prox-XAI (backbone-only aggregation + proximal)
# ─────────────────────────────────────────────────────────────────────────

def run_fedper_prox(clients_data, config, device='cpu'):
    """FedPer-Prox-XAI — backbone-only aggregation + proximal (ours)."""
    print(f"\n  [FedPer-Prox-XAI] backbone split + proximal (mu={config.training.mu_prox})...")
    n_clients = len(clients_data)

    models = [_new_model(config, device) for _ in range(n_clients)]
    global_backbone = {k: v.clone() for k, v in models[0].state_dict().items()
                       if k.startswith('backbone.')}

    for rnd in range(config.training.num_rounds):
        for m in models:
            sd = m.state_dict()
            sd.update(copy.deepcopy(global_backbone))
            m.load_state_dict(sd)

        for cid in range(n_clients):
            cdata = clients_data[cid]
            weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
            criterion = FocalLoss(alpha=weights, gamma=2.0)
            optimizer = torch.optim.AdamW(models[cid].parameters(), lr=config.training.lr,
                                           weight_decay=config.training.weight_decay)
            loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

            global_bb_params = [p.detach().clone() for p in models[cid].backbone.parameters()]

            models[cid].train()
            for _ in range(config.training.local_epochs):
                for X_b, y_b in loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    task_loss = criterion(models[cid](X_b), y_b)
                    prox_loss = compute_proximal_loss(
                        models[cid], global_bb_params, config.training.mu_prox)
                    loss = task_loss + prox_loss
                    loss.backward()
                    optimizer.step()

        n_samples = [len(c['y_train']) for c in clients_data]
        tau = config.training.aggregation_temp
        w_raw = [n ** tau for n in n_samples]
        w_sum = sum(w_raw)
        w = [wr / w_sum for wr in w_raw]
        global_backbone = _fedper_aggregate(models, w)

    all_metrics = []
    for cid in range(n_clients):
        m = evaluate_model(models[cid], clients_data[cid]['X_test'],
                           clients_data[cid]['y_test'],
                           config.model.num_classes, ALL_CLASSES, device)
        all_metrics.append(m)

    return _aggregate_client_metrics(all_metrics)


# ─────────────────────────────────────────────────────────────────────────
# Aggregation / reporting
# ─────────────────────────────────────────────────────────────────────────

def _aggregate_client_metrics(metrics_list):
    """Average metrics across clients."""
    keys = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted',
            'precision_macro', 'recall_macro', 'cohen_kappa', 'mcc']
    agg = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m]
        if vals:
            agg[k] = float(np.mean(vals))
            agg[f'{k}_std'] = float(np.std(vals))
    agg['n_clients'] = len(metrics_list)
    agg['per_client'] = metrics_list
    return agg


# Registry of all baselines, in reporting order. Adding/removing a method
# only requires editing this dict — run_all_baselines(), main(), and the
# table/CSV generators all read from it.
BASELINE_REGISTRY = [
    ('centralized',      'Centralized',                    run_centralized),
    ('local_only',       'Local-only',                     run_local_only),
    ('fedavg',           'FedAvg (2017)',                  run_fedavg),
    ('fedprox',          'FedProx (2020)',                 run_fedprox),
    ('fedper',           'FedPer (2019)',                  run_fedper),
    ('scaffold',         'SCAFFOLD (2020)',                run_scaffold),
    ('fednova',          'FedNova (2020)',                 run_fednova),
    ('moon',             'MOON (2021)',                    run_moon),
    ('ditto',            'Ditto (2021)',                   run_ditto),
    ('fedper_prox_xai',  '\\textbf{FedPer-Prox-XAI (Ours)}', run_fedper_prox),
]


def run_all_baselines(clients_data, config, device='cpu', methods=None):
    """Run the selected baselines (default: all) and return a comparison dict.

    `methods`, when given, restricts execution to those registry keys — this
    is what lets each terminal run a single method in parallel (see --methods
    in main()) without re-running the others.
    """
    selected = BASELINE_REGISTRY if methods is None else [
        entry for entry in BASELINE_REGISTRY if entry[0] in methods
    ]
    results = {}
    for key, _, fn in selected:
        t0 = time.time()
        results[key] = fn(clients_data, config, device)
        results[key]['wall_time_s'] = time.time() - t0
    return results


def generate_comparison_table_latex(all_results, out_path):
    """Generate LaTeX booktabs comparison table.

    Only emits rows for methods actually present in all_results, so a
    single-method run (e.g. --methods scaffold) produces a one-row table
    instead of a full table padded with zeros for the methods not run.
    """
    present = set(all_results['mean'].keys()) if 'mean' in all_results else set(all_results.keys())
    present.discard('seed')
    methods = [(key, label) for key, label, _ in BASELINE_REGISTRY if key in present]
    metrics = [
        ('accuracy', 'Accuracy'),
        ('f1_macro', 'F1-Macro'),
        ('f1_weighted', 'F1-Weighted'),
        ('precision_macro', 'Precision'),
        ('recall_macro', 'Recall'),
        ('cohen_kappa', "Cohen's $\\kappa$"),
        ('mcc', 'MCC'),
    ]

    lines = [
        '% Requires \\usepackage{booktabs}',
        '\\begin{table*}[t]',
        '\\centering',
        '\\caption{Performance comparison of FedPer-Prox-XAI against '
        'classic (2017--2019) and recent (2020--2021) federated '
        'baselines on IoMT Non-IID data (mean $\\pm$ std across clients/seeds).}',
        '\\label{tab:sota_comparison}',
        '\\begin{tabular}{l' + 'c' * len(metrics) + '}',
        '\\toprule',
        'Method & ' + ' & '.join(m[1] for m in metrics) + ' \\\\',
        '\\midrule',
    ]

    is_multi_seed = isinstance(all_results, dict) and 'mean' in all_results

    for key, label in methods:
        row_parts = [label]
        for mk, _ in metrics:
            if is_multi_seed:
                mean_val = all_results['mean'].get(key, {}).get(mk, 0)
                std_val = all_results['std'].get(key, {}).get(mk, 0)
                row_parts.append(f'{mean_val:.3f} $\\pm$ {std_val:.3f}')
            else:
                val = all_results.get(key, {}).get(mk, 0)
                std = all_results.get(key, {}).get(f'{mk}_std', 0)
                if std > 0:
                    row_parts.append(f'{val:.3f} $\\pm$ {std:.3f}')
                else:
                    row_parts.append(f'{val:.3f}')
        lines.append(' & '.join(row_parts) + ' \\\\')

    lines += [
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table*}',
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def generate_comparison_csv(all_results, out_path):
    """Generate CSV comparison table (same present-methods-only filtering as
    generate_comparison_table_latex — see its docstring)."""
    present = set(all_results['mean'].keys()) if 'mean' in all_results else set(all_results.keys())
    present.discard('seed')
    methods = [key for key, _, _ in BASELINE_REGISTRY if key in present]
    metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro',
               'recall_macro', 'cohen_kappa', 'mcc']

    is_multi_seed = isinstance(all_results, dict) and 'mean' in all_results

    header = 'method,' + ','.join(f'{m}_mean,{m}_std' for m in metrics)
    rows = [header]

    for method in methods:
        parts = [method]
        for mk in metrics:
            if is_multi_seed:
                mean_val = all_results['mean'].get(method, {}).get(mk, 0)
                std_val = all_results['std'].get(method, {}).get(mk, 0)
            else:
                mean_val = all_results.get(method, {}).get(mk, 0)
                std_val = all_results.get(method, {}).get(f'{mk}_std', 0)
            parts.extend([f'{mean_val:.4f}', f'{std_val:.4f}'])
        rows.append(','.join(parts))

    with open(out_path, 'w') as f:
        f.write('\n'.join(rows) + '\n')


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

VALID_METHOD_KEYS = [key for key, _, _ in BASELINE_REGISTRY]


def main():
    p = argparse.ArgumentParser(description='R3.6: SOTA baseline comparison')
    p.add_argument('--data-dir', type=str, default=f'{DATA_ROOT}/client_datasets',
                   help='Client data directory (with client_0/, client_1/, ...)')
    p.add_argument('--seeds', type=int, nargs='+', default=[42],
                   help='Random seeds for reproducibility')
    p.add_argument('--n-clients', type=int, default=4)
    p.add_argument('--synthetic', action='store_true',
                   help='Use synthetic data (if real data unavailable)')
    p.add_argument('--out-dir', type=str, default='./results/sota_comparison')
    p.add_argument('--methods', type=str, nargs='+', default=None,
                   choices=VALID_METHOD_KEYS,
                   help='Run only these methods (e.g. --methods scaffold). '
                        'Lets each terminal/job run one method in parallel; '
                        'omit to run the full registry in this process. '
                        f'Valid keys: {", ".join(VALID_METHOD_KEYS)}')
    args = p.parse_args()

    methods = args.methods  # None => all (handled by run_all_baselines)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # When running a subset of methods (parallel terminals), suffix every
    # output file with the method name(s) so concurrent jobs writing to the
    # SAME --out-dir never overwrite each other's results.
    file_suffix = '' if methods is None else '_' + '_'.join(methods)

    config = FLConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_registry = BASELINE_REGISTRY if methods is None else [
        e for e in BASELINE_REGISTRY if e[0] in methods
    ]

    print("=" * 70)
    print("  R3.6 — SOTA BASELINE COMPARISON (classic + 2020-2021 methods)")
    print("=" * 70)
    print(f"  Seeds     : {args.seeds}")
    print(f"  Clients   : {args.n_clients}")
    print(f"  Rounds    : {config.training.num_rounds}")
    print(f"  Epochs/R  : {config.training.local_epochs}")
    print(f"  Device    : {device}")
    print(f"  Data      : {'synthetic' if args.synthetic else args.data_dir}")
    print(f"  Methods   : {', '.join(k for k, _, _ in run_registry)}")

    data_path = Path(args.data_dir)
    use_synthetic = args.synthetic or not data_path.exists()
    if use_synthetic:
        print("\n  Using synthetic Non-IID data...")

    all_seed_results = []

    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"  SEED = {seed}")
        print(f"{'='*70}")
        set_all_seeds(seed)

        if use_synthetic:
            clients_data = generate_synthetic_data(
                n_clients=args.n_clients,
                n_features=config.model.input_dim,
                n_classes=config.model.num_classes,
                seed=seed,
            )
        else:
            clients_data = []
            for cid in range(args.n_clients):
                X_tr, y_tr, X_te, y_te = load_client_data(args.data_dir, cid, seed=seed)
                clients_data.append({
                    'X_train': X_tr, 'y_train': y_tr,
                    'X_test': X_te, 'y_test': y_te,
                })

        # FLConfig's defaults (input_dim=23, matching IoMT-TrafficData) do not
        # necessarily match the dataset actually found at --data-dir (e.g.
        # IoT-IDS2024 has 81 features). Infer both dimensions from the loaded
        # data itself and override the config so every baseline builds a
        # model with the right input/output shape.
        inferred_input_dim = clients_data[0]['X_train'].shape[1]
        inferred_num_classes = int(max(
            int(c['y_train'].max()) for c in clients_data
        )) + 1
        if (inferred_input_dim != config.model.input_dim
                or inferred_num_classes != config.model.num_classes):
            print(f"\n  [config] Overriding model dims from data: "
                  f"input_dim {config.model.input_dim} -> {inferred_input_dim}, "
                  f"num_classes {config.model.num_classes} -> {inferred_num_classes}")
            config.model.input_dim = inferred_input_dim
            config.model.num_classes = inferred_num_classes

        results = run_all_baselines(clients_data, config, device, methods=methods)
        results['seed'] = seed
        all_seed_results.append(results)

        print(f"\n  Summary (seed={seed}):")
        print(f"  {'Method':<22s} {'Accuracy':>10s} {'F1-Macro':>10s} {'MCC':>10s}")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
        for key, _, _ in run_registry:
            r = results[key]
            print(f"  {key:<22s} {r.get('accuracy',0):>10.4f} "
                  f"{r.get('f1_macro',0):>10.4f} {r.get('mcc',0):>10.4f}")

    if len(args.seeds) > 1:
        run_keys = [key for key, _, _ in run_registry]
        metrics_keys = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro',
                        'recall_macro', 'cohen_kappa', 'mcc']

        agg = {'mean': {}, 'std': {}}
        for method in run_keys:
            agg['mean'][method] = {}
            agg['std'][method] = {}
            for mk in metrics_keys:
                vals = [r[method].get(mk, 0) for r in all_seed_results]
                agg['mean'][method][mk] = float(np.mean(vals))
                agg['std'][method][mk] = float(np.std(vals))

        generate_comparison_table_latex(agg, out_dir / f'sota_comparison_table{file_suffix}.tex')
        generate_comparison_csv(agg, out_dir / f'sota_comparison{file_suffix}.csv')

        print(f"\n{'='*70}")
        print(f"  AGGREGATED RESULTS ({len(args.seeds)} seeds)")
        print(f"{'='*70}")
        print(f"  {'Method':<22s} {'Accuracy':>16s} {'F1-Macro':>16s} {'MCC':>16s}")
        print(f"  {'-'*22} {'-'*16} {'-'*16} {'-'*16}")
        for method in run_keys:
            acc = f"{agg['mean'][method]['accuracy']:.3f}±{agg['std'][method]['accuracy']:.3f}"
            f1 = f"{agg['mean'][method]['f1_macro']:.3f}±{agg['std'][method]['f1_macro']:.3f}"
            mcc = f"{agg['mean'][method]['mcc']:.3f}±{agg['std'][method]['mcc']:.3f}"
            print(f"  {method:<22s} {acc:>16s} {f1:>16s} {mcc:>16s}")
    else:
        generate_comparison_table_latex(all_seed_results[0], out_dir / f'sota_comparison_table{file_suffix}.tex')
        generate_comparison_csv(all_seed_results[0], out_dir / f'sota_comparison{file_suffix}.csv')

    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_dir / f'sota_comparison_full{file_suffix}.json', 'w') as f:
        json.dump(clean_for_json(all_seed_results), f, indent=2)

    print(f"\n  Saved to: {out_dir}/")
    print(f"    sota_comparison_full{file_suffix}.json")
    print(f"    sota_comparison_table{file_suffix}.tex")
    print(f"    sota_comparison{file_suffix}.csv")


if __name__ == '__main__':
    main()