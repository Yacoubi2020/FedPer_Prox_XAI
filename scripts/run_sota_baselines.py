# #!/usr/bin/env python3
# """R3.6 — SOTA baseline comparison for FedPer-Prox-XAI.

# Implements and evaluates the following baselines to compare against
# FedPer-Prox-XAI (Reviewer 3, Comment 6: compare with latest SOTA):

#   Traditional ML (centralized, pooled data):
#   1. Random Forest         — Breiman (2001)
#   2. XGBoost / GradientBoosting — Chen & Guestrin (2016) / Friedman (2001)
#   3. SVM (RBF kernel)      — Cortes & Vapnik (1995)
#   4. Decision Tree (CART)   — Breiman et al. (1984)
#   5. K-Nearest Neighbors   — Cover & Hart (1967)
#   6. Logistic Regression    — Cox (1958)

#   Deep Learning & Federated:
#   7. Centralized MLP       — Standard MLP trained on all data pooled together
#   8. Local-only MLP        — Each client trains independently (no federation)
#   9. FedAvg                — McMahan et al. (2017) — full model aggregation
#   10. FedProx              — Li et al. (2020) — FedAvg + proximal term
#   11. FedPer               — Arivazhagan et al. (2019) — split backbone/head
#   12. FedPer-Prox-XAI      — Our method (backbone+head split, proximal, SHAP)

# All baselines use the SAME data splits for fair comparison.

# Usage:
#   python -I scripts/run_sota_baselines.py
#   python -I scripts/run_sota_baselines.py --seeds 0 1 2 3 42 --data-dir data/client_datasets
# """
# import os, sys, argparse, json, time, copy
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from pathlib import Path
# from collections import defaultdict
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset

# from config import FLConfig, ALL_CLASSES
# from models import FedPerProxModel, FocalLoss, compute_proximal_loss
# from monitoring.advanced_metrics import compute_classification_metrics


# # ─────────────────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────────────────

# def set_all_seeds(seed: int):
#     """Full deterministic seeding."""
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ['PYTHONHASHSEED'] = str(seed)


# def load_client_data(data_dir: str, client_id: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """Load client partition from .npy files + 80/20 stratified split."""
#     from sklearn.model_selection import train_test_split
#     path = Path(data_dir) / f'client_{client_id}'
#     if not path.exists():
#         raise FileNotFoundError(f"Client data not found: {path}")

#     # Try pre-split files first, then full dataset
#     if (path / 'X_train.npy').exists():
#         X_train = np.load(path / 'X_train.npy')
#         y_train = np.load(path / 'y_train.npy')
#         X_test = np.load(path / 'X_test.npy')
#         y_test = np.load(path / 'y_test.npy')
#     elif (path / 'X.npy').exists():
#         X = np.load(path / 'X.npy')
#         y = np.load(path / 'y.npy')
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=seed, stratify=y)
#     else:
#         raise FileNotFoundError(f"No X.npy or X_train.npy in {path}")

#     return X_train, y_train, X_test, y_test


# def generate_synthetic_data(n_clients: int = 4, n_features: int = 23,
#                             n_classes: int = 7, samples_per_client: int = 5000,
#                             non_iid: bool = True, seed: int = 42):
#     """Generate synthetic Non-IID data for baselines when real data unavailable."""
#     rng = np.random.default_rng(seed)
#     clients = []

#     # Create class centers for more realistic separation
#     class_centers = rng.standard_normal((n_classes, n_features)) * 2.0

#     for c in range(n_clients):
#         # Non-IID: each client sees only a subset of classes (label skew)
#         if non_iid:
#             n_local_classes = max(3, n_classes - c)
#             local_classes = sorted(rng.choice(n_classes, n_local_classes, replace=False))
#         else:
#             local_classes = list(range(n_classes))

#         # Generate samples per class with quantity skew
#         X_parts, y_parts = [], []
#         for cls in local_classes:
#             n_cls = rng.integers(samples_per_client // (2 * len(local_classes)),
#                                 samples_per_client // len(local_classes))
#             X_cls = class_centers[cls] + rng.standard_normal((n_cls, n_features)) * 0.5
#             y_cls = np.full(n_cls, cls)
#             X_parts.append(X_cls)
#             y_parts.append(y_cls)

#         X = np.vstack(X_parts).astype(np.float32)
#         y = np.concatenate(y_parts).astype(np.int64)

#         # Shuffle
#         idx = rng.permutation(len(X))
#         X, y = X[idx], y[idx]

#         # Split 80/20
#         split = int(0.8 * len(X))
#         clients.append({
#             'X_train': X[:split], 'y_train': y[:split],
#             'X_test': X[split:], 'y_test': y[split:],
#         })

#     return clients


# def make_dataloader(X, y, batch_size, shuffle=True):
#     """Create PyTorch DataLoader."""
#     ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
#     return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# def compute_class_weights(y, n_classes):
#     """Compute inverse-frequency class weights."""
#     counts = np.bincount(y, minlength=n_classes).astype(float)
#     counts = np.clip(counts, 1, None)
#     weights = len(y) / (n_classes * counts)
#     return torch.FloatTensor(weights)


# def evaluate_model(model, X_test, y_test, n_classes, class_names=None, device='cpu'):
#     """Evaluate model and return metrics dict."""
#     model.eval()
#     with torch.no_grad():
#         X_t = torch.FloatTensor(X_test).to(device)
#         logits = model(X_t)
#         probs = torch.softmax(logits, dim=1).cpu().numpy()
#         preds = logits.argmax(dim=1).cpu().numpy()

#     metrics = compute_classification_metrics(
#         y_true=y_test, y_pred=preds, y_prob=probs,
#         class_names=class_names, num_classes=n_classes,
#     )
#     return metrics


# # ─────────────────────────────────────────────────────────────────────────
# # Traditional ML Baselines (centralized, pooled data)
# # ─────────────────────────────────────────────────────────────────────────

# def run_traditional_ml_baselines(clients_data, config):
#     """Train and evaluate traditional ML classifiers on pooled (centralized) data.

#     Returns a dict: { 'random_forest': metrics, 'xgboost': metrics, ... }
#     """
#     from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#     from sklearn.svm import SVC
#     from sklearn.tree import DecisionTreeClassifier
#     from sklearn.neighbors import KNeighborsClassifier
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.preprocessing import label_binarize

#     # Try importing XGBoost; fall back to GradientBoosting if not installed
#     try:
#         from xgboost import XGBClassifier
#         has_xgboost = True
#     except ImportError:
#         has_xgboost = False
#         print("    [!] XGBoost not installed — using sklearn GradientBoosting instead")

#     # Pool all client data
#     X_train = np.vstack([c['X_train'] for c in clients_data])
#     y_train = np.concatenate([c['y_train'] for c in clients_data])
#     X_test = np.vstack([c['X_test'] for c in clients_data])
#     y_test = np.concatenate([c['y_test'] for c in clients_data])
#     n_classes = config.model.num_classes
#     class_names = ALL_CLASSES

#     print(f"\n  [Traditional ML] Pooled data: {X_train.shape[0]} train, "
#           f"{X_test.shape[0]} test, {X_train.shape[1]} features, {n_classes} classes")

#     # Define classifiers
#     classifiers = {
#         'random_forest': RandomForestClassifier(
#             n_estimators=200,
#             max_depth=None,
#             min_samples_split=5,
#             class_weight='balanced',
#             n_jobs=-1,
#             random_state=42,
#         ),
#         'decision_tree': DecisionTreeClassifier(
#             max_depth=None,
#             min_samples_split=5,
#             class_weight='balanced',
#             random_state=42,
#         ),
#         'knn': KNeighborsClassifier(
#             n_neighbors=5,
#             n_jobs=-1,
#         ),
#         'logistic_regression': LogisticRegression(
#             max_iter=1000,
#             class_weight='balanced',
#             solver='lbfgs',
#             n_jobs=-1,
#             random_state=42,
#         ),
#         'svm': SVC(
#             kernel='rbf',
#             class_weight='balanced',
#             probability=True,
#             random_state=42,
#         ),
#     }

#     # Add XGBoost or GradientBoosting
#     if has_xgboost:
#         classifiers['xgboost'] = XGBClassifier(
#             n_estimators=200,
#             max_depth=6,
#             learning_rate=0.1,
#             use_label_encoder=False,
#             eval_metric='mlogloss',
#             n_jobs=-1,
#             random_state=42,
#         )
#     else:
#         classifiers['gradient_boosting'] = GradientBoostingClassifier(
#             n_estimators=200,
#             max_depth=6,
#             learning_rate=0.1,
#             random_state=42,
#         )

#     results = {}
#     for name, clf in classifiers.items():
#         print(f"\n    Training {name}...")
#         t0 = time.time()
#         clf.fit(X_train, y_train)
#         train_time = time.time() - t0

#         y_pred = clf.predict(X_test)

#         # Get probability predictions if available
#         if hasattr(clf, 'predict_proba'):
#             y_prob = clf.predict_proba(X_test)
#             # Ensure correct shape (n_samples, n_classes)
#             if y_prob.shape[1] < n_classes:
#                 full_prob = np.zeros((len(y_test), n_classes))
#                 for i, c in enumerate(clf.classes_):
#                     full_prob[:, int(c)] = y_prob[:, i]
#                 y_prob = full_prob
#         else:
#             # One-hot fallback
#             y_prob = np.eye(n_classes)[y_pred.astype(int)]

#         metrics = compute_classification_metrics(
#             y_true=y_test, y_pred=y_pred, y_prob=y_prob,
#             class_names=class_names, num_classes=n_classes,
#         )
#         metrics['train_time_s'] = train_time
#         results[name] = metrics

#         print(f"    {name}: Acc={metrics.get('accuracy',0):.4f} "
#               f"F1={metrics.get('f1_macro',0):.4f} "
#               f"MCC={metrics.get('mcc',0):.4f} "
#               f"({train_time:.1f}s)")

#     return results


# # ─────────────────────────────────────────────────────────────────────────
# # Baseline (DL): Centralized MLP
# # ─────────────────────────────────────────────────────────────────────────

# def run_centralized(clients_data, config, device='cpu'):
#     """Train a single model on pooled data from all clients."""
#     print("\n  [Centralized] Pooling all client data...")

#     # Pool all data
#     X_train = np.vstack([c['X_train'] for c in clients_data])
#     y_train = np.concatenate([c['y_train'] for c in clients_data])
#     X_test = np.vstack([c['X_test'] for c in clients_data])
#     y_test = np.concatenate([c['y_test'] for c in clients_data])

#     model = FedPerProxModel(
#         input_dim=config.model.input_dim,
#         num_classes=config.model.num_classes,
#         backbone_hidden=config.model.backbone_hidden,
#         head_hidden=config.model.head_hidden,
#         dropout=config.model.dropout,
#     ).to(device)

#     weights = compute_class_weights(y_train, config.model.num_classes).to(device)
#     criterion = FocalLoss(alpha=weights, gamma=2.0)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr,
#                                    weight_decay=config.training.weight_decay)

#     loader = make_dataloader(X_train, y_train, config.training.batch_size)

#     total_epochs = config.training.num_rounds * config.training.local_epochs
#     model.train()
#     t0 = time.time()
#     for epoch in range(total_epochs):
#         for X_b, y_b in loader:
#             X_b, y_b = X_b.to(device), y_b.to(device)
#             optimizer.zero_grad()
#             loss = criterion(model(X_b), y_b)
#             loss.backward()
#             optimizer.step()
#     train_time = time.time() - t0

#     metrics = evaluate_model(model, X_test, y_test, config.model.num_classes,
#                              ALL_CLASSES, device)
#     metrics['train_time_s'] = train_time
#     metrics['total_epochs'] = total_epochs
#     return metrics


# # ─────────────────────────────────────────────────────────────────────────
# # Baseline 2: Local-only (no federation)
# # ─────────────────────────────────────────────────────────────────────────

# def run_local_only(clients_data, config, device='cpu'):
#     """Each client trains independently — no federation."""
#     print("\n  [Local-only] Training each client independently...")
#     all_metrics = []
#     total_epochs = config.training.num_rounds * config.training.local_epochs

#     for cid, cdata in enumerate(clients_data):
#         model = FedPerProxModel(
#             input_dim=config.model.input_dim,
#             num_classes=config.model.num_classes,
#             backbone_hidden=config.model.backbone_hidden,
#             head_hidden=config.model.head_hidden,
#             dropout=config.model.dropout,
#         ).to(device)

#         weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
#         criterion = FocalLoss(alpha=weights, gamma=2.0)
#         optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr,
#                                        weight_decay=config.training.weight_decay)

#         loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

#         model.train()
#         for epoch in range(total_epochs):
#             for X_b, y_b in loader:
#                 X_b, y_b = X_b.to(device), y_b.to(device)
#                 optimizer.zero_grad()
#                 loss = criterion(model(X_b), y_b)
#                 loss.backward()
#                 optimizer.step()

#         m = evaluate_model(model, cdata['X_test'], cdata['y_test'],
#                            config.model.num_classes, ALL_CLASSES, device)
#         all_metrics.append(m)
#         print(f"    Client {cid}: acc={m['accuracy']:.4f} F1={m['f1_macro']:.4f}")

#     return _aggregate_client_metrics(all_metrics)


# # ─────────────────────────────────────────────────────────────────────────
# # Baseline 3: FedAvg
# # ─────────────────────────────────────────────────────────────────────────

# def _fedavg_aggregate(models, weights):
#     """Weighted average of ALL parameters (backbone + head)."""
#     global_state = copy.deepcopy(models[0].state_dict())
#     for key in global_state:
#         global_state[key] = sum(
#             w * models[i].state_dict()[key].float()
#             for i, w in enumerate(weights)
#         )
#     return global_state


# def run_fedavg(clients_data, config, device='cpu'):
#     """FedAvg — full model aggregation (McMahan et al., 2017)."""
#     print("\n  [FedAvg] Federated averaging (full model)...")
#     n_clients = len(clients_data)

#     # Initialize models
#     models = []
#     for _ in range(n_clients):
#         m = FedPerProxModel(
#             input_dim=config.model.input_dim,
#             num_classes=config.model.num_classes,
#             backbone_hidden=config.model.backbone_hidden,
#             head_hidden=config.model.head_hidden,
#             dropout=config.model.dropout,
#         ).to(device)
#         models.append(m)

#     # Global model (for initial state)
#     global_state = copy.deepcopy(models[0].state_dict())

#     for rnd in range(config.training.num_rounds):
#         # Distribute global model
#         for m in models:
#             m.load_state_dict(copy.deepcopy(global_state))

#         # Local training
#         for cid in range(n_clients):
#             cdata = clients_data[cid]
#             weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
#             criterion = FocalLoss(alpha=weights, gamma=2.0)
#             optimizer = torch.optim.AdamW(models[cid].parameters(), lr=config.training.lr,
#                                            weight_decay=config.training.weight_decay)
#             loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

#             models[cid].train()
#             for _ in range(config.training.local_epochs):
#                 for X_b, y_b in loader:
#                     X_b, y_b = X_b.to(device), y_b.to(device)
#                     optimizer.zero_grad()
#                     loss = criterion(models[cid](X_b), y_b)
#                     loss.backward()
#                     optimizer.step()

#         # Aggregate (uniform weights for FedAvg)
#         n_samples = [len(c['y_train']) for c in clients_data]
#         total = sum(n_samples)
#         w = [n / total for n in n_samples]
#         global_state = _fedavg_aggregate(models, w)

#     # Evaluate each client
#     all_metrics = []
#     for cid in range(n_clients):
#         models[cid].load_state_dict(copy.deepcopy(global_state))
#         m = evaluate_model(models[cid], clients_data[cid]['X_test'],
#                            clients_data[cid]['y_test'],
#                            config.model.num_classes, ALL_CLASSES, device)
#         all_metrics.append(m)

#     return _aggregate_client_metrics(all_metrics)


# # ─────────────────────────────────────────────────────────────────────────
# # Baseline 4: FedProx (full model + proximal term)
# # ─────────────────────────────────────────────────────────────────────────

# def run_fedprox(clients_data, config, device='cpu'):
#     """FedProx — FedAvg + proximal regularization (Li et al., 2020)."""
#     print(f"\n  [FedProx] FedAvg + proximal term (mu={config.training.mu_prox})...")
#     n_clients = len(clients_data)

#     models = []
#     for _ in range(n_clients):
#         m = FedPerProxModel(
#             input_dim=config.model.input_dim,
#             num_classes=config.model.num_classes,
#             backbone_hidden=config.model.backbone_hidden,
#             head_hidden=config.model.head_hidden,
#             dropout=config.model.dropout,
#         ).to(device)
#         models.append(m)

#     global_state = copy.deepcopy(models[0].state_dict())

#     for rnd in range(config.training.num_rounds):
#         # Distribute
#         for m in models:
#             m.load_state_dict(copy.deepcopy(global_state))

#         # Local training with proximal term
#         for cid in range(n_clients):
#             cdata = clients_data[cid]
#             weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
#             criterion = FocalLoss(alpha=weights, gamma=2.0)
#             optimizer = torch.optim.AdamW(models[cid].parameters(), lr=config.training.lr,
#                                            weight_decay=config.training.weight_decay)
#             loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

#             # Save global params for proximal term (ALL params for FedProx)
#             global_params = [p.detach().clone() for p in models[cid].parameters()]

#             models[cid].train()
#             for _ in range(config.training.local_epochs):
#                 for X_b, y_b in loader:
#                     X_b, y_b = X_b.to(device), y_b.to(device)
#                     optimizer.zero_grad()
#                     loss = criterion(models[cid](X_b), y_b)
#                     # Proximal term on ALL parameters
#                     prox = torch.tensor(0.0, device=device)
#                     for p_local, p_global in zip(models[cid].parameters(), global_params):
#                         prox = prox + ((p_local - p_global) ** 2).sum()
#                     loss = loss + (config.training.mu_prox / 2.0) * prox
#                     loss.backward()
#                     optimizer.step()

#         # Aggregate (full model)
#         n_samples = [len(c['y_train']) for c in clients_data]
#         total = sum(n_samples)
#         w = [n / total for n in n_samples]
#         global_state = _fedavg_aggregate(models, w)

#     # Evaluate
#     all_metrics = []
#     for cid in range(n_clients):
#         models[cid].load_state_dict(copy.deepcopy(global_state))
#         m = evaluate_model(models[cid], clients_data[cid]['X_test'],
#                            clients_data[cid]['y_test'],
#                            config.model.num_classes, ALL_CLASSES, device)
#         all_metrics.append(m)

#     return _aggregate_client_metrics(all_metrics)


# # ─────────────────────────────────────────────────────────────────────────
# # Baseline 5: FedPer (backbone-only aggregation, no prox)
# # ─────────────────────────────────────────────────────────────────────────

# def _fedper_aggregate(models, weights):
#     """Weighted average of BACKBONE parameters only."""
#     global_backbone = {}
#     for key in models[0].state_dict():
#         if key.startswith('backbone.'):
#             global_backbone[key] = sum(
#                 w * models[i].state_dict()[key].float()
#                 for i, w in enumerate(weights)
#             )
#     return global_backbone


# def run_fedper(clients_data, config, device='cpu'):
#     """FedPer — partial model sharing (Arivazhagan et al., 2019)."""
#     print("\n  [FedPer] Backbone-only aggregation (no proximal)...")
#     n_clients = len(clients_data)

#     models = []
#     for _ in range(n_clients):
#         m = FedPerProxModel(
#             input_dim=config.model.input_dim,
#             num_classes=config.model.num_classes,
#             backbone_hidden=config.model.backbone_hidden,
#             head_hidden=config.model.head_hidden,
#             dropout=config.model.dropout,
#         ).to(device)
#         models.append(m)

#     # Initial backbone state
#     global_backbone = {k: v.clone() for k, v in models[0].state_dict().items()
#                        if k.startswith('backbone.')}

#     for rnd in range(config.training.num_rounds):
#         # Distribute backbone only (heads stay local)
#         for m in models:
#             sd = m.state_dict()
#             sd.update(copy.deepcopy(global_backbone))
#             m.load_state_dict(sd)

#         # Local training (no proximal term)
#         for cid in range(n_clients):
#             cdata = clients_data[cid]
#             weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
#             criterion = FocalLoss(alpha=weights, gamma=2.0)
#             optimizer = torch.optim.AdamW(models[cid].parameters(), lr=config.training.lr,
#                                            weight_decay=config.training.weight_decay)
#             loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

#             models[cid].train()
#             for _ in range(config.training.local_epochs):
#                 for X_b, y_b in loader:
#                     X_b, y_b = X_b.to(device), y_b.to(device)
#                     optimizer.zero_grad()
#                     loss = criterion(models[cid](X_b), y_b)
#                     loss.backward()
#                     optimizer.step()

#         # Aggregate backbone only with temperature-softened weights
#         n_samples = [len(c['y_train']) for c in clients_data]
#         tau = config.training.aggregation_temp
#         w_raw = [n ** tau for n in n_samples]
#         w_sum = sum(w_raw)
#         w = [wr / w_sum for wr in w_raw]
#         global_backbone = _fedper_aggregate(models, w)

#     # Evaluate (each client uses its own head)
#     all_metrics = []
#     for cid in range(n_clients):
#         m = evaluate_model(models[cid], clients_data[cid]['X_test'],
#                            clients_data[cid]['y_test'],
#                            config.model.num_classes, ALL_CLASSES, device)
#         all_metrics.append(m)

#     return _aggregate_client_metrics(all_metrics)


# # ─────────────────────────────────────────────────────────────────────────
# # Our Method: FedPer-Prox-XAI (backbone-only aggregation + proximal)
# # ─────────────────────────────────────────────────────────────────────────

# def run_fedper_prox(clients_data, config, device='cpu'):
#     """FedPer-Prox-XAI — backbone-only aggregation + proximal (ours)."""
#     print(f"\n  [FedPer-Prox-XAI] backbone split + proximal (mu={config.training.mu_prox})...")
#     n_clients = len(clients_data)

#     models = []
#     for _ in range(n_clients):
#         m = FedPerProxModel(
#             input_dim=config.model.input_dim,
#             num_classes=config.model.num_classes,
#             backbone_hidden=config.model.backbone_hidden,
#             head_hidden=config.model.head_hidden,
#             dropout=config.model.dropout,
#         ).to(device)
#         models.append(m)

#     global_backbone = {k: v.clone() for k, v in models[0].state_dict().items()
#                        if k.startswith('backbone.')}

#     for rnd in range(config.training.num_rounds):
#         # Distribute backbone only
#         for m in models:
#             sd = m.state_dict()
#             sd.update(copy.deepcopy(global_backbone))
#             m.load_state_dict(sd)

#         # Local training with proximal term on backbone only
#         for cid in range(n_clients):
#             cdata = clients_data[cid]
#             weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
#             criterion = FocalLoss(alpha=weights, gamma=2.0)
#             optimizer = torch.optim.AdamW(models[cid].parameters(), lr=config.training.lr,
#                                            weight_decay=config.training.weight_decay)
#             loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

#             # Save global backbone for proximal term
#             global_bb_params = [p.detach().clone() for p in models[cid].backbone.parameters()]

#             models[cid].train()
#             for _ in range(config.training.local_epochs):
#                 for X_b, y_b in loader:
#                     X_b, y_b = X_b.to(device), y_b.to(device)
#                     optimizer.zero_grad()
#                     task_loss = criterion(models[cid](X_b), y_b)
#                     prox_loss = compute_proximal_loss(
#                         models[cid], global_bb_params, config.training.mu_prox)
#                     loss = task_loss + prox_loss
#                     loss.backward()
#                     optimizer.step()

#         # Aggregate backbone with temperature-softened weights
#         n_samples = [len(c['y_train']) for c in clients_data]
#         tau = config.training.aggregation_temp
#         w_raw = [n ** tau for n in n_samples]
#         w_sum = sum(w_raw)
#         w = [wr / w_sum for wr in w_raw]
#         global_backbone = _fedper_aggregate(models, w)

#     # Evaluate
#     all_metrics = []
#     for cid in range(n_clients):
#         m = evaluate_model(models[cid], clients_data[cid]['X_test'],
#                            clients_data[cid]['y_test'],
#                            config.model.num_classes, ALL_CLASSES, device)
#         all_metrics.append(m)

#     return _aggregate_client_metrics(all_metrics)


# # ─────────────────────────────────────────────────────────────────────────
# # Aggregation / reporting
# # ─────────────────────────────────────────────────────────────────────────

# def _aggregate_client_metrics(metrics_list):
#     """Average metrics across clients."""
#     keys = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted',
#             'precision_macro', 'recall_macro', 'cohen_kappa', 'mcc']
#     agg = {}
#     for k in keys:
#         vals = [m[k] for m in metrics_list if k in m]
#         if vals:
#             agg[k] = float(np.mean(vals))
#             agg[f'{k}_std'] = float(np.std(vals))
#     agg['n_clients'] = len(metrics_list)
#     agg['per_client'] = metrics_list
#     return agg


# def run_all_baselines(clients_data, config, device='cpu'):
#     """Run all baselines and return comparison dict."""
#     results = {}

#     # ── Traditional ML baselines (centralized) ──
#     print("\n" + "─" * 70)
#     print("  TRADITIONAL ML BASELINES")
#     print("─" * 70)
#     t0 = time.time()
#     ml_results = run_traditional_ml_baselines(clients_data, config)
#     for method_name, method_metrics in ml_results.items():
#         method_metrics['wall_time_s'] = time.time() - t0
#         results[method_name] = method_metrics

#     # ── Deep Learning baselines ──
#     print("\n" + "─" * 70)
#     print("  DEEP LEARNING & FEDERATED BASELINES")
#     print("─" * 70)

#     t0 = time.time()
#     results['centralized_mlp'] = run_centralized(clients_data, config, device)
#     results['centralized_mlp']['wall_time_s'] = time.time() - t0

#     t0 = time.time()
#     results['local_only'] = run_local_only(clients_data, config, device)
#     results['local_only']['wall_time_s'] = time.time() - t0

#     t0 = time.time()
#     results['fedavg'] = run_fedavg(clients_data, config, device)
#     results['fedavg']['wall_time_s'] = time.time() - t0

#     t0 = time.time()
#     results['fedprox'] = run_fedprox(clients_data, config, device)
#     results['fedprox']['wall_time_s'] = time.time() - t0

#     t0 = time.time()
#     results['fedper'] = run_fedper(clients_data, config, device)
#     results['fedper']['wall_time_s'] = time.time() - t0

#     t0 = time.time()
#     results['fedper_prox_xai'] = run_fedper_prox(clients_data, config, device)
#     results['fedper_prox_xai']['wall_time_s'] = time.time() - t0

#     return results


# def generate_comparison_table_latex(all_results, out_path):
#     """Generate LaTeX booktabs comparison table."""
#     methods = [
#         # Traditional ML
#         ('random_forest', 'Random Forest'),
#         ('xgboost', 'XGBoost'),
#         ('gradient_boosting', 'Gradient Boosting'),
#         ('svm', 'SVM (RBF)'),
#         ('decision_tree', 'Decision Tree'),
#         ('knn', 'KNN'),
#         ('logistic_regression', 'Logistic Regression'),
#         # Deep Learning & Federated
#         ('centralized_mlp', 'Centralized MLP'),
#         ('local_only', 'Local-only MLP'),
#         ('fedavg', 'FedAvg'),
#         ('fedprox', 'FedProx'),
#         ('fedper', 'FedPer'),
#         ('fedper_prox_xai', '\\textbf{FedPer-Prox-XAI (Ours)}'),
#     ]
#     # Filter to only methods present in results
#     is_multi_seed = isinstance(all_results, dict) and 'mean' in all_results
#     if is_multi_seed:
#         available = set(all_results['mean'].keys())
#     else:
#         available = set(all_results.keys()) - {'seed'}
#     methods = [(k, v) for k, v in methods if k in available]
#     metrics = [
#         ('accuracy', 'Accuracy'),
#         ('f1_macro', 'F1-Macro'),
#         ('f1_weighted', 'F1-Weighted'),
#         ('precision_macro', 'Precision'),
#         ('recall_macro', 'Recall'),
#         ('cohen_kappa', "Cohen's $\\kappa$"),
#         ('mcc', 'MCC'),
#     ]

#     lines = [
#         '% Requires \\usepackage{booktabs}',
#         '\\begin{table*}[t]',
#         '\\centering',
#         '\\caption{Performance comparison of FedPer-Prox-XAI against '
#         'baseline methods on IoMT Non-IID data (mean $\\pm$ std across clients/seeds).}',
#         '\\label{tab:sota_comparison}',
#         '\\begin{tabular}{l' + 'c' * len(metrics) + '}',
#         '\\toprule',
#         'Method & ' + ' & '.join(m[1] for m in metrics) + ' \\\\',
#         '\\midrule',
#     ]

#     # Check if we have multi-seed results
#     is_multi_seed = isinstance(all_results, dict) and 'mean' in all_results

#     # Traditional ML methods
#     ml_keys = {'random_forest', 'xgboost', 'gradient_boosting', 'svm',
#                 'decision_tree', 'knn', 'logistic_regression'}
#     ml_methods = [(k, v) for k, v in methods if k in ml_keys]
#     dl_methods = [(k, v) for k, v in methods if k not in ml_keys]

#     if ml_methods:
#         ncols = len(metrics)
#         lines.append(f'\\multicolumn{{{ncols+1}}}{{l}}{{\\textit{{Traditional ML (centralized)}}}}'
#                       ' \\\\')
#         for key, label in ml_methods:
#             row_parts = [label]
#             for mk, _ in metrics:
#                 if is_multi_seed:
#                     mean_val = all_results['mean'].get(key, {}).get(mk, 0)
#                     std_val = all_results['std'].get(key, {}).get(mk, 0)
#                     row_parts.append(f'{mean_val:.3f} $\\pm$ {std_val:.3f}')
#                 else:
#                     val = all_results.get(key, {}).get(mk, 0)
#                     std = all_results.get(key, {}).get(f'{mk}_std', 0)
#                     if std > 0:
#                         row_parts.append(f'{val:.3f} $\\pm$ {std:.3f}')
#                     else:
#                         row_parts.append(f'{val:.3f}')
#             lines.append(' & '.join(row_parts) + ' \\\\')
#         lines.append('\\midrule')

#     if dl_methods:
#         ncols = len(metrics)
#         lines.append(f'\\multicolumn{{{ncols+1}}}{{l}}{{\\textit{{Deep Learning \\& Federated}}}}'
#                       ' \\\\')
#         for key, label in dl_methods:
#             row_parts = [label]
#             for mk, _ in metrics:
#                 if is_multi_seed:
#                     mean_val = all_results['mean'].get(key, {}).get(mk, 0)
#                     std_val = all_results['std'].get(key, {}).get(mk, 0)
#                     row_parts.append(f'{mean_val:.3f} $\\pm$ {std_val:.3f}')
#                 else:
#                     val = all_results.get(key, {}).get(mk, 0)
#                     std = all_results.get(key, {}).get(f'{mk}_std', 0)
#                     if std > 0:
#                         row_parts.append(f'{val:.3f} $\\pm$ {std:.3f}')
#                     else:
#                         row_parts.append(f'{val:.3f}')
#             lines.append(' & '.join(row_parts) + ' \\\\')

#     lines += [
#         '\\bottomrule',
#         '\\end{tabular}',
#         '\\end{table*}',
#     ]

#     with open(out_path, 'w') as f:
#         f.write('\n'.join(lines) + '\n')


# def generate_comparison_csv(all_results, out_path):
#     """Generate CSV comparison table."""
#     all_possible_methods = [
#         'random_forest', 'xgboost', 'gradient_boosting', 'svm',
#         'decision_tree', 'knn', 'logistic_regression',
#         'centralized_mlp', 'local_only', 'fedavg', 'fedprox', 'fedper', 'fedper_prox_xai',
#     ]
#     metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro',
#                'recall_macro', 'cohen_kappa', 'mcc']

#     is_multi_seed = isinstance(all_results, dict) and 'mean' in all_results
#     if is_multi_seed:
#         available = set(all_results['mean'].keys())
#     else:
#         available = set(all_results.keys()) - {'seed'}
#     methods = [m for m in all_possible_methods if m in available]

#     header = 'method,' + ','.join(f'{m}_mean,{m}_std' for m in metrics)
#     rows = [header]

#     for method in methods:
#         parts = [method]
#         for mk in metrics:
#             if is_multi_seed:
#                 mean_val = all_results['mean'].get(method, {}).get(mk, 0)
#                 std_val = all_results['std'].get(method, {}).get(mk, 0)
#             else:
#                 mean_val = all_results.get(method, {}).get(mk, 0)
#                 std_val = all_results.get(method, {}).get(f'{mk}_std', 0)
#             parts.extend([f'{mean_val:.4f}', f'{std_val:.4f}'])
#         rows.append(','.join(parts))

#     with open(out_path, 'w') as f:
#         f.write('\n'.join(rows) + '\n')


# # ─────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────

# def main():
#     p = argparse.ArgumentParser(description='R3.6: SOTA baseline comparison')
#     p.add_argument('--data-dir', type=str, default='data/client_datasets',
#                    help='Client data directory (with client_0/, client_1/, ...)')
#     p.add_argument('--seeds', type=int, nargs='+', default=[42],
#                    help='Random seeds for reproducibility')
#     p.add_argument('--n-clients', type=int, default=4)
#     p.add_argument('--synthetic', action='store_true',
#                    help='Use synthetic data (if real data unavailable)')
#     p.add_argument('--out-dir', type=str, default='./results/sota_comparison')
#     args = p.parse_args()

#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     config = FLConfig()
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     print("=" * 70)
#     print("  R3.6 — SOTA BASELINE COMPARISON")
#     print("=" * 70)
#     print(f"  Seeds     : {args.seeds}")
#     print(f"  Clients   : {args.n_clients}")
#     print(f"  Rounds    : {config.training.num_rounds}")
#     print(f"  Epochs/R  : {config.training.local_epochs}")
#     print(f"  Device    : {device}")
#     print(f"  Data      : {'synthetic' if args.synthetic else args.data_dir}")

#     # Load or generate data
#     data_path = Path(args.data_dir)
#     use_synthetic = args.synthetic or not data_path.exists()
#     if use_synthetic:
#         print("\n  Using synthetic Non-IID data...")

#     all_seed_results = []

#     for seed in args.seeds:
#         print(f"\n{'='*70}")
#         print(f"  SEED = {seed}")
#         print(f"{'='*70}")
#         set_all_seeds(seed)

#         if use_synthetic:
#             clients_data = generate_synthetic_data(
#                 n_clients=args.n_clients,
#                 n_features=config.model.input_dim,
#                 n_classes=config.model.num_classes,
#                 seed=seed,
#             )
#         else:
#             clients_data = []
#             for cid in range(args.n_clients):
#                 X_tr, y_tr, X_te, y_te = load_client_data(args.data_dir, cid, seed=seed)
#                 clients_data.append({
#                     'X_train': X_tr, 'y_train': y_tr,
#                     'X_test': X_te, 'y_test': y_te,
#                 })
#             # Update input_dim from actual data
#             config.model.input_dim = clients_data[0]['X_train'].shape[1]
#             print(f"  Input dim: {config.model.input_dim}")

#         results = run_all_baselines(clients_data, config, device)
#         results['seed'] = seed
#         all_seed_results.append(results)

#         # Print summary for this seed
#         print(f"\n  Summary (seed={seed}):")
#         print(f"  {'Method':<25s} {'Accuracy':>10s} {'F1-Macro':>10s} {'MCC':>10s}")
#         print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
#         for key in sorted(results.keys()):
#             if key == 'seed':
#                 continue
#             r = results[key]
#             if isinstance(r, dict) and 'accuracy' in r:
#                 print(f"  {key:<25s} {r.get('accuracy',0):>10.4f} "
#                       f"{r.get('f1_macro',0):>10.4f} {r.get('mcc',0):>10.4f}")

#     # Aggregate across seeds
#     if len(args.seeds) > 1:
#         methods = [k for k in all_seed_results[0].keys() if k != 'seed']
#         metrics_keys = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro',
#                         'recall_macro', 'cohen_kappa', 'mcc']

#         agg = {'mean': {}, 'std': {}}
#         for method in methods:
#             agg['mean'][method] = {}
#             agg['std'][method] = {}
#             for mk in metrics_keys:
#                 vals = [r[method].get(mk, 0) for r in all_seed_results]
#                 agg['mean'][method][mk] = float(np.mean(vals))
#                 agg['std'][method][mk] = float(np.std(vals))

#         generate_comparison_table_latex(agg, out_dir / 'sota_comparison_table.tex')
#         generate_comparison_csv(agg, out_dir / 'sota_comparison.csv')

#         # Summary
#         print(f"\n{'='*70}")
#         print(f"  AGGREGATED RESULTS ({len(args.seeds)} seeds)")
#         print(f"{'='*70}")
#         print(f"  {'Method':<25s} {'Accuracy':>16s} {'F1-Macro':>16s} {'MCC':>16s}")
#         print(f"  {'-'*25} {'-'*16} {'-'*16} {'-'*16}")
#         for method in methods:
#             acc = f"{agg['mean'][method]['accuracy']:.3f}±{agg['std'][method]['accuracy']:.3f}"
#             f1 = f"{agg['mean'][method]['f1_macro']:.3f}±{agg['std'][method]['f1_macro']:.3f}"
#             mcc = f"{agg['mean'][method]['mcc']:.3f}±{agg['std'][method]['mcc']:.3f}"
#             print(f"  {method:<25s} {acc:>16s} {f1:>16s} {mcc:>16s}")
#     else:
#         generate_comparison_table_latex(all_seed_results[0], out_dir / 'sota_comparison_table.tex')
#         generate_comparison_csv(all_seed_results[0], out_dir / 'sota_comparison.csv')

#     # Save full JSON report
#     # Clean up non-serializable fields
#     def clean_for_json(obj):
#         if isinstance(obj, dict):
#             return {k: clean_for_json(v) for k, v in obj.items()}
#         if isinstance(obj, list):
#             return [clean_for_json(v) for v in obj]
#         if isinstance(obj, (np.integer,)):
#             return int(obj)
#         if isinstance(obj, (np.floating, np.float64)):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return obj

#     with open(out_dir / 'sota_comparison_full.json', 'w') as f:
#         json.dump(clean_for_json(all_seed_results), f, indent=2)

#     print(f"\n  Saved to: {out_dir}/")
#     print(f"    sota_comparison_full.json")
#     print(f"    sota_comparison_table.tex")
#     print(f"    sota_comparison.csv")


# if __name__ == '__main__':
#     main()



#!/usr/bin/env python3
"""R3.6 — SOTA baseline comparison for FedPer-Prox-XAI.

Implements and evaluates the following baselines to compare against
FedPer-Prox-XAI (Reviewer 3, Comment 6: compare with latest SOTA):

  Traditional ML (centralized, pooled data):
  1. Random Forest         — Breiman (2001)
  2. XGBoost / GradientBoosting — Chen & Guestrin (2016) / Friedman (2001)
  3. SVM (RBF kernel)      — Cortes & Vapnik (1995)
  4. Decision Tree (CART)   — Breiman et al. (1984)
  5. K-Nearest Neighbors   — Cover & Hart (1967)
  6. Logistic Regression    — Cox (1958)

  Deep Learning & Federated:
  7. Centralized MLP       — Standard MLP trained on all data pooled together
  8. Local-only MLP        — Each client trains independently (no federation)
  9. FedAvg                — McMahan et al. (2017) — full model aggregation
  10. FedProx              — Li et al. (2020) — FedAvg + proximal term
  11. FedPer               — Arivazhagan et al. (2019) — split backbone/head
  12. FedPer-Prox-XAI      — Our method (backbone+head split, proximal, SHAP)

All baselines use the SAME data splits for fair comparison.

Usage:
  python -I scripts/run_sota_baselines.py
  python -I scripts/run_sota_baselines.py --seeds 0 1 2 3 42 --data-dir data/client_datasets
"""
import os, sys, argparse, json, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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


def load_client_data(data_dir: str, client_id: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load client partition from .npy files + 80/20 stratified split."""
    from sklearn.model_selection import train_test_split
    path = Path(data_dir) / f'client_{client_id}'
    if not path.exists():
        raise FileNotFoundError(f"Client data not found: {path}")

    # Try pre-split files first, then full dataset
    if (path / 'X_train.npy').exists():
        X_train = np.load(path / 'X_train.npy')
        y_train = np.load(path / 'y_train.npy')
        X_test = np.load(path / 'X_test.npy')
        y_test = np.load(path / 'y_test.npy')
    elif (path / 'X.npy').exists():
        X = np.load(path / 'X.npy')
        y = np.load(path / 'y.npy')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
    else:
        raise FileNotFoundError(f"No X.npy or X_train.npy in {path}")

    return X_train, y_train, X_test, y_test


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

    metrics = compute_classification_metrics(
        y_true=y_test, y_pred=preds, y_prob=probs,
        class_names=class_names, num_classes=n_classes,
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────
# Traditional ML Baselines (centralized, pooled data)
# ─────────────────────────────────────────────────────────────────────────

def run_traditional_ml_baselines(clients_data, config):
    """Train and evaluate traditional ML classifiers on pooled (centralized) data.

    Returns a dict: { 'random_forest': metrics, 'xgboost': metrics, ... }
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import label_binarize

    # Try importing XGBoost; fall back to GradientBoosting if not installed
    try:
        from xgboost import XGBClassifier
        has_xgboost = True
    except ImportError:
        has_xgboost = False
        print("    [!] XGBoost not installed — using sklearn GradientBoosting instead")

    # Pool all client data
    X_train = np.vstack([c['X_train'] for c in clients_data])
    y_train = np.concatenate([c['y_train'] for c in clients_data])
    X_test = np.vstack([c['X_test'] for c in clients_data])
    y_test = np.concatenate([c['y_test'] for c in clients_data])
    n_classes = config.model.num_classes
    class_names = ALL_CLASSES

    print(f"\n  [Traditional ML] Pooled data: {X_train.shape[0]} train, "
          f"{X_test.shape[0]} test, {X_train.shape[1]} features, {n_classes} classes")

    # Define classifiers
    classifiers = {
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        ),
        'decision_tree': DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
            n_jobs=-1,
            random_state=42,
        ),
    }

    # Add XGBoost or GradientBoosting
    if has_xgboost:
        classifiers['xgboost'] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=42,
        )
    else:
        classifiers['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )

    results = {}
    for name, clf in classifiers.items():
        print(f"\n    Training {name}...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = clf.predict(X_test)

        # Get probability predictions if available
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(X_test)
            # Ensure correct shape (n_samples, n_classes)
            if y_prob.shape[1] < n_classes:
                full_prob = np.zeros((len(y_test), n_classes))
                for i, c in enumerate(clf.classes_):
                    full_prob[:, int(c)] = y_prob[:, i]
                y_prob = full_prob
        else:
            # One-hot fallback
            y_prob = np.eye(n_classes)[y_pred.astype(int)]

        metrics = compute_classification_metrics(
            y_true=y_test, y_pred=y_pred, y_prob=y_prob,
            class_names=class_names, num_classes=n_classes,
        )
        metrics['train_time_s'] = train_time
        results[name] = metrics

        print(f"    {name}: Acc={metrics.get('accuracy',0):.4f} "
              f"F1={metrics.get('f1_macro',0):.4f} "
              f"MCC={metrics.get('mcc',0):.4f} "
              f"({train_time:.1f}s)")

    return results


# ─────────────────────────────────────────────────────────────────────────
# Baseline (DL): Centralized MLP
# ─────────────────────────────────────────────────────────────────────────

def run_centralized(clients_data, config, device='cpu'):
    """Train a single model on pooled data from all clients."""
    print("\n  [Centralized] Pooling all client data...")

    # Pool all data
    X_train = np.vstack([c['X_train'] for c in clients_data])
    y_train = np.concatenate([c['y_train'] for c in clients_data])
    X_test = np.vstack([c['X_test'] for c in clients_data])
    y_test = np.concatenate([c['y_test'] for c in clients_data])

    model = FedPerProxModel(
        input_dim=config.model.input_dim,
        num_classes=config.model.num_classes,
        backbone_hidden=config.model.backbone_hidden,
        head_hidden=config.model.head_hidden,
        dropout=config.model.dropout,
    ).to(device)

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
        model = FedPerProxModel(
            input_dim=config.model.input_dim,
            num_classes=config.model.num_classes,
            backbone_hidden=config.model.backbone_hidden,
            head_hidden=config.model.head_hidden,
            dropout=config.model.dropout,
        ).to(device)

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

    # Initialize models
    models = []
    for _ in range(n_clients):
        m = FedPerProxModel(
            input_dim=config.model.input_dim,
            num_classes=config.model.num_classes,
            backbone_hidden=config.model.backbone_hidden,
            head_hidden=config.model.head_hidden,
            dropout=config.model.dropout,
        ).to(device)
        models.append(m)

    # Global model (for initial state)
    global_state = copy.deepcopy(models[0].state_dict())

    for rnd in range(config.training.num_rounds):
        # Distribute global model
        for m in models:
            m.load_state_dict(copy.deepcopy(global_state))

        # Local training
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

        # Aggregate (uniform weights for FedAvg)
        n_samples = [len(c['y_train']) for c in clients_data]
        total = sum(n_samples)
        w = [n / total for n in n_samples]
        global_state = _fedavg_aggregate(models, w)

    # Evaluate each client
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

    models = []
    for _ in range(n_clients):
        m = FedPerProxModel(
            input_dim=config.model.input_dim,
            num_classes=config.model.num_classes,
            backbone_hidden=config.model.backbone_hidden,
            head_hidden=config.model.head_hidden,
            dropout=config.model.dropout,
        ).to(device)
        models.append(m)

    global_state = copy.deepcopy(models[0].state_dict())

    for rnd in range(config.training.num_rounds):
        # Distribute
        for m in models:
            m.load_state_dict(copy.deepcopy(global_state))

        # Local training with proximal term
        for cid in range(n_clients):
            cdata = clients_data[cid]
            weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
            criterion = FocalLoss(alpha=weights, gamma=2.0)
            optimizer = torch.optim.AdamW(models[cid].parameters(), lr=config.training.lr,
                                           weight_decay=config.training.weight_decay)
            loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

            # Save global params for proximal term (ALL params for FedProx)
            global_params = [p.detach().clone() for p in models[cid].parameters()]

            models[cid].train()
            for _ in range(config.training.local_epochs):
                for X_b, y_b in loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    loss = criterion(models[cid](X_b), y_b)
                    # Proximal term on ALL parameters
                    prox = torch.tensor(0.0, device=device)
                    for p_local, p_global in zip(models[cid].parameters(), global_params):
                        prox = prox + ((p_local - p_global) ** 2).sum()
                    loss = loss + (config.training.mu_prox / 2.0) * prox
                    loss.backward()
                    optimizer.step()

        # Aggregate (full model)
        n_samples = [len(c['y_train']) for c in clients_data]
        total = sum(n_samples)
        w = [n / total for n in n_samples]
        global_state = _fedavg_aggregate(models, w)

    # Evaluate
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

    models = []
    for _ in range(n_clients):
        m = FedPerProxModel(
            input_dim=config.model.input_dim,
            num_classes=config.model.num_classes,
            backbone_hidden=config.model.backbone_hidden,
            head_hidden=config.model.head_hidden,
            dropout=config.model.dropout,
        ).to(device)
        models.append(m)

    # Initial backbone state
    global_backbone = {k: v.clone() for k, v in models[0].state_dict().items()
                       if k.startswith('backbone.')}

    for rnd in range(config.training.num_rounds):
        # Distribute backbone only (heads stay local)
        for m in models:
            sd = m.state_dict()
            sd.update(copy.deepcopy(global_backbone))
            m.load_state_dict(sd)

        # Local training (no proximal term)
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

        # Aggregate backbone only with temperature-softened weights
        n_samples = [len(c['y_train']) for c in clients_data]
        tau = config.training.aggregation_temp
        w_raw = [n ** tau for n in n_samples]
        w_sum = sum(w_raw)
        w = [wr / w_sum for wr in w_raw]
        global_backbone = _fedper_aggregate(models, w)

    # Evaluate (each client uses its own head)
    all_metrics = []
    for cid in range(n_clients):
        m = evaluate_model(models[cid], clients_data[cid]['X_test'],
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

    models = []
    for _ in range(n_clients):
        m = FedPerProxModel(
            input_dim=config.model.input_dim,
            num_classes=config.model.num_classes,
            backbone_hidden=config.model.backbone_hidden,
            head_hidden=config.model.head_hidden,
            dropout=config.model.dropout,
        ).to(device)
        models.append(m)

    global_backbone = {k: v.clone() for k, v in models[0].state_dict().items()
                       if k.startswith('backbone.')}

    for rnd in range(config.training.num_rounds):
        # Distribute backbone only
        for m in models:
            sd = m.state_dict()
            sd.update(copy.deepcopy(global_backbone))
            m.load_state_dict(sd)

        # Local training with proximal term on backbone only
        for cid in range(n_clients):
            cdata = clients_data[cid]
            weights = compute_class_weights(cdata['y_train'], config.model.num_classes).to(device)
            criterion = FocalLoss(alpha=weights, gamma=2.0)
            optimizer = torch.optim.AdamW(models[cid].parameters(), lr=config.training.lr,
                                           weight_decay=config.training.weight_decay)
            loader = make_dataloader(cdata['X_train'], cdata['y_train'], config.training.batch_size)

            # Save global backbone for proximal term
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

        # Aggregate backbone with temperature-softened weights
        n_samples = [len(c['y_train']) for c in clients_data]
        tau = config.training.aggregation_temp
        w_raw = [n ** tau for n in n_samples]
        w_sum = sum(w_raw)
        w = [wr / w_sum for wr in w_raw]
        global_backbone = _fedper_aggregate(models, w)

    # Evaluate
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


def run_all_baselines(clients_data, config, device='cpu'):
    """Run all baselines and return comparison dict."""
    results = {}

    # ── Traditional ML baselines (centralized) ──
    print("\n" + "─" * 70)
    print("  TRADITIONAL ML BASELINES")
    print("─" * 70)
    t0 = time.time()
    # ml_results = run_traditional_ml_baselines(clients_data, config)
    # for method_name, method_metrics in ml_results.items():
    #     method_metrics['wall_time_s'] = time.time() - t0
    #     results[method_name] = method_metrics

    # ── Deep Learning baselines ──
    print("\n" + "─" * 70)
    print("  DEEP LEARNING & FEDERATED BASELINES")
    print("─" * 70)

    t0 = time.time()
    results['centralized_mlp'] = run_centralized(clients_data, config, device)
    results['centralized_mlp']['wall_time_s'] = time.time() - t0

    t0 = time.time()
    results['local_only'] = run_local_only(clients_data, config, device)
    results['local_only']['wall_time_s'] = time.time() - t0

    t0 = time.time()
    results['fedavg'] = run_fedavg(clients_data, config, device)
    results['fedavg']['wall_time_s'] = time.time() - t0

    t0 = time.time()
    results['fedprox'] = run_fedprox(clients_data, config, device)
    results['fedprox']['wall_time_s'] = time.time() - t0

    t0 = time.time()
    results['fedper'] = run_fedper(clients_data, config, device)
    results['fedper']['wall_time_s'] = time.time() - t0

    t0 = time.time()
    results['fedper_prox_xai'] = run_fedper_prox(clients_data, config, device)
    results['fedper_prox_xai']['wall_time_s'] = time.time() - t0

    return results


def generate_comparison_table_latex(all_results, out_path):
    """Generate LaTeX booktabs comparison table."""
    methods = [
        # Traditional ML
        ('random_forest', 'Random Forest'),
        ('xgboost', 'XGBoost'),
        ('decision_tree', 'Decision Tree'),
        ('knn', 'KNN'),
        ('logistic_regression', 'Logistic Regression'),
        # Deep Learning & Federated
        ('centralized_mlp', 'Centralized MLP'),
        ('local_only', 'Local-only MLP'),
        ('fedavg', 'FedAvg'),
        ('fedprox', 'FedProx'),
        ('fedper', 'FedPer'),
        ('fedper_prox_xai', '\\textbf{FedPer-Prox-XAI (Ours)}'),
    ]
    # Filter to only methods present in results
    is_multi_seed = isinstance(all_results, dict) and 'mean' in all_results
    if is_multi_seed:
        available = set(all_results['mean'].keys())
    else:
        available = set(all_results.keys()) - {'seed'}
    methods = [(k, v) for k, v in methods if k in available]
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
        'baseline methods on IoMT Non-IID data (mean $\\pm$ std across clients/seeds).}',
        '\\label{tab:sota_comparison}',
        '\\begin{tabular}{l' + 'c' * len(metrics) + '}',
        '\\toprule',
        'Method & ' + ' & '.join(m[1] for m in metrics) + ' \\\\',
        '\\midrule',
    ]

    # Check if we have multi-seed results
    is_multi_seed = isinstance(all_results, dict) and 'mean' in all_results

    # Traditional ML methods
    ml_keys = {'random_forest', 'xgboost',
                'decision_tree', 'knn', 'logistic_regression'}
    ml_methods = [(k, v) for k, v in methods if k in ml_keys]
    dl_methods = [(k, v) for k, v in methods if k not in ml_keys]

    if ml_methods:
        ncols = len(metrics)
        lines.append(f'\\multicolumn{{{ncols+1}}}{{l}}{{\\textit{{Traditional ML (centralized)}}}}'
                      ' \\\\')
        for key, label in ml_methods:
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
        lines.append('\\midrule')

    if dl_methods:
        ncols = len(metrics)
        lines.append(f'\\multicolumn{{{ncols+1}}}{{l}}{{\\textit{{Deep Learning \\& Federated}}}}'
                      ' \\\\')
        for key, label in dl_methods:
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
    """Generate CSV comparison table."""
    all_possible_methods = [
        'random_forest', 'xgboost',
        'decision_tree', 'knn', 'logistic_regression',
        'centralized_mlp', 'local_only', 'fedavg', 'fedprox', 'fedper', 'fedper_prox_xai',
    ]
    metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro',
               'recall_macro', 'cohen_kappa', 'mcc']

    is_multi_seed = isinstance(all_results, dict) and 'mean' in all_results
    if is_multi_seed:
        available = set(all_results['mean'].keys())
    else:
        available = set(all_results.keys()) - {'seed'}
    methods = [m for m in all_possible_methods if m in available]

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

def main():
    p = argparse.ArgumentParser(description='R3.6: SOTA baseline comparison')
    p.add_argument('--data-dir', type=str, default='data/client_datasets',
                   help='Client data directory (with client_0/, client_1/, ...)')
    p.add_argument('--seeds', type=int, nargs='+', default=[42],
                   help='Random seeds for reproducibility')
    p.add_argument('--n-clients', type=int, default=4)
    p.add_argument('--synthetic', action='store_true',
                   help='Use synthetic data (if real data unavailable)')
    p.add_argument('--out-dir', type=str, default='./results/sota_comparison')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = FLConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("  R3.6 — SOTA BASELINE COMPARISON")
    print("=" * 70)
    print(f"  Seeds     : {args.seeds}")
    print(f"  Clients   : {args.n_clients}")
    print(f"  Rounds    : {config.training.num_rounds}")
    print(f"  Epochs/R  : {config.training.local_epochs}")
    print(f"  Device    : {device}")
    print(f"  Data      : {'synthetic' if args.synthetic else args.data_dir}")

    # Load or generate data
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
            # Update input_dim from actual data
            config.model.input_dim = clients_data[0]['X_train'].shape[1]
            print(f"  Input dim: {config.model.input_dim}")

        results = run_all_baselines(clients_data, config, device)
        results['seed'] = seed
        all_seed_results.append(results)

        # Print summary for this seed
        print(f"\n  Summary (seed={seed}):")
        print(f"  {'Method':<25s} {'Accuracy':>10s} {'F1-Macro':>10s} {'MCC':>10s}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        for key in sorted(results.keys()):
            if key == 'seed':
                continue
            r = results[key]
            if isinstance(r, dict) and 'accuracy' in r:
                print(f"  {key:<25s} {r.get('accuracy',0):>10.4f} "
                      f"{r.get('f1_macro',0):>10.4f} {r.get('mcc',0):>10.4f}")

    # Aggregate across seeds
    if len(args.seeds) > 1:
        methods = [k for k in all_seed_results[0].keys() if k != 'seed']
        metrics_keys = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro',
                        'recall_macro', 'cohen_kappa', 'mcc']

        agg = {'mean': {}, 'std': {}}
        for method in methods:
            agg['mean'][method] = {}
            agg['std'][method] = {}
            for mk in metrics_keys:
                vals = [r[method].get(mk, 0) for r in all_seed_results]
                agg['mean'][method][mk] = float(np.mean(vals))
                agg['std'][method][mk] = float(np.std(vals))

        generate_comparison_table_latex(agg, out_dir / 'sota_comparison_table.tex')
        generate_comparison_csv(agg, out_dir / 'sota_comparison.csv')

        # Summary
        print(f"\n{'='*70}")
        print(f"  AGGREGATED RESULTS ({len(args.seeds)} seeds)")
        print(f"{'='*70}")
        print(f"  {'Method':<25s} {'Accuracy':>16s} {'F1-Macro':>16s} {'MCC':>16s}")
        print(f"  {'-'*25} {'-'*16} {'-'*16} {'-'*16}")
        for method in methods:
            acc = f"{agg['mean'][method]['accuracy']:.3f}±{agg['std'][method]['accuracy']:.3f}"
            f1 = f"{agg['mean'][method]['f1_macro']:.3f}±{agg['std'][method]['f1_macro']:.3f}"
            mcc = f"{agg['mean'][method]['mcc']:.3f}±{agg['std'][method]['mcc']:.3f}"
            print(f"  {method:<25s} {acc:>16s} {f1:>16s} {mcc:>16s}")
    else:
        generate_comparison_table_latex(all_seed_results[0], out_dir / 'sota_comparison_table.tex')
        generate_comparison_csv(all_seed_results[0], out_dir / 'sota_comparison.csv')

    # Save full JSON report
    # Clean up non-serializable fields
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

    with open(out_dir / 'sota_comparison_full.json', 'w') as f:
        json.dump(clean_for_json(all_seed_results), f, indent=2)

    print(f"\n  Saved to: {out_dir}/")
    print(f"    sota_comparison_full.json")
    print(f"    sota_comparison_table.tex")
    print(f"    sota_comparison.csv")


if __name__ == '__main__':
    main()
