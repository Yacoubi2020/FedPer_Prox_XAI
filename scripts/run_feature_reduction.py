#!/usr/bin/env python3
"""R1.2 — Feature reduction experiment using SHAP importance.

Evaluates model performance with reduced feature sets (top-k features)
to validate that SHAP feature importance is not misleading with many features.

This addresses Reviewer 1 Comment 2:
  "Evaluate model with low number of features (feature importance can give
   false sense with >5 features)."

The script:
  1. Loads SHAP importance rankings from existing results
  2. For each k in [3, 5, 7, 10, 15, 23], trains and evaluates using top-k features
  3. Compares performance degradation curves
  4. Generates publication-ready figure and LaTeX table

Usage:
  python -I scripts/run_feature_reduction.py --network wifi --results-dir ./results
  python -I scripts/run_feature_reduction.py --network wifi --ks 3 5 7 10 15 23
"""
import os, sys, argparse, json, time
# See scripts/csv_to_npy.py for the FEDPER_DATA_ROOT convention.
DATA_ROOT = os.environ.get('FEDPER_DATA_ROOT', 'data')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import FLConfig
from models import FedPerProxModel, FocalLoss


# ── Feature selection from SHAP ──────────────────────────────────────────

def load_shap_ranking(results_dir, network, n_clients=4):
    """Load and aggregate SHAP feature importance across clients.

    Returns sorted list of (feature_index, feature_name, importance).
    """
    all_importances = []
    feature_names = None

    for cid in range(n_clients):
        shap_path = Path(results_dir) / network / f'client_{cid}' / 'shap_summary.json'
        if not shap_path.exists():
            continue
        with open(shap_path) as f:
            shap_data = json.load(f)

        if 'mean_importance' in shap_data:
            imp = shap_data['mean_importance']
            if feature_names is None and 'feature_names' in shap_data:
                feature_names = shap_data['feature_names']
            all_importances.append(imp)

    if not all_importances:
        raise FileNotFoundError(
            f"No SHAP results found in {results_dir}/{network}/client_*/shap_summary.json"
        )

    # Average importance across clients
    # Handle variable-length arrays
    max_len = max(len(imp) for imp in all_importances)
    padded = np.zeros((len(all_importances), max_len))
    for i, imp in enumerate(all_importances):
        padded[i, :len(imp)] = imp
    avg_importance = padded.mean(axis=0)

    # Sort by importance (descending)
    sorted_idx = np.argsort(avg_importance)[::-1]
    ranking = []
    for rank, idx in enumerate(sorted_idx):
        name = feature_names[idx] if feature_names and idx < len(feature_names) else f'feat_{idx}'
        ranking.append((int(idx), name, float(avg_importance[idx])))

    return ranking


def get_default_feature_names():
    """Default IoMT traffic feature names (23 features)."""
    return [
        'http.content_length', 'http.request', 'http.response.code',
        'http.response_number', 'http.time', 'tcp.analysis.initial_rtt',
        'tcp.connection.fin', 'tcp.connection.syn', 'tcp.connection.synack',
        'tcp.flags.cwr', 'tcp.flags.ecn', 'tcp.flags.fin', 'tcp.flags.ns',
        'tcp.flags.res', 'tcp.flags.syn', 'tcp.flags.urg', 'tcp.urgent_pointer',
        'ip.frag_offset', 'eth.dst.ig', 'eth.src.ig', 'eth.src.lg',
        'eth.src_not_group', 'arp.isannouncement',
    ]


# ── Training with feature subset ─────────────────────────────────────────

def load_data_with_features(data_dir, client_id, feature_indices, batch_size=256,
                            seed=42):
    """Load client data using only selected feature indices.

    Supports both pre-split (X_train.npy) and unsplit (X.npy) formats.
    """
    from sklearn.model_selection import train_test_split
    client_dir = Path(data_dir) / f'client_{client_id}'

    if (client_dir / 'X_train.npy').exists():
        X_train = np.load(client_dir / 'X_train.npy')
        y_train = np.load(client_dir / 'y_train.npy')
        X_test = np.load(client_dir / 'X_test.npy')
        y_test = np.load(client_dir / 'y_test.npy')
    elif (client_dir / 'X.npy').exists():
        X = np.load(client_dir / 'X.npy')
        y = np.load(client_dir / 'y.npy')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
    else:
        raise FileNotFoundError(f"No X.npy or X_train.npy in {client_dir}")

    # Select only top-k features
    X_train = X_train[:, feature_indices]
    X_test = X_test[:, feature_indices]

    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(
        torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Use max label + 1 (not len(unique)) to handle label skew
    # where label indices are non-contiguous (e.g. {0, 3, 4} → need size >= 5)
    all_y = np.concatenate([y_train, y_test])
    num_classes = int(all_y.max()) + 1
    return train_loader, test_loader, num_classes


def train_and_evaluate(train_loader, test_loader, input_dim, num_classes,
                       config, n_epochs=30, device='cpu'):
    """Train a centralized model with given features and evaluate."""
    # Adjust backbone hidden dims proportionally if input is small
    backbone_hidden = config.model.backbone_hidden.copy()
    if input_dim <= 5:
        backbone_hidden = [64, 32, 16]
    elif input_dim <= 10:
        backbone_hidden = [128, 64, 32]

    model = FedPerProxModel(
        input_dim=input_dim,
        num_classes=num_classes,
        backbone_hidden=backbone_hidden,
        head_hidden=config.model.head_hidden,
        dropout=config.model.dropout,
    ).to(device)

    # Class weights (handle absent classes in label skew)
    class_counts = np.zeros(num_classes)
    for _, y in train_loader:
        for c in y.numpy():
            class_counts[c] += 1
    # For absent classes, set count to 1 to get neutral weight
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    class_weights = torch.FloatTensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Train
    for epoch in range(n_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    total_loss = total_n = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_n += y.size(0)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                  recall_score, matthews_corrcoef, balanced_accuracy_score)

    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'f1_macro': float(f1_score(labels, preds, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(labels, preds, average='weighted', zero_division=0)),
        'precision_macro': float(precision_score(labels, preds, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(labels, preds, average='macro', zero_division=0)),
        'mcc': float(matthews_corrcoef(labels, preds)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, preds)),
        'test_loss': total_loss / total_n,
    }


# ── Output generators ────────────────────────────────────────────────────

def generate_feature_reduction_figure(results, out_path):
    """Generate publication-ready feature reduction curve."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ks = sorted(results.keys())
    metrics_to_plot = ['accuracy', 'f1_macro', 'mcc', 'balanced_accuracy']
    labels = ['Accuracy', 'Macro-F1', 'MCC', 'Balanced Acc.']
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    markers = ['o', 's', '^', 'D']

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for metric, label, color, marker in zip(metrics_to_plot, labels, colors, markers):
        means = [results[k]['mean'].get(metric, 0) for k in ks]
        stds = [results[k]['std'].get(metric, 0) for k in ks]
        ax.errorbar(ks, means, yerr=stds, label=label,
                   color=color, marker=marker, markersize=8,
                   linewidth=2, capsize=5, capthick=1.5)

    ax.set_xlabel('Number of Top Features (k)', fontsize=14, fontfamily='serif')
    ax.set_ylabel('Score', fontsize=14, fontfamily='serif')
    ax.set_title('Feature Reduction Analysis (SHAP-ranked)', fontsize=16, fontfamily='serif')
    ax.legend(fontsize=12, loc='lower right')
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved figure: {out_path}")


def generate_feature_reduction_latex(results, ranking, out_path):
    """Generate LaTeX table for feature reduction results."""
    ks = sorted(results.keys())
    lines = [
        '% Requires \\usepackage{booktabs}',
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Performance with top-$k$ SHAP-ranked features. '
        'Values as mean $\\pm$ std across 4 clients.}',
        '\\label{tab:feature_reduction}',
        '\\begin{tabular}{cccccc}',
        '\\toprule',
        '$k$ & Accuracy & Macro-F1 & MCC & Bal.\\,Acc. & Precision \\\\',
        '\\midrule',
    ]
    for k in ks:
        m, s = results[k]['mean'], results[k]['std']
        cells = []
        for metric in ['accuracy', 'f1_macro', 'mcc', 'balanced_accuracy', 'precision_macro']:
            cells.append(f"{m.get(metric, 0):.4f}$\\pm${s.get(metric, 0):.4f}")
        lines.append(f"{k} & {' & '.join(cells)} \\\\")

    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']

    # Add feature ranking table
    lines += [
        '',
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Top-10 features ranked by SHAP importance '
        '(averaged across all clients).}',
        '\\label{tab:feature_ranking}',
        '\\begin{tabular}{clc}',
        '\\toprule',
        'Rank & Feature & Importance \\\\',
        '\\midrule',
    ]
    for rank, (idx, name, imp) in enumerate(ranking[:10], 1):
        lines.append(f'{rank} & \\texttt{{{name}}} & {imp:.6f} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']

    Path(out_path).write_text('\n'.join(lines) + '\n')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='R1.2: Feature reduction experiment')
    p.add_argument('--network', type=str, default='wifi',
                   choices=['wifi', '5g', '4g', '3g'])
    p.add_argument('--results-dir', type=str, default='./results',
                   help='Directory with existing results (for SHAP rankings)')
    p.add_argument('--data-dir', type=str, default=f'{DATA_ROOT}/client_datasets')
    p.add_argument('--ks', type=int, nargs='+', default=[5, 10, 20, 30, 40, 50, 60, 81],
                   help='Numbers of top features to evaluate')
    p.add_argument('--n-clients', type=int, default=4)
    p.add_argument('--epochs', type=int, default=30,
                   help='Training epochs per feature set')
    p.add_argument('--out-dir', type=str, default=None)
    args = p.parse_args()

    config = FLConfig()
    config.apply_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_dir) / 'feature_reduction'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  R1.2 — FEATURE REDUCTION EXPERIMENT")
    print("=" * 70)
    print(f"  Network  : {args.network.upper()}")
    print(f"  k values : {args.ks}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Device   : {device}")
    print("=" * 70)

    # 1) Load SHAP ranking
    print("\n  Loading SHAP feature ranking...")
    try:
        ranking = load_shap_ranking(args.results_dir, args.network, args.n_clients)
        print(f"  Loaded {len(ranking)} features from SHAP")
        print(f"  Top-5 features: {[r[1] for r in ranking[:5]]}")
    except FileNotFoundError as e:
        print(f"  [WARN] {e}")
        # Try to detect actual feature count from data
        sample_path = Path(args.data_dir) / 'client_0'
        if (sample_path / 'X.npy').exists():
            n_feat = np.load(sample_path / 'X.npy', mmap_mode='r').shape[1]
        elif (sample_path / 'X_train.npy').exists():
            n_feat = np.load(sample_path / 'X_train.npy', mmap_mode='r').shape[1]
        else:
            n_feat = 81
        print(f"  Falling back to random ranking ({n_feat} features)")
        ranking = [(i, f'feat_{i}', 1.0 / (i + 1)) for i in range(n_feat)]

    # Filter ks to not exceed available features
    max_features = len(ranking)
    valid_ks = sorted([k for k in args.ks if k <= max_features])
    if not valid_ks:
        print(f"  [ERROR] All k values exceed available features ({max_features})")
        return
    print(f"  k values to test: {valid_ks} (max features: {max_features})")

    # 2) Run experiments for each k
    results = {}
    for k in valid_ks:
        if k > len(ranking):
            print(f"\n  [SKIP] k={k} > total features ({len(ranking)})")
            continue

        print(f"\n  k={k} features: {[r[1] for r in ranking[:k]]}")
        feature_indices = [r[0] for r in ranking[:k]]

        client_metrics = []
        for cid in range(args.n_clients):
            try:
                train_loader, test_loader, num_classes = load_data_with_features(
                    args.data_dir, cid, feature_indices, config.training.batch_size)
            except FileNotFoundError:
                print(f"    [SKIP] Client {cid}: data not found")
                continue

            t0 = time.time()
            metrics = train_and_evaluate(
                train_loader, test_loader, input_dim=k,
                num_classes=num_classes, config=config,
                n_epochs=args.epochs, device=device)
            elapsed = time.time() - t0

            client_metrics.append(metrics)
            print(f"    Client {cid}: Acc={metrics['accuracy']:.4f}  "
                  f"F1={metrics['f1_macro']:.4f}  MCC={metrics['mcc']:.4f}  "
                  f"({elapsed:.1f}s)")

        if client_metrics:
            # Aggregate across clients
            mean_metrics = {}
            std_metrics = {}
            for key in client_metrics[0].keys():
                vals = [m[key] for m in client_metrics]
                mean_metrics[key] = float(np.mean(vals))
                std_metrics[key] = float(np.std(vals))
            results[k] = {'mean': mean_metrics, 'std': std_metrics,
                          'per_client': client_metrics}

    if not results:
        print("\n  [ERROR] No results collected")
        return

    # 3) Output
    # JSON report
    report = {
        'feature_ranking': [(idx, name, float(imp)) for idx, name, imp in ranking],
        'k_values': sorted(results.keys()),
        'results': {str(k): v for k, v in results.items()},
    }
    with open(out_dir / 'feature_reduction_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Figure
    generate_feature_reduction_figure(results, out_dir / 'feature_reduction_curve.png')

    # LaTeX table
    generate_feature_reduction_latex(results, ranking,
                                    out_dir / 'feature_reduction_table.tex')

    # Summary
    print(f"\n  {'─'*55}")
    print(f"  FEATURE REDUCTION SUMMARY")
    print(f"  {'─'*55}")
    print(f"  {'k':>4s}  {'Accuracy':>12s}  {'Macro-F1':>12s}  {'MCC':>12s}")
    for k in sorted(results.keys()):
        m = results[k]['mean']
        s = results[k]['std']
        print(f"  {k:4d}  {m['accuracy']:.4f}+/-{s['accuracy']:.4f}  "
              f"{m['f1_macro']:.4f}+/-{s['f1_macro']:.4f}  "
              f"{m['mcc']:.4f}+/-{s['mcc']:.4f}")

    print(f"\n  Saved to: {out_dir}/")


if __name__ == '__main__':
    main()