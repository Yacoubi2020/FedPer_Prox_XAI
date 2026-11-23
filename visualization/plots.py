"""Publication-ready figures: training, SHAP, attack profiles, multi-network,
classification & representation-space metrics.

All labels are in English and styled for direct inclusion in a journal article
(IEEE / Elsevier). Colour palette is the colour-blind-safe Okabe-Ito set; fonts
use Liberation Serif (Times-compatible). Figures are exported at 300 DPI.
"""
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (headless Raspberry Pi)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Publication style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#2b2b2b',
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': '#d0d0d0',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.7,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '#cccccc',
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
})

# Okabe-Ito colour-blind-safe palette (per-client colours)
COLORS = ['#0072B2', '#009E73', '#E69F00', '#D55E00', '#CC79A7',
          '#56B4E9', '#F0E442', '#000000']
# Per-metric accent colours
C_PRECISION, C_RECALL, C_F1 = '#0072B2', '#009E73', '#E69F00'
C_RATIO, C_SIL = '#CC79A7', '#009E73'
NETWORK_COLORS = {'wifi': '#0072B2', '5g': '#009E73', '4g': '#E69F00', '3g': '#D55E00'}


def _finalize(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [saved] {save_path}")


# ── Loading ──────────────────────────────────────────────────────────────────

def _discover_client_ids(network, results_dir):
    """Liste les IDs de clients réellement présents (dossiers client_*), triés.

    Robuste aux IDs non contigus (0,2,3) et à un nombre quelconque de clients :
    on ne suppose plus une plage fixe 0..n_clients-1.
    """
    net_dir = Path(results_dir) / network
    if not net_dir.exists():
        return []
    ids = []
    for d in net_dir.glob('client_*'):
        if d.is_dir():
            try:
                ids.append(int(d.name.split('_')[1]))
            except (IndexError, ValueError):
                continue
    return sorted(ids)


def load_all_client_shap(network, results_dir, n_clients=None):
    results = {}
    for cid in _discover_client_ids(network, results_dir):
        path = Path(results_dir) / network / f'client_{cid}' / 'shap_results.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                results[cid] = pickle.load(f)
    return results


def load_training_history(network, results_dir, n_clients=None):
    history = {}
    for cid in _discover_client_ids(network, results_dir):
        path = Path(results_dir) / network / f'client_{cid}' / 'metrics.json'
        if path.exists():
            with open(path) as f:
                history[cid] = json.load(f)
    return history


def load_evaluation_metrics(network, results_dir, n_clients=None):
    metrics = {}
    for cid in _discover_client_ids(network, results_dir):
        path = Path(results_dir) / network / f'client_{cid}' / 'evaluation_metrics.json'
        if path.exists():
            with open(path) as f:
                metrics[cid] = json.load(f)
    return metrics


# ── Training / SHAP plots ─────────────────────────────────────────────────────

def plot_training_history(history, save_path):
    if not history:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Federated Training History')
    for cid in sorted(history.keys()):
        h = history[cid]
        rounds = list(range(1, len(h.get('train_acc', [])) + 1))
        color = COLORS[cid % len(COLORS)]
        kw = dict(marker='o', color=color, label=f'Client {cid}')
        if h.get('train_acc'):
            axes[0, 0].plot(rounds, h['train_acc'], **kw)
        if h.get('train_loss'):
            axes[0, 1].plot(rounds, h['train_loss'], **kw)
        if h.get('energy_j'):
            axes[1, 0].plot(rounds, h['energy_j'], **kw)
        if h.get('comm_mb'):
            axes[1, 1].plot(rounds, h['comm_mb'], **kw)
    titles = ['Training accuracy', 'Training loss',
              'Cumulative energy', 'Cumulative communication']
    ylabels = ['Accuracy', 'Loss', 'Energy (J)', 'Data transferred (MB)']
    for ax, t, y in zip(axes.flatten(), titles, ylabels):
        ax.set_title(t); ax.set_xlabel('Communication round'); ax.set_ylabel(y)
        ax.legend()
    _finalize(fig, save_path)


def plot_shap_per_client(shap_results, save_path):
    n = len(shap_results)
    if n == 0:
        return
    # R1.3: Enlarged figure for better readability (was 6.5*n x 8)
    n_cols = min(n, 2)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 10 * n_rows))
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    fig.suptitle('SHAP Feature Importance per Client', fontsize=18, fontweight='bold', y=1.01)
    for idx, (cid, r) in enumerate(sorted(shap_results.items())):
        ax = axes[idx]
        imp = np.asarray(r['mean_importance'])
        names = r['feature_names']
        top_k = min(15, len(imp))
        top_idx = np.argsort(imp)[::-1][:top_k]
        ax.barh(range(top_k), imp[top_idx][::-1],
                color=COLORS[cid % len(COLORS)], edgecolor='white', linewidth=0.4)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([names[i] for i in reversed(top_idx)], fontsize=12)
        ax.set_title(f'Client {cid} (n = {r["n_samples"]:,})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Mean |SHAP value|', fontsize=13)
        ax.tick_params(axis='x', labelsize=11)
        ax.grid(axis='x')
    # Hide empty subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    _finalize(fig, save_path)


def plot_shap_consensus(shap_results, save_path):
    if not shap_results:
        return
    first = next(iter(shap_results.values()))
    feature_names = first['feature_names']
    n_features = len(feature_names)
    clients_sorted = sorted(shap_results.keys())
    matrix = np.zeros((len(clients_sorted), n_features))
    for i, cid in enumerate(clients_sorted):
        imp = np.asarray(shap_results[cid]['mean_importance'])
        m = min(len(imp), n_features)
        norm = imp[:m] / (imp[:m].sum() + 1e-10)
        matrix[i, :m] = norm

    mean_imp = matrix.mean(axis=0)
    cv = matrix.std(axis=0) / (mean_imp + 1e-10)

    # R1.3: Enlarged figure (was 21 x 8)
    fig, axes = plt.subplots(1, 3, figsize=(26, 11))
    fig.suptitle('Cross-Client SHAP Consensus', fontsize=18, fontweight='bold', y=1.01)
    top_k = min(20, n_features)
    top_idx = np.argsort(mean_imp)[::-1][:top_k]
    sns.heatmap(matrix[:, top_idx].T,
                xticklabels=[f'C{c}' for c in clients_sorted],
                yticklabels=[feature_names[i] for i in top_idx],
                cmap='YlOrRd', ax=axes[0], annot=True, fmt='.2f',
                annot_kws={'size': 11},
                cbar_kws={'label': 'Normalized importance'})
    axes[0].set_title('Normalized importance (top 20)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Client', fontsize=13)
    axes[0].tick_params(axis='both', labelsize=11)

    sc = axes[1].scatter(mean_imp, cv, c=mean_imp, cmap='YlOrRd', s=90,
                         edgecolor='#444', linewidth=0.4)
    axes[1].set_xlabel('Mean importance', fontsize=13)
    axes[1].set_ylabel('Coefficient of variation', fontsize=13)
    axes[1].set_title('Consensus vs. divergence', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=11)
    fig.colorbar(sc, ax=axes[1], label='Mean importance')

    cons = np.argsort(mean_imp)[::-1][:15]
    axes[2].barh(range(15), mean_imp[cons][::-1], color=C_RECALL,
                 edgecolor='white', linewidth=0.4)
    axes[2].set_yticks(range(15))
    axes[2].set_yticklabels([feature_names[i] for i in reversed(cons)], fontsize=12)
    axes[2].set_title('Top consensus features', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Normalized importance', fontsize=13)
    axes[2].tick_params(axis='x', labelsize=11)
    axes[2].grid(axis='x')
    _finalize(fig, save_path)


def plot_global_importance(shap_results, save_path):
    if not shap_results:
        return
    first = next(iter(shap_results.values()))
    feature_names = first['feature_names']
    n_features = len(feature_names)
    total = np.zeros(n_features); total_w = 0
    for r in shap_results.values():
        imp = np.asarray(r['mean_importance'])
        m = min(len(imp), n_features)
        w = r['n_samples']
        total[:m] += imp[:m] * w
        total_w += w
    global_imp = total / total_w
    top_k = min(25, n_features)
    top_idx = np.argsort(global_imp)[::-1][:top_k]
    # R1.3: Enlarged figure (was 9 x 9)
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.barh(range(top_k), global_imp[top_idx][::-1], color=COLORS[0],
            edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in reversed(top_idx)], fontsize=12)
    ax.set_title('Top-25 Global Features (FL-Aggregated SHAP)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Global SHAP importance (weighted by sample count)', fontsize=13)
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(axis='x')
    _finalize(fig, save_path)


def plot_attack_profiles(shap_results, save_path):
    if not shap_results:
        return
    first = next(iter(shap_results.values()))
    feature_names = first['feature_names']
    n_features = len(feature_names)
    attack_imp = {}
    for r in shap_results.values():
        for attack, imp_arr in r['class_importance'].items():
            arr = np.asarray(imp_arr)
            if attack not in attack_imp:
                attack_imp[attack] = np.zeros(n_features)
            m = min(len(arr), n_features)
            attack_imp[attack][:m] += arr[:m]
    n_attacks = len(attack_imp)
    if n_attacks == 0:
        return
    # R1.3: Enlarged figure (was 6*n_cols x 4*n_rows)
    n_cols = 3
    n_rows = (n_attacks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle('SHAP Profiles per Attack Class', fontsize=18, fontweight='bold', y=1.01)
    for idx, (attack, imp) in enumerate(sorted(attack_imp.items())):
        ax = axes[idx]
        top_idx = np.argsort(imp)[::-1][:10]
        top_k = len(top_idx)
        ax.barh(range(top_k), imp[top_idx][::-1], color=COLORS[3],
                edgecolor='white', linewidth=0.4)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([feature_names[i] for i in reversed(top_idx)], fontsize=11)
        ax.set_title(attack, fontsize=14, fontweight='bold')
        ax.set_xlabel('SHAP importance', fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
        ax.grid(axis='x')
    for j in range(n_attacks, len(axes)):
        axes[j].set_visible(False)
    _finalize(fig, save_path)


def plot_network_comparison(network_results, save_path):
    """Compare energy / communication / accuracy / time across network profiles."""
    if not network_results:
        return
    networks = list(network_results.keys())
    metrics = {'energy_j': [], 'comm_mb': [], 'mean_acc': [], 'time_s': []}
    for net in networks:
        clients = network_results[net]
        metrics['energy_j'].append(sum(c.get('energy_j', 0) for c in clients.values()))
        metrics['comm_mb'].append(sum(c.get('comm_mb', 0) for c in clients.values()))
        accs = [c.get('final_acc', 0) for c in clients.values()]
        metrics['mean_acc'].append(float(np.mean(accs)) if accs else 0)
        metrics['time_s'].append(sum(c.get('total_time_s', 0) for c in clients.values()))

    labels = [n.upper() for n in networks]
    colors_net = [NETWORK_COLORS.get(n, COLORS[0]) for n in networks]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle('Multi-Network Comparison (WiFi / 5G / 4G / 3G)')
    specs = [('energy_j', 'Total energy', 'Energy (J)', None),
             ('comm_mb', 'Total communication', 'Data transferred (MB)', None),
             ('mean_acc', 'Mean accuracy', 'Accuracy', (0, 1)),
             ('time_s', 'Total wall-clock time', 'Time (s)', None)]
    for ax, (key, title, ylab, ylim) in zip(axes, specs):
        ax.bar(labels, metrics[key], color=colors_net, edgecolor='white', linewidth=0.6)
        ax.set_title(title); ax.set_ylabel(ylab); ax.grid(axis='y')
        if ylim:
            ax.set_ylim(*ylim)
    _finalize(fig, save_path)


# ── Classification & representation metrics ────────────────────────────────────

def _present_classes(clf):
    """Indices + names of classes actually present in the test set (support > 0)."""
    names = clf.get('class_names', [])
    per_class = clf.get('per_class', {})
    present = [i for i, n in enumerate(names)
               if per_class.get(n, {}).get('support', 0) > 0]
    return present, [names[i] for i in present]


def _cm_active_classes(clf):
    """Classes to display in a confusion matrix: union of classes that appear as a
    TRUE label (row sum > 0) or as a PREDICTED label (column sum > 0). This guarantees
    misclassifications are never hidden, even when a client owns a class subset (Non-IID).
    """
    cm = np.array(clf['confusion_matrix'], dtype=float)
    names = clf.get('class_names', [])
    # pad names so every matrix index has a label
    while len(names) < cm.shape[0]:
        names.append(f'class_{len(names)}')
    row_active = cm.sum(axis=1) > 0          # classes présentes en vérité terrain
    col_active = cm.sum(axis=0) > 0          # classes vers lesquelles le modèle prédit
    active = np.where(row_active | col_active)[0].tolist()
    return active, [names[i] for i in active]


def plot_confusion_matrices(eval_metrics, save_path):
    """Row-normalized confusion matrices (inter-class) — one per client."""
    if not eval_metrics:
        return
    cids = sorted(eval_metrics.keys())
    n = len(cids)
    ncols = n if n <= 3 else 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 5.8 * nrows))
    fig.suptitle('Normalized Confusion Matrices (Inter-Class)')
    axes = np.atleast_1d(axes).flatten()
    for ax, cid in zip(axes, cids):
        clf = eval_metrics[cid]['classification']
        active, names = _cm_active_classes(clf)
        cm = np.array(clf['confusion_matrix'], dtype=float)[np.ix_(active, active)]
        # re-normalisation par ligne sur le sous-ensemble affiché
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
                    xticklabels=names, yticklabels=names, ax=ax, square=True,
                    linewidths=0.5, linecolor='white',
                    cbar_kws={'label': 'Proportion'})
        ax.set_title(f'Client {cid}  (Acc = {clf["accuracy"]:.3f}, '
                     f'Macro-F1 = {clf["f1_macro"]:.3f})')
        ax.set_xlabel('Predicted label'); ax.set_ylabel('True label')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
    for ax in axes[n:]:
        ax.axis('off')
    _finalize(fig, save_path)


def plot_per_class_metrics(eval_metrics, save_path):
    """Precision / Recall / F1 per class (intra-class) — one panel per client."""
    if not eval_metrics:
        return
    cids = sorted(eval_metrics.keys())
    fig, axes = plt.subplots(len(cids), 1, figsize=(11, 3.1 * len(cids)))
    fig.suptitle('Per-Class Performance (Intra-Class)')
    axes = np.atleast_1d(axes)
    for ax, cid in zip(axes, cids):
        clf = eval_metrics[cid]['classification']
        _, names = _present_classes(clf)
        prec = [clf['per_class'][n]['precision'] for n in names]
        rec = [clf['per_class'][n]['recall'] for n in names]
        f1 = [clf['per_class'][n]['f1'] for n in names]
        x = np.arange(len(names)); w = 0.27
        ax.bar(x - w, prec, w, label='Precision', color=C_PRECISION,
               edgecolor='white', linewidth=0.5)
        ax.bar(x, rec, w, label='Recall', color=C_RECALL,
               edgecolor='white', linewidth=0.5)
        ax.bar(x + w, f1, w, label='F1-score', color=C_F1,
               edgecolor='white', linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
        ax.set_title(f'Client {cid}')
        ax.legend(loc='lower right', ncol=3)
        ax.grid(axis='y')
    _finalize(fig, save_path)


def plot_separability(eval_metrics, save_path):
    """Representation-space separability: intra-class compactness vs. global ratio."""
    if not eval_metrics:
        return
    cids = sorted(eval_metrics.keys())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Representation-Space Separability (Backbone Embeddings)')

    # (1) Intra-class compactness, aligned by class name (Non-IID -> different subsets)
    ax = axes[0]
    all_names = []
    for cid in cids:
        for nm in eval_metrics[cid]['representation']['intra_class']['per_class']:
            if nm not in all_names:
                all_names.append(nm)
    xpos = np.arange(len(all_names))
    for cid in cids:
        per = eval_metrics[cid]['representation']['intra_class']['per_class']
        vals = [per[nm]['mean_dist_to_centroid'] if nm in per else np.nan
                for nm in all_names]
        ax.plot(xpos, vals, marker='o', label=f'Client {cid}',
                color=COLORS[cid % len(COLORS)])
    ax.set_xticks(xpos); ax.set_xticklabels(all_names, rotation=30, ha='right')
    ax.set_ylabel('Mean distance to class centroid')
    ax.set_title('Intra-class compactness (lower is better)')
    ax.legend()

    # (2) Fisher separability ratio + silhouette per client
    ax = axes[1]
    x = np.arange(len(cids)); w = 0.35
    ratios = [eval_metrics[c]['representation']['separability_ratio'] for c in cids]
    sils = [eval_metrics[c]['representation'].get('silhouette_score', 0) for c in cids]
    b1 = ax.bar(x - w / 2, ratios, w, label='Inter/intra ratio (Fisher)',
                color=C_RATIO, edgecolor='white', linewidth=0.5)
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w / 2, sils, w, label='Silhouette',
                 color=C_SIL, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f'C{c}' for c in cids])
    ax.set_xlabel('Client')
    ax.set_ylabel('Inter/intra separability ratio (higher is better)', color=C_RATIO)
    ax2.set_ylabel('Silhouette score $\\in[0,1]$ (higher is better)', color=C_SIL)
    ax2.set_ylim(0, 1)
    ax.set_ylim(0, max(ratios) * 1.15 if ratios else 1)
    ax.tick_params(axis='y', colors=C_RATIO)
    ax2.tick_params(axis='y', colors=C_SIL)
    ax2.grid(False)
    ax.set_title('Global separability per client')
    ax.legend(handles=[b1, b2], labels=['Inter/intra ratio (Fisher)', 'Silhouette'],
              loc='upper left')
    _finalize(fig, save_path)


def plot_metrics_summary(eval_metrics, save_path):
    """Summary dashboard of bounded [0,1] evaluation metrics per client."""
    if not eval_metrics:
        return
    cids = sorted(eval_metrics.keys())
    keys = [
        ('accuracy', 'Accuracy', 'classification'),
        ('f1_macro', 'Macro-F1', 'classification'),
        ('f1_weighted', 'Weighted-F1', 'classification'),
        ('mcc', 'MCC', 'classification'),
        ('balanced_accuracy', 'Balanced acc.', 'classification'),
        ('silhouette_score', 'Silhouette', 'representation'),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(keys)); w = 0.8 / max(len(cids), 1)
    for i, cid in enumerate(cids):
        vals = [eval_metrics[cid][grp].get(k, 0) or 0 for k, _, grp in keys]
        bars = ax.bar(x + i * w, vals, w, label=f'Client {cid}',
                      color=COLORS[cid % len(COLORS)], edgecolor='white', linewidth=0.5)
        ax.bar_label(bars, fmt='%.2f', fontsize=6, padding=1)
    ax.set_xticks(x + w * (len(cids) - 1) / 2)
    ax.set_xticklabels([lbl for _, lbl, _ in keys])
    ax.set_ylim(0, 1.08)
    ax.set_title('Per-Client Evaluation Metrics Summary')
    ax.set_ylabel('Score')
    ax.legend(ncol=min(len(cids), 4))
    ax.grid(axis='y')
    _finalize(fig, save_path)


# ── Embedding projection (t-SNE / UMAP) ────────────────────────────────────────

def load_embedding_samples(network, results_dir):
    """Load per-client embedding samples saved during evaluation (embeddings_sample.npz)."""
    samples = {}
    for cid in _discover_client_ids(network, results_dir):
        path = Path(results_dir) / network / f'client_{cid}' / 'embeddings_sample.npz'
        if path.exists():
            data = np.load(path, allow_pickle=True)
            samples[cid] = {
                'embeddings': data['embeddings'],
                'labels': data['labels'],
                'class_names': list(data['class_names']),
            }
    return samples


def _project_2d(X, method='tsne', seed=42, perplexity=30):
    """Reduce X (N x D) to 2D via t-SNE or UMAP. Falls back to t-SNE if UMAP missing."""
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    if method == 'umap':
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(n_components=2, random_state=seed,
                                n_neighbors=min(15, max(2, n - 1)))
            return reducer.fit_transform(X), 'UMAP'
        except Exception:
            print("  [info] umap-learn unavailable -> falling back to t-SNE")
    from sklearn.manifold import TSNE
    perp = float(min(perplexity, max(5, (n - 1) // 3)))
    reducer = TSNE(n_components=2, random_state=seed, perplexity=perp,
                   init='pca', learning_rate='auto')
    return reducer.fit_transform(X), 't-SNE'


def _scatter_by_class(ax, emb2d, labels, all_class_names, title):
    classes = sorted(np.unique(labels).tolist())
    for c in classes:
        m = labels == c
        name = all_class_names[c] if c < len(all_class_names) else f'class_{c}'
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=12, alpha=0.7,
                   color=COLORS[c % len(COLORS)], label=name,
                   edgecolor='none', rasterized=True)
    ax.set_title(title)
    ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


def _scatter_by_client(ax, emb2d, client_ids, title):
    for cid in sorted(np.unique(client_ids).tolist()):
        m = client_ids == cid
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=12, alpha=0.6,
                   color=COLORS[cid % len(COLORS)], label=f'Client {cid}',
                   edgecolor='none', rasterized=True)
    ax.set_title(title)
    ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


def plot_embedding_projection(samples, save_path, method='tsne',
                              seed=42, perplexity=30, max_aggregate=4000):
    """2D projection of backbone embeddings, coloured by class.

    Produces one panel per client (each client's own latent space) plus a final
    aggregated panel pooling all clients — qualitative evidence that the shared
    backbone yields class-separable representations under Non-IID partitioning.
    """
    if not samples:
        return
    cids = sorted(samples.keys())
    # union of class names (use the longest list as reference)
    all_class_names = max((samples[c]['class_names'] for c in cids), key=len)

    n_panels = len(cids) + (1 if len(cids) > 1 else 0)
    ncols = min(n_panels, 3)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    method_label = None

    for ax, cid in zip(axes, cids):
        emb2d, method_label = _project_2d(samples[cid]['embeddings'], method, seed, perplexity)
        _scatter_by_class(ax, emb2d, samples[cid]['labels'], all_class_names,
                          f'Client {cid}')

    # aggregated panel
    if len(cids) > 1:
        Xs = np.concatenate([samples[c]['embeddings'] for c in cids])
        ys = np.concatenate([samples[c]['labels'] for c in cids])
        if len(Xs) > max_aggregate:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(Xs), size=max_aggregate, replace=False)
            Xs, ys = Xs[idx], ys[idx]
        emb2d, method_label = _project_2d(Xs, method, seed, perplexity)
        _scatter_by_class(axes[len(cids)], emb2d, ys, all_class_names,
                          'All clients (aggregated)')

    for ax in axes[n_panels:]:
        ax.axis('off')

    # single shared legend (de-duplicated)
    handles, labels = [], []
    for ax in axes[:n_panels]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 6),
               bbox_to_anchor=(0.5, -0.02), markerscale=1.6, frameon=True)
    fig.suptitle(f'Backbone Embedding Projection ({method_label}, coloured by class)')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [saved] {save_path}")


def plot_embedding_client_invariance(samples, save_path, method='tsne',
                                     seed=42, perplexity=30, max_points=5000):
    """Client-invariance view: ONE aggregated 2D projection of pooled backbone
    embeddings, shown twice on the SAME coordinates — left coloured by class,
    right coloured by client.

    Reading (key argument under label/quantity skew):
      - classes separate (left) AND clients intermix (right)  -> backbone learned
        client-invariant, attack-discriminative features (desired for FedPer + FedProx);
      - clients form isolated blobs (right)                   -> backbone overfits to
        client identity (undesirable).
    """
    if not samples:
        return
    cids = sorted(samples.keys())
    if len(cids) < 2:
        return  # invariance only meaningful with >= 2 clients
    all_class_names = max((samples[c]['class_names'] for c in cids), key=len)

    X = np.concatenate([samples[c]['embeddings'] for c in cids])
    y = np.concatenate([samples[c]['labels'] for c in cids])
    who = np.concatenate([np.full(len(samples[c]['labels']), c) for c in cids])

    if len(X) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_points, replace=False)
        X, y, who = X[idx], y[idx], who[idx]

    # projection calculée UNE seule fois -> comparaison équitable des deux colorations
    emb2d, method_label = _project_2d(X, method, seed, perplexity)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(f'Backbone Embedding — Client Invariance ({method_label})')
    _scatter_by_class(axes[0], emb2d, y, all_class_names, 'Aggregated — coloured by class')
    _scatter_by_client(axes[1], emb2d, who, 'Aggregated — coloured by client')
    axes[0].legend(loc='best', markerscale=1.6, ncol=2, fontsize=8)
    axes[1].legend(loc='best', markerscale=1.6, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [saved] {save_path}")
