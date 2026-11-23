# #!/usr/bin/env python3
# """R1.1 — Multi-seed experiment runner.

# Runs the full FedPer-Prox-XAI pipeline with multiple random seeds
# (default: 0, 1, 2, 3, 42) and aggregates results as mean +/- std.

# This addresses Reviewer 1 Comment 1:
#   "Run experiment across 4 different seeds besides 42 for range of performance."

# The script:
#   1. For each seed, sets all random seeds (torch, numpy, python)
#   2. Runs the full FL pipeline (server + 4 clients) locally
#   3. Collects per-seed evaluation metrics
#   4. Aggregates and outputs mean +/- std for all key metrics
#   5. Generates a LaTeX table and CSV summary

# Usage:
#   python -I scripts/run_multi_seed.py --seeds 0 1 2 3 42 --rounds 30 --network wifi
#   python -I scripts/run_multi_seed.py --all-networks --rounds 30
# """
# import os, sys, argparse, json, time, subprocess, signal
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from pathlib import Path
# from collections import defaultdict

# import numpy as np


# # ── Reproducibility ────────────────────────────────────────────────────────

# def set_all_seeds(seed: int):
#     """Set all random seeds for full reproducibility."""
#     import random
#     import torch
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ['PYTHONHASHSEED'] = str(seed)


# # ── Single-seed pipeline runner ──────────────────────────────────────────

# def run_single_seed(seed, network, rounds, local_epochs, n_clients,
#                     server_port, base_results_dir, data_dir, mu, temp):
#     """Run one complete FL experiment (server + clients) with a given seed."""
#     results_dir = str(Path(base_results_dir) / f'seed_{seed}')
#     os.makedirs(results_dir, exist_ok=True)

#     project_root = str(Path(__file__).resolve().parent.parent)
#     env = os.environ.copy()
#     env['PYTHONPATH'] = project_root
#     env['FL_SEED'] = str(seed)

#     # 1) Start server
#     server_cmd = [
#         sys.executable, '-I',
#         os.path.join(project_root, 'scripts', 'run_server.py'),
#         '--rounds', str(rounds),
#         '--network', network,
#         '--address', f'0.0.0.0:{server_port}',
#         '--min-clients', str(n_clients),
#         '--local-epochs', str(local_epochs),
#         '--mu', str(mu),
#         '--temp', str(temp),
#         '--results-dir', results_dir,
#     ]
#     log_dir = Path(results_dir) / 'logs'
#     log_dir.mkdir(parents=True, exist_ok=True)

#     with open(log_dir / 'server.log', 'w') as server_log:
#         server_proc = subprocess.Popen(
#             server_cmd, env=env, stdout=server_log, stderr=subprocess.STDOUT,
#             preexec_fn=os.setsid,
#         )
#     time.sleep(8)  # Wait for server startup

#     if server_proc.poll() is not None:
#         print(f"    [ERROR] Server crashed for seed {seed}")
#         return None

#     # 2) Start clients
#     client_procs = []
#     for cid in range(n_clients):
#         client_cmd = [
#             sys.executable, '-I',
#             os.path.join(project_root, 'scripts', 'run_client.py'),
#             '--client-id', str(cid),
#             '--server-addr', f'127.0.0.1:{server_port}',
#             '--network', network,
#             '--results-dir', results_dir,
#             '--data-dir', data_dir,
#         ]
#         client_env = env.copy()
#         client_env['FL_SEED'] = str(seed)
#         client_env['USE_FOCAL_LOSS'] = '0'
#         with open(log_dir / f'client_{cid}.log', 'w') as cl:
#             p = subprocess.Popen(
#                 client_cmd, env=client_env, stdout=cl, stderr=subprocess.STDOUT,
#             )
#             client_procs.append(p)

#     # 3) Wait for completion
#     server_proc.wait()
#     for p in client_procs:
#         p.wait()

#     return results_dir


# # ── Metrics collection ────────────────────────────────────────────────────

# METRIC_KEYS = [
#     ('accuracy', 'classification'),
#     ('f1_macro', 'classification'),
#     ('f1_weighted', 'classification'),
#     ('mcc', 'classification'),
#     ('balanced_accuracy', 'classification'),
#     ('precision_macro', 'classification'),
#     ('recall_macro', 'classification'),
#     ('roc_auc_macro_ovr', 'classification'),
#     ('cohen_kappa', 'classification'),
#     ('silhouette_score', 'representation'),
#     ('davies_bouldin_score', 'representation'),
#     ('calinski_harabasz_score', 'representation'),
#     ('separability_ratio', 'representation'),
# ]


# def collect_seed_metrics(results_dir, network, n_clients):
#     """Collect evaluation metrics for one seed."""
#     metrics = {}
#     for cid in range(n_clients):
#         eval_path = Path(results_dir) / network / f'client_{cid}' / 'evaluation_metrics.json'
#         summary_path = Path(results_dir) / network / f'client_{cid}' / 'final_summary.json'

#         if eval_path.exists():
#             with open(eval_path) as f:
#                 m = json.load(f)
#             row = {}
#             for key, group in METRIC_KEYS:
#                 v = m.get(group, {}).get(key)
#                 if v is not None:
#                     row[key] = float(v)
#             metrics[cid] = row
#         elif summary_path.exists():
#             # Fallback to summary
#             with open(summary_path) as f:
#                 s = json.load(f)
#             metrics[cid] = {
#                 'accuracy': s.get('test_acc'),
#                 'f1_macro': s.get('f1_macro'),
#                 'f1_weighted': s.get('f1_weighted'),
#                 'mcc': s.get('mcc'),
#                 'silhouette_score': s.get('silhouette_score'),
#                 'separability_ratio': s.get('separability_ratio'),
#             }
#             metrics[cid] = {k: float(v) for k, v in metrics[cid].items() if v is not None}

#     return metrics


# def aggregate_across_seeds(all_seed_metrics, n_clients):
#     """Aggregate metrics across seeds: compute mean +/- std per client and globally."""
#     # Structure: {seed: {client_id: {metric_key: value}}}
#     all_keys = set()
#     for sm in all_seed_metrics.values():
#         for cm in sm.values():
#             all_keys.update(cm.keys())

#     # Per-client aggregation
#     per_client = {}
#     for cid in range(n_clients):
#         per_client[cid] = {}
#         for key in sorted(all_keys):
#             vals = []
#             for seed, sm in all_seed_metrics.items():
#                 if cid in sm and key in sm[cid]:
#                     vals.append(sm[cid][key])
#             if vals:
#                 per_client[cid][key] = {
#                     'mean': float(np.mean(vals)),
#                     'std': float(np.std(vals)),
#                     'min': float(np.min(vals)),
#                     'max': float(np.max(vals)),
#                     'values': vals,
#                     'n_seeds': len(vals),
#                 }

#     # Global (across all clients and seeds)
#     global_agg = {}
#     for key in sorted(all_keys):
#         vals = []
#         for seed, sm in all_seed_metrics.items():
#             for cid in range(n_clients):
#                 if cid in sm and key in sm[cid]:
#                     vals.append(sm[cid][key])
#         if vals:
#             global_agg[key] = {
#                 'mean': float(np.mean(vals)),
#                 'std': float(np.std(vals)),
#                 'min': float(np.min(vals)),
#                 'max': float(np.max(vals)),
#                 'n_values': len(vals),
#             }

#     return per_client, global_agg


# # ── Output generators ────────────────────────────────────────────────────

# def write_multi_seed_csv(global_agg, per_client, seeds, out_path):
#     """Write CSV with multi-seed aggregated results."""
#     import csv
#     with open(out_path, 'w', newline='') as f:
#         w = csv.writer(f)
#         w.writerow(['Metric', 'Mean', 'Std', 'Min', 'Max', 'N_seeds_x_clients'])
#         for key in sorted(global_agg.keys()):
#             g = global_agg[key]
#             w.writerow([key, f"{g['mean']:.6f}", f"{g['std']:.6f}",
#                        f"{g['min']:.6f}", f"{g['max']:.6f}", g['n_values']])


# def write_multi_seed_latex(global_agg, seeds, network, out_path):
#     """Write LaTeX table for multi-seed results."""
#     display_metrics = [
#         ('accuracy', 'Accuracy', 4),
#         ('f1_macro', 'Macro-F1', 4),
#         ('f1_weighted', 'Weighted-F1', 4),
#         ('mcc', 'MCC', 4),
#         ('balanced_accuracy', 'Bal. Acc.', 4),
#         ('precision_macro', 'Precision', 4),
#         ('recall_macro', 'Recall', 4),
#         ('roc_auc_macro_ovr', 'ROC-AUC', 4),
#         ('silhouette_score', 'Silhouette', 3),
#         ('separability_ratio', 'Sep. Ratio', 3),
#     ]

#     lines = [
#         '% Requires \\usepackage{booktabs}',
#         '\\begin{table}[t]',
#         '\\centering',
#         f'\\caption{{Multi-seed robustness analysis of FedPer-Prox-XAI '
#         f'on {network.upper()} network (seeds: {", ".join(map(str, seeds))}). '
#         f'Values reported as mean $\\pm$ std across {len(seeds)} seeds $\\times$ 4 clients.}}',
#         f'\\label{{tab:multi_seed_{network}}}',
#         '\\begin{tabular}{lcc}',
#         '\\toprule',
#         'Metric & Mean $\\pm$ Std & Range \\\\',
#         '\\midrule',
#     ]
#     for key, name, dec in display_metrics:
#         if key in global_agg:
#             g = global_agg[key]
#             lines.append(
#                 f'{name} & {g["mean"]:.{dec}f} $\\pm$ {g["std"]:.{dec}f} '
#                 f'& [{g["min"]:.{dec}f}, {g["max"]:.{dec}f}] \\\\'
#             )
#     lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
#     Path(out_path).write_text('\n'.join(lines) + '\n')


# def write_multi_seed_json(all_seed_metrics, per_client, global_agg, seeds, out_path):
#     """Write full JSON report."""
#     report = {
#         'seeds': seeds,
#         'per_seed_per_client': {
#             str(s): {str(c): m for c, m in cm.items()}
#             for s, cm in all_seed_metrics.items()
#         },
#         'per_client_aggregated': {
#             str(c): m for c, m in per_client.items()
#         },
#         'global_aggregated': global_agg,
#     }
#     with open(out_path, 'w') as f:
#         json.dump(report, f, indent=2)


# # ── Main ──────────────────────────────────────────────────────────────────

# def main():
#     p = argparse.ArgumentParser(description='R1.1: Multi-seed robustness experiment')
#     p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 42],
#                    help='Random seeds to test (default: 0 1 2 3 42)')
#     p.add_argument('--rounds', type=int, default=30)
#     p.add_argument('--local-epochs', type=int, default=3)
#     p.add_argument('--network', type=str, default='wifi',
#                    choices=['wifi', '5g', '4g', '3g'])
#     p.add_argument('--all-networks', action='store_true',
#                    help='Run across all network types')
#     p.add_argument('--n-clients', type=int, default=4)
#     p.add_argument('--server-port', type=int, default=8080)
#     p.add_argument('--results-dir', type=str, default='./results_multi_seed')
#     p.add_argument('--data-dir', type=str, default='./data/client_datasets')
#     p.add_argument('--mu', type=float, default=0.01)
#     p.add_argument('--temp', type=float, default=0.5)
#     p.add_argument('--collect-only', action='store_true',
#                    help='Skip running, just collect existing results')
#     args = p.parse_args()

#     networks = ['wifi', '5g', '4g', '3g'] if args.all_networks else [args.network]

#     print("=" * 70)
#     print("  R1.1 — MULTI-SEED ROBUSTNESS EXPERIMENT")
#     print("=" * 70)
#     print(f"  Seeds      : {args.seeds}")
#     print(f"  Networks   : {[n.upper() for n in networks]}")
#     print(f"  Rounds     : {args.rounds}")
#     print(f"  Clients    : {args.n_clients}")
#     print("=" * 70)

#     for network in networks:
#         print(f"\n{'='*60}")
#         print(f"  Network: {network.upper()}")
#         print(f"{'='*60}")

#         net_results_dir = Path(args.results_dir) / network
#         all_seed_metrics = {}

#         for i, seed in enumerate(args.seeds):
#             print(f"\n  [{i+1}/{len(args.seeds)}] Seed = {seed}")

#             seed_dir = net_results_dir / f'seed_{seed}'

#             if not args.collect_only:
#                 print(f"    Running FL pipeline (seed={seed}, {args.rounds} rounds)...")
#                 t0 = time.time()
#                 result = run_single_seed(
#                     seed=seed, network=network, rounds=args.rounds,
#                     local_epochs=args.local_epochs, n_clients=args.n_clients,
#                     server_port=args.server_port,
#                     base_results_dir=str(seed_dir),
#                     data_dir=args.data_dir, mu=args.mu, temp=args.temp,
#                 )
#                 elapsed = time.time() - t0
#                 if result is None:
#                     print(f"    [SKIP] Seed {seed} failed after {elapsed:.1f}s")
#                     continue
#                 print(f"    Completed in {elapsed:.1f}s")

#             # Collect metrics
#             metrics = collect_seed_metrics(str(seed_dir), network, args.n_clients)
#             if metrics:
#                 all_seed_metrics[seed] = metrics
#                 avg_acc = np.mean([m.get('accuracy', 0) for m in metrics.values()])
#                 avg_f1 = np.mean([m.get('f1_macro', 0) for m in metrics.values()])
#                 print(f"    Avg Accuracy={avg_acc:.4f}  Avg Macro-F1={avg_f1:.4f}")
#             else:
#                 print(f"    [WARN] No metrics found for seed {seed}")

#         if not all_seed_metrics:
#             print(f"\n  [ERROR] No results collected for {network}")
#             continue

#         # Aggregate
#         per_client, global_agg = aggregate_across_seeds(
#             all_seed_metrics, args.n_clients)

#         # Output
#         out_dir = net_results_dir / 'aggregated'
#         out_dir.mkdir(parents=True, exist_ok=True)

#         write_multi_seed_csv(global_agg, per_client, args.seeds,
#                             out_dir / 'multi_seed_summary.csv')
#         write_multi_seed_latex(global_agg, args.seeds, network,
#                               out_dir / 'multi_seed_table.tex')
#         write_multi_seed_json(all_seed_metrics, per_client, global_agg,
#                              args.seeds, out_dir / 'multi_seed_full.json')

#         # Print summary
#         print(f"\n  {'─'*55}")
#         print(f"  MULTI-SEED SUMMARY ({network.upper()}) — {len(all_seed_metrics)} seeds")
#         print(f"  {'─'*55}")
#         for key in ['accuracy', 'f1_macro', 'f1_weighted', 'mcc',
#                      'silhouette_score', 'separability_ratio']:
#             if key in global_agg:
#                 g = global_agg[key]
#                 print(f"    {key:25s}  {g['mean']:.4f} +/- {g['std']:.4f}  "
#                       f"[{g['min']:.4f}, {g['max']:.4f}]")

#         print(f"\n  Saved to: {out_dir}/")

#     print(f"\n{'='*70}")
#     print("  MULTI-SEED EXPERIMENT COMPLETE")
#     print(f"{'='*70}")


# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
"""R1.1 — Multi-seed experiment runner.

Runs the full FedPer-Prox-XAI pipeline with multiple random seeds
(default: 0, 1, 2, 3, 42) and aggregates results as mean +/- std.

This addresses Reviewer 1 Comment 1:
  "Run experiment across 4 different seeds besides 42 for range of performance."

The script:
  1. For each seed, sets all random seeds (torch, numpy, python)
  2. Runs the full FL pipeline (server + 4 clients) locally
  3. Collects per-seed evaluation metrics
  4. Aggregates and outputs mean +/- std for all key metrics
  5. Generates a LaTeX table and CSV summary

Usage:
  python -I scripts/run_multi_seed.py --seeds 0 1 2 3 42 --rounds 30 --network wifi
  python -I scripts/run_multi_seed.py --all-networks --rounds 30
"""
import os, sys, argparse, json, time, subprocess, signal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# See scripts/csv_to_npy.py for the FEDPER_DATA_ROOT convention.
DATA_ROOT = os.environ.get('FEDPER_DATA_ROOT', 'data')
from pathlib import Path
from collections import defaultdict

import numpy as np


# ── Reproducibility ────────────────────────────────────────────────────────

def set_all_seeds(seed: int):
    """Set all random seeds for full reproducibility."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ── Single-seed pipeline runner ──────────────────────────────────────────

def run_single_seed(seed, network, rounds, local_epochs, n_clients,
                    server_port, base_results_dir, data_dir, mu, temp):
    """Run one complete FL experiment (server + clients) with a given seed."""
    results_dir = str(Path(base_results_dir) / f'seed_{seed}')
    os.makedirs(results_dir, exist_ok=True)

    project_root = str(Path(__file__).resolve().parent.parent)
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root
    env['FL_SEED'] = str(seed)

    # 1) Start server
    server_cmd = [
        sys.executable, '-I',
        os.path.join(project_root, 'scripts', 'run_server.py'),
        '--rounds', str(rounds),
        '--network', network,
        '--address', f'0.0.0.0:{server_port}',
        '--min-clients', str(n_clients),
        '--local-epochs', str(local_epochs),
        '--mu', str(mu),
        '--temp', str(temp),
        '--results-dir', results_dir,
    ]
    log_dir = Path(results_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / 'server.log', 'w') as server_log:
        server_proc = subprocess.Popen(
            server_cmd, env=env, stdout=server_log, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
    time.sleep(8)  # Wait for server startup

    if server_proc.poll() is not None:
        print(f"    [ERROR] Server crashed for seed {seed}")
        return None

    # 2) Start clients
    client_procs = []
    for cid in range(n_clients):
        client_cmd = [
            sys.executable, '-I',
            os.path.join(project_root, 'scripts', 'run_client.py'),
            '--client-id', str(cid),
            '--server-addr', f'127.0.0.1:{server_port}',
            '--network', network,
            '--results-dir', results_dir,
            '--data-dir', data_dir,
        ]
        client_env = env.copy()
        client_env['FL_SEED'] = str(seed)
        client_env['USE_FOCAL_LOSS'] = '0'
        with open(log_dir / f'client_{cid}.log', 'w') as cl:
            p = subprocess.Popen(
                client_cmd, env=client_env, stdout=cl, stderr=subprocess.STDOUT,
            )
            client_procs.append(p)

    # 3) Wait for completion
    server_proc.wait()
    for p in client_procs:
        p.wait()

    return results_dir


# ── Metrics collection ────────────────────────────────────────────────────

METRIC_KEYS = [
    ('accuracy', 'classification'),
    ('f1_macro', 'classification'),
    ('f1_weighted', 'classification'),
    ('mcc', 'classification'),
    ('balanced_accuracy', 'classification'),
    ('precision_macro', 'classification'),
    ('recall_macro', 'classification'),
    ('roc_auc_macro_ovr', 'classification'),
    ('cohen_kappa', 'classification'),
    ('silhouette_score', 'representation'),
    ('davies_bouldin_score', 'representation'),
    ('calinski_harabasz_score', 'representation'),
    ('separability_ratio', 'representation'),
]


def collect_seed_metrics(results_dir, network, n_clients):
    """Collect evaluation metrics for one seed."""
    metrics = {}
    for cid in range(n_clients):
        eval_path = Path(results_dir) / network / f'client_{cid}' / 'evaluation_metrics.json'
        summary_path = Path(results_dir) / network / f'client_{cid}' / 'final_summary.json'

        if eval_path.exists():
            with open(eval_path) as f:
                m = json.load(f)
            row = {}
            for key, group in METRIC_KEYS:
                v = m.get(group, {}).get(key)
                if v is not None:
                    row[key] = float(v)
            metrics[cid] = row
        elif summary_path.exists():
            # Fallback to summary
            with open(summary_path) as f:
                s = json.load(f)
            metrics[cid] = {
                'accuracy': s.get('test_acc'),
                'f1_macro': s.get('f1_macro'),
                'f1_weighted': s.get('f1_weighted'),
                'mcc': s.get('mcc'),
                'silhouette_score': s.get('silhouette_score'),
                'separability_ratio': s.get('separability_ratio'),
            }
            metrics[cid] = {k: float(v) for k, v in metrics[cid].items() if v is not None}

    return metrics


def aggregate_across_seeds(all_seed_metrics, n_clients):
    """Aggregate metrics across seeds: compute mean +/- std per client and globally."""
    # Structure: {seed: {client_id: {metric_key: value}}}
    all_keys = set()
    for sm in all_seed_metrics.values():
        for cm in sm.values():
            all_keys.update(cm.keys())

    # Per-client aggregation
    per_client = {}
    for cid in range(n_clients):
        per_client[cid] = {}
        for key in sorted(all_keys):
            vals = []
            for seed, sm in all_seed_metrics.items():
                if cid in sm and key in sm[cid]:
                    vals.append(sm[cid][key])
            if vals:
                per_client[cid][key] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                    'values': vals,
                    'n_seeds': len(vals),
                }

    # Global (across all clients and seeds)
    global_agg = {}
    for key in sorted(all_keys):
        vals = []
        for seed, sm in all_seed_metrics.items():
            for cid in range(n_clients):
                if cid in sm and key in sm[cid]:
                    vals.append(sm[cid][key])
        if vals:
            global_agg[key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'n_values': len(vals),
            }

    return per_client, global_agg


# ── Output generators ────────────────────────────────────────────────────

def write_multi_seed_csv(global_agg, per_client, seeds, out_path):
    """Write CSV with multi-seed aggregated results."""
    import csv
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Mean', 'Std', 'Min', 'Max', 'N_seeds_x_clients'])
        for key in sorted(global_agg.keys()):
            g = global_agg[key]
            w.writerow([key, f"{g['mean']:.6f}", f"{g['std']:.6f}",
                       f"{g['min']:.6f}", f"{g['max']:.6f}", g['n_values']])


def write_multi_seed_latex(global_agg, seeds, network, out_path):
    """Write LaTeX table for multi-seed results."""
    display_metrics = [
        ('accuracy', 'Accuracy', 4),
        ('f1_macro', 'Macro-F1', 4),
        ('f1_weighted', 'Weighted-F1', 4),
        ('mcc', 'MCC', 4),
        ('balanced_accuracy', 'Bal. Acc.', 4),
        ('precision_macro', 'Precision', 4),
        ('recall_macro', 'Recall', 4),
        ('roc_auc_macro_ovr', 'ROC-AUC', 4),
        ('silhouette_score', 'Silhouette', 3),
        ('separability_ratio', 'Sep. Ratio', 3),
    ]

    lines = [
        '% Requires \\usepackage{booktabs}',
        '\\begin{table}[t]',
        '\\centering',
        f'\\caption{{Multi-seed robustness analysis of FedPer-Prox-XAI '
        f'on {network.upper()} network (seeds: {", ".join(map(str, seeds))}). '
        f'Values reported as mean $\\pm$ std across {len(seeds)} seeds $\\times$ 4 clients.}}',
        f'\\label{{tab:multi_seed_{network}}}',
        '\\begin{tabular}{lcc}',
        '\\toprule',
        'Metric & Mean $\\pm$ Std & Range \\\\',
        '\\midrule',
    ]
    for key, name, dec in display_metrics:
        if key in global_agg:
            g = global_agg[key]
            lines.append(
                f'{name} & {g["mean"]:.{dec}f} $\\pm$ {g["std"]:.{dec}f} '
                f'& [{g["min"]:.{dec}f}, {g["max"]:.{dec}f}] \\\\'
            )
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    Path(out_path).write_text('\n'.join(lines) + '\n')


def write_multi_seed_json(all_seed_metrics, per_client, global_agg, seeds, out_path):
    """Write full JSON report."""
    report = {
        'seeds': seeds,
        'per_seed_per_client': {
            str(s): {str(c): m for c, m in cm.items()}
            for s, cm in all_seed_metrics.items()
        },
        'per_client_aggregated': {
            str(c): m for c, m in per_client.items()
        },
        'global_aggregated': global_agg,
    }
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='R1.1: Multi-seed robustness experiment')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 42],
                   help='Random seeds to test (default: 0 1 2 3 42)')
    p.add_argument('--rounds', type=int, default=30)
    p.add_argument('--local-epochs', type=int, default=3)
    p.add_argument('--network', type=str, default='wifi',
                   choices=['wifi', '5g', '4g', '3g'])
    p.add_argument('--all-networks', action='store_true',
                   help='Run across all network types')
    p.add_argument('--n-clients', type=int, default=4)
    p.add_argument('--server-port', type=int, default=8080)
    p.add_argument('--results-dir', type=str, default='./results_multi_seed')
    p.add_argument('--data-dir', type=str, default=f'{DATA_ROOT}/client_datasets')
    p.add_argument('--mu', type=float, default=0.01)
    p.add_argument('--temp', type=float, default=0.5)
    p.add_argument('--collect-only', action='store_true',
                   help='Skip running, just collect existing results')
    args = p.parse_args()

    networks = ['wifi', '5g', '4g', '3g'] if args.all_networks else [args.network]

    print("=" * 70)
    print("  R1.1 — MULTI-SEED ROBUSTNESS EXPERIMENT")
    print("=" * 70)
    print(f"  Seeds      : {args.seeds}")
    print(f"  Networks   : {[n.upper() for n in networks]}")
    print(f"  Rounds     : {args.rounds}")
    print(f"  Clients    : {args.n_clients}")
    print("=" * 70)

    for network in networks:
        print(f"\n{'='*60}")
        print(f"  Network: {network.upper()}")
        print(f"{'='*60}")

        net_results_dir = Path(args.results_dir) / network
        all_seed_metrics = {}

        if not args.collect_only:
            # ── PARALLEL EXECUTION: launch ALL seeds at once ──
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _run_one_seed(seed, port):
                """Worker: run one seed on its own port."""
                seed_dir = net_results_dir / f'seed_{seed}'
                t0 = time.time()
                result = run_single_seed(
                    seed=seed, network=network, rounds=args.rounds,
                    local_epochs=args.local_epochs, n_clients=args.n_clients,
                    server_port=port,
                    base_results_dir=str(seed_dir),
                    data_dir=args.data_dir, mu=args.mu, temp=args.temp,
                )
                elapsed = time.time() - t0
                return seed, result, elapsed

            base_port = args.server_port
            futures = {}
            print(f"\n  Launching {len(args.seeds)} seeds IN PARALLEL "
                  f"(ports {base_port}–{base_port + len(args.seeds) - 1})...")

            with ThreadPoolExecutor(max_workers=len(args.seeds)) as executor:
                for i, seed in enumerate(args.seeds):
                    port = base_port + i
                    f = executor.submit(_run_one_seed, seed, port)
                    futures[f] = seed
                    print(f"    Started seed={seed} on port {port}")

                for f in as_completed(futures):
                    seed, result, elapsed = f.result()
                    if result is None:
                        print(f"    [FAIL] Seed {seed} crashed after {elapsed:.1f}s")
                    else:
                        print(f"    [DONE] Seed {seed} completed in {elapsed:.1f}s")

        # ── Collect metrics for all seeds ──
        for seed in args.seeds:
            seed_dir = net_results_dir / f'seed_{seed}'
            metrics = collect_seed_metrics(str(seed_dir), network, args.n_clients)
            if metrics:
                all_seed_metrics[seed] = metrics
                avg_acc = np.mean([m.get('accuracy', 0) for m in metrics.values()])
                avg_f1 = np.mean([m.get('f1_macro', 0) for m in metrics.values()])
                print(f"    Seed {seed}: Avg Accuracy={avg_acc:.4f}  Avg Macro-F1={avg_f1:.4f}")
            else:
                print(f"    [WARN] No metrics found for seed {seed}")

        if not all_seed_metrics:
            print(f"\n  [ERROR] No results collected for {network}")
            continue

        # Aggregate
        per_client, global_agg = aggregate_across_seeds(
            all_seed_metrics, args.n_clients)

        # Output
        out_dir = net_results_dir / 'aggregated'
        out_dir.mkdir(parents=True, exist_ok=True)

        write_multi_seed_csv(global_agg, per_client, args.seeds,
                            out_dir / 'multi_seed_summary.csv')
        write_multi_seed_latex(global_agg, args.seeds, network,
                              out_dir / 'multi_seed_table.tex')
        write_multi_seed_json(all_seed_metrics, per_client, global_agg,
                             args.seeds, out_dir / 'multi_seed_full.json')

        # Print summary
        print(f"\n  {'─'*55}")
        print(f"  MULTI-SEED SUMMARY ({network.upper()}) — {len(all_seed_metrics)} seeds")
        print(f"  {'─'*55}")
        for key in ['accuracy', 'f1_macro', 'f1_weighted', 'mcc',
                     'silhouette_score', 'separability_ratio']:
            if key in global_agg:
                g = global_agg[key]
                print(f"    {key:25s}  {g['mean']:.4f} +/- {g['std']:.4f}  "
                      f"[{g['min']:.4f}, {g['max']:.4f}]")

        print(f"\n  Saved to: {out_dir}/")

    print(f"\n{'='*70}")
    print("  MULTI-SEED EXPERIMENT COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
