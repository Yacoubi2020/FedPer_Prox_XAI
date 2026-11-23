#!/usr/bin/env python3
"""Aggregate evaluation_metrics.json across networks x clients into paper-ready tables.

Outputs (in --out-dir, default ./results/tables):
  - metrics_per_client.csv : one row per (network, client)
  - metrics_summary.csv     : one row per network (mean +/- std across clients)
  - metrics_table.tex       : LaTeX booktabs table (network rows, mean +/- std)

Usage:
  python scripts/make_metrics_table.py --results-dir ./results
  python scripts/make_metrics_table.py --results-dir ./results --networks wifi 5g --clients 0 1 2
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

# (json_key, json_group, column header, decimals)
METRICS = [
    ('accuracy', 'classification', 'Accuracy', 4),
    ('f1_macro', 'classification', 'Macro-F1', 4),
    ('f1_weighted', 'classification', 'Weighted-F1', 4),
    ('mcc', 'classification', 'MCC', 4),
    ('balanced_accuracy', 'classification', 'Bal.\\,Acc', 4),
    ('roc_auc_macro_ovr', 'classification', 'ROC-AUC', 4),
    ('silhouette_score', 'representation', 'Silhouette', 3),
    ('davies_bouldin_score', 'representation', 'DB', 3),
    ('calinski_harabasz_score', 'representation', 'CH', 1),
    ('separability_ratio', 'representation', 'Sep.\\,ratio', 3),
]


def discover_clients(net_dir, want):
    ids = []
    for d in sorted(net_dir.glob('client_*')):
        try:
            cid = int(d.name.split('_')[1])
        except (IndexError, ValueError):
            continue
        if want is None or cid in want:
            ids.append(cid)
    return ids


def collect(results_dir, networks, clients):
    rows = []  # list of dicts: network, client, <metric keys>
    for net in networks:
        net_dir = Path(results_dir) / net
        if not net_dir.exists():
            continue
        for cid in discover_clients(net_dir, clients):
            f = net_dir / f'client_{cid}' / 'evaluation_metrics.json'
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            row = {'network': net, 'client': cid}
            for key, grp, _, _ in METRICS:
                v = m.get(grp, {}).get(key, None)
                row[key] = v
            rows.append(row)
    return rows


def write_per_client_csv(rows, path):
    cols = ['network', 'client'] + [k for k, *_ in METRICS]
    with open(path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['Network', 'Client'] + [h for _, _, h, _ in METRICS])
        for r in rows:
            w.writerow([r['network'], r['client']] +
                       ['' if r.get(k) is None else r[k] for k, *_ in METRICS])


def summarize(rows, networks):
    """Return {network: {metric_key: (mean, std, n)}}."""
    summary = {}
    for net in networks:
        sub = [r for r in rows if r['network'] == net]
        if not sub:
            continue
        summary[net] = {}
        for key, *_ in METRICS:
            vals = [r[key] for r in sub if r.get(key) is not None]
            if vals:
                summary[net][key] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
            else:
                summary[net][key] = (None, None, 0)
    return summary


def write_summary_csv(summary, path):
    with open(path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['Network'] + [h for _, _, h, _ in METRICS])
        for net, d in summary.items():
            cells = []
            for key, _, _, dec in METRICS:
                mean, std, n = d[key]
                cells.append('' if mean is None else f'{mean:.{dec}f}+/-{std:.{dec}f}')
            w.writerow([net.upper()] + cells)


def write_latex(summary, path):
    headers = [h for _, _, h, _ in METRICS]
    col_fmt = 'l' + 'c' * len(headers)
    lines = [
        '% Requires \\usepackage{booktabs}',
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Per-network classification and representation-space separability '
        'metrics for the proposed FedPer-Prox-XAI pipeline (mean~$\\pm$~std across clients).}',
        '\\label{tab:fedper_metrics}',
        '\\resizebox{\\linewidth}{!}{%',
        f'\\begin{{tabular}}{{{col_fmt}}}',
        '\\toprule',
        'Network & ' + ' & '.join(headers) + ' \\\\',
        '\\midrule',
    ]
    for net, d in summary.items():
        cells = []
        for key, _, _, dec in METRICS:
            mean, std, n = d[key]
            cells.append('--' if mean is None else f'{mean:.{dec}f}$\\pm${std:.{dec}f}')
        lines.append(f'{net.upper()} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}%', '}', '\\end{table}']
    Path(path).write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', default='./results')
    p.add_argument('--networks', nargs='+', default=['wifi', '5g', '4g', '3g'])
    p.add_argument('--clients', type=int, nargs='+', default=None)
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results_dir) / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect(args.results_dir, args.networks, args.clients)
    if not rows:
        print('No evaluation_metrics.json found. Run the clients first.')
        return
    present_nets = [n for n in args.networks if any(r['network'] == n for r in rows)]
    summary = summarize(rows, present_nets)

    write_per_client_csv(rows, out_dir / 'metrics_per_client.csv')
    write_summary_csv(summary, out_dir / 'metrics_summary.csv')
    write_latex(summary, out_dir / 'metrics_table.tex')

    print(f'  [saved] {out_dir / "metrics_per_client.csv"}  ({len(rows)} rows)')
    print(f'  [saved] {out_dir / "metrics_summary.csv"}')
    print(f'  [saved] {out_dir / "metrics_table.tex"}')
    print('\nPreview (summary):')
    for net, d in summary.items():
        acc = d['accuracy'][0]; f1 = d['f1_macro'][0]; sil = d['silhouette_score'][0]
        print(f'  {net.upper():5s}  Acc={acc:.4f}  Macro-F1={f1:.4f}  Silhouette={sil}')


if __name__ == '__main__':
    main()
