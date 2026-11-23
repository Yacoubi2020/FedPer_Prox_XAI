#!/usr/bin/env python3
"""
data/client_1.csv ... client_4.csv  ->  data/client_datasets/client_N/{X.npy, y.npy, stats.pkl}
- Features alignées sur les colonnes communes à tous les clients
- Labels encodés sur la liste GLOBALE des classes (même index pour tous)
- Normalisation StandardScaler (scaler sauvegardé)
"""
import argparse, json, os, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Single-folder data separation: if FEDPER_DATA_ROOT is set, raw CSVs and
# generated client_datasets default to living under that one external folder
# (e.g. shared HPC storage) instead of inside the repo's own data/ directory.
# Anyone launching a run then only needs read access to $FEDPER_DATA_ROOT;
# the repo itself stays code-only. Unset, behaviour is unchanged (./data).
DATA_ROOT = os.environ.get('FEDPER_DATA_ROOT', 'data')

def detect_label_col(df, wanted=None):
    if wanted and wanted in df.columns:
        return wanted
    for c in ['Attack_type', 'attack_type', 'Label', 'label', 'target', 'Target', 'class']:
        if c in df.columns:
            return c
    return df.columns[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-dir', default=f'{DATA_ROOT}/raw')
    ap.add_argument('--out-dir', default=f'{DATA_ROOT}/client_datasets')
    ap.add_argument('--n-clients', type=int, default=4)
    ap.add_argument('--label-col', default=None)
    ap.add_argument('--scaler', choices=['global', 'per-client'], default='global')
    args = ap.parse_args()

    csv_dir, out_dir = Path(args.csv_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Lecture
    dfs = {}
    for i in range(1, args.n_clients + 1):
        p = csv_dir / f'client_{i}.csv'
        dfs[i] = pd.read_csv(p, low_memory=False)
        print(f'client_{i}.csv : {len(dfs[i]):,} lignes')

    label_col = detect_label_col(dfs[1], args.label_col)
    print(f'Colonne label : {label_col}')

    # 2) Classes globales + features communes
    classes = sorted({str(c) for d in dfs.values() for c in d[label_col].unique()})
    class_to_idx = {c: k for k, c in enumerate(classes)}
    print(f'{len(classes)} classes : {class_to_idx}')

    common = None
    for d in dfs.values():
        num = set(d.drop(columns=[label_col]).select_dtypes(include=[np.number]).columns)
        common = num if common is None else common & num
    feat_cols = [c for c in dfs[1].columns if c in common]   # ordre stable
    print(f'{len(feat_cols)} features communes')

    # 3) Matrices brutes
    raw = {}
    for i, d in dfs.items():
        X = d[feat_cols].apply(pd.to_numeric, errors='coerce')
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median()).fillna(0.0)
        y = d[label_col].astype(str).map(class_to_idx).to_numpy(dtype=np.int64)
        raw[i] = (X.to_numpy(dtype=np.float32), y)

    # 4) Normalisation
    if args.scaler == 'global':
        scaler = StandardScaler().fit(np.vstack([X for X, _ in raw.values()]))
        scalers = {i: scaler for i in raw}
        with open(out_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        scalers = {i: StandardScaler().fit(X) for i, (X, _) in raw.items()}

    # 5) Sauvegarde .npy
    all_stats = {}
    for i, (X, y) in raw.items():
        X = scalers[i].transform(X).astype(np.float32)
        cdir = out_dir / f'client_{i - 1}'          # client_0 ... client_3 (attendu par run_client.py)
        cdir.mkdir(exist_ok=True)
        np.save(cdir / 'X.npy', X)
        np.save(cdir / 'y.npy', y)

        present = sorted(set(y.tolist()))
        stats = {
            'n_samples': int(len(X)),
            'n_features': int(X.shape[1]),
            'input_dim': int(X.shape[1]),
            'n_classes': len(present),
            'class_names': [classes[k] for k in present],
            'class_distribution': {classes[k]: int((y == k).sum()) for k in present},
        }
        with open(cdir / 'stats.pkl', 'wb') as f:
            pickle.dump(stats, f)
        all_stats[i - 1] = stats
        print(f'client_{i - 1}/ : X{X.shape} y{y.shape} | {stats["class_distribution"]}')

    with open(out_dir / 'all_stats.pkl', 'wb') as f:
        pickle.dump(all_stats, f)
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump({'classes': classes, 'features': feat_cols, 'label_col': label_col}, f, indent=2)
    print(f'\nTerminé -> {out_dir}/')

if __name__ == '__main__':
    main()