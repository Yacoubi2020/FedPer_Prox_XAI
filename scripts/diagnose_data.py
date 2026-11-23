#!/usr/bin/env python3
"""Diagnostic rapide des donnees client pour debugger les resultats faibles."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
from pathlib import Path

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/client_datasets')

print("=" * 60)
print("DIAGNOSTIC DONNEES")
print("=" * 60)

for cid in range(4):
    cdir = data_dir / f'client_{cid}'
    if not cdir.exists():
        print(f"\n[SKIP] client_{cid} — dossier inexistant")
        continue

    print(f"\n--- Client {cid} ---")

    # Charger X et y
    X = np.load(cdir / 'X.npy')
    y = np.load(cdir / 'y.npy')

    print(f"  X.shape = {X.shape}  dtype = {X.dtype}")
    print(f"  y.shape = {y.shape}  dtype = {y.dtype}")
    print(f"  y unique values = {np.unique(y)}")
    print(f"  y min={y.min()} max={y.max()}")

    # Distribution des classes
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        print(f"    class {c}: {n:,} samples ({100*n/len(y):.1f}%)")

    # Stats features
    print(f"  X range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  X mean:  {X.mean():.4f}  std: {X.std():.4f}")

    # NaN / Inf check
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  *** PROBLEME: {n_nan} NaN, {n_inf} Inf dans X ***")
    else:
        print(f"  Pas de NaN/Inf")

    # Stats par feature (premiers et derniers)
    print(f"  Feature stats (min/max/mean/std):")
    for i in [0, 1, 2, X.shape[1]//2, X.shape[1]-1]:
        col = X[:, i]
        print(f"    feat[{i:2d}]: min={col.min():.4f} max={col.max():.4f} "
              f"mean={col.mean():.4f} std={col.std():.4f}")

    # Charger stats.pkl si dispo
    sp = cdir / 'stats.pkl'
    if sp.exists():
        with open(sp, 'rb') as f:
            stats = pickle.load(f)
        print(f"  stats.pkl keys: {list(stats.keys())}")
        if 'class_names' in stats:
            print(f"  class_names: {stats['class_names']}")

print("\n" + "=" * 60)
print("FIN DIAGNOSTIC")
print("=" * 60)
