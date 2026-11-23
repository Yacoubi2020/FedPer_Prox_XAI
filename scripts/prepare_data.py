#!/usr/bin/env python3
"""Prépare les partitions de données depuis les CSV clients.

DEPRECATED — DO NOT USE on the datasets in this repo.

This script (via data.partition_and_save) assumes the CSV's LAST column is the
label and uses a single hardcoded class list (data.ALL_CLASSES). On the real
IoMT-TrafficData / IoT-IDS2024 exports, the label lives in a column named
`class` (or similar), not in the last column, and the class list differs
between the two datasets. Using this script on those CSVs silently produces
wrong labels (everything falls back to class 0) rather than failing loudly.

The script that was actually used to generate the client_datasets/ shipped in
this repository, and the one documented in the top-level README, is
scripts/csv_to_npy.py: it auto-detects the label column and derives the class
list from the data itself. Kept here only for historical reference.
"""
import os, sys, argparse

print(
    "\n[DEPRECATED] scripts/prepare_data.py assumes the label is the CSV's "
    "last column, which is NOT true for this project's datasets (the label "
    "column is 'class'). Use scripts/csv_to_npy.py instead. Aborting.\n",
    file=sys.stderr,
)
sys.exit(1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

from data import partition_and_save, ALL_CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv-dir', type=str, default='data/')
    p.add_argument('--output-dir', type=str, default='data/client_datasets')
    p.add_argument('--n-clients', type=int, default=4)
    args = p.parse_args()

    print("=" * 70)
    print("📊 PRÉPARATION DES DONNÉES IoMT Traffic")
    print("=" * 70)
    print(f"  Répertoire CSV : {args.csv_dir}")
    print(f"  Clients        : {args.n_clients}")
    print(f"  Classes        : {ALL_CLASSES}")

    csv_files = []
    for i in range(1, args.n_clients + 1):
        path = Path(args.csv_dir) / f'client_{i}.csv'
        if not path.exists():
            print(f"❌ Manquant : {path}")
            sys.exit(1)
        csv_files.append(str(path))
        print(f"  ✅ Client {i-1}: {path}")

    stats = partition_and_save(csv_files, args.output_dir, ALL_CLASSES)
    print("\n" + "=" * 70)
    print("✅ DONNÉES PRÊTES")
    print("=" * 70)
    for cid, s in stats.items():
        print(f"  Client {cid}: {s['n_samples']:,} samples | classes: {s['class_names']}")


if __name__ == '__main__':
    main()
