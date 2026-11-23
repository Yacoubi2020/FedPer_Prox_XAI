#!/usr/bin/env python3
"""Génère des CSV IoMT artificiels pour tester rapidement le pipeline."""
import os, sys, argparse
# See scripts/csv_to_npy.py for the FEDPER_DATA_ROOT convention.
DATA_ROOT = os.environ.get('FEDPER_DATA_ROOT', 'data')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import pandas as pd

ALL_CLASSES = ['Normal', 'SlowRead', 'SlowLoris', 'RUDY', 'ARP',
               'cam', 'Netscan', 'ApacheKiller', 'Malaria']

FEATURE_COLS = [
    'http.content_length', 'http.request', 'http.response.code',
    'http.response_number', 'http.time', 'tcp.analysis.initial_rtt',
    'tcp.connection.fin', 'tcp.connection.syn', 'tcp.connection.synack',
    'tcp.flags.cwr', 'tcp.flags.ecn', 'tcp.flags.fin', 'tcp.flags.ns',
    'tcp.flags.res', 'tcp.flags.syn', 'tcp.flags.urg', 'tcp.urgent_pointer',
    'ip.frag_offset', 'eth.dst.ig', 'eth.src.ig', 'eth.src.lg',
    'eth.src_not_group', 'arp.isannouncement',
]


def make_client(n, cid, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURE_COLS)))
    weights = [0.40, 0.10, 0.10, 0.08, 0.08, 0.07, 0.07, 0.05, 0.05]
    classes = ALL_CLASSES if cid < 2 else ALL_CLASSES[:7]
    w = np.array(weights[:len(classes)]); w /= w.sum()
    labels = rng.choice(classes, size=n, p=w)
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df['is_malicious'] = (labels != 'Normal').astype(int)
    df['class'] = labels
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=2000)
    p.add_argument('--n-clients', type=int, default=4)
    p.add_argument('--output-dir', type=str, default=f'{DATA_ROOT}/')
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"🔧 Génération données artificielles : {args.samples} samples × {args.n_clients} clients\n")
    for i in range(1, args.n_clients + 1):
        df = make_client(args.samples, cid=i-1, seed=i*42)
        path = out / f'client_{i}.csv'
        df.to_csv(path, index=False)
        print(f"  ✅ {path} — {len(df):,} lignes")
    print(f"\n🚀 Suite : python scripts/prepare_data.py --csv-dir {args.output_dir}")


if __name__ == '__main__':
    main()
