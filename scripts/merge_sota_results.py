#!/usr/bin/env python3
"""Merge per-method SOTA comparison results (one JSON per terminal/job,
produced by run_sota_baselines.py --methods <key>) into the single
combined table/CSV that run_sota_baselines.py would have produced if run
with --methods omitted (all methods in one process).

Usage:
  python -I scripts/merge_sota_results.py --out-dir results/sota_comparison
"""
import argparse
import json
from pathlib import Path

# Re-use the same table/CSV writers as the main script so the merged
# output is byte-for-byte the same format.
from run_sota_baselines_v2 import generate_comparison_table_latex, generate_comparison_csv


def main():
    p = argparse.ArgumentParser(description='Merge per-method SOTA result JSONs')
    p.add_argument('--out-dir', type=str, default='./results/sota_comparison',
                   help='Directory containing the per-method sota_comparison_full_<method>.json files')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    json_files = sorted(out_dir.glob('sota_comparison_full_*.json'))
    if not json_files:
        raise SystemExit(f"No per-method result files found in {out_dir} "
                          f"(expected sota_comparison_full_<method>.json)")

    print(f"Found {len(json_files)} per-method result file(s):")
    for jf in json_files:
        print(f"  - {jf.name}")

    # Each file is a list (one entry per seed) of {method_key: {...}, 'seed': s}.
    # Merge all method_key -> metrics dicts per seed into one combined
    # per-seed results list, matching the shape run_all_baselines() returns.
    per_seed_merged = {}  # seed -> {method_key: metrics, ...}
    for jf in json_files:
        with open(jf) as f:
            seed_entries = json.load(f)
        for entry in seed_entries:
            seed = entry['seed']
            per_seed_merged.setdefault(seed, {'seed': seed})
            for key, val in entry.items():
                if key == 'seed':
                    continue
                per_seed_merged[seed][key] = val

    seeds = sorted(per_seed_merged.keys())
    all_seed_results = [per_seed_merged[s] for s in seeds]
    print(f"\nMerged seeds: {seeds}")
    methods_found = sorted({k for r in all_seed_results for k in r if k != 'seed'})
    print(f"Merged methods: {methods_found}")

    if len(seeds) > 1:
        metrics_keys = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro',
                        'recall_macro', 'cohen_kappa', 'mcc']
        agg = {'mean': {}, 'std': {}}
        import numpy as np
        for method in methods_found:
            agg['mean'][method] = {}
            agg['std'][method] = {}
            for mk in metrics_keys:
                vals = [r[method].get(mk, 0) for r in all_seed_results if method in r]
                if vals:
                    agg['mean'][method][mk] = float(np.mean(vals))
                    agg['std'][method][mk] = float(np.std(vals))
        generate_comparison_table_latex(agg, out_dir / 'sota_comparison_table_merged.tex')
        generate_comparison_csv(agg, out_dir / 'sota_comparison_merged.csv')
    else:
        generate_comparison_table_latex(all_seed_results[0], out_dir / 'sota_comparison_table_merged.tex')
        generate_comparison_csv(all_seed_results[0], out_dir / 'sota_comparison_merged.csv')

    with open(out_dir / 'sota_comparison_full_merged.json', 'w') as f:
        json.dump(all_seed_results, f, indent=2)

    print(f"\nWrote merged outputs to {out_dir}/:")
    print(f"  sota_comparison_table_merged.tex")
    print(f"  sota_comparison_merged.csv")
    print(f"  sota_comparison_full_merged.json")


if __name__ == '__main__':
    main()