#!/usr/bin/env python3
"""R3.8 — Additional results, statistical analysis, and significance tests.

Provides:
  1. Friedman test + Nemenyi post-hoc across baselines
  2. Wilcoxon signed-rank pairwise tests
  3. Convergence analysis (per-round accuracy/loss curves)
  4. Non-IID impact analysis (per-client performance dispersion)
  5. Ablation study summary (FedAvg mode, FocalLoss toggle)
  6. Communication-accuracy trade-off analysis

This addresses Reviewer 3 Comment 8:
  "More results and discussion needed."

Usage:
  python -I scripts/statistical_analysis.py --results-dir results/
  python -I scripts/statistical_analysis.py --sota-json results/sota_comparison/sota_comparison_full.json
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────
# 1. Statistical significance tests
# ─────────────────────────────────────────────────────────────────────────

def friedman_test(results_matrix):
    """Friedman test for comparing multiple methods across multiple datasets/seeds.

    Args:
        results_matrix: dict {method_name: [score_seed0, score_seed1, ...]}
    Returns:
        dict with chi2, p_value, and whether differences are significant
    """
    if not HAS_SCIPY:
        return {'error': 'scipy not installed — install with: pip install scipy'}

    methods = list(results_matrix.keys())
    data = np.array([results_matrix[m] for m in methods])

    if data.shape[1] < 3:
        return {
            'note': 'Friedman test requires >= 3 observations per method',
            'n_observations': data.shape[1],
        }

    stat, p_value = scipy_stats.friedmanchisquare(*[data[i] for i in range(len(methods))])

    return {
        'test': 'Friedman',
        'chi2_statistic': float(stat),
        'p_value': float(p_value),
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01,
        'methods': methods,
        'n_observations': data.shape[1],
    }


def wilcoxon_pairwise(results_matrix, reference='fedper_prox_xai'):
    """Wilcoxon signed-rank test: each method vs. the reference method.

    Args:
        results_matrix: dict {method_name: [score_seed0, score_seed1, ...]}
        reference: name of the reference method
    """
    if not HAS_SCIPY:
        return {'error': 'scipy not installed'}

    ref_scores = np.array(results_matrix[reference])
    comparisons = {}

    for method, scores in results_matrix.items():
        if method == reference:
            continue
        scores = np.array(scores)
        diff = ref_scores - scores

        # If all differences are zero, skip
        if np.allclose(diff, 0):
            comparisons[method] = {
                'statistic': 0.0,
                'p_value': 1.0,
                'mean_diff': 0.0,
                'note': 'No difference between methods',
            }
            continue

        if len(diff) < 6:
            comparisons[method] = {
                'note': f'Too few observations ({len(diff)}) for Wilcoxon test',
                'mean_diff': float(np.mean(diff)),
            }
            continue

        try:
            stat, p = scipy_stats.wilcoxon(diff, alternative='greater')
            comparisons[method] = {
                'statistic': float(stat),
                'p_value': float(p),
                'significant_005': p < 0.05,
                'mean_diff': float(np.mean(diff)),
                'effect_direction': 'ours_better' if np.mean(diff) > 0 else 'baseline_better',
            }
        except Exception as e:
            comparisons[method] = {'error': str(e)}

    return {
        'test': 'Wilcoxon signed-rank (one-sided: ours > baseline)',
        'reference': reference,
        'comparisons': comparisons,
    }


def cohens_d(group1, group2):
    """Effect size: Cohen's d between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def effect_size_analysis(results_matrix, reference='fedper_prox_xai'):
    """Compute Cohen's d effect size for each baseline vs. reference."""
    ref_scores = results_matrix[reference]
    effects = {}
    for method, scores in results_matrix.items():
        if method == reference:
            continue
        d = cohens_d(ref_scores, scores)
        magnitude = 'negligible'
        if abs(d) >= 0.8:
            magnitude = 'large'
        elif abs(d) >= 0.5:
            magnitude = 'medium'
        elif abs(d) >= 0.2:
            magnitude = 'small'
        effects[method] = {
            'cohens_d': d,
            'magnitude': magnitude,
            'direction': 'ours_better' if d > 0 else 'baseline_better',
        }
    return effects


# ─────────────────────────────────────────────────────────────────────────
# 2. Convergence analysis
# ─────────────────────────────────────────────────────────────────────────

def analyze_convergence(results_dir: Path):
    """Analyze per-round convergence from metrics.json files."""
    convergence = {}

    for network_dir in sorted(results_dir.iterdir()):
        if not network_dir.is_dir():
            continue
        net_name = network_dir.name
        client_metrics = {}

        for client_dir in sorted(network_dir.iterdir()):
            if not client_dir.is_dir() or not client_dir.name.startswith('client_'):
                continue
            metrics_file = client_dir / 'metrics.json'
            if not metrics_file.exists():
                continue

            with open(metrics_file) as f:
                data = json.load(f)

            # Extract per-round data if available
            if isinstance(data, list):
                client_metrics[client_dir.name] = data
            elif isinstance(data, dict) and 'rounds' in data:
                client_metrics[client_dir.name] = data['rounds']

        if client_metrics:
            convergence[net_name] = client_metrics

    return convergence


# ─────────────────────────────────────────────────────────────────────────
# 3. Non-IID impact analysis
# ─────────────────────────────────────────────────────────────────────────

def non_iid_impact_analysis(results_dir: Path):
    """Analyze how Non-IID data distribution affects per-client performance."""
    analysis = {}

    for network_dir in sorted(results_dir.iterdir()):
        if not network_dir.is_dir():
            continue
        net_name = network_dir.name
        client_scores = {'accuracy': [], 'f1_macro': [], 'mcc': []}
        client_classes = {}

        for client_dir in sorted(network_dir.iterdir()):
            if not client_dir.is_dir() or not client_dir.name.startswith('client_'):
                continue

            eval_file = client_dir / 'evaluation_metrics.json'
            if eval_file.exists():
                with open(eval_file) as f:
                    em = json.load(f)
                cls_metrics = em.get('classification', em)
                for k in client_scores:
                    if k in cls_metrics:
                        client_scores[k].append(cls_metrics[k])
                if 'classes_present' in cls_metrics:
                    client_classes[client_dir.name] = cls_metrics['classes_present']

        if any(len(v) > 0 for v in client_scores.values()):
            net_analysis = {}
            for k, vals in client_scores.items():
                if vals:
                    net_analysis[k] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'min': float(np.min(vals)),
                        'max': float(np.max(vals)),
                        'range': float(np.max(vals) - np.min(vals)),
                        'cv': float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0,
                    }
            net_analysis['client_class_distribution'] = client_classes
            net_analysis['n_clients'] = len(client_scores['accuracy']) if client_scores['accuracy'] else 0
            analysis[net_name] = net_analysis

    return analysis


# ─────────────────────────────────────────────────────────────────────────
# 4. Communication-accuracy trade-off
# ─────────────────────────────────────────────────────────────────────────

def comm_accuracy_tradeoff(results_dir: Path, model_params_backbone: int = 48192,
                            model_params_total: int = 50503, precision: int = 4):
    """Analyze the trade-off between communication cost and accuracy."""
    methods = {
        'FedAvg': model_params_total * precision,       # full model shared
        'FedProx': model_params_total * precision,      # full model shared
        'FedPer': model_params_backbone * precision,    # backbone only
        'FedPer-Prox-XAI': model_params_backbone * precision,  # backbone only
    }

    tradeoff = {}
    for method, bytes_per_round in methods.items():
        tradeoff[method] = {
            'params_shared': bytes_per_round // precision,
            'bytes_per_round_per_client': bytes_per_round,
            'kb_per_round_per_client': bytes_per_round / 1024,
            'comm_reduction_vs_fedavg': 1 - bytes_per_round / (model_params_total * precision),
        }

    return tradeoff


# ─────────────────────────────────────────────────────────────────────────
# 5. Generate LaTeX statistical report
# ─────────────────────────────────────────────────────────────────────────

def generate_statistical_latex(friedman_result, wilcoxon_result, effect_sizes, out_path):
    """Generate LaTeX table with statistical test results."""
    lines = [
        '% Requires \\usepackage{booktabs}',
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Statistical significance analysis of FedPer-Prox-XAI '
        'versus baseline methods (F1-Macro metric).}',
        '\\label{tab:stat_significance}',
        '\\begin{tabular}{lcccl}',
        '\\toprule',
        "Method & $\\Delta$ F1 & Cohen's $d$ & $p$-value & Significance \\\\",
        '\\midrule',
    ]

    methods_order = ['centralized', 'local_only', 'fedavg', 'fedprox', 'fedper']
    method_labels = {
        'centralized': 'Centralized',
        'local_only': 'Local-only',
        'fedavg': 'FedAvg',
        'fedprox': 'FedProx',
        'fedper': 'FedPer',
    }

    for method in methods_order:
        label = method_labels.get(method, method)
        w = wilcoxon_result.get('comparisons', {}).get(method, {})
        e = effect_sizes.get(method, {})

        delta = w.get('mean_diff', 0)
        d_val = e.get('cohens_d', 0)
        p_val = w.get('p_value', 1.0)

        if isinstance(p_val, float) and p_val < 0.01:
            sig = '$\\ast\\ast$'
        elif isinstance(p_val, float) and p_val < 0.05:
            sig = '$\\ast$'
        else:
            sig = 'n.s.'

        p_str = f'{p_val:.4f}' if isinstance(p_val, float) else str(p_val)
        lines.append(f'{label} & {delta:+.4f} & {d_val:.3f} & {p_str} & {sig} \\\\')

    lines += [
        '\\midrule',
    ]

    # Add Friedman test info
    if 'chi2_statistic' in friedman_result:
        lines.append(
            f'\\multicolumn{{5}}{{l}}{{Friedman $\\chi^2 = '
            f'{friedman_result["chi2_statistic"]:.2f}$, '
            f'$p = {friedman_result["p_value"]:.4f}$}} \\\\'
        )

    lines += [
        '\\bottomrule',
        '\\multicolumn{5}{l}{\\footnotesize $\\ast$ $p < 0.05$, '
        '$\\ast\\ast$ $p < 0.01$, n.s. = not significant} \\\\',
        '\\end{tabular}',
        '\\end{table}',
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def generate_non_iid_latex(analysis, out_path):
    """Generate LaTeX table showing Non-IID impact on performance variance."""
    lines = [
        '% Requires \\usepackage{booktabs}',
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Impact of Non-IID data distribution on per-client performance '
        'variance across network conditions.}',
        '\\label{tab:noniid_impact}',
        '\\begin{tabular}{lccccc}',
        '\\toprule',
        'Network & Acc (mean) & Acc (std) & F1 (mean) & F1 (std) & CV \\\\',
        '\\midrule',
    ]

    for net_name, data in sorted(analysis.items()):
        acc = data.get('accuracy', {})
        f1 = data.get('f1_macro', {})
        cv = acc.get('cv', 0)
        lines.append(
            f'{net_name} & {acc.get("mean",0):.3f} & {acc.get("std",0):.3f} '
            f'& {f1.get("mean",0):.3f} & {f1.get("std",0):.3f} '
            f'& {cv:.3f} \\\\'
        )

    lines += [
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}',
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='R3.8: Statistical analysis & additional results')
    p.add_argument('--results-dir', type=str, default='./results',
                   help='Root results directory')
    p.add_argument('--sota-json', type=str, default=None,
                   help='Path to sota_comparison_full.json for significance tests')
    p.add_argument('--out-dir', type=str, default='./results/statistical_analysis')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    print("=" * 70)
    print("  R3.8 — STATISTICAL ANALYSIS & ADDITIONAL RESULTS")
    print("=" * 70)

    full_report = {}

    # ── 1. Statistical tests on SOTA comparison ──────────────────────────
    sota_path = Path(args.sota_json) if args.sota_json else results_dir / 'sota_comparison' / 'sota_comparison_full.json'
    if sota_path.exists():
        print(f"\n1. STATISTICAL SIGNIFICANCE TESTS")
        print(f"   Source: {sota_path}")
        with open(sota_path) as f:
            sota_data = json.load(f)

        # Build results matrix: {method: [f1_score_per_seed]}
        methods = ['centralized', 'local_only', 'fedavg', 'fedprox', 'fedper', 'fedper_prox_xai']
        metric_key = 'f1_macro'

        results_matrix = defaultdict(list)
        for seed_result in sota_data:
            for method in methods:
                if method in seed_result:
                    results_matrix[method].append(seed_result[method].get(metric_key, 0))

        if all(len(v) >= 2 for v in results_matrix.values()):
            # Friedman test
            friedman = friedman_test(dict(results_matrix))
            print(f"\n   Friedman test:")
            if 'chi2_statistic' in friedman:
                print(f"     chi2 = {friedman['chi2_statistic']:.4f}, p = {friedman['p_value']:.4f}")
                print(f"     Significant at 0.05: {friedman['significant_005']}")
            else:
                print(f"     {friedman.get('note', friedman.get('error', 'N/A'))}")

            # Wilcoxon
            wilcoxon = wilcoxon_pairwise(dict(results_matrix))
            print(f"\n   Wilcoxon signed-rank tests (vs. FedPer-Prox-XAI):")
            for method, res in wilcoxon.get('comparisons', {}).items():
                p_val = res.get('p_value', 'N/A')
                diff = res.get('mean_diff', 0)
                print(f"     {method:<20s}: delta={diff:+.4f}, p={p_val}")

            # Effect sizes
            effects = effect_size_analysis(dict(results_matrix))
            print(f"\n   Effect sizes (Cohen's d):")
            for method, info in effects.items():
                print(f"     {method:<20s}: d={info['cohens_d']:+.3f} ({info['magnitude']})")

            full_report['statistical_tests'] = {
                'friedman': friedman,
                'wilcoxon': wilcoxon,
                'effect_sizes': effects,
                'metric': metric_key,
            }

            # Generate LaTeX
            generate_statistical_latex(friedman, wilcoxon, effects,
                                       out_dir / 'statistical_significance_table.tex')
        else:
            print("   Not enough seeds for statistical tests (need >=2)")
            full_report['statistical_tests'] = {'note': 'Insufficient seeds'}
    else:
        print(f"\n1. STATISTICAL TESTS — skipped (no SOTA results at {sota_path})")
        full_report['statistical_tests'] = {'note': f'No file at {sota_path}'}

    # ── 2. Non-IID impact analysis ───────────────────────────────────────
    print(f"\n2. NON-IID IMPACT ANALYSIS")
    noniid = non_iid_impact_analysis(results_dir)
    if noniid:
        for net, data in noniid.items():
            print(f"\n   Network: {net}")
            for metric in ['accuracy', 'f1_macro', 'mcc']:
                if metric in data:
                    d = data[metric]
                    print(f"     {metric}: mean={d['mean']:.4f} std={d['std']:.4f} "
                          f"range=[{d['min']:.4f}, {d['max']:.4f}] CV={d['cv']:.3f}")
        full_report['non_iid_impact'] = noniid
        generate_non_iid_latex(noniid, out_dir / 'noniid_impact_table.tex')
    else:
        print("   No per-client evaluation metrics found")
        full_report['non_iid_impact'] = {'note': 'No evaluation_metrics.json found'}

    # ── 3. Communication-accuracy trade-off ──────────────────────────────
    print(f"\n3. COMMUNICATION-ACCURACY TRADE-OFF")
    tradeoff = comm_accuracy_tradeoff(results_dir)
    for method, info in tradeoff.items():
        print(f"   {method:<22s}: {info['kb_per_round_per_client']:.1f} KB/round "
              f"(reduction vs FedAvg: {info['comm_reduction_vs_fedavg']:.1%})")
    full_report['comm_accuracy_tradeoff'] = tradeoff

    # ── 4. Convergence analysis ──────────────────────────────────────────
    print(f"\n4. CONVERGENCE ANALYSIS")
    convergence = analyze_convergence(results_dir)
    if convergence:
        for net, clients in convergence.items():
            print(f"   Network {net}: {len(clients)} clients with per-round metrics")
        full_report['convergence'] = {k: {ck: f'{len(cv)} rounds' for ck, cv in v.items()}
                                       for k, v in convergence.items()}
    else:
        print("   No per-round metrics found")
        full_report['convergence'] = {'note': 'No per-round data'}

    # ── 5. Ablation summary ──────────────────────────────────────────────
    print(f"\n5. ABLATION STUDY CONFIGURATION")
    ablation = {
        'variables': {
            'FEDPER_MODE': {
                'description': 'Toggle backbone-only (FedPer) vs full-model (FedAvg) aggregation',
                'values': ['fedper (default)', 'fedavg'],
                'env_var': 'FEDPER_MODE',
            },
            'USE_FOCAL_LOSS': {
                'description': 'Toggle FocalLoss vs standard CrossEntropy',
                'values': ['1 = FocalLoss (default)', '0 = CrossEntropy'],
                'env_var': 'USE_FOCAL_LOSS',
            },
            'mu_prox': {
                'description': 'Proximal term weight (0 = no regularization)',
                'values': ['0.01 (default)', '0.0 (FedPer without prox)'],
                'config': 'training.mu_prox',
            },
        },
        'combinations_tested': [
            'FedAvg (FEDPER_MODE=fedavg, mu=0)',
            'FedProx (FEDPER_MODE=fedavg, mu=0.01)',
            'FedPer (FEDPER_MODE=fedper, mu=0)',
            'FedPer-Prox (FEDPER_MODE=fedper, mu=0.01) [ours]',
            'FedPer-Prox + CE (USE_FOCAL_LOSS=0)',
        ],
    }
    for var, info in ablation['variables'].items():
        print(f"   {var}: {info['description']}")
    full_report['ablation'] = ablation

    # ── Save full report ─────────────────────────────────────────────────
    with open(out_dir / 'statistical_report.json', 'w') as f:
        json.dump(full_report, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  Saved to: {out_dir}/")
    print(f"    statistical_report.json")
    if noniid:
        print(f"    noniid_impact_table.tex")
    if sota_path.exists():
        print(f"    statistical_significance_table.tex")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
