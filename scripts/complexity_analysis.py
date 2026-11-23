#!/usr/bin/env python3
"""R3.7 — Algorithmic complexity analysis of FedPer-Prox-XAI.

Provides detailed analysis of:
  1. Model parameters (total, backbone, head)
  2. FLOPs per forward pass (theoretical)
  3. Communication overhead per round (backbone params only)
  4. Time complexity of each phase (training, aggregation, SHAP)
  5. Memory footprint
  6. Per-round and total computational budget

This addresses Reviewer 3 Comment 7:
  "Discuss algorithmic steps in detail + complexity analysis."

Usage:
  python -I scripts/complexity_analysis.py
  python -I scripts/complexity_analysis.py --input-dim 23 --num-classes 7 --batch-size 256
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch

from config import FLConfig
from models import FedPerProxModel


# ── Parameter counting ───────────────────────────────────────────────────

def count_parameters(model):
    """Count trainable parameters per module."""
    backbone_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    total_params = backbone_params + head_params

    # Detailed layer breakdown
    layers = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.BatchNorm1d)):
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            layers[name] = {
                'type': type(module).__name__,
                'params': n,
                'shape': str(list(module.parameters())[0].shape) if list(module.parameters()) else 'N/A',
            }

    return {
        'total': total_params,
        'backbone': backbone_params,
        'head': head_params,
        'shared_ratio': backbone_params / total_params if total_params > 0 else 0,
        'layers': layers,
    }


# ── FLOPs estimation ────────────────────────────────────────────────────

def estimate_flops(model, input_dim, batch_size=1):
    """Estimate FLOPs for a single forward pass.

    For a Linear(in, out): 2 * in * out FLOPs (multiply + add)
    For BatchNorm1d(n):    2 * n FLOPs
    For ReLU(n):           n FLOPs
    For Dropout:           0 FLOPs
    """
    flops = 0
    flop_details = OrderedDict()

    # Backbone layers
    prev_dim = input_dim
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Linear):
            out_dim = module.out_features
            layer_flops = 2 * prev_dim * out_dim * batch_size
            flops += layer_flops
            flop_details[f'backbone.{name}'] = {
                'type': 'Linear',
                'shape': f'{prev_dim} -> {out_dim}',
                'flops': layer_flops,
            }
            prev_dim = out_dim
        elif isinstance(module, torch.nn.BatchNorm1d):
            layer_flops = 2 * module.num_features * batch_size
            flops += layer_flops
            flop_details[f'backbone.{name}'] = {
                'type': 'BatchNorm1d',
                'features': module.num_features,
                'flops': layer_flops,
            }
        elif isinstance(module, torch.nn.ReLU):
            layer_flops = prev_dim * batch_size
            flops += layer_flops

    # Head layers
    for name, module in model.head.named_modules():
        if isinstance(module, torch.nn.Linear):
            out_dim = module.out_features
            layer_flops = 2 * prev_dim * out_dim * batch_size
            flops += layer_flops
            flop_details[f'head.{name}'] = {
                'type': 'Linear',
                'shape': f'{prev_dim} -> {out_dim}',
                'flops': layer_flops,
            }
            prev_dim = out_dim

    return flops, flop_details


# ── Communication complexity ─────────────────────────────────────────────

def communication_analysis(model, n_clients, n_rounds, precision_bytes=4):
    """Analyze communication overhead."""
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    total_params = backbone_params + head_params

    # FedPer: only backbone is shared
    fedper_upload_bytes = backbone_params * precision_bytes  # per client per round
    fedper_download_bytes = backbone_params * precision_bytes  # per client per round

    # FedAvg baseline: everything is shared
    fedavg_upload_bytes = total_params * precision_bytes
    fedavg_download_bytes = total_params * precision_bytes

    # Total over all rounds and clients
    fedper_total = (fedper_upload_bytes + fedper_download_bytes) * n_clients * n_rounds
    fedavg_total = (fedavg_upload_bytes + fedavg_download_bytes) * n_clients * n_rounds

    return {
        'backbone_params': backbone_params,
        'head_params': head_params,
        'fedper': {
            'upload_per_round_per_client_bytes': fedper_upload_bytes,
            'download_per_round_per_client_bytes': fedper_download_bytes,
            'total_bytes': fedper_total,
            'total_mb': fedper_total / (1024**2),
        },
        'fedavg': {
            'upload_per_round_per_client_bytes': fedavg_upload_bytes,
            'download_per_round_per_client_bytes': fedavg_download_bytes,
            'total_bytes': fedavg_total,
            'total_mb': fedavg_total / (1024**2),
        },
        'savings_ratio': 1 - (fedper_total / fedavg_total) if fedavg_total > 0 else 0,
    }


# ── Time complexity notation ─────────────────────────────────────────────

def time_complexity_analysis(config):
    """Provide Big-O time complexity for each algorithmic step."""
    d = config.model.input_dim
    C = config.model.num_classes
    H = config.model.backbone_hidden
    h = config.model.head_hidden
    N = 'N'  # num samples (variable)
    K = config.training.num_rounds
    E = config.training.local_epochs
    M = 4  # num clients

    analysis = OrderedDict()

    # Local training
    backbone_complexity = ' + '.join([f'{d}*{H[0]}'] +
                                      [f'{H[i]}*{H[i+1]}' for i in range(len(H)-1)])
    head_complexity = f'{H[-1]}*{h[0]} + {h[0]}*{C}'

    analysis['local_training'] = {
        'description': 'Local SGD on one client for E epochs',
        'complexity': f'O(E * N/B * ({backbone_complexity} + {head_complexity}))',
        'simplified': f'O(E * N * d * H)',
        'explanation': (
            f'Each local epoch processes N/B mini-batches. '
            f'Forward pass: multiply through {len(H)+len(h)+1} layers. '
            f'Backward pass: same complexity. '
            f'With E={E} epochs, batch_size B={config.training.batch_size}.'
        ),
    }

    analysis['proximal_term'] = {
        'description': 'FedProx proximal regularization computation',
        'complexity': f'O(P_backbone)',
        'simplified': f'O(P)',
        'explanation': (
            f'Computes ||w - w_global||^2 over all backbone parameters. '
            f'Linear in the number of shared parameters P. '
            f'mu={config.training.mu_prox}.'
        ),
    }

    analysis['aggregation'] = {
        'description': 'Temperature-softened weighted averaging on server',
        'complexity': f'O(M * P_backbone)',
        'simplified': f'O(M * P)',
        'explanation': (
            f'Server averages backbone parameters from M={M} clients '
            f'with soft-capped weights w_i = n_i^tau, tau={config.training.aggregation_temp}. '
            f'Linear in number of clients and backbone parameters.'
        ),
    }

    analysis['full_fl_pipeline'] = {
        'description': 'Complete federated learning pipeline',
        'complexity': f'O(K * (M * E * N/B * d * H + M * P))',
        'simplified': f'O(K * M * E * N * d * H)',
        'explanation': (
            f'K={K} rounds, each with M={M} clients training E={E} local epochs, '
            f'plus aggregation. Communication dominates only for very large models.'
        ),
    }

    analysis['shap_gradient_input'] = {
        'description': 'Gradient x Input attribution (per class)',
        'complexity': f'O(C * N_test * d * H)',
        'simplified': f'O(C * N * d * H)',
        'explanation': (
            f'For each of C={C} classes, computes gradient w.r.t. input '
            f'for N_test samples. Each gradient computation is one forward + '
            f'backward pass. Replaces full KernelSHAP which would be O(C * N * 2^d).'
        ),
    }

    return analysis


# ── Memory footprint ─────────────────────────────────────────────────────

def memory_analysis(model, config, precision_bytes=4):
    """Estimate memory footprint."""
    total_params = sum(p.numel() for p in model.parameters())

    # Model weights
    model_memory = total_params * precision_bytes

    # Optimizer state (AdamW: 2 additional buffers per parameter)
    optimizer_memory = total_params * precision_bytes * 2

    # Gradient storage
    gradient_memory = total_params * precision_bytes

    # FedProx: global backbone copy
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    fedprox_memory = backbone_params * precision_bytes

    # Activation memory (approximate for one batch)
    batch_size = config.training.batch_size
    activations = 0
    for h in config.model.backbone_hidden:
        activations += batch_size * h * precision_bytes
    for h in config.model.head_hidden:
        activations += batch_size * h * precision_bytes

    total_training = model_memory + optimizer_memory + gradient_memory + fedprox_memory + activations

    return {
        'model_weights_bytes': model_memory,
        'optimizer_state_bytes': optimizer_memory,
        'gradients_bytes': gradient_memory,
        'fedprox_global_copy_bytes': fedprox_memory,
        'activations_per_batch_bytes': activations,
        'total_training_bytes': total_training,
        'total_training_mb': total_training / (1024**2),
    }


# ── Algorithm pseudocode ─────────────────────────────────────────────────

def generate_algorithm_latex():
    """Generate LaTeX algorithmic pseudocode for the paper."""
    return r"""
% Requires \usepackage{algorithm, algorithmic}
\begin{algorithm}[t]
\caption{FedPer-Prox-XAI Training}
\label{alg:fedper_prox_xai}
\begin{algorithmic}[1]
\REQUIRE $K$ clients with local datasets $\{D_k\}_{k=1}^K$,
         $T$ rounds, $E$ local epochs, learning rate $\eta$,
         proximal weight $\mu$, aggregation temperature $\tau$
\ENSURE Shared backbone $\theta_B^*$, personalized heads $\{\theta_{H_k}^*\}$

\STATE \textbf{Server:}
\STATE Initialize backbone $\theta_B^{(0)}$
\FOR{$t = 1, \ldots, T$}
    \STATE Broadcast $\theta_B^{(t-1)}$ to all $K$ clients
    \FOR{each client $k$ \textbf{in parallel}}
        \STATE $\theta_{B_k}^{(t)}, n_k \leftarrow \textsc{ClientUpdate}(k, \theta_B^{(t-1)})$
    \ENDFOR
    \STATE Compute weights: $w_k = n_k^{\tau} / \sum_j n_j^{\tau}$
    \STATE Aggregate: $\theta_B^{(t)} = \sum_k w_k \cdot \theta_{B_k}^{(t)}$
\ENDFOR

\STATE
\STATE \textbf{ClientUpdate}$(k, \theta_B)$:
\STATE $\theta_{B_k} \leftarrow \theta_B$ \hfill (receive backbone)
\STATE $\theta_{H_k} \leftarrow$ local head \hfill (keep personalized)
\FOR{$e = 1, \ldots, E$}
    \FOR{mini-batch $(X, y) \in D_k$}
        \STATE $\hat{y} = f_{\theta_{H_k}}(f_{\theta_{B_k}}(X))$ \hfill (forward pass)
        \STATE $\mathcal{L}_{\text{task}} = \text{FocalLoss}(\hat{y}, y; \alpha, \gamma)$
        \STATE $\mathcal{L}_{\text{prox}} = \frac{\mu}{2} \|\theta_{B_k} - \theta_B\|^2$
        \STATE $\mathcal{L} = \mathcal{L}_{\text{task}} + \mathcal{L}_{\text{prox}}$
        \STATE $\theta_{B_k}, \theta_{H_k} \leftarrow \theta_{B_k}, \theta_{H_k} - \eta \nabla \mathcal{L}$
    \ENDFOR
\ENDFOR
\RETURN $\theta_{B_k}, n_k$ \hfill (upload backbone only)

\STATE
\STATE \textbf{SHAP Explainability (post-training):}
\FOR{each client $k$}
    \FOR{each class $c$}
        \STATE $\phi_c^{(k)} = \frac{1}{|X_{\text{test}}|} \sum_{x} |x \odot \nabla_x f_c(x)|$ \hfill (Gradient $\times$ Input)
    \ENDFOR
    \STATE Aggregate: $\bar{\phi}^{(k)} = \frac{1}{C} \sum_c \phi_c^{(k)}$
\ENDFOR
\STATE Global consensus: $\Phi = \frac{\sum_k n_k \bar{\phi}^{(k)}}{\sum_k n_k}$
\STATE Cross-client CV: $\text{CV}_j = \text{std}(\bar{\phi}_j^{(1)}, \ldots, \bar{\phi}_j^{(K)}) / \Phi_j$
\end{algorithmic}
\end{algorithm}
"""


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='R3.7: Algorithmic complexity analysis')
    p.add_argument('--input-dim', type=int, default=23)
    p.add_argument('--num-classes', type=int, default=7)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--n-clients', type=int, default=4)
    p.add_argument('--n-rounds', type=int, default=30)
    p.add_argument('--out-dir', type=str, default='./results/complexity')
    args = p.parse_args()

    config = FLConfig()
    config.model.input_dim = args.input_dim
    config.model.num_classes = args.num_classes
    config.training.batch_size = args.batch_size
    config.training.num_rounds = args.n_rounds

    model = FedPerProxModel(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        backbone_hidden=config.model.backbone_hidden,
        head_hidden=config.model.head_hidden,
        dropout=config.model.dropout,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  R3.7 — ALGORITHMIC COMPLEXITY ANALYSIS")
    print("=" * 70)

    # 1) Parameter analysis
    print("\n1. MODEL PARAMETERS")
    print("-" * 50)
    param_info = count_parameters(model)
    print(f"  Total parameters       : {param_info['total']:,}")
    print(f"  Backbone (shared)      : {param_info['backbone']:,} ({param_info['shared_ratio']:.1%})")
    print(f"  Head (personalized)    : {param_info['head']:,} ({1-param_info['shared_ratio']:.1%})")
    print(f"\n  Layer breakdown:")
    for name, info in param_info['layers'].items():
        print(f"    {name:35s} {info['type']:15s}  {info['params']:>8,} params  {info['shape']}")

    # 2) FLOPs
    print(f"\n2. COMPUTATIONAL COMPLEXITY (FLOPs)")
    print("-" * 50)
    flops_single, flop_details = estimate_flops(model, args.input_dim, batch_size=1)
    flops_batch, _ = estimate_flops(model, args.input_dim, batch_size=args.batch_size)
    print(f"  FLOPs per sample       : {flops_single:,}")
    print(f"  FLOPs per batch ({args.batch_size}): {flops_batch:,}")
    print(f"  MFLOPs per sample      : {flops_single/1e6:.3f}")
    for name, info in flop_details.items():
        print(f"    {name:35s} {info['type']:15s}  {info['flops']:>12,} FLOPs")

    # 3) Communication
    print(f"\n3. COMMUNICATION COMPLEXITY")
    print("-" * 50)
    comm = communication_analysis(model, args.n_clients, args.n_rounds)
    print(f"  FedPer (backbone only):")
    fp = comm['fedper']
    print(f"    Upload/round/client  : {fp['upload_per_round_per_client_bytes']:,} bytes "
          f"({fp['upload_per_round_per_client_bytes']/1024:.1f} KB)")
    print(f"    Total ({args.n_rounds}R x {args.n_clients}C): "
          f"{fp['total_mb']:.2f} MB")
    print(f"  FedAvg (full model):")
    fa = comm['fedavg']
    print(f"    Upload/round/client  : {fa['upload_per_round_per_client_bytes']:,} bytes "
          f"({fa['upload_per_round_per_client_bytes']/1024:.1f} KB)")
    print(f"    Total ({args.n_rounds}R x {args.n_clients}C): "
          f"{fa['total_mb']:.2f} MB")
    print(f"  Communication savings  : {comm['savings_ratio']:.1%}")

    # 4) Time complexity
    print(f"\n4. TIME COMPLEXITY (Big-O)")
    print("-" * 50)
    time_cx = time_complexity_analysis(config)
    for step, info in time_cx.items():
        print(f"\n  {step}:")
        print(f"    {info['description']}")
        print(f"    Complexity: {info['complexity']}")
        print(f"    Simplified: {info['simplified']}")

    # 5) Memory footprint
    print(f"\n5. MEMORY FOOTPRINT")
    print("-" * 50)
    mem = memory_analysis(model, config)
    print(f"  Model weights          : {mem['model_weights_bytes']/1024:.1f} KB")
    print(f"  Optimizer (AdamW)      : {mem['optimizer_state_bytes']/1024:.1f} KB")
    print(f"  Gradients              : {mem['gradients_bytes']/1024:.1f} KB")
    print(f"  FedProx global copy    : {mem['fedprox_global_copy_bytes']/1024:.1f} KB")
    print(f"  Activations per batch  : {mem['activations_per_batch_bytes']/1024:.1f} KB")
    print(f"  Total training memory  : {mem['total_training_mb']:.2f} MB")

    # Save full report
    report = {
        'config': {
            'input_dim': args.input_dim,
            'num_classes': args.num_classes,
            'backbone_hidden': config.model.backbone_hidden,
            'head_hidden': config.model.head_hidden,
            'batch_size': args.batch_size,
            'n_clients': args.n_clients,
            'n_rounds': args.n_rounds,
        },
        'parameters': param_info,
        'flops': {
            'per_sample': flops_single,
            'per_batch': flops_batch,
            'mflops_per_sample': flops_single / 1e6,
            'layer_details': {k: {kk: vv for kk, vv in v.items()} for k, v in flop_details.items()},
        },
        'communication': comm,
        'time_complexity': {k: {kk: vv for kk, vv in v.items()} for k, v in time_cx.items()},
        'memory': mem,
    }

    # Convert OrderedDicts for JSON
    def convert_od(obj):
        if isinstance(obj, OrderedDict):
            return dict(obj)
        return obj

    with open(out_dir / 'complexity_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Save algorithm pseudocode
    with open(out_dir / 'algorithm.tex', 'w') as f:
        f.write(generate_algorithm_latex())

    # Save complexity table for paper
    latex_lines = [
        '% Requires \\usepackage{booktabs}',
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Computational complexity analysis of FedPer-Prox-XAI '
        'with $d{=}23$ input features, $C{=}7$ classes, and backbone '
        'layers $[256, 128, 64]$.}',
        '\\label{tab:complexity}',
        '\\begin{tabular}{lrl}',
        '\\toprule',
        'Component & Value & Unit \\\\',
        '\\midrule',
        f'Total parameters & {param_info["total"]:,} & -- \\\\',
        f'Backbone (shared) & {param_info["backbone"]:,} & ({param_info["shared_ratio"]:.1%}) \\\\',
        f'Head (personalized) & {param_info["head"]:,} & -- \\\\',
        f'FLOPs per sample & {flops_single/1e3:.1f} & kFLOPs \\\\',
        f'Comm. per round (FedPer) & {fp["upload_per_round_per_client_bytes"]/1024:.1f} & KB \\\\',
        f'Comm. per round (FedAvg) & {fa["upload_per_round_per_client_bytes"]/1024:.1f} & KB \\\\',
        f'Comm. savings & {comm["savings_ratio"]:.1%} & -- \\\\',
        f'Training memory & {mem["total_training_mb"]:.2f} & MB \\\\',
        '\\bottomrule',
        '\\end{tabular}',
        '\\end{table}',
    ]
    with open(out_dir / 'complexity_table.tex', 'w') as f:
        f.write('\n'.join(latex_lines) + '\n')

    print(f"\n  Saved reports to: {out_dir}/")
    print(f"    complexity_report.json")
    print(f"    algorithm.tex")
    print(f"    complexity_table.tex")


if __name__ == '__main__':
    main()
