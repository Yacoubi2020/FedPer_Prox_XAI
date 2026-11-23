#!/usr/bin/env python3
"""Génère TOUS les graphes après les expériences (training + SHAP + multi-réseau)."""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

from visualization import (
    load_all_client_shap, load_training_history, load_evaluation_metrics,
    load_embedding_samples,
    plot_training_history, plot_shap_per_client, plot_shap_consensus,
    plot_global_importance, plot_attack_profiles, plot_network_comparison,
    plot_confusion_matrices, plot_per_class_metrics,
    plot_separability, plot_metrics_summary, plot_embedding_projection,
    plot_embedding_client_invariance,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--network', type=str, default=None)
    p.add_argument('--results-dir', type=str, default='./results')
    p.add_argument('--n-clients', type=int, default=4)
    p.add_argument('--clients', type=int, nargs='+', default=None,
                   help="IDs explicites des clients à tracer (ex: --clients 0 1 2). "
                        "Ignore les dossiers client_* résiduels d'anciens runs.")
    p.add_argument('--proj', type=str, default='tsne', choices=['tsne', 'umap', 'none'],
                   help="Méthode de projection des embeddings (t-SNE par défaut, "
                        "UMAP si umap-learn installé, none pour désactiver).")
    p.add_argument('--perplexity', type=float, default=30,
                   help="Perplexité t-SNE (40-50 = amas plus compacts). Sans effet sur UMAP.")
    p.add_argument('--all-networks', action='store_true')
    args = p.parse_args()

    networks = ['wifi', '5g', '4g', '3g'] if args.all_networks else [args.network or '5g']

    def _filter(d):
        """Restreint un dict {client_id: ...} aux IDs demandés via --clients."""
        if args.clients is None:
            return d
        return {k: v for k, v in d.items() if k in args.clients}

    for net in networks:
        print(f"\n{'='*70}\n📊 Réseau : {net.upper()}\n{'='*70}")
        out = Path(args.results_dir) / net / 'figures'
        out.mkdir(parents=True, exist_ok=True)

        history = _filter(load_training_history(net, args.results_dir))
        shap_results = _filter(load_all_client_shap(net, args.results_dir))
        eval_metrics = _filter(load_evaluation_metrics(net, args.results_dir))
        embed_samples = _filter(load_embedding_samples(net, args.results_dir))

        if args.clients is not None:
            print(f"  🎯 Clients retenus : {sorted(set(args.clients))}")

        if history:
            plot_training_history(history, out / 'training_history.png')
        if shap_results:
            plot_shap_per_client(shap_results, out / 'shap_per_client.png')
            plot_shap_consensus(shap_results, out / 'shap_consensus.png')
            plot_global_importance(shap_results, out / 'global_importance.png')
            plot_attack_profiles(shap_results, out / 'attack_profiles.png')
        if eval_metrics:
            plot_confusion_matrices(eval_metrics, out / 'confusion_matrices.png')
            plot_per_class_metrics(eval_metrics, out / 'per_class_metrics.png')
            plot_separability(eval_metrics, out / 'separability.png')
            plot_metrics_summary(eval_metrics, out / 'metrics_summary.png')
        if embed_samples and args.proj != 'none':
            plot_embedding_projection(embed_samples, out / f'embedding_{args.proj}.png',
                                      method=args.proj, perplexity=args.perplexity)
            plot_embedding_client_invariance(
                embed_samples, out / f'embedding_invariance_{args.proj}.png',
                method=args.proj, perplexity=args.perplexity)
        if not history and not shap_results and not eval_metrics:
            print(f"  ⚠️  Aucun résultat trouvé pour {net}")

    # Comparaison multi-réseau
    if args.all_networks:
        print(f"\n{'='*70}\n📊 COMPARAISON MULTI-RÉSEAU\n{'='*70}")
        compare_dir = Path(args.results_dir) / 'comparison_figures'
        compare_dir.mkdir(parents=True, exist_ok=True)
        netcompare = {}
        for net in networks:
            netcompare[net] = {}
            for path in sorted((Path(args.results_dir) / net).glob('client_*/final_summary.json')):
                cid = int(path.parent.name.split('_')[1])
                if args.clients is not None and cid not in args.clients:
                    continue
                with open(path) as f:
                    netcompare[net][cid] = json.load(f)
        if any(netcompare.values()):
            plot_network_comparison(netcompare, compare_dir / 'network_comparison.png')
        else:
            print("  ⚠️  Aucun résumé final trouvé pour la comparaison")

    print(f"\n✅ Tous les graphes générés dans {args.results_dir}/")


if __name__ == '__main__':
    main()
