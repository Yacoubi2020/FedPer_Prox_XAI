#!/usr/bin/env python3
"""Lance l'analyse SHAP locale sur un client APRÈS l'entraînement.
Avec affichage de progression en temps réel et mode rapide.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Forcer le mode unbuffered AVANT toute impression
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

from xai import compute_shap_local


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--client-id', type=int, required=True)
    p.add_argument('--network', type=str, required=True,
                   choices=['wifi', '5g', '4g', '3g'])
    p.add_argument('--results-dir', type=str, default='./results')
    p.add_argument('--data-dir', type=str, default='./data/client_datasets')
    p.add_argument('--n-background', type=int, default=50,
                   help='Nb samples background (50=rapide, 100=standard)')
    p.add_argument('--n-test', type=int, default=100,
                   help='Nb samples à expliquer (100=rapide, 300=standard)')
    p.add_argument('--fast', action='store_true',
                   help='Mode rapide avec GradientExplainer (10× plus rapide)')
    p.add_argument('--kernel', action='store_true',
                   help='Utilise KernelExplainer (modèle-agnostique)')
    args = p.parse_args()

    print(f"[SHAP] Démarrage client {args.client_id} - réseau {args.network}", flush=True)

    try:
        compute_shap_local(
            client_id=args.client_id,
            network=args.network,
            results_dir=args.results_dir,
            data_dir=args.data_dir,
            n_background=args.n_background,
            n_test=args.n_test,
            fast_mode=args.fast,
            use_kernel=args.kernel,
        )
    except Exception as e:
        print(f"\n❌ ERREUR pour client {args.client_id}:", flush=True)
        print(f"   {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()