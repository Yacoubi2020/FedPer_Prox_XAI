#!/usr/bin/env python3
"""Lance le serveur Flower FedPer-Prox-XAI."""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import flwr as fl
from config import FLConfig
from flower_components import FedPerProxStrategy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=30)
    p.add_argument('--network', type=str, default='5g', choices=['wifi', '5g', '4g', '3g'])
    p.add_argument('--address', type=str, default='0.0.0.0:8080')
    p.add_argument('--min-clients', type=int, default=4)
    p.add_argument('--local-epochs', type=int, default=3)
    p.add_argument('--mu', type=float, default=0.01)
    p.add_argument('--temp', type=float, default=0.5)
    p.add_argument('--results-dir', type=str, default='./results')
    args = p.parse_args()

    config = FLConfig()
    config.apply_seed()  # R1.1: apply seed for reproducibility
    config.training.num_rounds = args.rounds
    config.training.local_epochs = args.local_epochs
    config.training.mu_prox = args.mu
    config.training.aggregation_temp = args.temp
    config.network.network_type = args.network
    config.logging.results_dir = args.results_dir

    out_dir = Path(args.results_dir) / args.network / 'server'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"🖥️  SERVEUR FLOWER - FedPer-Prox-XAI")
    print("=" * 70)
    print(f"  Adresse     : {args.address}")
    print(f"  Rounds      : {args.rounds}")
    print(f"  Min clients : {args.min_clients}")
    print(f"  Réseau      : {args.network.upper()}")
    print(f"  FedProx μ   : {args.mu}")
    print(f"  Agg. temp   : {args.temp}")
    print(f"  Epochs/round: {args.local_epochs}")
    print(f"  Seed        : {config.seed}")
    print("=" * 70)
    print("\n⏳ En attente des clients...\n")

    strategy = FedPerProxStrategy(
        aggregation_temp=args.temp,
        local_epochs=args.local_epochs,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 1024,
    )

    # Sauvegarder l'historique
    with open(out_dir / 'server_history.json', 'w') as f:
        json.dump(strategy.round_history, f, indent=2)
    print(f"\n✅ Historique serveur sauvegardé : {out_dir/'server_history.json'}")


if __name__ == '__main__':
    main()
