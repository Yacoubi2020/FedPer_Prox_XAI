"""Stratégie Flower FedPer-Prox avec agrégation soft-capped."""
import numpy as np
from typing import List, Tuple, Dict, Optional
import flwr as fl
from flwr.common import Parameters, Scalar, FitRes, FitIns, EvaluateIns, EvaluateRes
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedPerProxStrategy(FedAvg):
    """
    Agrégation soft-capped : pondération par n_samples^temperature.
      temp=1.0 → pondération classique (par n_samples)
      temp=0.0 → moyenne égalitaire
      temp=0.5 → compromis (recommandé pour Non-IID)
    """

    def __init__(self, aggregation_temp: float = 0.5, local_epochs: int = 3, **kwargs):
        # Forcer l'utilisation de TOUS les clients
        kwargs.setdefault('fraction_fit', 1.0)
        kwargs.setdefault('fraction_evaluate', 1.0)
        super().__init__(**kwargs)
        self.aggregation_temp = aggregation_temp
        self.local_epochs = local_epochs
        self.round_history = []

    def configure_fit(self, server_round, parameters, client_manager):
        """Sélectionne TOUS les clients disponibles pour le fit."""
        config = {
            'server_round': server_round,
            'local_epochs': self.local_epochs,
        }
        fit_ins = FitIns(parameters, config)

        # Attendre que tous les clients soient prêts
        client_manager.wait_for(num_clients=self.min_fit_clients, timeout=600)

        # Échantillonner TOUS les clients disponibles
        n_available = client_manager.num_available()
        clients = client_manager.sample(
            num_clients=n_available,
            min_num_clients=self.min_fit_clients,
        )
        print(f"  📡 Round {server_round}: {len(clients)} clients sélectionnés / {n_available} disponibles")
        return [(c, fit_ins) for c in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Sélectionne TOUS les clients pour l'évaluation."""
        evaluate_ins = EvaluateIns(parameters, {'server_round': server_round})
        client_manager.wait_for(num_clients=self.min_evaluate_clients, timeout=600)
        n_available = client_manager.num_available()
        clients = client_manager.sample(
            num_clients=n_available,
            min_num_clients=self.min_evaluate_clients,
        )
        return [(c, evaluate_ins) for c in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        if failures:
            print(f"  ⚠️  {len(failures)} échec(s) côté client")

        # Calcul des poids soft-capped
        weights_list = []
        params_list = []
        client_metrics = []
        for client_proxy, fit_res in results:
            n = fit_res.num_examples
            w = n ** self.aggregation_temp
            weights_list.append(w)
            params_list.append(parameters_to_ndarrays(fit_res.parameters))
            client_metrics.append({
                'num_examples': n,
                'metrics': fit_res.metrics,
            })

        # Normalisation
        total_w = sum(weights_list)
        weights_list = [w / total_w for w in weights_list]

        # Agrégation pondérée des layers backbone
        n_layers = len(params_list[0])
        agg = []
        for layer_idx in range(n_layers):
            layer = sum(w * p[layer_idx] for w, p in zip(weights_list, params_list))
            agg.append(layer)

        # Métriques agrégées
        agg_loss = np.mean([m['metrics'].get('loss', 0) for m in client_metrics])
        agg_acc = np.mean([m['metrics'].get('accuracy', 0) for m in client_metrics])
        total_energy = sum(m['metrics'].get('energy_j', 0) for m in client_metrics)
        total_comm = sum(m['metrics'].get('comm_mb', 0) for m in client_metrics)

        print(f"\n  📊 Round {server_round} → {len(results)} clients | "
              f"agg_loss={agg_loss:.4f} | agg_acc={agg_acc:.4f} | "
              f"total_E={total_energy:.2f}J | total_Comm={total_comm:.3f}MB")

        self.round_history.append({
            'round': server_round,
            'n_clients': len(results),
            'agg_loss': float(agg_loss),
            'agg_acc': float(agg_acc),
            'total_energy_j': float(total_energy),
            'total_comm_mb': float(total_comm),
        })

        return ndarrays_to_parameters(agg), {
            'agg_loss': float(agg_loss),
            'agg_acc': float(agg_acc),
            'total_energy_j': float(total_energy),
            'total_comm_mb': float(total_comm),
        }

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        total_n = sum(r.num_examples for _, r in results)
        losses = [r.loss * r.num_examples for _, r in results]
        accs = [r.metrics.get('accuracy', 0) * r.num_examples for _, r in results]
        f1s = [r.metrics.get('f1_macro', 0) * r.num_examples for _, r in results]
        agg_acc = sum(accs) / total_n
        agg_f1 = sum(f1s) / total_n
        print(f"  📈 Round {server_round} eval → acc={agg_acc:.4f} | f1_macro={agg_f1:.4f}")
        return sum(losses) / total_n, {
            'accuracy': agg_acc,
            'f1_macro': agg_f1,
        }