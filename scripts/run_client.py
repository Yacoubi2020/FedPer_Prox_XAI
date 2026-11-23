# """Client Flower FedPer-Prox-XAI avec support ablation study + multi-seed (R1.1)."""
# import os, sys, argparse
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import flwr as fl
# from typing import Dict, List, Tuple
# from flwr.common import NDArrays, Scalar
# from pathlib import Path

# from config import FLConfig
# from models import FedPerProxModel, compute_proximal_loss, FocalLoss
# from monitoring import EnergyMonitor, CommMonitor, MetricsLogger


# class FedPerProxClient(fl.client.NumPyClient):
#     def __init__(self, client_id, train_loader, test_loader, client_stats, config):
#         self.client_id = client_id
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.client_stats = client_stats
#         self.config = config
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # Modèle
#         self.model = FedPerProxModel(
#             input_dim=config.model.input_dim,
#             num_classes=config.model.num_classes,
#             backbone_hidden=config.model.backbone_hidden,
#             head_hidden=config.model.head_hidden,
#             dropout=config.model.dropout,
#         ).to(self.device)

#         # Pondération de classe
#         class_counts = np.zeros(config.model.num_classes)
#         for _, y in train_loader:
#             for c in y.numpy():
#                 class_counts[c] += 1
#         weights = 1.0 / (class_counts + 1e-6)
#         weights = weights / weights.sum() * config.model.num_classes
#         self.class_weights = torch.FloatTensor(weights).to(self.device)

#         # ─── Ablation : choix de la loss ─────────────────────────────────
#         use_focal = os.environ.get('USE_FOCAL_LOSS', '1') == '1'
#         if use_focal:
#             self.criterion = FocalLoss(alpha=self.class_weights, gamma=2.0)
#             print(f"  🔧 Loss : FocalLoss (γ=2.0)")
#         else:
#             self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
#             print(f"  🔧 Loss : CrossEntropyLoss")

#         # ─── Ablation : mode FedPer vs FedAvg ────────────────────────────
#         fedper_val = os.environ.get('FEDPER_MODE', '1').lower().strip()
#         self.fedper_mode = fedper_val in ('1', 'true', 'yes', 'fedper')
#         if self.fedper_mode:
#             print(f"  🔧 Mode : FedPer (backbone partagé + head perso)")
#         else:
#             print(f"  🔧 Mode : FedAvg (tout le modèle partagé)")

#         # Monitoring
#         self.energy_monitor = EnergyMonitor(
#             tdp_watts=config.energy.tdp_watts,
#             use_hardware_sensor=config.energy.use_hardware_sensor,
#             voltage_v=config.energy.voltage_v,
#         )
#         self.comm_monitor = CommMonitor(network_type=config.network.network_type)
#         self.logger = MetricsLogger(client_id, config.logging.results_dir)

#         self._global_backbone_params = None
#         self._current_round = 0

#         print(f"✅ Client {client_id} initialisé | {client_stats['n_samples']:,} samples "
#               f"| réseau: {config.network.network_type.upper()}")

#     # ── Flower API ──────────────────────────────────────────────────────────

#     def get_parameters(self, config):
#         if self.fedper_mode:
#             return self.model.get_backbone_params()
#         else:
#             # Mode FedAvg : on partage TOUT le modèle (backbone + head)
#             return [p.detach().cpu().numpy() for p in self.model.parameters()]

#     def set_parameters(self, parameters):
#         """Convertit numpy→Tensor et met à jour le(s) module(s)."""
#         tensors = [torch.from_numpy(p.copy()) if isinstance(p, np.ndarray)
#                    else p.clone().detach()
#                    for p in parameters]

#         if self.fedper_mode:
#             self.model.set_backbone_params(tensors)
#             self._global_backbone_params = [t.clone().detach().to(self.device) for t in tensors]
#         else:
#             # Tout le modèle
#             for p_model, p_new in zip(self.model.parameters(), tensors):
#                 p_model.data.copy_(p_new.to(p_model.device))
#             n_bb = len(list(self.model.backbone.parameters()))
#             self._global_backbone_params = [t.clone().detach().to(self.device)
#                                             for t in tensors[:n_bb]]

#     def fit(self, parameters, config):
#         import time
#         round_start = time.time()
#         server_round = config.get('server_round', self._current_round + 1)
#         self._current_round = server_round
#         local_epochs = int(config.get('local_epochs', self.config.training.local_epochs))

#         print(f"\n  🏋️  Client {self.client_id} | Round {server_round} | {local_epochs} epochs")

#         # Phase download
#         with self.energy_monitor.measure('download'):
#             self.set_parameters(parameters)
#             self.comm_monitor.track_download(server_round, parameters)

#         # Phase training
#         with self.energy_monitor.measure('training'):
#             train_loss, train_acc = self._train(local_epochs)

#         # Phase upload
#         new_params = self.get_parameters(config)
#         with self.energy_monitor.measure('upload'):
#             self.comm_monitor.track_upload(server_round, new_params)

#         # Logging
#         e_sum = self.energy_monitor.get_summary()
#         c_sum = self.comm_monitor.get_summary()
#         round_time = time.time() - round_start

#         print(f"     loss={train_loss:.4f} | acc={train_acc:.4f} | "
#               f"E={e_sum['total_energy_j']:.2f}J | Comm={c_sum['total_mb_compressed']:.3f}MB")

#         self.logger.log_round(server_round, {
#             'train_loss': train_loss, 'train_acc': train_acc,
#             'energy_j': e_sum['total_energy_j'],
#             'comm_mb': c_sum['total_mb_compressed'],
#             'round_time_s': round_time,
#         })

#         return new_params, self.client_stats['n_samples'], {
#             'loss': float(train_loss), 'accuracy': float(train_acc),
#             'energy_j': float(e_sum['total_energy_j']),
#             'comm_mb': float(c_sum['total_mb_compressed']),
#         }

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         with self.energy_monitor.measure('evaluation'):
#             test_loss, test_acc = self._evaluate()
#         # Compute F1-macro for federated evaluation
#         self.model.eval()
#         all_preds, all_labels = [], []
#         with torch.no_grad():
#             for X, y in self.test_loader:
#                 X = X.to(self.device)
#                 preds = self.model(X).argmax(1).cpu().numpy()
#                 all_preds.extend(preds)
#                 all_labels.extend(y.numpy())
#         from sklearn.metrics import f1_score
#         f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
#         return float(test_loss), self.client_stats['n_samples'], {
#             'accuracy': float(test_acc),
#             'f1_macro': float(f1),
#         }

#     # ── Helpers ─────────────────────────────────────────────────────────────

#     def _train(self, local_epochs):
#         self.model.train()
#         optimizer = optim.AdamW(
#             self.model.parameters(),
#             lr=self.config.training.lr,
#             weight_decay=self.config.training.weight_decay,
#         )
#         total_loss = total_correct = total_n = 0
#         for _ in range(local_epochs):
#             for X, y in self.train_loader:
#                 X, y = X.to(self.device), y.to(self.device)
#                 optimizer.zero_grad()
#                 logits = self.model(X)
#                 task_loss = self.criterion(logits, y)
#                 prox_loss = compute_proximal_loss(
#                     self.model, self._global_backbone_params,
#                     self.config.training.mu_prox,
#                 )
#                 loss = task_loss + prox_loss
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#                 optimizer.step()

#                 total_loss += task_loss.item() * y.size(0)
#                 total_correct += (logits.argmax(1) == y).sum().item()
#                 total_n += y.size(0)
#         return total_loss / total_n, total_correct / total_n

#     def _evaluate(self):
#         self.model.eval()
#         total_loss = total_correct = total_n = 0
#         with torch.no_grad():
#             for X, y in self.test_loader:
#                 X, y = X.to(self.device), y.to(self.device)
#                 logits = self.model(X)
#                 loss = self.criterion(logits, y)
#                 total_loss += loss.item() * y.size(0)
#                 total_correct += (logits.argmax(1) == y).sum().item()
#                 total_n += y.size(0)
#         return total_loss / total_n, total_correct / total_n

#     def save_model(self, path):
#         torch.save({
#             'backbone': self.model.backbone.state_dict(),
#             'head': self.model.head.state_dict(),
#             'stats': self.client_stats,
#         }, path)

#     def compute_full_metrics(self, embeddings_save_path=None):
#         """Compute comprehensive evaluation metrics (classification + representation)."""
#         from monitoring.advanced_metrics import compute_all_metrics
#         all_logits, all_labels, all_embeds = self._collect_outputs()

#         metrics = compute_all_metrics(
#             all_logits, all_labels, all_embeds,
#             class_names=self.config.data.ALL_CLASSES,
#         )

#         if embeddings_save_path and all_embeds is not None:
#             np.savez(embeddings_save_path,
#                      embeddings=all_embeds, labels=all_labels)

#         return metrics

#     def _collect_outputs(self):
#         """Collect predictions, labels and embeddings from test set."""
#         self.model.eval()
#         all_logits, all_labels, all_embeds = [], [], []
#         with torch.no_grad():
#             for X, y in self.test_loader:
#                 X = X.to(self.device)
#                 logits = self.model(X)
#                 embed = self.model.backbone(X)
#                 all_logits.append(logits.cpu().numpy())
#                 all_labels.append(y.numpy())
#                 all_embeds.append(embed.cpu().numpy())
#         return (np.concatenate(all_logits),
#                 np.concatenate(all_labels),
#                 np.concatenate(all_embeds))


# # ── Main entry point ──────────────────────────────────────────────────────

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument('--client-id', type=int, required=True)
#     p.add_argument('--server-addr', type=str, default='127.0.0.1:8080')
#     p.add_argument('--data-dir', type=str, default='./data/client_datasets')
#     p.add_argument('--network', type=str, default='wifi',
#                    choices=['wifi', '5g', '4g', '3g'])
#     p.add_argument('--results-dir', type=str, default='./results')
#     p.add_argument('--tdp', type=float, default=15.0)
#     p.add_argument('--use-sensor', action='store_true')
#     args = p.parse_args()

#     config = FLConfig()
#     config.apply_seed()  # R1.1: apply seed for reproducibility

#     config.client.client_id = args.client_id
#     config.client.server_address = args.server_addr
#     config.network.network_type = args.network
#     config.energy.tdp_watts = args.tdp
#     config.energy.use_hardware_sensor = args.use_sensor
#     config.logging.results_dir = str(Path(args.results_dir) / args.network)

#     print("=" * 70)
#     print(f"  CLIENT {args.client_id} - FedPer-Prox-XAI (seed={config.seed})")
#     print("=" * 70)
#     print(f"  Serveur : {args.server_addr}")
#     print(f"  Reseau  : {args.network.upper()}")
#     print(f"  Seed    : {config.seed}")
#     print("=" * 70)

#     # Load data — try the data module first, fallback to inline loader
#     print(f"\n  Chargement donnees client {args.client_id}...")
#     try:
#         from data import load_client_partition
#         train_loader, test_loader, stats = load_client_partition(
#             client_id=args.client_id,
#             data_dir=args.data_dir,
#             batch_size=config.training.batch_size,
#         )
#         print(f"  [OK] Chargement via data.load_client_partition")
#     except (ImportError, Exception) as e:
#         print(f"  [INFO] data module indisponible ({e}), fallback inline loader")
#         import pickle
#         client_dir = Path(args.data_dir) / f'client_{args.client_id}'
#         try:
#             X_all = np.load(client_dir / 'X.npy')
#             y_all = np.load(client_dir / 'y.npy')
#         except FileNotFoundError as e2:
#             print(f"  [ERROR] Donnees introuvables: {e2}")
#             print(f"  Executez d'abord: python scripts/prepare_data.py")
#             sys.exit(1)

#         # Load stats if available
#         stats_path = client_dir / 'stats.pkl'
#         saved_stats = {}
#         if stats_path.exists():
#             with open(stats_path, 'rb') as f:
#                 saved_stats = pickle.load(f)

#         # Train/test split (80/20, stratified, deterministic with seed)
#         from sklearn.model_selection import train_test_split
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_all, y_all, test_size=0.2, random_state=config.seed, stratify=y_all)

#         train_ds = torch.utils.data.TensorDataset(
#             torch.FloatTensor(X_train), torch.LongTensor(y_train))
#         test_ds = torch.utils.data.TensorDataset(
#             torch.FloatTensor(X_test), torch.LongTensor(y_test))
#         train_loader = torch.utils.data.DataLoader(
#             train_ds, batch_size=config.training.batch_size, shuffle=True)
#         test_loader = torch.utils.data.DataLoader(
#             test_ds, batch_size=config.training.batch_size, shuffle=False)

#         present_classes = sorted(set(y_all.tolist()))
#         stats = {
#             'input_dim': X_all.shape[1],
#             'n_samples': len(X_all),
#             'n_train': len(X_train),
#             'n_test': len(X_test),
#             'class_names': saved_stats.get('class_names',
#                            [config.data.ALL_CLASSES[c] for c in present_classes
#                             if c < len(config.data.ALL_CLASSES)]),
#             'classes_present': present_classes,
#         }

#     print(f"  {stats['n_samples']:,} samples | features={stats['input_dim']}")
#     print(f"  Classes: {stats['class_names']}")

#     config.model.input_dim = stats['input_dim']
#     config.model.num_classes = len(config.data.ALL_CLASSES)

#     # Create client
#     client = FedPerProxClient(
#         client_id=args.client_id,
#         train_loader=train_loader, test_loader=test_loader,
#         client_stats=stats, config=config,
#     )

#     print(f"  Connexion a {args.server_addr}...\n")
#     fl.client.start_numpy_client(
#         server_address=args.server_addr,
#         client=client,
#         grpc_max_message_length=config.client.grpc_max_message_length,
#     )

#     # Save final model + report
#     out_dir = Path(config.logging.results_dir) / f'client_{args.client_id}'
#     out_dir.mkdir(parents=True, exist_ok=True)
#     client.save_model(out_dir / 'final_model.pt')

#     # Full evaluation
#     print(f"\n  Calcul des metriques d'evaluation completes...")
#     try:
#         eval_metrics = client.compute_full_metrics(
#             embeddings_save_path=out_dir / 'embeddings_sample.npz')
#         client.logger.save_evaluation_metrics(eval_metrics)
#         clf = eval_metrics.get('classification', {})
#         rep = eval_metrics.get('representation', {})
#         print(f"    F1-macro={clf.get('f1_macro')} | "
#               f"F1-weighted={clf.get('f1_weighted')} | MCC={clf.get('mcc')}")
#         print(f"    Separabilite={rep.get('separability_ratio')} | "
#               f"silhouette={rep.get('silhouette_score')}")
#     except Exception as e:
#         print(f"    [WARN] Evaluation avancee echouee: {e}")
#         print(f"    Fallback: calcul des metriques de base...")
#         import traceback; traceback.print_exc()
#         try:
#             from sklearn.metrics import (accuracy_score, f1_score,
#                                          matthews_corrcoef, precision_score,
#                                          recall_score, classification_report)
#             import torch as _th
#             _model = client.model
#             _model.eval()
#             _ys, _ps = [], []
#             with _th.no_grad():
#                 for _X, _y in client.test_loader:
#                     _X = _X.to(client.device)
#                     _logits = _model(_X)
#                     _ps.append(_logits.argmax(1).cpu().numpy())
#                     _ys.append(_y.numpy())
#             y_true = np.concatenate(_ys)
#             y_pred = np.concatenate(_ps)
#             num_classes = client.config.model.num_classes
#             class_names = list(client.config.data.ALL_CLASSES[:num_classes])
#             acc = accuracy_score(y_true, y_pred)
#             f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
#             f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
#             mcc = matthews_corrcoef(y_true, y_pred)
#             prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
#             rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
#             report = classification_report(y_true, y_pred, target_names=class_names,
#                                            zero_division=0, output_dict=True)
#             eval_metrics = {
#                 'classification': {
#                     'accuracy': acc,
#                     'f1_macro': f1_mac,
#                     'f1_weighted': f1_w,
#                     'mcc': mcc,
#                     'precision_macro': prec,
#                     'recall_macro': rec,
#                     'classification_report': report,
#                 },
#                 'representation': {},
#             }
#             client.logger.save_evaluation_metrics(eval_metrics)
#             print(f"    F1-macro={f1_mac:.4f} | F1-weighted={f1_w:.4f} | MCC={mcc:.4f}")
#             print(f"    Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f}")
#         except Exception as e2:
#             print(f"    [ERREUR] Fallback aussi echoue: {e2}")
#             import traceback; traceback.print_exc()

#     client.logger.save_final_report({
#         'seed': config.seed,
#         'energy_summary': client.energy_monitor.get_summary(),
#         'comm_summary': client.comm_monitor.get_summary(),
#         'client_stats': {k: v for k, v in stats.items() if k != 'X'},
#     })

#     print(f"\n  Client {args.client_id} termine (seed={config.seed})!")
#     print(f"    Energie : {client.energy_monitor.get_total_energy_j():.2f}J")
#     print(f"    Comm    : {client.comm_monitor.get_total_mb():.3f}MB")


# if __name__ == '__main__':
#     main()

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# See scripts/csv_to_npy.py for the FEDPER_DATA_ROOT convention.
DATA_ROOT = os.environ.get('FEDPER_DATA_ROOT', 'data')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from typing import Dict, List, Tuple
from flwr.common import NDArrays, Scalar
from pathlib import Path

from config import FLConfig
from models import FedPerProxModel, compute_proximal_loss, FocalLoss
from monitoring import EnergyMonitor, CommMonitor, MetricsLogger


class FedPerProxClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_loader, test_loader, client_stats, config):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_stats = client_stats
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Modèle
        self.model = FedPerProxModel(
            input_dim=config.model.input_dim,
            num_classes=config.model.num_classes,
            backbone_hidden=config.model.backbone_hidden,
            head_hidden=config.model.head_hidden,
            dropout=config.model.dropout,
        ).to(self.device)

        # Pondération de classe
        class_counts = np.zeros(config.model.num_classes)
        for _, y in train_loader:
            for c in y.numpy():
                class_counts[c] += 1
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * config.model.num_classes
        self.class_weights = torch.FloatTensor(weights).to(self.device)

        # ─── Ablation : choix de la loss ─────────────────────────────────
        use_focal = os.environ.get('USE_FOCAL_LOSS', '1') == '1'
        if use_focal:
            self.criterion = FocalLoss(alpha=self.class_weights, gamma=2.0)
            print(f"  🔧 Loss : FocalLoss (γ=2.0)")
        else:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            print(f"  🔧 Loss : CrossEntropyLoss")

        # ─── Ablation : mode FedPer vs FedAvg ────────────────────────────
        fedper_val = os.environ.get('FEDPER_MODE', '1').lower().strip()
        self.fedper_mode = fedper_val in ('1', 'true', 'yes', 'fedper')
        if self.fedper_mode:
            print(f"  🔧 Mode : FedPer (backbone partagé + head perso)")
        else:
            print(f"  🔧 Mode : FedAvg (tout le modèle partagé)")

        # Monitoring
        self.energy_monitor = EnergyMonitor(
            tdp_watts=config.energy.tdp_watts,
            use_hardware_sensor=config.energy.use_hardware_sensor,
            voltage_v=config.energy.voltage_v,
        )
        self.comm_monitor = CommMonitor(network_type=config.network.network_type)
        self.logger = MetricsLogger(client_id, config.logging.results_dir)

        self._global_backbone_params = None
        self._current_round = 0

        print(f"✅ Client {client_id} initialisé | {client_stats['n_samples']:,} samples "
              f"| réseau: {config.network.network_type.upper()}")

    # ── Flower API ──────────────────────────────────────────────────────────

    def get_parameters(self, config):
        if self.fedper_mode:
            return self.model.get_backbone_params()
        else:
            # Mode FedAvg : on partage TOUT le modèle (backbone + head)
            return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        """Convertit numpy→Tensor et met à jour le(s) module(s)."""
        tensors = [torch.from_numpy(p.copy()) if isinstance(p, np.ndarray)
                   else p.clone().detach()
                   for p in parameters]

        if self.fedper_mode:
            self.model.set_backbone_params(tensors)
            self._global_backbone_params = [t.clone().detach().to(self.device) for t in tensors]
        else:
            # Tout le modèle
            for p_model, p_new in zip(self.model.parameters(), tensors):
                p_model.data.copy_(p_new.to(p_model.device))
            n_bb = len(list(self.model.backbone.parameters()))
            self._global_backbone_params = [t.clone().detach().to(self.device)
                                            for t in tensors[:n_bb]]

    def fit(self, parameters, config):
        import time
        round_start = time.time()
        server_round = config.get('server_round', self._current_round + 1)
        self._current_round = server_round
        local_epochs = int(config.get('local_epochs', self.config.training.local_epochs))

        print(f"\n  🏋️  Client {self.client_id} | Round {server_round} | {local_epochs} epochs")

        # Phase download
        with self.energy_monitor.measure('download'):
            self.set_parameters(parameters)
            self.comm_monitor.track_download(server_round, parameters)

        # Phase training
        with self.energy_monitor.measure('training'):
            train_loss, train_acc = self._train(local_epochs)

        # Phase upload
        new_params = self.get_parameters(config)
        with self.energy_monitor.measure('upload'):
            self.comm_monitor.track_upload(server_round, new_params)

        # Logging
        e_sum = self.energy_monitor.get_summary()
        c_sum = self.comm_monitor.get_summary()
        round_time = time.time() - round_start

        print(f"     loss={train_loss:.4f} | acc={train_acc:.4f} | "
              f"E={e_sum['total_energy_j']:.2f}J | Comm={c_sum['total_mb_compressed']:.3f}MB")

        self.logger.log_round(server_round, {
            'train_loss': train_loss, 'train_acc': train_acc,
            'energy_j': e_sum['total_energy_j'],
            'comm_mb': c_sum['total_mb_compressed'],
            'round_time_s': round_time,
        })

        return new_params, self.client_stats['n_samples'], {
            'loss': float(train_loss), 'accuracy': float(train_acc),
            'energy_j': float(e_sum['total_energy_j']),
            'comm_mb': float(c_sum['total_mb_compressed']),
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        with self.energy_monitor.measure('evaluation'):
            test_loss, test_acc = self._evaluate()
        # Compute F1-macro for federated evaluation
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                preds = self.model(X).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        return float(test_loss), self.client_stats['n_samples'], {
            'accuracy': float(test_acc),
            'f1_macro': float(f1),
        }

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _train(self, local_epochs):
        self.model.train()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        total_loss = total_correct = total_n = 0
        for _ in range(local_epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X)
                task_loss = self.criterion(logits, y)
                prox_loss = compute_proximal_loss(
                    self.model, self._global_backbone_params,
                    self.config.training.mu_prox,
                )
                loss = task_loss + prox_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += task_loss.item() * y.size(0)
                total_correct += (logits.argmax(1) == y).sum().item()
                total_n += y.size(0)
        return total_loss / total_n, total_correct / total_n

    def _evaluate(self):
        self.model.eval()
        total_loss = total_correct = total_n = 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * y.size(0)
                total_correct += (logits.argmax(1) == y).sum().item()
                total_n += y.size(0)
        return total_loss / total_n, total_correct / total_n

    def save_model(self, path):
        torch.save({
            'backbone': self.model.backbone.state_dict(),
            'head': self.model.head.state_dict(),
            'stats': self.client_stats,
        }, path)

    def compute_full_metrics(self, embeddings_save_path=None):
        """Compute comprehensive evaluation metrics (classification + representation)."""
        from monitoring.advanced_metrics import compute_all_metrics
        all_logits, all_labels, all_embeds = self._collect_outputs()

        # ── Handle label skew: clients may have a subset of classes ──
        present = sorted(set(all_labels.astype(int).tolist()))
        n_model_classes = all_logits.shape[1]

        if len(present) < n_model_classes:
            # Keep only logit columns for present classes & remap labels 0..K-1
            class_names = [self.config.data.ALL_CLASSES[c]
                           for c in present if c < len(self.config.data.ALL_CLASSES)]
            remap = {old: new for new, old in enumerate(present)}
            all_labels = np.array([remap[int(l)] for l in all_labels])
            all_logits = all_logits[:, present]
            # Also slice embeddings-save labels (original labels for analysis)
            original_labels = all_labels.copy()
        else:
            class_names = list(self.config.data.ALL_CLASSES[:n_model_classes])
            original_labels = all_labels

        metrics = compute_all_metrics(
            all_logits, all_labels, all_embeds,
            class_names=class_names,
        )

        if embeddings_save_path and all_embeds is not None:
            np.savez(embeddings_save_path,
                     embeddings=all_embeds, labels=original_labels)

        return metrics

    def _collect_outputs(self):
        """Collect predictions, labels and embeddings from test set."""
        self.model.eval()
        all_logits, all_labels, all_embeds = [], [], []
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                logits = self.model(X)
                embed = self.model.backbone(X)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(y.numpy())
                all_embeds.append(embed.cpu().numpy())
        return (np.concatenate(all_logits),
                np.concatenate(all_labels),
                np.concatenate(all_embeds))


# ── Main entry point ──────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--client-id', type=int, required=True)
    p.add_argument('--server-addr', type=str, default='127.0.0.1:8080')
    p.add_argument('--data-dir', type=str, default=f'{DATA_ROOT}/client_datasets')
    p.add_argument('--network', type=str, default='wifi',
                   choices=['wifi', '5g', '4g', '3g'])
    p.add_argument('--results-dir', type=str, default='./results')
    p.add_argument('--tdp', type=float, default=15.0)
    p.add_argument('--use-sensor', action='store_true')
    args = p.parse_args()

    config = FLConfig()
    config.apply_seed()  # R1.1: apply seed for reproducibility

    config.client.client_id = args.client_id
    config.client.server_address = args.server_addr
    config.network.network_type = args.network
    config.energy.tdp_watts = args.tdp
    config.energy.use_hardware_sensor = args.use_sensor
    config.logging.results_dir = str(Path(args.results_dir) / args.network)

    print("=" * 70)
    print(f"  CLIENT {args.client_id} - FedPer-Prox-XAI (seed={config.seed})")
    print("=" * 70)
    print(f"  Serveur : {args.server_addr}")
    print(f"  Reseau  : {args.network.upper()}")
    print(f"  Seed    : {config.seed}")
    print("=" * 70)

    # Load data — try the data module first, fallback to inline loader
    print(f"\n  Chargement donnees client {args.client_id}...")
    try:
        from data import load_client_partition
        train_loader, test_loader, stats = load_client_partition(
            client_id=args.client_id,
            data_dir=args.data_dir,
            batch_size=config.training.batch_size,
        )
        print(f"  [OK] Chargement via data.load_client_partition")
    except (ImportError, Exception) as e:
        print(f"  [INFO] data module indisponible ({e}), fallback inline loader")
        import pickle
        client_dir = Path(args.data_dir) / f'client_{args.client_id}'
        try:
            X_all = np.load(client_dir / 'X.npy')
            y_all = np.load(client_dir / 'y.npy')
        except FileNotFoundError as e2:
            print(f"  [ERROR] Donnees introuvables: {e2}")
            print(f"  Executez d'abord: python scripts/prepare_data.py")
            sys.exit(1)

        # Load stats if available
        stats_path = client_dir / 'stats.pkl'
        saved_stats = {}
        if stats_path.exists():
            with open(stats_path, 'rb') as f:
                saved_stats = pickle.load(f)

        # Train/test split (80/20, stratified, deterministic with seed)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=config.seed, stratify=y_all)

        train_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_ds = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.LongTensor(y_test))
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.training.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=config.training.batch_size, shuffle=False)

        present_classes = sorted(set(y_all.tolist()))
        stats = {
            'input_dim': X_all.shape[1],
            'n_samples': len(X_all),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'class_names': saved_stats.get('class_names',
                           [config.data.ALL_CLASSES[c] for c in present_classes
                            if c < len(config.data.ALL_CLASSES)]),
            'classes_present': present_classes,
        }

    print(f"  {stats['n_samples']:,} samples | features={stats['input_dim']}")
    print(f"  Classes: {stats['class_names']}")

    config.model.input_dim = stats['input_dim']
    config.model.num_classes = len(config.data.ALL_CLASSES)
    _max_lbl = max(int(y.max()) for _, y in train_loader)
    if _max_lbl >= config.model.num_classes:
        config.model.num_classes = _max_lbl + 1
        print(f"  [INFO] Label skew: num_classes -> {config.model.num_classes}")

    # Create client
    client = FedPerProxClient(
        client_id=args.client_id,
        train_loader=train_loader, test_loader=test_loader,
        client_stats=stats, config=config,
    )

    print(f"  Connexion a {args.server_addr}...\n")
    fl.client.start_numpy_client(
        server_address=args.server_addr,
        client=client,
        grpc_max_message_length=config.client.grpc_max_message_length,
    )

    # Save final model + report
    out_dir = Path(config.logging.results_dir) / f'client_{args.client_id}'
    out_dir.mkdir(parents=True, exist_ok=True)
    client.save_model(out_dir / 'final_model.pt')

    # Full evaluation
    print(f"\n  Calcul des metriques d'evaluation completes...")
    try:
        eval_metrics = client.compute_full_metrics(
            embeddings_save_path=out_dir / 'embeddings_sample.npz')
        client.logger.save_evaluation_metrics(eval_metrics)
        clf = eval_metrics.get('classification', {})
        rep = eval_metrics.get('representation', {})
        print(f"    F1-macro={clf.get('f1_macro')} | "
              f"F1-weighted={clf.get('f1_weighted')} | MCC={clf.get('mcc')}")
        print(f"    Separabilite={rep.get('separability_ratio')} | "
              f"silhouette={rep.get('silhouette_score')}")
    except Exception as e:
        print(f"    [WARN] Evaluation avancee echouee: {e}")
        print(f"    Fallback: calcul des metriques de base...")
        import traceback; traceback.print_exc()
        try:
            from sklearn.metrics import (accuracy_score, f1_score,
                                         matthews_corrcoef, precision_score,
                                         recall_score, classification_report)
            import torch as _th
            _model = client.model
            _model.eval()
            _ys, _ps = [], []
            with _th.no_grad():
                for _X, _y in client.test_loader:
                    _X = _X.to(client.device)
                    _logits = _model(_X)
                    _ps.append(_logits.argmax(1).cpu().numpy())
                    _ys.append(_y.numpy())
            y_true = np.concatenate(_ys)
            y_pred = np.concatenate(_ps)
            # Detect actual classes present (handles label skew)
            present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
            all_classes = list(client.config.data.ALL_CLASSES)
            class_names = [all_classes[c] for c in present_labels
                           if c < len(all_classes)]
            acc = accuracy_score(y_true, y_pred)
            f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            report = classification_report(y_true, y_pred,
                                           labels=present_labels,
                                           target_names=class_names,
                                           zero_division=0, output_dict=True)
            eval_metrics = {
                'classification': {
                    'accuracy': acc,
                    'f1_macro': f1_mac,
                    'f1_weighted': f1_w,
                    'mcc': mcc,
                    'precision_macro': prec,
                    'recall_macro': rec,
                    'classification_report': report,
                },
                'representation': {},
            }
            client.logger.save_evaluation_metrics(eval_metrics)
            print(f"    F1-macro={f1_mac:.4f} | F1-weighted={f1_w:.4f} | MCC={mcc:.4f}")
            print(f"    Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f}")
        except Exception as e2:
            print(f"    [ERREUR] Fallback aussi echoue: {e2}")
            import traceback; traceback.print_exc()

    client.logger.save_final_report({
        'seed': config.seed,
        'energy_summary': client.energy_monitor.get_summary(),
        'comm_summary': client.comm_monitor.get_summary(),
        'client_stats': {k: v for k, v in stats.items() if k != 'X'},
    })

    print(f"\n  Client {args.client_id} termine (seed={config.seed})!")
    print(f"    Energie : {client.energy_monitor.get_total_energy_j():.2f}J")
    print(f"    Comm    : {client.comm_monitor.get_total_mb():.3f}MB")


if __name__ == '__main__':
    main()