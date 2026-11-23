"""Client Flower FedPer-Prox-XAI."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from typing import Dict, List, Tuple
from flwr.common import NDArrays, Scalar

from models import FedPerProxModel, compute_proximal_loss, FocalLoss
from monitoring import EnergyMonitor, CommMonitor, MetricsLogger
from monitoring.advanced_metrics import compute_all_metrics


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

        # Pondération de classe (Focal Loss)
        class_counts = np.zeros(config.model.num_classes)
        for _, y in train_loader:
            for c in y.numpy():
                class_counts[c] += 1
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * config.model.num_classes
        self.class_weights = torch.FloatTensor(weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

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
        return self.model.get_backbone_params()

    def set_parameters(self, parameters):
        """Convertit numpy→Tensor et met à jour backbone + snapshot pour FedProx."""
        tensors = [torch.from_numpy(p.copy()) if isinstance(p, np.ndarray)
                   else p.clone().detach()
                   for p in parameters]
        self.model.set_backbone_params(tensors)
        self._global_backbone_params = [t.clone().detach().to(self.device) for t in tensors]

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
        new_params = self.model.get_backbone_params()
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
        server_round = int(config.get('server_round', self._current_round))
        with self.energy_monitor.measure('evaluation'):
            y_true, y_pred, y_prob, _ = self._collect_outputs(with_embeddings=False)

        test_loss = self._cross_entropy(y_true, y_prob)
        test_acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
        f1_macro = self._quick_f1_macro(y_true, y_pred)

        # Logging du round d'évaluation (alimente test_loss/test_acc, jusque-là vides)
        self.logger.log_round(server_round, {
            'test_loss': float(test_loss), 'test_acc': float(test_acc),
        })

        return float(test_loss), self.client_stats['n_samples'], {
            'accuracy': float(test_acc),
            'f1_macro': float(f1_macro),
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

    def _collect_outputs(self, with_embeddings: bool = True):
        """Passe une fois sur le test set et renvoie (y_true, y_pred, y_prob, embeddings).

        Les embeddings sont la sortie du backbone partagé — base des métriques
        de séparabilité inter/intra-classe dans l'espace de représentation.
        """
        self.model.eval()
        ys, preds, probs, embs = [], [], [], []
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                feats = self.model.backbone(X)            # représentation latente
                logits = self.model.head(feats)
                prob = torch.softmax(logits, dim=1)
                ys.append(y.numpy())
                preds.append(prob.argmax(1).cpu().numpy())
                probs.append(prob.cpu().numpy())
                if with_embeddings:
                    embs.append(feats.cpu().numpy())
        y_true = np.concatenate(ys) if ys else np.array([])
        y_pred = np.concatenate(preds) if preds else np.array([])
        y_prob = np.concatenate(probs) if probs else np.array([])
        embeddings = np.concatenate(embs) if (with_embeddings and embs) else None
        return y_true, y_pred, y_prob, embeddings

    @staticmethod
    def _cross_entropy(y_true, y_prob):
        """CE moyenne à partir des probas (équivalent au test_loss historique)."""
        if len(y_true) == 0:
            return 0.0
        p = np.clip(y_prob[np.arange(len(y_true)), y_true], 1e-12, 1.0)
        return float(-np.log(p).mean())

    @staticmethod
    def _quick_f1_macro(y_true, y_pred):
        """F1-macro léger (sur classes présentes) sans dépendance lourde par round."""
        if len(y_true) == 0:
            return 0.0
        f1s = []
        for c in np.unique(y_true):
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))
            denom = 2 * tp + fp + fn
            f1s.append((2 * tp / denom) if denom > 0 else 0.0)
        return float(np.mean(f1s)) if f1s else 0.0

    def compute_full_metrics(self, embeddings_save_path=None, max_embed_points=2000):
        """Évaluation complète finale : classification + séparabilité de représentation.

        Renvoie le dict {'classification': {...}, 'representation': {...}} produit par
        monitoring.advanced_metrics.compute_all_metrics. Utilisé à la fin de
        l'entraînement pour le rapport publiable.

        Si embeddings_save_path est fourni, un échantillon stratifié d'embeddings
        du backbone (pour t-SNE / UMAP) est sauvegardé au format .npz.
        """
        # IMPORTANT : les labels sont encodés en indices GLOBAUX (0..num_classes-1,
        # voir data_loader: y = global_classes.index(c)). On doit donc passer la liste
        # GLOBALE complète des classes, alignée sur ces indices — surtout pas le
        # sous-ensemble local du client (client_stats['class_names']), qui désaligne
        # les noms et fait disparaître les classes 4,5,6... des métriques par classe.
        num_classes = self.config.model.num_classes
        class_names = list(self.config.data.ALL_CLASSES[:num_classes])
        y_true, y_pred, y_prob, embeddings = self._collect_outputs(with_embeddings=True)
        if embeddings_save_path is not None and embeddings is not None:
            self._save_embedding_sample(
                embeddings_save_path, embeddings, y_true, class_names, max_embed_points)
        return compute_all_metrics(
            y_true=y_true, y_pred=y_pred, y_prob=y_prob,
            embeddings=embeddings, class_names=class_names,
            num_classes=num_classes,
        )

    @staticmethod
    def _save_embedding_sample(path, embeddings, y_true, class_names, max_points):
        """Sous-échantillonne (stratifié par classe) et sauvegarde X, y, noms en .npz."""
        embeddings = np.asarray(embeddings)
        y_true = np.asarray(y_true)
        n = len(y_true)
        if n > max_points:
            rng = np.random.default_rng(42)
            # échantillonnage stratifié : proportionnel par classe, >= 1 par classe
            idx_all = []
            classes, counts = np.unique(y_true, return_counts=True)
            for c, cnt in zip(classes, counts):
                ci = np.where(y_true == c)[0]
                take = max(1, int(round(max_points * cnt / n)))
                take = min(take, len(ci))
                idx_all.append(rng.choice(ci, size=take, replace=False))
            idx = np.concatenate(idx_all)
            rng.shuffle(idx)
            embeddings, y_true = embeddings[idx], y_true[idx]
        np.savez_compressed(
            str(path), embeddings=embeddings.astype(np.float32),
            labels=y_true.astype(np.int64), class_names=np.array(class_names, dtype=object))

    def save_model(self, path):
        torch.save({
            'backbone': self.model.backbone.state_dict(),
            'head': self.model.head.state_dict(),
            'stats': self.client_stats,
        }, path)
