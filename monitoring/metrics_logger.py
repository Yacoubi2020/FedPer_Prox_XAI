"""Logger central pour métriques par round (sauvegarde JSON pour visualisation)."""
import json
import time
from pathlib import Path
from typing import Dict, Any


class MetricsLogger:
    def __init__(self, client_id: int, results_dir: str):
        self.client_id = client_id
        self.results_dir = Path(results_dir) / f'client_{client_id}'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': [],
            'energy_j': [], 'comm_mb': [],
            'round_time_s': [],
        }
        self.start_time = time.time()

    def log_round(self, round_num: int, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if k in self.history:
                self.history[k].append(v)
        # Sauvegarde incrémentale
        with open(self.results_dir / 'metrics.json', 'w') as f:
            json.dump(self.history, f, indent=2)

    def save_evaluation_metrics(self, eval_metrics: Dict[str, Any]):
        """Sauvegarde les métriques d'évaluation complètes (classification + représentation)."""
        self.evaluation_metrics = eval_metrics
        with open(self.results_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(eval_metrics, f, indent=2, default=str)

    def save_final_report(self, final_metrics: Dict[str, Any]):
        report = {
            'client_id': self.client_id,
            'total_time_s': time.time() - self.start_time,
            'history': self.history,
            'final_metrics': final_metrics,
            'evaluation_metrics': getattr(self, 'evaluation_metrics', None),
        }
        with open(self.results_dir / 'final_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        clf = (getattr(self, 'evaluation_metrics', None) or {}).get('classification', {})
        rep = (getattr(self, 'evaluation_metrics', None) or {}).get('representation', {})
        with open(self.results_dir / 'final_summary.json', 'w') as f:
            json.dump({
                'client_id': self.client_id,
                'final_acc': self.history['train_acc'][-1] if self.history['train_acc'] else 0,
                'final_loss': self.history['train_loss'][-1] if self.history['train_loss'] else 0,
                'test_acc': clf.get('accuracy'),
                'f1_macro': clf.get('f1_macro'),
                'f1_weighted': clf.get('f1_weighted'),
                'mcc': clf.get('mcc'),
                'separability_ratio': rep.get('separability_ratio'),
                'silhouette_score': rep.get('silhouette_score'),
                'energy_j': final_metrics.get('energy_summary', {}).get('total_energy_j', 0),
                'comm_mb': final_metrics.get('comm_summary', {}).get('total_mb_compressed', 0),
                'total_rounds': len(self.history['train_acc']),
            }, f, indent=2)
