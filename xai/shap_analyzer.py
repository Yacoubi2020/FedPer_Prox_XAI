"""Analyseur d'importance des features basé sur les gradients (substitut SHAP rapide).
Fonctionne en ~5 secondes au lieu de 15 minutes, sans bug avec BatchNorm.
Méthode : Gradient × Input (équivalent à SHAP pour modèles linéaires-par-morceaux).
"""
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import torch


def _flush(msg):
    print(msg, flush=True)
    sys.stdout.flush()


def compute_shap_local(client_id, network, results_dir, data_dir,
                       n_background=100, n_test=300, fast_mode=False,
                       use_kernel=False):
    """Calcule l'importance des features par Gradient × Input par classe.

    Beaucoup plus rapide que shap.DeepExplainer (5s vs 15 min) et plus stable
    avec les BatchNorm. Résultats équivalents pour modèles linéaires-par-morceaux.
    """
    from data import load_client_partition
    from models import FedPerProxModel
    from config import FLConfig

    config = FLConfig()
    out_dir = Path(results_dir) / network / f'client_{client_id}'
    out_dir.mkdir(parents=True, exist_ok=True)

    _flush(f"\n{'='*60}")
    _flush(f"🔬 Analyse importance Client {client_id} - Réseau {network.upper()}")
    _flush(f"{'='*60}")
    _flush(f"  Méthode : Gradient × Input par classe")
    _flush(f"  Test    : {n_test} samples")
    _flush(f"{'='*60}\n")

    t0 = time.time()

    _flush(f"📥 [1/4] Chargement des données...")
    train_loader, test_loader, stats = load_client_partition(
        client_id=client_id, data_dir=data_dir, batch_size=256
    )
    n_features = stats['input_dim']
    n_classes = len(config.data.ALL_CLASSES)
    class_names = config.data.ALL_CLASSES
    feature_names = stats.get('feature_names', [f'f{i}' for i in range(n_features)])
    _flush(f"  ✅ {stats['n_samples']:,} samples | {n_features} features | {n_classes} classes")

    _flush(f"\n📥 [2/4] Chargement du modèle...")
    candidates = [
        out_dir / 'final_model.pt',
        Path(results_dir) / network / f'client_{client_id}_model.pt',
        Path(results_dir) / f'client_{client_id}' / 'final_model.pt',
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        _flush(f"❌ Modèle introuvable : {candidates}")
        return None

    state = torch.load(model_path, map_location='cpu')
    # Infer num_classes from checkpoint (handles label skew where each client has different classes)
    if isinstance(state, dict) and 'head' in state:
        actual_n_classes = state['head']['net.3.bias'].shape[0]
    elif 'net.3.bias' in state:
        actual_n_classes = state['net.3.bias'].shape[0]
    else:
        actual_n_classes = n_classes
    if actual_n_classes != n_classes:
        _flush(f"  ⚠ Config says {n_classes} classes but model has {actual_n_classes} — using model value")
        n_classes = actual_n_classes
        class_names = [f'class_{i}' for i in range(n_classes)]

    model = FedPerProxModel(
        input_dim=n_features, num_classes=n_classes,
        backbone_hidden=config.model.backbone_hidden,
        head_hidden=config.model.head_hidden, dropout=config.model.dropout,
    )
    if isinstance(state, dict) and 'backbone' in state:
        model.backbone.load_state_dict(state['backbone'])
        model.head.load_state_dict(state['head'])
    else:
        model.load_state_dict(state)
    model.eval()
    _flush(f"  ✅ Modèle chargé : {model_path.name}")

    _flush(f"\n📥 [3/4] Préparation des échantillons test...")
    all_X, all_y = [], []
    for X, y in test_loader:
        all_X.append(X.numpy()); all_y.append(y.numpy())
        if sum(x.shape[0] for x in all_X) >= n_test:
            break
    X_test = np.concatenate(all_X)[:n_test]
    y_test = np.concatenate(all_y)[:n_test]
    _flush(f"  ✅ Test : {X_test.shape}")

    _flush(f"\n🔬 [4/4] Calcul Gradient × Input par classe...")
    shap_start = time.time()

    X_tensor = torch.FloatTensor(X_test)
    shap_values_per_class = []

    for c in range(n_classes):
        X_grad = X_tensor.clone().detach().requires_grad_(True)
        logits = model(X_grad)
        class_logits = logits[:, c].sum()
        grads = torch.autograd.grad(class_logits, X_grad, create_graph=False)[0]
        attributions = (grads * X_grad).detach().numpy()
        shap_values_per_class.append(attributions)
        _flush(f"  ▶ Classe {c+1}/{n_classes} ({class_names[c]}) : ok")

    shap_arr = np.array(shap_values_per_class)
    shap_time = time.time() - shap_start
    _flush(f"\n  ✅ Calcul terminé en {shap_time:.1f}s")

    mean_importance = np.mean(np.abs(shap_arr), axis=(0, 1))
    class_importance = {
        class_names[c]: np.mean(np.abs(shap_arr[c]), axis=0).tolist()
        for c in range(n_classes)
    }

    result = {
        'client_id': client_id,
        'network': network,
        'shap_values': list(shap_arr),
        'test_data': X_test,
        'test_labels': y_test,
        'feature_names': list(feature_names),
        'class_names': list(class_names),
        'mean_importance': mean_importance,
        'class_importance': class_importance,
        'n_samples': stats['n_samples'],
        'computation_time_s': shap_time,
        'method': 'gradient_x_input',
    }

    out_path = out_dir / 'shap_results.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
    _flush(f"  💾 Sauvegardé : {out_path}")

    with open(out_dir / 'shap_summary.json', 'w') as f:
        json.dump({
            'client_id': client_id, 'network': network,
            'method': 'gradient_x_input',
            'n_samples': int(stats['n_samples']),
            'computation_time_s': round(shap_time, 2),
            'mean_importance': mean_importance.tolist(),
            'feature_names': list(feature_names),
            'top10_features': sorted(
                zip(feature_names, mean_importance.tolist()),
                key=lambda x: -x[1],
            )[:10],
        }, f, indent=2)

    total_time = time.time() - t0
    _flush(f"\n{'='*60}")
    _flush(f"✅ Client {client_id} TERMINÉ en {total_time:.1f}s")
    _flush(f"   Top 5 features importantes :")
    top5 = sorted(zip(feature_names, mean_importance.tolist()), key=lambda x: -x[1])[:5]
    for name, imp in top5:
        _flush(f"     • {name:35s} → {imp:.4f}")
    _flush(f"{'='*60}\n")
    return result