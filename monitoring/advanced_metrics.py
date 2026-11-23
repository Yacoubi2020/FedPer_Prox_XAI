"""Métriques avancées d'évaluation pour FedPer-Prox-XAI.

Deux familles de métriques, pensées pour un article Q1 IDS/FL :

1) Métriques de CLASSIFICATION (sur les prédictions du modèle) :
   - F1 macro / micro / weighted, précision & rappel macro, balanced accuracy
   - Cohen's kappa, MCC (robustes au déséquilibre de classes)
   - INTRA-CLASSE : précision/rappel/F1/support par classe + FPR/FNR par classe
   - INTER-CLASSE : matrice de confusion (brute + normalisée par ligne)
   - ROC-AUC macro (One-vs-Rest) si les probabilités sont fournies

2) Métriques de SÉPARABILITÉ dans l'espace de représentation (sortie backbone) :
   - INTRA-CLASSE : compacité = distance moyenne des échantillons au centroïde de leur classe
   - INTER-CLASSE : séparation = distances entre centroïdes de classes (moyenne/min/max)
   - Ratio de séparabilité (Fisher-like) = inter / intra
   - Silhouette, Davies-Bouldin, Calinski-Harabasz
"""
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def _round(x, n=4):
    """Arrondit récursivement (scalaires, listes, dict) pour un JSON propre."""
    if isinstance(x, dict):
        return {k: _round(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, n) for v in x]
    if isinstance(x, (np.floating, float)):
        return round(float(x), n)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
) -> Dict:
    """Métriques de classification globales + par classe (intra) + confusion (inter).

    Args:
        y_true:  labels réels, shape (N,)
        y_pred:  labels prédits, shape (N,)
        y_prob:  probabilités softmax, shape (N, C) — optionnel (pour ROC-AUC)
        class_names: noms des classes (longueur C)
        num_classes: nombre total de classes attendu (sinon déduit)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1) if len(y_true) else 0
    labels = list(range(num_classes))               # toutes les classes globales
    present = sorted(int(c) for c in np.unique(y_true))  # classes vues par ce client
    if class_names is None:
        class_names = [f"class_{i}" for i in labels]

    # ── Métriques globales ────────────────────────────────────────────────
    # En Non-IID, un client ne voit qu'un sous-ensemble des classes globales.
    # On distingue deux conventions :
    #   *_present : moyenne macro sur les classes RÉELLEMENT présentes (vue par client)
    #   *_global  : moyenne macro sur les 9 classes globales (pénalise les classes absentes)
    metrics: Dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        # F1-macro : convention "classes présentes" comme valeur principale
        "f1_macro": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
        "f1_macro_global": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }
    p_macro, r_macro, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present, average="macro", zero_division=0
    )
    metrics["precision_macro"] = p_macro
    metrics["recall_macro"] = r_macro
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred, labels=labels)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["classes_present"] = present

    # ── ROC-AUC macro (One-vs-Rest), restreint aux classes présentes ──────
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        metrics["roc_auc_macro_ovr"] = None
        try:
            if len(present) > 1 and y_prob.shape[1] == num_classes:
                # On ne garde que les colonnes des classes présentes, puis renormalise
                sub = y_prob[:, present]
                sub = sub / np.clip(sub.sum(axis=1, keepdims=True), 1e-12, None)
                remap = {c: i for i, c in enumerate(present)}
                y_remap = np.array([remap[int(v)] for v in y_true])
                auc = roc_auc_score(
                    y_remap, sub, multi_class="ovr", average="macro",
                    labels=list(range(len(present))),
                )
                metrics["roc_auc_macro_ovr"] = float(auc)
        except Exception as e:  # noqa: BLE001
            metrics["roc_auc_note"] = f"non calculable: {e}"

    # ── INTRA-CLASSE : précision/rappel/F1/support par classe ─────────────
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    # ── INTER-CLASSE : matrice de confusion (brute + normalisée par ligne) ─
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)

    # FPR / FNR par classe (dérivés de la matrice de confusion) — clé pour un IDS
    total = cm.sum()
    per_class: Dict = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        per_class[name] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "fpr": float(fpr),
            "fnr": float(fnr),
        }

    metrics["per_class"] = per_class
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_normalized"] = cm_norm.tolist()
    metrics["class_names"] = class_names

    return _round(metrics)


def compute_representation_metrics(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    class_names: Optional[List[str]] = None,
    max_samples_for_cluster_scores: int = 5000,
    metric: str = "euclidean",
    seed: int = 42,
) -> Dict:
    """Séparabilité inter/intra-classe dans l'espace latent (sortie du backbone).

    - INTRA : pour chaque classe, distance moyenne des points à leur centroïde.
    - INTER : distances entre centroïdes de classes.
    - Ratio de Fisher (séparabilité) = inter_moyen / intra_moyen.
    - Silhouette / Davies-Bouldin / Calinski-Harabasz (échantillonnés si N grand).

    Args:
        embeddings: sortie du backbone, shape (N, D)
        y_true: labels, shape (N,)
        max_samples_for_cluster_scores: sous-échantillonnage pour les scores O(N²)
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(y_true).ravel()
    present = np.unique(y)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(int(y.max()) + 1)] if len(y) else []

    # ── Centroïdes + compacité intra-classe ───────────────────────────────
    centroids: Dict[int, np.ndarray] = {}
    intra_per_class: Dict = {}
    for c in present:
        Xc = X[y == c]
        mu = Xc.mean(axis=0)
        centroids[int(c)] = mu
        # distance euclidienne moyenne au centroïde (compacité)
        d = np.linalg.norm(Xc - mu, axis=1)
        name = class_names[int(c)] if int(c) < len(class_names) else f"class_{int(c)}"
        intra_per_class[name] = {
            "mean_dist_to_centroid": float(d.mean()),
            "std_dist_to_centroid": float(d.std()),
            "n_samples": int(len(Xc)),
        }

    intra_values = [v["mean_dist_to_centroid"] for v in intra_per_class.values()]
    mean_intra = float(np.mean(intra_values)) if intra_values else 0.0

    # ── Séparation inter-classe (distances entre centroïdes) ──────────────
    cids = list(centroids.keys())
    inter_pairs: Dict[str, float] = {}
    inter_dists: List[float] = []
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            ci, cj = cids[i], cids[j]
            dist = float(np.linalg.norm(centroids[ci] - centroids[cj]))
            ni = class_names[ci] if ci < len(class_names) else f"class_{ci}"
            nj = class_names[cj] if cj < len(class_names) else f"class_{cj}"
            inter_pairs[f"{ni}__vs__{nj}"] = dist
            inter_dists.append(dist)

    mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
    min_inter = float(np.min(inter_dists)) if inter_dists else 0.0
    max_inter = float(np.max(inter_dists)) if inter_dists else 0.0

    # ── Ratio de séparabilité (Fisher-like) : > 1 = classes bien séparées ─
    separability_ratio = mean_inter / mean_intra if mean_intra > 0 else 0.0

    result: Dict = {
        "embedding_dim": int(X.shape[1]) if X.ndim == 2 else None,
        "n_classes_present": int(len(present)),
        "intra_class": {
            "per_class": intra_per_class,
            "mean_compactness": mean_intra,  # plus petit = classes plus compactes
        },
        "inter_class": {
            "pairwise_centroid_distance": inter_pairs,
            "mean_separation": mean_inter,  # plus grand = classes plus éloignées
            "min_separation": min_inter,
            "max_separation": max_inter,
        },
        "separability_ratio": separability_ratio,  # inter / intra (Fisher-like)
    }

    # ── Scores de clustering (sous-échantillonnés si N trop grand) ────────
    if len(present) > 1 and len(X) > len(present):
        if len(X) > max_samples_for_cluster_scores:
            idx = rng.choice(len(X), size=max_samples_for_cluster_scores, replace=False)
            Xs, ys = X[idx], y[idx]
            sampled = True
        else:
            Xs, ys, sampled = X, y, False
        try:
            result["silhouette_score"] = float(silhouette_score(Xs, ys, metric=metric))
            result["davies_bouldin_score"] = float(davies_bouldin_score(Xs, ys))
            result["calinski_harabasz_score"] = float(calinski_harabasz_score(Xs, ys))
            result["cluster_scores_subsampled"] = sampled
            result["cluster_scores_n_used"] = int(len(Xs))
        except Exception as e:  # noqa: BLE001
            result["cluster_scores_note"] = f"non calculable: {e}"

    return _round(result)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    embeddings: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
) -> Dict:
    """Regroupe classification + représentation dans un seul dictionnaire."""
    out: Dict = {
        "classification": compute_classification_metrics(
            y_true, y_pred, y_prob, class_names, num_classes
        )
    }
    if embeddings is not None:
        out["representation"] = compute_representation_metrics(
            embeddings, y_true, class_names
        )
    return out
