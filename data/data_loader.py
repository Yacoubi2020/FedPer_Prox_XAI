"""
Chargement et partitionnement des données IoMT Traffic.

Fournit:
  - load_client_partition(client_id, data_dir, batch_size)
      → (train_loader, test_loader, stats)
  - partition_and_save(csv_files, output_dir, class_names)
      → dict[client_id → stats]
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Classes du dataset IoMT Traffic ─────────────────────────────────────
ALL_CLASSES = [
    'Normal', 'DoS', 'Vulnerability_Analysis', 'Malware',
    'Routing', 'MITM', 'Injection',
]


# ── Dataset PyTorch ─────────────────────────────────────────────────────
class IoMTDataset(Dataset):
    """Wrapper léger autour de tenseurs (X, y)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Création des DataLoaders ────────────────────────────────────────────
def create_dataloaders(X, y, batch_size=256, test_split=0.2):
    """Split 80/20 stratifié → train + test DataLoaders."""
    # Sécurité : si une classe a <2 samples, pas de stratification
    _, counts = np.unique(y, return_counts=True)
    stratify = y if counts.min() >= 2 else None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_split,
        random_state=42,
        stratify=stratify,
    )

    train_loader = DataLoader(
        IoMTDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        IoMTDataset(X_te, y_te),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


# ── Charger une partition client (X.npy / y.npy / stats.pkl) ───────────
def load_client_partition(client_id, data_dir, batch_size=256):
    """
    Charge les données d'un client depuis data_dir/client_<id>/.

    Fichiers attendus :
        X.npy      – features (n_samples, n_features), float32, standardisé
        y.npy      – labels   (n_samples,), int encodé 0..C-1
        stats.pkl  – dict avec input_dim, n_samples, class_names, etc.
    """
    client_path = Path(data_dir) / f'client_{client_id}'

    X = np.load(client_path / 'X.npy')
    y = np.load(client_path / 'y.npy')

    with open(client_path / 'stats.pkl', 'rb') as f:
        stats = pickle.load(f)

    train_loader, test_loader = create_dataloaders(X, y, batch_size)

    # Compléter stats si nécessaire
    if 'n_train' not in stats:
        stats['n_train'] = len(train_loader.dataset)
    if 'n_test' not in stats:
        stats['n_test'] = len(test_loader.dataset)

    return train_loader, test_loader, stats


# ── Partitionnement depuis CSV (DEPRECATED — voir scripts/csv_to_npy.py) ──
def partition_and_save(csv_files, output_dir, class_names=None):
    """
    DEPRECATED for this project's datasets: assumes the label is the CSV's
    last column and uses a single hardcoded class list, both of which are
    wrong for IoMT-TrafficData / IoT-IDS2024 (label column is `class`, class
    list differs per dataset). Use scripts/csv_to_npy.py instead, which
    auto-detects the label column and the class list from the data. Kept
    here only because scripts/prepare_data.py (also deprecated) imports it.

    Lit N fichiers CSV (un par client), standardise les features,
    encode les labels et sauvegarde X.npy, y.npy, stats.pkl par client.

    Parameters
    ----------
    csv_files : list[str]
        Chemins des CSV clients (client_1.csv, client_2.csv, …).
    output_dir : str
        Répertoire de sortie (data/client_datasets/).
    class_names : list[str], optional
        Noms des classes. Si None, utilise ALL_CLASSES.

    Returns
    -------
    dict[int → dict] : stats par client_id.
    """
    if class_names is None:
        class_names = ALL_CLASSES

    output_dir = Path(output_dir)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(class_names)

    all_stats = {}

    # Première passe : fit le scaler sur TOUS les CSV
    print("\n  Phase 1 : Lecture de tous les CSV pour fit du scaler...")
    all_dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, low_memory=False)
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)

    # Séparer features et label
    label_col = combined.columns[-1]  # dernière colonne = label

    # Filtrer les colonnes : garder uniquement les colonnes numériques
    # (exclure IP addresses, timestamps, strings, etc.)
    candidate_cols = [c for c in combined.columns if c != label_col]
    numeric_cols = []
    dropped_cols = []
    for col in candidate_cols:
        try:
            combined[col] = pd.to_numeric(combined[col], errors='coerce')
            # Si trop de NaN après conversion (>50%), la colonne est non-numérique
            if combined[col].isna().mean() > 0.5:
                dropped_cols.append(col)
            else:
                numeric_cols.append(col)
        except Exception:
            dropped_cols.append(col)

    if dropped_cols:
        print(f"    Colonnes non-numériques exclues ({len(dropped_cols)}): "
              f"{dropped_cols[:5]}{'...' if len(dropped_cols)>5 else ''}")

    feature_cols = numeric_cols

    # Remplacer les NaN restantes par 0
    combined[feature_cols] = combined[feature_cols].fillna(0)

    # Appliquer aussi le fillna sur chaque df individuel
    for i, df in enumerate(all_dfs):
        for col in feature_cols:
            all_dfs[i][col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    scaler = StandardScaler()
    scaler.fit(combined[feature_cols].values.astype(np.float32))

    print(f"    {len(combined)} samples totaux | {len(feature_cols)} features numeriques | "
          f"label col: '{label_col}'")

    # Deuxième passe : transform + save par client
    print("  Phase 2 : Standardisation et sauvegarde par client...")
    for client_id, df in enumerate(all_dfs):
        client_dir = output_dir / f'client_{client_id}'
        client_dir.mkdir(parents=True, exist_ok=True)

        X = scaler.transform(df[feature_cols].values.astype(np.float32))

        # Encoder les labels
        raw_labels = df[label_col].values
        y = np.zeros(len(raw_labels), dtype=np.int64)
        for i, lab in enumerate(raw_labels):
            lab_str = str(lab).strip()
            if lab_str in class_names:
                y[i] = class_names.index(lab_str)
            else:
                # Essayer un match partiel
                matched = False
                for ci, cn in enumerate(class_names):
                    if lab_str.lower() == cn.lower():
                        y[i] = ci
                        matched = True
                        break
                if not matched:
                    print(f"    [WARN] Label inconnu: '{lab_str}' → classe 0")
                    y[i] = 0

        present_classes = sorted(set(y.tolist()))

        stats = {
            'input_dim': X.shape[1],
            'n_samples': len(X),
            'class_names': [class_names[c] for c in present_classes
                            if c < len(class_names)],
            'classes_present': present_classes,
            'class_distribution': {
                class_names[c]: int((y == c).sum())
                for c in present_classes if c < len(class_names)
            },
        }

        np.save(client_dir / 'X.npy', X)
        np.save(client_dir / 'y.npy', y)
        with open(client_dir / 'stats.pkl', 'wb') as f:
            pickle.dump(stats, f)

        all_stats[client_id] = stats
        print(f"    Client {client_id}: {stats['n_samples']:,} samples → {client_dir}")

    return all_stats
