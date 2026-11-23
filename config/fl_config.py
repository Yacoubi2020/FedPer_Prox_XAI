import os
import random
import random
"""Configuration centrale du projet FedPer-Prox-XAI Flower"""
from dataclasses import dataclass, field
from typing import List


# Classes du dataset IoMT Traffic
ALL_CLASSES = ['Normal', 'DoS', 'Vulnerability_Analysis', 'Malware', 'Routing',
               'MITM', 'Injection']


@dataclass
class ModelConfig:
    input_dim: int = 23
    num_classes: int = 9
    backbone_hidden: List[int] = field(default_factory=lambda: [256, 128, 64])
    head_hidden: List[int] = field(default_factory=lambda: [32])
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    num_rounds: int = 10
    local_epochs: int = 3
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    mu_prox: float = 0.01           # FedProx proximal term
    aggregation_temp: float = 0.5   # Soft-capping temperature


@dataclass
class NetworkConfig:
    network_type: str = '5g'  # wifi/5g/4g/3g


@dataclass
class EnergyConfig:
    tdp_watts: float = 15.0
    use_hardware_sensor: bool = False
    voltage_v: float = 5.1


@dataclass
class ClientConfig:
    client_id: int = 0
    server_address: str = '0.0.0.0:8080'
    grpc_max_message_length: int = 1024 * 1024 * 1024  # 1GB


@dataclass
class DataConfig:
    # If FEDPER_DATA_ROOT is set (e.g. a shared HPC folder outside this repo),
    # data_dir defaults there instead of the repo-relative ./data — see
    # scripts/csv_to_npy.py for the full convention. Each script's own
    # --data-dir CLI flag is what actually gets used at runtime; this field
    # exists for anyone importing DataConfig directly.
    data_dir: str = os.environ.get('FEDPER_DATA_ROOT', './data') + '/client_datasets'
    ALL_CLASSES: List[str] = field(default_factory=lambda: ALL_CLASSES.copy())


@dataclass
class XAIConfig:
    enabled: bool = True
    background_samples: int = 100
    test_samples: int = 300


@dataclass
class LoggingConfig:
    results_dir: str = './results'


@dataclass
class FLConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    data: DataConfig = field(default_factory=DataConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42

    def apply_seed(self):
        """Set all random seeds for full reproducibility (R1.1)."""
        import numpy as np
        import torch
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(self.seed)

