"""Suivi communications Flower avec profils réseau réalistes."""
import time
import pickle
import zlib
from dataclasses import dataclass, field
from typing import List, Dict, Any

NETWORK_CONFIGS = {
    'wifi': dict(bandwidth_mbps=100.0, latency_ms=10.0, packet_loss=0.001,
                 energy_per_mb_upload=0.05, energy_per_mb_download=0.03),
    '5g':   dict(bandwidth_mbps=150.0, latency_ms=5.0, packet_loss=0.0005,
                 energy_per_mb_upload=0.08, energy_per_mb_download=0.05),
    '4g':   dict(bandwidth_mbps=50.0, latency_ms=30.0, packet_loss=0.002,
                 energy_per_mb_upload=0.12, energy_per_mb_download=0.08),
    '3g':   dict(bandwidth_mbps=10.0, latency_ms=100.0, packet_loss=0.005,
                 energy_per_mb_upload=0.18, energy_per_mb_download=0.12),
}


@dataclass
class CommRecord:
    round_num: int
    direction: str
    raw_bytes: int
    compressed_bytes: int
    energy_j: float
    time_ms: float
    timestamp: float = field(default_factory=time.time)


class CommMonitor:
    def __init__(self, network_type='wifi', enable_compression=True):
        if network_type not in NETWORK_CONFIGS:
            raise ValueError(f"Réseau inconnu : {network_type}")
        self.cfg = NETWORK_CONFIGS[network_type]
        self.network_type = network_type
        self.enable_compression = enable_compression
        self.records: List[CommRecord] = []

    def _serialize(self, data):
        raw = pickle.dumps(data)
        comp = zlib.compress(raw, level=6) if self.enable_compression else raw
        return len(raw), len(comp)

    def _transfer_time_ms(self, n_bytes):
        mb = (n_bytes * 8) / (1024 * 1024)
        bw_time = mb / self.cfg['bandwidth_mbps']
        lat = self.cfg['latency_ms'] / 1000.0
        return (bw_time + lat) * (1 + self.cfg['packet_loss']) * 1000.0

    def _energy_j(self, n_bytes, direction):
        mb = n_bytes / (1024 * 1024)
        key = 'energy_per_mb_upload' if direction == 'upload' else 'energy_per_mb_download'
        return mb * self.cfg[key]

    def track(self, round_num, data, direction):
        raw_b, comp_b = self._serialize(data)
        rec = CommRecord(
            round_num=round_num, direction=direction,
            raw_bytes=raw_b, compressed_bytes=comp_b,
            energy_j=self._energy_j(comp_b, direction),
            time_ms=self._transfer_time_ms(comp_b),
        )
        self.records.append(rec)
        return rec

    def track_upload(self, round_num, data):
        return self.track(round_num, data, 'upload')

    def track_download(self, round_num, data):
        return self.track(round_num, data, 'download')

    def get_total_mb(self, compressed=True):
        key = 'compressed_bytes' if compressed else 'raw_bytes'
        return sum(getattr(r, key) for r in self.records) / (1024 * 1024)

    def get_total_energy_j(self):
        return sum(r.energy_j for r in self.records)

    def get_summary(self):
        return {
            'network_type': self.network_type,
            'total_mb_raw': round(sum(r.raw_bytes for r in self.records) / (1024 * 1024), 4),
            'total_mb_compressed': round(self.get_total_mb(), 4),
            'total_energy_j': round(self.get_total_energy_j(), 4),
            'n_uploads': sum(1 for r in self.records if r.direction == 'upload'),
            'n_downloads': sum(1 for r in self.records if r.direction == 'download'),
            'bandwidth_mbps': self.cfg['bandwidth_mbps'],
            'latency_ms': self.cfg['latency_ms'],
        }
