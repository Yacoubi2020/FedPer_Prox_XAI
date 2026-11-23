"""Monitoring énergétique pour Raspberry Pi 5 (INA219 ou estimation TDP)."""
import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class EnergyRecord:
    phase: str
    duration_s: float
    energy_j: float
    power_w: float
    timestamp: float = field(default_factory=time.time)


class EnergyMonitor:
    def __init__(self, tdp_watts=15.0, battery_capacity_mah=10000.0,
                 use_hardware_sensor=False, voltage_v=5.1):
        self.tdp_watts = tdp_watts
        self.voltage_v = voltage_v
        self.battery_capacity_mah = battery_capacity_mah
        self.use_hardware_sensor = use_hardware_sensor
        self.records: List[EnergyRecord] = []
        self._sensor = None
        self._lock = threading.Lock()
        if use_hardware_sensor:
            self._init_sensor()

    def _init_sensor(self):
        try:
            from ina219 import INA219
            self._sensor = INA219(shunt_ohms=0.1, address=0x40)
            self._sensor.configure()
            print("✅ Capteur INA219 initialisé")
        except Exception as e:
            print(f"⚠️  INA219 indisponible: {e} → mode estimation logicielle")
            self._sensor = None
            self.use_hardware_sensor = False

    def _read_power_w(self):
        if self._sensor:
            try:
                return self._sensor.power() / 1000.0
            except Exception:
                pass
        return self.tdp_watts

    class _PhaseContext:
        def __init__(self, monitor, phase):
            self.monitor = monitor
            self.phase = phase

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            duration = time.time() - self.start
            power = self.monitor._read_power_w()
            energy = power * duration
            with self.monitor._lock:
                self.monitor.records.append(
                    EnergyRecord(phase=self.phase, duration_s=duration,
                                 energy_j=energy, power_w=power)
                )

    def measure(self, phase):
        return self._PhaseContext(self, phase)

    def get_total_energy_j(self):
        return sum(r.energy_j for r in self.records)

    def get_summary(self):
        total = self.get_total_energy_j()
        by_phase = {}
        for r in self.records:
            by_phase[r.phase] = by_phase.get(r.phase, 0) + r.energy_j
        return {
            'total_energy_j': round(total, 4),
            'total_energy_mah': round((total / (self.voltage_v * 3600)) * 1000, 4),
            'energy_by_phase': {k: round(v, 4) for k, v in by_phase.items()},
            'n_records': len(self.records),
            'tdp_watts': self.tdp_watts,
            'hardware_sensor': self.use_hardware_sensor,
        }
