# Architecture and complexity notes

This document mirrors Section 3 (Proposed Method) and Section 3.6 (Computational and
Communication Complexity) of the paper, and records what was verified directly against
the code in this repository (not assumed from the paper alone).

## Components (verified against code)

- **Shared backbone** (`models/fedper_model.py`, $P_B = 48{,}192$ parameters for the
  IoMT-TrafficData feature count): Dense→BN→ReLU→Dropout(0.3) ×2, then Dense(128→64).
  Exchanged with the server every round (`flower_components/client.py`).
- **Personalized head**: Dense(64→32)→ReLU→Dense(32→C). Trained and kept locally.
- **Aggregation** (`flower_components/strategy.py::aggregate_fit`): confirmed to compute
  $\alpha_k = n_k^{\tau} / \sum_j n_j^{\tau}$ with $\tau = 0.5$ (`aggregation_temp`), i.e.
  exactly the temperature-softened rule described in the paper — this matches, no
  discrepancy found here.
- **Proximal term**: $\mu = 0.01$, applied in the client's local loss (`models/fedper_model.py`,
  `compute_proximal_loss`).

## Verified discrepancies between code and manuscript (resolve before/with publication)

These were confirmed by reading the actual source and outputs in this repository, not
inferred from the paper's wording:

1. **Explainability method.** `xai/shap_analyzer.py` computes Gradient × Input per class,
   not Shapley values — its own docstring and output (`"method": "gradient_x_input"` in
   every `shap_summary.json`/`shap_results.pkl`) say so explicitly. The manuscript
   describes "four-tier SHAP explainability" throughout. See the README's "A note on the
   explainability method" section for the three ways to reconcile this (run real SHAP,
   rename the method throughout and reposition the contribution on the four tiers, or
   both with a rank-correlation check).

2. **Energy measurement.** `monitoring/energy_monitor.py` defaults to
   `use_hardware_sensor=False`, i.e. energy = nominal TDP (15 W) × phase duration, not a
   sensor reading. Confirmed in the shipped `final_report.json` files
   (`energy_summary.hardware_sensor: false`). The manuscript's Section 3.7 already
   describes this accurately ("Energy is derived from the wall-clock duration ... 
   multiplied by the device's nominal power draw"), so this is consistent — just make
   sure no other passage in the paper describes energy as "measured"/"instrumented" in a
   way that implies a hardware reading.

3. **Data-preparation script.** The README previously documented `scripts/prepare_data.py`
   as the way to turn raw CSVs into `client_datasets/`. That script assumes the label is
   the CSV's last column, which is false for these datasets (label column is `class`) —
   it has been disabled. `scripts/csv_to_npy.py` is the script that was actually used to
   produce the `client_datasets/` shipped here (verified: its auto-detected label column,
   class list, and per-client sample counts match exactly) and is now the one documented
   in the top-level README.

## Non-discrepancies worth noting

- `config/fl_config.py`'s `ModelConfig.input_dim = 23` is a stale default — the real
  IoMT-TrafficData feature count is 24 (23 features + `is_malicious`, which is kept as a
  feature rather than dropped). This is harmless: `scripts/run_client.py` overwrites
  `config.model.input_dim` from the loaded partition's actual shape before building the
  model, so the stale default is never used to build a model in practice.
- `results/<config>/5g/` contains a 5-round run — a single-machine smoke test from the
  documented validation workflow, not reported experimental data. Only
  `results/<config>/wifi/` (30 rounds) corresponds to the paper's reported numbers for
  this configuration.

## Complexity (from the paper's Section 3.6)

| Quantity | Complexity | This deployment ($K{=}4$, $T{=}30$, $E{=}3$) |
|---|---|---|
| Per-client, per-round computation | $\mathcal{O}(E \cdot \lvert\mathcal{D}_k\rvert / B \cdot (P_B + P_H))$ | — |
| Server aggregation, per round | $\mathcal{O}(K \cdot P_B)$ | negligible vs. client training |
| Communication, per round | $\mathcal{O}(K \cdot P_B)$ | ≈ 1.4 MB |
| Communication, total ($T$ rounds) | $\mathcal{O}(T \cdot K \cdot P_B)$ | ≈ 43 MB (measured: 40.9–55.7 MB) |
| Attribution, tiers 1–3 | $\mathcal{O}(K \cdot \lvert\mathcal{S}\rvert \cdot (P_B+P_H))$ | computed once, post-training |
| Attribution, tier 4 (consensus–divergence) | $\mathcal{O}(K^2 \cdot d)$ | negligible at $K=4$ |

Communication cost is fixed by the architecture (backbone size) and is therefore
invariant to the non-IID regime; per-round computation and TDP-derived energy scale with
each client's local data volume, which is why energy varies by up to 4.9× across regimes
on the same dataset while communication does not.
