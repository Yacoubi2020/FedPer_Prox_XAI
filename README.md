# FedPer-Prox-XAI

Official implementation of **"FedPer-Prox-XAI: An Explainable Personalized Federated Learning Framework for Robust Intrusion Detection in Heterogeneous IoMT Networks"** (under review).

Federated intrusion detection for heterogeneous IoMT environments, combining FedPer (personalized classification heads), FedProx (proximal regularization), temperature-softened aggregation, and multi-tier gradient-based feature attribution. Deployed on a physical Raspberry Pi cluster and evaluated over four network profiles.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Server (Flower Strategy: FedPer-Prox)           │
│   • Temperature-softened aggregation of backbone │
│   • Round coordination                           │
└────────────┬──────────────────────────────────────┘
             │ gRPC
   ┌─────────┼─────────┬─────────┐
   ▼         ▼         ▼         ▼
 Client 0  Client 1  Client 2  Client 3
 Backbone  Backbone  Backbone  Backbone   ← shared, exchanged every round
 + Head    + Head    + Head    + Head     ← personalized, stays local
 + attrib. + attrib. + attrib. + attrib.  ← local feature attribution
```

- **Shared backbone**: Dense(in→256)→BN→ReLU→Dropout(0.3)→Dense(256→128)→BN→ReLU→Dropout(0.3)→Dense(128→64). Only this is exchanged with the server.
- **Personalized head**: Dense(64→32)→ReLU→Dense(32→C). Trained locally, never transmitted.
- **Aggregation** (`flower_components/strategy.py`): weighted average with $\alpha_k \propto n_k^{\tau}$, $\tau = 0.5$ (`aggregation_temp`), reducing the influence of data-rich clients under quantity skew relative to plain sample-proportional FedAvg.
- **Proximal term** (`models/fedper_model.py`): FedProx penalty $\mu \lVert \theta_B - \theta_B^{global}\rVert^2$, $\mu = 0.01$.

## Repository structure

```
FedPer-Prox-XAI/
├── config/fl_config.py        # All hyperparameters (model, training, energy, XAI...)
├── data/
│   ├── raw/<dataset>/<regime>/    # Put your client_1.csv … client_4.csv here (gitignored)
│   ├── client_datasets/<dataset>_<regime>/   # Generated per-client partitions (gitignored)
│   └── data_loader.py          # load_client_partition() used at runtime
├── flower_components/          # Flower client (client.py) and strategy (strategy.py)
├── models/fedper_model.py      # Backbone / personalized head / FocalLoss (see note below)
├── monitoring/                 # Energy, communication, and metrics logging
├── xai/shap_analyzer.py        # Four-tier feature-attribution pipeline (see note below)
├── visualization/plots.py      # Figure generation
├── scripts/                    # CLI entry points (see "Workflow" below)
├── results/iomt_traffic_quantity_skew/   # Artifacts backing the paper's tables/figures
├── results_multi_seed/iomt_traffic_quantity_skew/  # Per-seed logs (5-seed robustness study)
└── requirements.txt
```

The four experimental configurations reported in the paper are **IoMT-TrafficData** and **IoT-IDS2024**, each under **label skew** and **quantity skew**. This repository ships the code (identical across all four configurations — only the data and `config/fl_config.py`'s `ALL_CLASSES` / `input_dim` change) together with the full result artifacts for **IoMT-TrafficData, Quantity Skew**; run the same scripts on the other three `data/raw/<dataset>/<regime>/` folders to reproduce the rest.

## Installation

```bash
git clone https://github.com/<your-username>/FedPer-Prox-XAI.git
cd FedPer-Prox-XAI
chmod +x setup.sh && ./setup.sh
# or manually:
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Separating data from code

The datasets (raw CSVs and generated `client_datasets/`) can live in **one external
folder outside this repository** — useful on a shared machine or HPC cluster, where you
want the code checked out per-user but the (large, possibly access-controlled) data kept
in a single place that every job just needs read access to.

Set one environment variable and every script in this repo resolves its data paths from
it automatically, instead of defaulting to `./data` inside the repo:

```bash
export FEDPER_DATA_ROOT=/path/to/shared/FedPer-data   # e.g. a group-readable HPC folder
```

Lay that folder out exactly like `data/` is laid out here:

```
$FEDPER_DATA_ROOT/
├── raw/
│   ├── iomt_traffic/{label_skew,quantity_skew}/client_1.csv … client_4.csv
│   └── iot_ids2024/{label_skew,quantity_skew}/client_1.csv … client_4.csv
└── client_datasets/
    ├── iomt_traffic_label_skew/client_0 … client_3/{X.npy,y.npy,stats.pkl}
    ├── iomt_traffic_quantity_skew/...
    ├── iot_ids2024_label_skew/...
    └── iot_ids2024_quantity_skew/...
```

Once `FEDPER_DATA_ROOT` is exported (e.g. in `~/.bashrc`, or at the top of a SLURM job
script), you no longer need to pass `--csv-dir`/`--out-dir`/`--data-dir` at all for the
common case — every script's default already points there:

```bash
python scripts/csv_to_npy.py --csv-dir "$FEDPER_DATA_ROOT/raw/iomt_traffic/quantity_skew" \
                              --out-dir "$FEDPER_DATA_ROOT/client_datasets/iomt_traffic_quantity_skew"
python scripts/run_client.py --client-id 0 --data-dir "$FEDPER_DATA_ROOT/client_datasets/iomt_traffic_quantity_skew" ...
```

Practical effect: cloning this repo gives you code only (a few MB); pointing
`FEDPER_DATA_ROOT` at the shared folder gives every clone/job access to the same data
without copying it per-user or per-job. On an HPC cluster, grant group read permission
(`chmod -R g+rX "$FEDPER_DATA_ROOT"`, or an ACL) on that one folder instead of on each
user's checkout.

If `FEDPER_DATA_ROOT` is unset, nothing changes — every script falls back to the
repo-relative `./data/...` paths used elsewhere in this README.

## Workflow

### 1. Preparing the data

Place the four raw client CSVs for a configuration under `data/raw/<dataset>/<regime>/` (see `data/raw/README.md`), then run:

```bash
python scripts/csv_to_npy.py \
  --csv-dir  data/raw/<dataset>/<regime> \
  --out-dir  data/client_datasets/<dataset>_<regime>
```

**Use `scripts/csv_to_npy.py`, not `scripts/prepare_data.py`.** The latter (and the
`data.partition_and_save` function it calls) assumes the label is the CSV's *last*
column, which is not the case here — the real label column is `class`. On these
datasets it would silently produce wrong labels rather than failing, so it has been
disabled (it now exits with an error pointing here). `csv_to_npy.py` auto-detects the
label column and derives the class list from the data itself.

Before switching datasets, update `config/fl_config.py`'s `DataConfig.ALL_CLASSES` to
match the dataset you're running (used as a fallback for class names in the
explainability output):
- IoMT-TrafficData (9 classes): `Normal, ApacheKiller, RUDY, SlowRead, SlowLoris, ARP, cam, Malaria, Netscan`
- IoT-IDS2024 (7 classes): `Normal, DoS, Vulnerability_Analysis, Malware, Routing, MITM, Injection`

For a quick smoke test without real data:
```bash
python scripts/generate_fake_data.py --samples 2000
```

### 2. Validate on a single machine

```bash
DATA_DIR=data/client_datasets/<dataset>_<regime>   # e.g. iomt_traffic_quantity_skew

# Terminal 1 — server
python scripts/run_server.py --rounds 5 --network wifi

# Terminals 2-5 — clients (K=4)
python scripts/run_client.py --client-id 0 --server-addr 127.0.0.1:8080 --network wifi --data-dir $DATA_DIR
python scripts/run_client.py --client-id 1 --server-addr 127.0.0.1:8080 --network wifi --data-dir $DATA_DIR
python scripts/run_client.py --client-id 2 --server-addr 127.0.0.1:8080 --network wifi --data-dir $DATA_DIR
python scripts/run_client.py --client-id 3 --server-addr 127.0.0.1:8080 --network wifi --data-dir $DATA_DIR
```

This is a smoke test (few rounds) to confirm the pipeline runs end-to-end before a full deployment — the paper's reported results use 30 rounds (`--rounds 30`).

### 3. Physical edge deployment (Raspberry Pi cluster)

Install Tailscale on each device, retrieve the Tailscale IPs, edit `scripts/run_all_networks.sh`
(`SERVER_IP`, `CLIENT_IPS`, `CLIENT_USERS`), set up password-less SSH, then:

```bash
bash scripts/run_all_networks.sh 30
```

This runs the full 30-round training across the four network profiles (WiFi/5G/4G/3G), computes
feature attributions on each device, retrieves results to the server, and generates the figures.

### 4. Additional analyses

```bash
python scripts/run_multi_seed.py        # 5-seed robustness study (Section 5.7)
python scripts/run_sota_baselines_v2.py # 13-baseline SOTA comparison (Section 5.6)
python scripts/run_feature_reduction.py # SHAP-guided feature reduction (Section 5.8)
python scripts/complexity_analysis.py   # Complexity table (Section 3.6)
python scripts/statistical_analysis.py  # Non-IID impact statistics
python scripts/visualize_all.py --network wifi
```

## A note on the explainability method

`xai/shap_analyzer.py` computes **Gradient × Input** attributions per class, not Shapley
values from `shap.DeepExplainer`/`KernelExplainer` — its own output is labelled
`"method": "gradient_x_input"`. This is used as a fast proxy (~5s vs. ~15min per client)
that coincides with Shapley values for piecewise-linear models, but is not exactly
equivalent for an MLP with BatchNorm and dropout. `shap` is already in
`requirements.txt` for anyone who wants to swap in a genuine SHAP explainer for tiers
1–3 of the four-tier pipeline (per-client, per-class, globally aggregated); tier 4
(cross-client consensus–divergence) is a novel aggregation step independent of the
underlying attribution method. Make sure the wording used in any accompanying paper
matches whichever attribution method is actually run.

## A note on the energy figures

Per-round energy (`monitoring/energy_monitor.py`) defaults to a TDP-based model
(nominal power draw × phase duration, `tdp_watts: 15.0`, `use_hardware_sensor: False`
in `config/fl_config.py`), not a hardware sensor reading. The module also supports a
real INA219 shunt sensor (`use_hardware_sensor: True`); the results shipped in this
repository were produced with the TDP model. Communication cost, by contrast, is an
exact byte count of the serialized backbone payload, not an estimate.

## Metrics saved

Per client (`results/<config>/<network>/client_X/`):
- `metrics.json` — per-round history (loss, accuracy, energy, communication)
- `final_report.json`, `final_summary.json` — full / condensed run report
- `final_model.pt` — trained model (backbone + head) — not committed to this repo; regenerate by re-running training
- `shap_results.pkl`, `shap_summary.json` — feature-attribution output (see note above)

## Adjustable parameters

All hyperparameters live in `config/fl_config.py`: `num_rounds`, `local_epochs`,
`batch_size`, `lr`, `mu_prox` (FedProx), `aggregation_temp` (soft-capping), backbone/head
sizes, dropout, XAI sample sizes, and the energy/network settings described above.

## Citation

```bibtex
@article{fedperproxxai,
  title   = {FedPer-Prox-XAI: An Explainable Personalized Federated Learning Framework for Robust Intrusion Detection in Heterogeneous IoMT Networks},
  author  = {ANONYME},
  journal = {Array},
  year    = {2026},
  note    = {Under review}
}
```

