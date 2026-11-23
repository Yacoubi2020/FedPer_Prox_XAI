Results for the **IoMT-Traffic, Quantity Skew** configuration.

- `wifi/` — the configuration actually reported in the paper: 30 communication rounds
  (`total_rounds: 30` in each client's `final_summary.json`). Figures in `wifi/figures/`
  correspond to the ones in the paper's Results section for this configuration.
- `5g/` — a 5-round smoke-test run from the single-machine validation step of the
  workflow (see the top-level README, "Validate on a single machine"), **not** a
  reported experimental result. Kept for reference only.
- `multi_seed/`, `sota_comparison/`, `complexity/`, `feature_reduction/`,
  `statistical_analysis/` — the artifacts backing, respectively, the paper's
  5-seed robustness tables, the 13-baseline SOTA comparison, the complexity
  analysis (Section 3.6), the feature-reduction study, and the non-IID impact
  statistics.
- `results_multi_seed/iomt_traffic_quantity_skew/` (top-level sibling of `results/`) — the
  full per-seed logs and per-client metrics for the 5-seed robustness study
  (seeds 0, 1, 2, 3, 42).
