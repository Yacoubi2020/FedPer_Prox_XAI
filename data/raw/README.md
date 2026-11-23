Place the four client CSVs for a given (dataset × regime) configuration in the matching
subfolder below, named `client_1.csv` … `client_4.csv`:

```
data/raw/iomt_traffic/label_skew/
data/raw/iomt_traffic/quantity_skew/      <- this is the configuration this repo ships
                                              pre-computed client_datasets/ results for
data/raw/iot_ids2024/label_skew/
data/raw/iot_ids2024/quantity_skew/
```

Then run, once per configuration:

```bash
python scripts/csv_to_npy.py \
  --csv-dir  data/raw/<dataset>/<regime> \
  --out-dir  data/client_datasets/<dataset>_<regime>
```

See the top-level README's "Preparing the data" section for why `scripts/csv_to_npy.py`
(not `scripts/prepare_data.py`) is the script to use.

To keep the datasets in one external folder instead of inside this repo (e.g. on shared
HPC storage), see the top-level README's "Separating data from code" section and set
`FEDPER_DATA_ROOT` — this `data/raw/` folder then becomes unused.
