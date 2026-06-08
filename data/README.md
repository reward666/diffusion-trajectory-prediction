# Data Layout

Only the clean leader dataset is part of the current main line.

Local raw and intermediate data:

```text
data/raw/ngsim
data/processed/ngsim_us101_leader_clean
```

Main split directory after preprocessing:

```text
data/splits_us101_leader_clean
```

This split is grouped by:

```text
Location + source_file + Vehicle_ID
```

All windows from one vehicle stay in exactly one split. Normalization
statistics are computed from the training split only.
