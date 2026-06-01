# Data Layout

The repository includes the compact `us-101` train, validation, and test
chunks required for the first diffusion-model run:

```text
data/splits_us101_v2
```

The raw NGSIM CSV, expanded track CSV files, and intermediate preprocessing
chunks remain local and are intentionally ignored by Git.

The current split is an MVP split grouped by:

```text
Location + source_file + Vehicle_ID
```

All windows from one vehicle stay in exactly one split. For final reported
experiments, replace this with a stricter time-block split with a boundary
gap between train, validation, and test segments.

