# Vehicle Trajectory Diffusion

NGSIM trajectory prediction with conditional diffusion. The current project
keeps a clean leader baseline and a graph-interaction line for Lankershim:

```text
clean leader data -> leader encoder -> Temporal CNN diffusion denoiser
lankershim graph data -> LSTM node encoder + edge attention -> Temporal CNN diffusion denoiser
```

Raw CSV files, intermediate processed files, split chunks, checkpoints, and
reports are local artifacts and are ignored by Git.

## Setup

Use Python 3.10 or 3.11. On the GPU server, install a CUDA-enabled PyTorch
build that matches the driver, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data

Place NGSIM CSV files under:

```text
data/raw/ngsim
```

Build the clean `us-101` leader dataset:

```bash
python scripts/prepare_ngsim.py \
  --raw-dir data/raw/ngsim \
  --output-dir data/processed/ngsim_us101_leader_clean \
  --location us-101 \
  --stride 20 \
  --chunk-size 10000

python scripts/split_ngsim.py \
  --input-npz data/processed/ngsim_us101_leader_clean/samples_chunks \
  --output-dir data/splits_us101_leader_clean \
  --prefix ngsim_us101_leader_clean

python scripts/check_dataloader.py \
  --split-dir data/splits_us101_leader_clean \
  --prefix ngsim_us101_leader_clean \
  --future-representation delta
```

The clean leader version invalidates geometrically inconsistent leader matches
and adds:

```text
leader_closing_speed
leader_ttc
leader_inverse_gap
```

## Training

```bash
python scripts/train_diffusion.py \
  --config configs/ngsim_leader_clean_diffusion.yaml
```

Checkpoints are written to:

```text
outputs/checkpoints_leader_clean/
```

## Evaluation

```bash
python scripts/eval_diffusion.py \
  --config configs/ngsim_leader_clean_diffusion.yaml \
  --checkpoint outputs/checkpoints_leader_clean/diffusion_best.pt \
  --num-samples 6 \
  --num-plots 20
```

Outputs:

```text
outputs/reports/diffusion_test_metrics.json
outputs/figures/diffusion_test/
```

## Lankershim Graph Line

This line models nearby vehicles as graph neighbors. Each sample stores:

```text
ego_past
neighbor_past
edge_attr
neighbor_mask
future
```

Edges are built from vehicles within a distance threshold at the observation
end frame. The first graph model uses a shared LSTM for node histories and
edge-aware attention pooling for interaction aggregation.

Prepare graph samples:

```bash
python scripts/prepare_ngsim_graph.py \
  --raw-dir data/raw/ngsim \
  --output-dir data/processed/ngsim_lankershim_graph \
  --location lankershim \
  --radius-m 30 \
  --max-neighbors 12 \
  --stride 20 \
  --chunk-size 10000

python scripts/split_ngsim.py \
  --input-npz data/processed/ngsim_lankershim_graph/samples_chunks \
  --output-dir data/splits_lankershim_graph \
  --prefix ngsim_lankershim_graph

python scripts/check_dataloader.py \
  --split-dir data/splits_lankershim_graph \
  --prefix ngsim_lankershim_graph \
  --future-representation delta \
  --dataset-type graph
```

Train:

```bash
python scripts/train_diffusion.py \
  --config configs/ngsim_lankershim_graph_diffusion.yaml
```

Evaluate:

```bash
python scripts/eval_diffusion.py \
  --config configs/ngsim_lankershim_graph_diffusion.yaml \
  --checkpoint outputs/checkpoints_lankershim_graph/diffusion_best.pt \
  --num-samples 6 \
  --num-plots 20
```
