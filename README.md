# Vehicle Trajectory Diffusion

Minimal NGSIM trajectory-prediction pipeline using a conditional DDPM model.

## Server Setup

Use Python 3.10 or 3.11. Install a CUDA-enabled PyTorch build that matches the
server driver by following:

```text
https://pytorch.org/get-started/locally/
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Place NGSIM CSV files under:

```text
data/raw/ngsim
```

The CSV files may contain multiple locations. Samples remain isolated by
`Location`, source file, and `Vehicle_ID`.

Prepare and split the data. The current first-run setup uses only `us-101`
and keeps every vehicle inside exactly one split:

```bash
python scripts/prepare_ngsim.py \
  --raw-dir data/raw/ngsim \
  --output-dir data/processed/ngsim_us101_leader \
  --location us-101 \
  --stride 20 \
  --chunk-size 10000

python scripts/split_ngsim.py \
  --input-npz data/processed/ngsim_us101_leader/samples_chunks \
  --output-dir data/splits_us101_leader \
  --prefix ngsim_us101_leader

python scripts/check_dataloader.py \
  --split-dir data/splits_us101_leader \
  --prefix ngsim_us101_leader \
  --future-representation delta
```

The committed split chunks are sufficient for training on a server. Raw CSV
files and expanded intermediate files stay local and are ignored by Git.

## Training

Start the minimal diffusion training pipeline:

```bash
python scripts/train_diffusion.py \
  --config configs/ngsim_diffusion.yaml
```

The current model diffuses framewise future displacements (`delta_x`,
`delta_y`) and uses a temporal 1D CNN denoiser. The encoder receives ego
history and explicit leader-history features matched within the same
`Location`, source file, and frame. Predictions are integrated back into
positions for evaluation and visualization.

Outputs:

```text
outputs/checkpoints/diffusion_best.pt
outputs/checkpoints/diffusion_last.pt
outputs/reports/diffusion_train_log.json
```

To resume training, set `training.resume_from` in
`configs/ngsim_diffusion.yaml` to a checkpoint path.

## Evaluation And Static Visualization

Evaluate the best checkpoint with six diffusion samples per trajectory:

```bash
python scripts/eval_diffusion.py \
  --config configs/ngsim_diffusion.yaml \
  --checkpoint outputs/checkpoints/diffusion_best.pt \
  --num-samples 6
```

For a quick first check:

```bash
python scripts/eval_diffusion.py \
  --max-trajectories 512 \
  --num-samples 6 \
  --num-plots 12
```

Outputs:

```text
outputs/reports/diffusion_test_metrics.json
outputs/figures/diffusion_test/*.png
```

## Constant-Velocity Baseline

Compare diffusion predictions with a simple constant-velocity extrapolation:

```bash
python scripts/eval_constant_velocity.py \
  --config configs/ngsim_diffusion.yaml \
  --fps 10 \
  --velocity-window 5
```

Output:

```text
outputs/reports/constant_velocity_metrics.json
```

## Rule-Based Candidate Scoring

Rank diffusion candidates without ground truth using motion smoothness,
initial-velocity consistency, lateral drift, and leader-gap risk:

```bash
python scripts/eval_scored_diffusion.py \
  --config configs/ngsim_diffusion.yaml \
  --checkpoint outputs/checkpoints_leader/diffusion_best.pt \
  --num-samples 6
```

Output:

```text
outputs/reports/scored_diffusion_test_metrics.json
```
