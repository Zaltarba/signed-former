# signed-former

Auto-research loop for multivariate time series forecasting. Claude runs as an autonomous agent that iteratively proposes and evaluates architectural changes, using git history and experiment logs as memory.

## Goal

Design a **lightweight signed-attention Transformer** where every representation stays interpretable as a time series. The attention matrix is treated as a signed cross-variate correlation matrix (`softmax(+s) - softmax(-s)`), built from causal patch tokens with lead-lag awareness.

Baseline: **iTransformer** (inverted token embedding, full attention, FFN) — `model/iTransformer.py`.

## Dataset

**PEMS08** — traffic speed sensor network, 170 variates.

| Setting | Value |
|---------|-------|
| Input length | 96 steps |
| Forecast horizon | 12 steps |
| Sensors used | 40 (first 40 of 170, `keep_ratio=0.2`) |
| Metric | MSE (test set, best-val-epoch checkpoint) |

## Setup

```bash
uv sync
```

Place `PEMS08.npz` in `./dataset/PEMS/`.

## Running an experiment

```bash
bash experiment.sh
```

Trains for up to 2 epochs or 5 minutes (whichever comes first). Logs MSE to `results.tsv`.

## Auto-research loop

Invoke with `/loop` inside Claude Code. Each iteration:

1. Reads `program.md` (constraints), `results.tsv` and `git log` (history), `model/CustomModel.py` (current arch)
2. Proposes one focused architectural change
3. Edits `model/CustomModel.py` (and optionally `experiment.sh` for hyperparams)
4. Runs `bash experiment.sh`, parses `mse:{value}` from stdout
5. Keeps the commit if MSE improved over the best in `results.tsv`, otherwise `git reset --hard HEAD~1`
6. Appends one row to `results.tsv`

## Research directions (priority order)

1. Causal patch tokens via dilated conv — verify they outperform flat linear patch
2. Detrend before attention — signed attention on residuals + separate trend forecast
3. Lead-lag encoding — shift key/query patch index by `k` steps to model lagged correlations
4. Head specialisation — different heads capture correlations at different temporal scales
5. Parameter reduction — replace Linear projectors with depthwise conv

## File roles

| File | Purpose |
|------|---------|
| `model/CustomModel.py` | Architecture under research — only file the agent modifies (besides `experiment.sh`) |
| `model/iTransformer.py` | Reference baseline — never modified |
| `experiment.sh` | Launches timed training run |
| `program.md` | Dataset config, constraints, and research directions |
| `results.tsv` | Append-only experiment log: `commit_hash`, `MSE`, `gpu_mem_gb`, `status`, `description` |

## Tunable parameters

The agent may change: `d_model`, `n_heads`, `e_layers`, `d_ff`, `patch_len`, `stride`, `n_stacks`, `attention_window`, `learning_rate` (0.0001–0.01).

Fixed: batch size (32), loss (MSE), normalization (`use_norm=True`), data pipeline.
