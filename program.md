# Program

## Dataset

```
data:            PEMS
root_path:       ./dataset/PEMS/
data_path:       PEMS08.npz
features:        M
seq_len:         96
pred_len:        12
enc_in:          170     # full sensor count; auto-reduced to floor(170 * keep_ratio) = 40
freq:            h       # unused for PEMS (no time marks)
keep_ratio:      0.2     # first 40 of 170 sensors used across train/val/test
```

`keep_ratio` is applied identically to train/val/test (always the first N columns). The scaler is fit only on the kept variates. `enc_in`, `dec_in`, and `c_out` are automatically adjusted in `run.py`.

## Fixed Settings (do not change)

- Optimizer: Adam, lr=0.001
- Batch size: 32
- train_epochs: 2  (calibrate manually — increase if 5 min budget is not reached)
- time_budget: 300  (seconds; hard stop at epoch boundary if epochs haven't finished)
- Loss: MSE
- Early stopping patience: 3 (whichever triggers first: early stopping or time budget)
- Normalization: use_norm=True

## What the Agent May Change

Modify `model/CustomModel.py` freely:

- **Dimensions**: d_model, n_heads, e_layers, d_ff, 
- **Patch config**: patch_len, stride, n_stacks (set in `experiment.sh` and forwarded via configs)
- **Building blocks**: add, remove, or replace encoder layers
- **Attention**: swap or modify the attention mechanism (e.g. attention_window)
- **Embedding**: change how variates or time steps are embedded
- **Normalization**: add, remove, or reorder norm layers
- **Positional encoding**: modify or remove

The agent may also edit the corresponding args in `experiment.sh` to change `patch_len`, `stride`, `n_stacks`, `attention_window`, `d_model`, `n_heads`, `e_layers`.

Do not touch: optimizer, learning rate, batch size, keep_ratio, data pipeline, any file other than `model/CustomModel.py` and `experiment.sh`.

## Iteration Procedure

1. Read `results.tsv` and `git log --oneline -20` — identify which changes improved MSE
2. Form a hypothesis: one focused change and why it might help
3. Implement it in `model/CustomModel.py`
4. Run `bash experiment.sh`
5. Record result in `results.tsv`
6. Keep commit if MSE improved over best so far, otherwise `git reset --hard HEAD~1`

## Baseline

Starting architecture: iTransformer — inverted token embedding, full attention, FFN.  
Reference: `model/iTransformer.py`

---

## Research Directions

### Core design philosophy

Build a **lightweight Transformer** where every intermediate representation remains interpretable as a time series. Parameter count should stay low; prefer causal convolutions over large linear projections.

Key invariant: a token of shape `(n_heads, patch_len)` must be a causally-constructed segment of a time series — not a generic embedding vector. Each head represents an independent view of the same series segment.

### Token construction

A patch token for variate `i` at time `t` is built from `patch_len` consecutive time steps ending at `t`, extracted causally (no future leakage). Temporal structure inside the token must be preserved — construction via causal conv is preferred over a raw linear projection.

### Attention as a correlation matrix

The attention matrix should be interpretable as a **signed cross-variate correlation matrix**:
- Positive attention weight = positive correlation between two variate patches
- Negative attention weight = negative correlation
- This motivates the signed attention: `softmax(+s) - softmax(-s)`

Correlation is best estimated on **detrended / residual series** — apply trend decomposition before attention if possible, run attention on the residual.

### Lead-lag relationships

The model should be able to capture **lead-lag** structure: variate A at time `t` predicts variate B at time `t+k`. This requires the attention to operate across patches at different time offsets, not just within the same patch position.

### Reference papers

- **iTransformer**: variate-as-token, attention over variates
- **PatchTST**: patch-as-token, attention over time
- **N-BEATS**: stacked residual forecasting blocks with backcast/forecast decomposition
- **CMoS**: correlation-based multivariate structure

### Hypotheses to explore (priority order)

1. Causal patch tokens built via dilated conv (current) — verify they outperform a flat linear patch
2. Detrend before attention: run signed attention on residuals, add trend forecast separately
3. Lead-lag encoding: shift the key/query patch index by `k` steps to explicitly model lagged correlations
4. Head specialisation: encourage different heads to capture correlations at different temporal scales (short vs. long lag)
5. Parameter reduction: replace Linear projectors with depthwise conv where possible 