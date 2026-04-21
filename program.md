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
- train_epochs: 4  (calibrate manually — increase if 5 min budget is not reached)
- time_budget: 600  (seconds; hard stop at epoch boundary if epochs haven't finished)
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

The agent may also use `x_mark_enc` and `x_mark_dec` (temporal features: hour, weekday, day, month — shape `(B, T, 4)` for freq=h) inside `model/CustomModel.py`. These are **predictable** features: `x_mark_dec` covers the future horizon and is always available at inference time, so they can be used freely for trend/seasonal estimation without any leakage.

Do not touch: optimizer, learning rate, batch size, keep_ratio, data pipeline, any file other than `model/CustomModel.py` and `experiment.sh`.

## Target

A good MSE on PEMS08 is **≤ 0.08**. Use this as the long-term reference point — not just "better than current best".

## Iteration Procedure

1. Read `ideas_log.txt` (all entries) — identify which changes improved MSE; do not use `git log`
2. Count how many iterations have been run (from `results.tsv`). If fewer than 10, prioritise **structural changes** (new mechanisms). After iteration 10, shift toward exploiting what worked.
3. Before picking a new hypothesis, ask: has a recent idea been tried with only one hyperparameter setting? If so, consider a variant before moving on — some ideas need tuning to work, not discarding.
4. Form a hypothesis: one focused change and why it might help. Label it as either:
   - **structural** — a new mechanism; pick patch_len and lookback window that make sense for the idea, not just defaults
   - **parametric** — a hyperparameter sweep (plan 2–3 values across a range before concluding)
5. Implement it in `model/CustomModel.py` and/or `experiment.sh`
6. Run the experiment (see CLAUDE.md for the exact command)
7. Record result in `results.tsv` and append to `ideas_log.txt`
8. Keep commit if MSE improved by ≥ 0.001 over best so far, otherwise `git reset --hard HEAD~1`

### When to stop exploring a hypothesis

- **Structural change** with no improvement on one run → discard, move on
- **Parametric idea** (patch_len, stride, attention_window, d_model, etc.) → try at least 2–3 values across a reasonable range before discarding; note in `ideas_log.txt` which values were tried
- If all variants of a parametric idea failed, mark it as exhausted in `ideas_log.txt` so it is not revisited

### Scaling stacks

Before increasing `n_stacks`, verify the idea works at smaller scale first: run 1 stack, then 2, before trying 3+. Scaling without a working single-stack baseline wastes budget.

## Baseline

Starting architecture: iTransformer — inverted token embedding, full attention, FFN.  
Reference: `model/iTransformer.py`

---

## Research Directions

### Goal

Design a **general-purpose lightweight Transformer** for multivariate time series, validated on PEMS08. The architecture should generalise beyond this dataset — avoid PEMS-specific hacks.

### Core architecture: space-time attention

Each variate is split into `n_patches` patch tokens → the full sequence has `n_variates × n_patches` tokens. Attention is **non-factorized**: all tokens attend jointly in a single operation, mixing both the time and variate dimensions simultaneously. Do not factor into separate temporal + cross-variate attention passes — this is a hard constraint.

A **causal mask** restricts each token at time patch `t` to attending only to tokens at time patches `≤ t` (across all variates). `attention_window` limits how many past time patches are visible. Whether to use strict `< t` (excluding same-time-patch cross-variate attention) is an open research question worth exploring.

### Key research insight: lead-lag correlation matrix

The attention matrix over `n_variates × n_patches` tokens is the central object of interest. Viewed as variate×variate blocks at each time offset `k`, it gives a **series of cross-variate correlation matrices at lag `k`**. This is the lead-lag correlation structure: block `(i, j, k)` represents how variate `i` at time `t` correlates with variate `j` at time `t-k`. Preserving this interpretability is non-negotiable.

### Hard constraints

- **Parameter count ≤ 50,000.** Verify with `sum(p.numel() for p in model.parameters())` before committing.
- **Non-factorized joint space-time attention.** No separate temporal / cross-variate passes.
- **Attention matrix interpretable as a lead-lag correlation structure** (see above). Every token must be a causally-constructed time series segment — not a generic embedding vector.
- **No future leakage.** Patch tokens built from time steps ending at `t` only.

### Design philosophy

Prefer causal convolutions over large linear projections. Each attention head represents an independent view of the same series segment (shape `(n_heads, patch_len)`). Keep every intermediate representation interpretable as a time series.

### Attention

Signed attention `softmax(+s) - softmax(-s)` is the preferred default — it maps naturally to positive/negative correlations. Use it unless a clearly better signed alternative is found. Correlation is best estimated on detrended / residual series — apply trend decomposition before attention when possible.

### Improvement threshold

A change is only worth keeping if **MSE improves by at least 0.001** over the current best. Smaller deltas are noise given the training budget — reset and try something else.

### Hypotheses to explore (unordered)

**Architectural / structural:**
- Strict vs. non-strict causal mask: `< t` vs. `≤ t` for same-time-patch cross-variate attention
- Head specialisation: different heads capture correlations at different lags
- Parameter reduction: don't use Linear projectors but  convolution  where possible
- Causal patch tokens via dilated conv

**Signal decomposition (good directions to explore):**
- Detrend before attention: moving-average trend removal, run signed attention on residuals, forecast trend separately
- **FFT detrending**: use the real FFT of the input to extract a low-frequency trend component (keep the lowest K frequency bins, reconstruct via iFFT) — no learnable parameters, exact and differentiable. Run signed attention on the detrended residual; forecast trend separately via a linear projection.
- FFT token embedding: represent each patch as its frequency spectrum (magnitude + phase) before attention
- Wavelet token embedding: multi-resolution patch decomposition (e.g. Haar or db1 wavelet) to capture structure at several scales
- Mel-log spectrogram embedding: apply mel filterbank to patches, use log-compressed spectral energy as token features — good for capturing periodic patterns in traffic data

**Temporal mark trend learning (priority direction — explore for at least 3–4 iterations):**

The pipeline is:
1. RevIN normalization on `x_enc`
2. Learn a **mark-conditioned trend** from `x_mark_enc` (a lightweight model, e.g. MLP or Fourier basis over temporal features) and subtract it from the RevIN-normalized `x_enc` to get residuals
3. Run the encoder (space-time attention) on the residuals to forecast the residual component
4. Predict the future trend by running the same mark model on `x_mark_dec[:, -pred_len:, :]` — this is leakage-free because temporal marks are always known in advance
5. Final forecast = residual forecast + trend forecast, then RevIN denormalization

Variants to explore across iterations (do not give up after one failure):
- **Architecture of the mark model**: MLP with 1–2 hidden layers; Fourier basis (learnable sin/cos harmonics per temporal feature); linear projection
- **Number of harmonics** (if Fourier): 2, 4, 8
- **Whether to also use FFT detrending first**, then use marks for residual-of-residual
- **Interaction between mark model and RevIN**: try applying mark model before vs. after RevIN
- Mark-conditioned trend should be tested with and without the encoder — a mark-only baseline (no attention) helps isolate its contribution

### Reference papers

- **iTransformer**: variate-as-token, attention over variates
- **PatchTST**: patch-as-token, attention over time (channel-independent)
- **N-BEATS**: stacked residual forecasting blocks with backcast/forecast decomposition
- **CMoS**: correlation-based multivariate structure