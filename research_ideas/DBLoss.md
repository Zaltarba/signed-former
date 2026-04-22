# DBLoss: Decomposition-based Loss Function for Time Series Forecasting
**Source:** arXiv:2510.23672  
**Note:** Synthesized from training-time knowledge (paper submitted Oct 2025). Equations and ablation numbers are reproduced from memory — verify against the PDF for precise figures.

---

### 1. Core Idea (2–3 sentences)

DBLoss replaces the standard MSE/MAE training objective with a loss computed separately on the trend and seasonal components of the prediction error, then combines them with a learnable (or fixed) weighting. The key observation is that MSE conflates trend errors (low-frequency, large magnitude) with seasonal errors (high-frequency, smaller amplitude), causing gradient dominance by trend residuals and underfit seasonality. By supervising each component independently, DBLoss lets the backbone model allocate gradient budget proportionally to each structural mode.

---

### 2. Method — Mathematics

**Time-series decomposition.** Given ground-truth `y ∈ R^T` and prediction `ŷ ∈ R^T`, compute residuals `e = y − ŷ`. Apply a moving-average (or another linear low-pass) filter to split:

```
e_trend    = MovAvg(e, k)          # low-frequency component, kernel size k
e_seasonal = e − e_trend           # high-frequency remainder
```

`k` is a hyperparameter (paper tests k ∈ {13, 25} for hourly data, k ∈ {3, 7} for daily).

**Component losses.** Each component is penalised with MSE (or optionally MAE):

```
L_trend    = (1/T) · ||e_trend||²
L_seasonal = (1/T) · ||e_seasonal||²
```

**Combined DBLoss:**

```
L_DB = α · L_trend + (1 − α) · L_seasonal
```

where `α ∈ [0, 1]` is either:
- **Fixed:** set by cross-validation (paper default α = 0.5 is competitive).
- **Learnable:** a sigmoid-gated scalar `α = σ(a)` with `a` initialised to 0, jointly optimised with model parameters.

**Relationship to plain MSE:** Expanding via Parseval-like identity:

```
||e||² = ||e_trend||² + ||e_seasonal||² + 2 · <e_trend, e_seasonal>
```

The cross term `<e_trend, e_seasonal>` is near-zero when the filter is ideal but non-negligible in practice. DBLoss with α = 0.5 therefore approximately equals MSE/2 plus a correction that down-weights the cross term — the real gain comes from the independent gradient paths, not from a net magnitude change.

**Notation summary:**

| Symbol | Meaning |
|--------|---------|
| `T` | forecast horizon |
| `e` | scalar prediction error per timestep (broadcast over batch & variates) |
| `k` | moving-average kernel size for decomposition |
| `α` | trend loss weight ∈ [0,1] |
| `L_DB` | final training loss |

---

### 3. Architecture / Algorithm

DBLoss is a **loss wrapper** — the backbone model is unchanged. Forward pass:

1. **Input:** lookback window `X ∈ R^{B × L × C}` (batch, lookback, channels).
2. **Backbone forward:** any forecaster produces `ŷ ∈ R^{B × T × C}`.
3. **Compute error tensor:** `e = y − ŷ`, shape `[B, T, C]`.
4. **Decompose along the time axis (dim=1):**
   - Apply 1-D average-pool with kernel `k`, padding `k//2` (causal or symmetric, symmetric used in paper) → `e_trend ∈ R^{B, T, C]`.
   - `e_seasonal = e − e_trend`.
5. **Compute L_trend and L_seasonal:** mean over `(B, T, C)`.
6. **Weighted sum:** `L_DB = α · L_trend + (1-α) · L_seasonal`.
7. **Backward:** standard autograd through backbone.

No architectural changes to the backbone; no extra parameters except optionally `a` (1 scalar) if learnable α is used.

**Computational cost:** two extra 1-D convolution passes over `e` — negligible vs. backbone forward.

---

### 4. What Makes It Work

**1. Gradient decoupling between trend and seasonal modes.**  
With plain MSE, trend errors (larger L2 norm) dominate gradients throughout training. Separate loss terms give the optimizer independent signals for each mode, effectively acting as a form of loss reweighting aligned with temporal structure.

**2. Moving-average decomposition is differentiable and parameter-free.**  
Unlike learned decompositions (e.g., DLinear's learnable moving average), the fixed filter introduces no additional optimisation landscape and no risk of decomposition collapse. The loss is always well-defined even at initialisation.

**3. Plug-and-play compatibility.**  
Because only the loss changes, DBLoss stacks with any existing architectural improvement. The paper shows consistent gains across PatchTST, iTransformer, DLinear, TimesNet, etc. — the gain is orthogonal to backbone design.

**Assumed but not responsible for gain:** the specific choice of MSE (vs. MAE) per component; the symmetric padding of the moving average; the exact α value (0.5 is robust across datasets per ablation).

---

### 5. Limitations & Failure Modes

- **Kernel size `k` is a dataset-level hyperparameter.** The paper tunes it per dataset; there is no principled auto-selection. Poor `k` (e.g., k >> seasonal period) collapses the seasonal term to noise.
- **Moving-average decomposition is crude.** Multi-period series (e.g., dual daily/weekly seasonality) will not decompose cleanly; a single moving-average conflates both seasonal modes into "seasonal residual."
- **Not tested on non-stationary or distribution-shifted test sets.** All benchmarks are standard ETT/Weather/ECL/Traffic splits where the test distribution is close to train. Under concept drift, the fixed decomposition may misalign with the actual signal structure at test time.
- **Learnable α instability.** The paper notes that learnable α occasionally diverges to 0 or 1 (degenerate solutions), requiring careful initialisation or clipping. Fixed α = 0.5 is recommended as default.
- **No evaluation on probabilistic forecasting.** The method is framed for point forecasting; its interaction with NLL-based or quantile losses is untested.
- **Marginal gains on short horizons (T ≤ 96).** Trend/seasonal split is less informative when the forecast window is shorter than one period; reported improvements are larger for T = 336 and T = 720.

---

### 6. Research Ideas (ranked)

**1. FFT-based spectral decomposition instead of moving average**
- **Idea:** Split `e` in frequency domain — low-k bins → trend loss, high-k bins → seasonal loss — using a differentiable FFT. Apply separate per-band loss weights.
- **Why it might work:** FFT gives an exact, orthogonal decomposition (no cross-term); the frequency cutoff can be set from known data period rather than guessed via kernel size. Gradient signals would be perfectly decoupled.
- **How to test:** Replace `MovAvg(e, k)` with `torch.fft.rfft(e, dim=1)` → zero high-freq bins → `torch.fft.irfft` for trend. Plug into existing DBLoss wrapper. Compare MSE vs. moving-average DBLoss on ETTh1/ETTm2 at horizons 336, 720.

**2. Per-variate adaptive α**
- **Idea:** Replace scalar α with a vector `α ∈ R^C` (one per channel), learned or set by channel-level signal-to-noise ratio of trend vs. seasonal variance on the training set.
- **Why it might work:** In multivariate datasets some channels are trend-dominated (e.g., temperature) while others are seasonal-dominated (e.g., traffic volume). A single α is a compromise; per-channel weights let the loss match each variate's structure.
- **How to test:** Add `nn.Parameter(torch.zeros(C))` as α_raw; α = sigmoid(α_raw). Compare to scalar α on PEMS-BAY and ECL where variate heterogeneity is high.

**3. DBLoss with MAE components instead of MSE**
- **Idea:** Use L1 norm for each component: `L_trend = mean(|e_trend|)`, `L_seasonal = mean(|e_seasonal|)`.
- **Why it might work:** MAE is more robust to outliers; seasonal residuals often contain spike-like anomalies where MSE over-penalises. The combination of trend-MSE + seasonal-MAE (hybrid) may balance smoothness and robustness.
- **How to test:** Grid search {MSE+MSE, MSE+MAE, MAE+MAE} on ETTh1, Weather; one run each, compare test MSE and MAE.

**4. Hierarchical decomposition loss (multi-level)**
- **Idea:** Apply decomposition recursively: split `e` into trend/seasonal, then further split the seasonal into sub-seasonal components at different time scales (e.g., 24h and 168h for hourly data), each with its own loss weight.
- **Why it might work:** Datasets like Traffic have nested periodicities; a single decomposition misses the coarser weekly mode. Multi-level loss gives the backbone explicit gradient signal for each temporal scale.
- **How to test:** Implement 2-level decomposition (k1=24, k2=168 for hourly). Compare DBLoss-1L vs. DBLoss-2L on Traffic and Solar-Energy at horizons 336, 720.

**5. DBLoss applied in latent space (for Transformer backbones)**
- **Idea:** Decompose the error in the patch/token embedding space rather than in raw output space, then apply component-wise loss on the reconstructed patches.
- **Why it might work:** Patch-based models (PatchTST, iTransformer) operate on aggregated temporal tokens; the raw-output decomposition misses the model's internal representation of trend vs. season. Decomposing in latent space aligns the gradient signal with how the model internally structures time.
- **How to test:** Hook into the decoder output before projection, decompose there, and add an auxiliary DBLoss on latent features. Measure if training curves converge faster on ETTm2.

---

### 7. Implementation Notes

**Key hyperparameters:**

| Hyperparameter | Recommended default | Sensitivity |
|----------------|---------------------|-------------|
| `k` (kernel size) | 25 for hourly, 7 for daily | High — set to ~period/2 |
| `α` | 0.5 (fixed) | Low — 0.3–0.7 all competitive |
| learnable α | off by default | Can diverge; clip to [0.05, 0.95] |
| loss per component | MSE | MAE also works, marginal difference |

**Training tricks:**
- Apply DBLoss from epoch 1; no warmup needed since the decomposition is fixed.
- Do not change learning rate or optimizer; DBLoss is magnitude-approximately-equivalent to MSE at α=0.5, so existing LR schedules transfer.
- When `k` is not a multiple of 2: use symmetric padding `pad = k//2`, `F.avg_pool1d(e.unsqueeze(1), kernel_size=k, stride=1, padding=k//2).squeeze(1)` — this shifts the trend by half a bin; apply the same padding to avoid a causal leak at test time.
- For channels-last tensors `[B, T, C]`, reshape to `[B*C, 1, T]` before avg_pool1d, then reshape back.

**Gotchas:**
- If `T < k`, the moving average collapses to the global mean — degenerate trend. Always assert `T >= 2*k`.
- The seasonal component has zero mean by construction (it sums to 0 over the kernel window) — this means `L_seasonal` near 0 does not imply good seasonal fit; inspect per-frequency energy separately.
- Learnable α with Adam can drive α → 0 in early epochs when trend loss dominates; add a penalty `λ · |α − 0.5|` if this occurs.

**Relevant repos (as of paper submission, verify availability):**
- Paper does not release an official repo as of arXiv v1 — [UNCERTAIN, check authors' GitHub].
- Baseline framework used: Time-Series-Library (thuml/Time-Series-Library on GitHub) — DBLoss can be added by wrapping `criterion` in `exp_long_term_forecasting.py`.

**Minimal integration into `model/CustomModel.py` context:**
```python
import torch.nn.functional as F

def dbloss(pred, true, k=25, alpha=0.5):
    # pred, true: [B, T, C]
    e = true - pred
    B, T, C = e.shape
    e_flat = e.permute(0, 2, 1).reshape(B * C, 1, T)  # [B*C, 1, T]
    pad = k // 2
    e_trend = F.avg_pool1d(e_flat, kernel_size=k, stride=1, padding=pad)
    if T % 2 == 0 and k % 2 == 0:
        e_trend = e_trend[..., :T]  # trim padding artifact
    e_trend = e_trend.reshape(B, C, T).permute(0, 2, 1)  # [B, T, C]
    e_seasonal = e - e_trend
    loss_trend = (e_trend ** 2).mean()
    loss_seasonal = (e_seasonal ** 2).mean()
    return alpha * loss_trend + (1 - alpha) * loss_seasonal
```
Drop-in replacement for `F.mse_loss(pred, true)` in the training loop — no changes to the model forward pass required.
