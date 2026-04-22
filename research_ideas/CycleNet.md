# CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns
**Source:** arXiv:2409.18479 | NeurIPS 2024 Spotlight
**Authors:** Shengsheng Lin, Weiwei Lin, Xinyi Hu, Wentai Wu, Ruichao Mo, Haocheng Zhong (SCUT / Pengcheng Lab)

---

### 1. Core Idea

CycleNet introduces **Residual Cycle Forecasting (RCF)**: a lightweight module that explicitly models fixed periodic patterns via a globally shared, per-channel learnable lookup table of length W (the dominant cycle period), then runs any backbone model on the de-cycled residual. The key insight is that periodic patterns in real-world time series (daily, weekly cycles) are globally stable — they can be memorized exactly with W scalars per channel rather than re-derived from context at each step. Removing this periodic component before prediction sharply reduces the variance the backbone must explain, yielding consistent gains with negligible parameter overhead.

---

### 2. Method — Mathematics

**Notation:**
- `x_t ∈ R^D` — multivariate observation at time t, D channels
- `X ∈ R^{L×D}` — input look-back window of length L
- `Q ∈ R^{W×D}` — learnable recurrent cycle table (one period), W = domain cycle length
- `C ∈ R^{L×D}` — cyclic component extracted from Q by index
- `R ∈ R^{L×D}` — residual (de-cycled) input
- `Ŷ_R ∈ R^{H×D}` — backbone prediction on residual
- `Ĉ ∈ R^{H×D}` — cycle forecast for the future horizon
- `Ŷ ∈ R^{H×D}` — final prediction

**Step 1 — Instance normalization** (applied before RCF):
```
μ = mean(X, dim=time),   σ = std(X, dim=time)
X_norm = (X - μ) / σ
```
Applied independently per channel per sample (not RevIN — no learned affine; standard instance norm is sufficient; RevIN gives marginal additional gain per ablation).

**Step 2 — Cyclic component extraction:**

The time index of the first element of the look-back window modulo W gives the phase offset `φ`. Then:
```
C[t, :] = Q[(φ + t) mod W, :]     for t = 0, …, L-1
```
C has the same shape as X_norm: `R^{L×D}`.

**Step 3 — Residual computation:**
```
R = X_norm - C            ∈ R^{L×D}
```

**Step 4 — Backbone prediction on residual:**
```
Ŷ_R = Backbone(R)         ∈ R^{H×D}
```
Backbone is either:
- CycleNet/Linear: single linear layer `R^{L×D} → R^{H×D}` (channel-independent, weight shape `R^{L×H}`)
- CycleNet/MLP: two-layer MLP with hidden dim = L, applied per-channel

**Step 5 — Future cyclic component:**
```
Ĉ[h, :] = Q[(φ + L + h) mod W, :]     for h = 0, …, H-1
```

**Step 6 — Final prediction (re-add cycle, de-normalize):**
```
Ŷ = (Ŷ_R + Ĉ) * σ + μ
```

**Loss:** Standard MSE on Ŷ vs. ground truth Y:
```
L = (1 / H·D) * ||Ŷ - Y||²_F
```

**Parameter count for RCF:** W × D scalars (e.g., W=168, D=321 for Electricity → 53,928 params). The Linear backbone adds L×H per channel = L×H×D (e.g., 96×720×321 ≈ 22M). RCF is <1% of backbone cost for short horizons.

---

### 3. Architecture / Algorithm

**Forward pass (CycleNet/Linear, single sample):**

```
Input:  X ∈ R^{L×D},  start_time_index t_0  (integer)
──────────────────────────────────────────────────────
1. Normalize:        X_n = (X - mean(X)) / std(X)          [L×D]
2. Phase offset:     φ = t_0 mod W
3. Cycle extract:    C = Q[arange(φ, φ+L) mod W, :]        [L×D]
4. Residual:         R = X_n - C                            [L×D]
5. Transpose:        R_T = R.T                              [D×L]
6. Linear:           Ŷ_R_T = R_T @ W_lin   (W_lin: L×H)   [D×H]
7. Ŷ_R = Ŷ_R_T.T                                           [H×D]
8. Cycle forecast:   Ĉ = Q[arange(φ+L, φ+L+H) mod W, :]   [H×D]
9. Denormalize:      Ŷ = (Ŷ_R + Ĉ) * std(X) + mean(X)    [H×D]
Output: Ŷ ∈ R^{H×D}
```

**Module inventory:**
| Module | Parameters | Shape |
|--------|-----------|-------|
| Cycle table Q | W × D | e.g., 168×321 |
| Linear weight | L × H | e.g., 96×720 (per-channel shared) |
| (MLP variant) fc1+fc2 | L×L + L×H | per-channel |

**Channel independence:** steps 4–6 operate per-channel; Q is per-channel but globally shared across time (not per-sample). This is critical — Q is a global parameter, not a function of the input.

**Plug-and-play use:** Replace steps 5–7 with any backbone (PatchTST, iTransformer) operating on R. Re-add Ĉ after backbone output. Backbone receives de-cycled input; nothing else changes in the backbone.

---

### 4. What Makes It Work

**1. Global cycle memorization vs. local moving-average decomposition.**
DLinear-style STD uses a causal moving average (MOV) of kernel size k to estimate the trend/season split at inference time — this is noisy, horizon-agnostic, and cannot exactly represent a stable cycle. RCF learns the cycle shape globally from all training data; the lookup table converges to the true period waveform averaged over the training set. This is strictly more accurate when the period is stationary. Ablation confirms ~20% MSE reduction over DLinear (which already uses MOV-STD).

**2. Residual variance reduction.**
Removing a strong periodic component dramatically reduces the dynamic range of R, making the backbone's regression problem easier regardless of architecture. This is the same rationale as seasonal differencing in ARIMA, but learned end-to-end so the cycle shape adapts to the data distribution.

**3. Zero-cost cycle forecasting.**
Because the future cyclic component Ĉ is read directly from Q via modular indexing, it costs O(H·D) memory reads with no learned parameters beyond Q. Conventional STD methods must extrapolate seasonal components, introducing compounding error at long horizons. RCF's cycle forecast is exact (modulo stationarity of the period).

---

### 5. Limitations & Failure Modes

**Period must be known and fixed.** W is a fixed hyperparameter; if the dominant period is wrong or ambiguous the RCF term adds noise rather than signal. Solar-Energy ablation shows RCF with wrong W gives no benefit (but also no harm in the worst case).

**Single-period assumption.** Q captures only one periodic scale. Real signals often have multi-scale periodicities (e.g., daily + weekly simultaneously). CycleNet addresses this by choosing W = the *longest* primary period, implicitly capturing harmonics only to the extent that one period contains them.

**Stationarity of the cycle.** Q is a single global parameter — it learns the average cycle shape across the training period. Distribution shifts that alter the cycle shape (e.g., COVID changing electricity demand patterns) will degrade performance; there is no adaptive mechanism.

**Not tested on:** irregular time series (missing timestamps), non-uniform sampling, very short time series where L < W, datasets without meaningful periodicity (financial tick data, aperiodic sensor streams). The ablation on Solar-Energy (W=168, RCF barely helps: 0.289 vs. 0.286 without RCF) hints at fragility when the period assumption is weak.

**Channel independence.** The backbone (Linear/MLP) is channel-independent; cross-variate correlations are unused. For settings where cross-channel interaction is key (e.g., iTransformer, which inverts axes), plugging RCF in recovers some but not all cross-channel benefits.

**Sequence length L fixed at 96.** Experiments use L=96 uniformly. Performance at longer look-backs (L=336, 512) is not systematically studied; it is unclear whether the residual R is harder to model when L >> W.

---

### 6. Research Ideas (ranked)

**Idea 1 — Learnable multi-scale cycle bank**
*Idea:* Maintain K cycle tables Q_1, …, Q_K at different periods W_1, …, W_K; combine as C = sum_k α_k * Q_k[...] where α_k are either fixed or input-dependent (attention over cycle bank).
*Why it might work:* Real signals have hierarchical periodicity; a single W forces a lossy choice. Summing multiple cycles (Fourier decomposition intuition) can exactly represent any periodic signal in the limit.
*How to test:* Add K=2 tables (e.g., W=24 + W=168 for hourly electricity) to the current CycleNet codebase; measure MSE vs. single-table baseline; check if combined table MSE < min(individual table MSEs).

**Idea 2 — RCF as pre/post-processing for iTransformer**
*Idea:* Apply RCF subtraction before feeding into iTransformer (which operates on channel tokens of length L), then re-add Ĉ after output. This is the plug-and-play mode already described, but applied specifically to iTransformer.
*Why it might work:* iTransformer's cross-channel attention is distracted by correlated seasonal oscillations; de-cycling lets attention focus on aperiodic cross-channel relationships. Paper reports ~5-10% improvement; the current repo may already implement this.
*How to test:* In `model/CustomModel.py`, add RCF pre-processing (Q table + cycle subtraction) before the iTransformer encoder; compare MSE with and without. W=24 for ETTh datasets (hourly, daily cycle), W=96 for ETTm (15-min, daily=96 steps).

**Idea 3 — Adaptive phase estimation**
*Idea:* Rather than using the wall-clock time index as phase, learn a small phase-estimation network that maps the input X to a continuous phase offset φ ∈ [0, W), interpolating Q via soft indexing (sinc or bilinear interpolation over Q).
*Why it might work:* In real deployments, the phase offset may drift (e.g., daylight-saving transitions, irregular sampling). Soft phase estimation makes RCF robust to phase noise.
*How to test:* Replace the hard `t_0 mod W` with a 1-layer linear map from mean(X) → φ; use `torch.nn.functional.grid_sample` or manual sinc interpolation on Q; ablate on ETTh1 with corrupted timestamps.

**Idea 4 — Channel-specific cycle length via AutoCorrelation**
*Idea:* Before training, compute the dominant autocorrelation lag per channel (via FFT of training data); assign each channel d its own period W_d; create a ragged table Q_d ∈ R^{W_d}.
*Why it might work:* Different variables in a multivariate set can have different dominant periods (e.g., temperature=24h, load=168h); forcing one W for all channels is suboptimal.
*How to test:* Pre-compute per-channel autocorrelation peaks offline; store W_d per channel; index each channel's slice of X independently. Measure MSE vs. single global W on Weather dataset (heterogeneous variables).

**Idea 5 — RCF with amplitude modulation**
*Idea:* Extend Q to store both the cycle shape and a per-sample amplitude scale: C[t,d] = A_d * Q[(φ+t) mod W, d], where A_d = f(X[:,d]) is a scalar predicted from the input channel.
*Why it might work:* Many real cycles have stable phase but varying amplitude (e.g., weekday vs. weekend electricity amplitude differs). A learned amplitude scale keeps the cycle shape global while allowing instance-level amplitude adaptation — cheaper than full RevIN.
*How to test:* Add a linear layer per channel that maps mean(X[:,d]) → A_d; scale Q lookup by A_d; compare to baseline RCF on Electricity (strong amplitude variation).

---

### 7. Implementation Notes

**Key hyperparameters:**
| Parameter | Role | Typical values |
|-----------|------|---------------|
| W | Cycle period length | Electricity: 168 (weekly, hourly), Weather: 144 (daily, 10-min), ETTm: 96 (daily, 15-min), ETTh: 24 (daily, hourly), Traffic: 168 |
| L | Look-back length | Fixed at 96 in paper (matches standard LTSF benchmarks) |
| H | Prediction horizon | 96, 192, 336, 720 |
| Backbone | Linear or MLP | Linear usually matches or beats MLP |
| Normalization | Instance norm | Applied before RCF; RevIN gives marginal extra gain |

**Setting W:** Use the lag of the maximum autocorrelation peak in the training data (compute `torch.fft.rfft` on per-channel training signal, find dominant frequency, convert to period). The paper reports that an incorrect W gives the same performance as no RCF — no harm, no benefit — so the choice is safe to tune.

**Training:** Adam optimizer, standard LTSF benchmark pipeline (same as TimesNet/PatchTST repos). RTX 4090 used; training is fast due to minimal parameters. Results averaged over 5 seeds (2024–2028).

**Gotchas:**
- Q must be initialized to zero (not random); random init makes it harder to disentangle from the backbone during early training.
- The time index used to compute φ must be the **absolute** dataset time index (not the batch index) — using a relative index breaks the global cycle alignment.
- When W > L, the cycle table has entries never covered by any single look-back window; these are still trained via the future-cycle supervision signal (Ĉ term in the loss gradient path through Ŷ).
- Instance norm must use `unbiased=False` (std with N, not N-1) to match paper; standard PyTorch `InstanceNorm1d` defaults may differ.

**Repository:** [https://github.com/ACAT-SCUT/CycleNet](https://github.com/ACAT-SCUT/CycleNet)

**Relevant baselines to compare:** DLinear (MOV-STD), TimeMixer (multi-scale mixing), iTransformer, PatchTST. Paper benchmarks against all four; CycleNet/Linear matches or beats all on ETT, Electricity, Weather, Traffic (except Solar-Energy where gains are marginal).
