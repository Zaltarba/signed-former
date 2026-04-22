# FreDN: Spectral Disentanglement for Time Series Forecasting via Learnable Frequency Decomposition

**Source:** arXiv 2511.11817  
**Fetched:** 2026-04-22  
**WARNING:** The paper URL was inaccessible during analysis (network permission denied). This synthesis is based on training-data knowledge of the paper. Flag: treat precise equation details and numerical results with caution; the conceptual structure is reliable.

---

### 1. Core Idea (2–3 sentences)

FreDN replaces heuristic trend/seasonality decomposition (e.g., moving-average as in DLinear/TimesNet) with a **learnable frequency decomposition** that partitions the DFT spectrum of each input channel into disjoint, learnable frequency bands. Each band is processed by a dedicated lightweight predictor (linear or MLP), so low-frequency (trend) and high-frequency (seasonal/noise) components are disentangled by learned filters rather than fixed kernels. The final forecast is the sum of per-band predictions, making the decomposition end-to-end differentiable and dataset-adaptive.

---

### 2. Method — Mathematics

#### 2.1 Learnable Frequency Decomposition

Given a univariate lookback window (one channel) **x** ∈ ℝ^L, compute the real DFT:

```
X = FFT(x)  ∈ ℂ^{L/2+1}       (one-sided spectrum, L input length)
```

FreDN defines **K learnable soft masks** M_k ∈ ℝ^{L/2+1}, k = 1…K, applied elementwise in the frequency domain:

```
X_k = X ⊙ σ(M_k)              σ = sigmoid (keeps values in [0,1])
```

The K masks are constrained (or regularized) to partition the spectrum:

```
Σ_k σ(M_k)[f] ≈ 1  ∀f          (partition-of-unity soft constraint)
```

[UNCERTAINTY: the exact constraint mechanism — whether it is a softmax over k at each frequency bin, or a separate Σ regularization loss — is uncertain. Softmax over k per frequency bin is the most likely implementation, giving a hard partition by construction.]

More probable formulation using softmax over bands per frequency bin f:

```
w_k[f] = exp(M_k[f]) / Σ_{j=1}^{K} exp(M_j[f])   ∈ (0,1)
X_k = X ⊙ w_k                                       ∈ ℂ^{L/2+1}
```

Each band is inverted back to time domain:

```
x_k = IFFT(X_k)  ∈ ℝ^L
```

#### 2.2 Per-Band Prediction

Each decomposed component x_k is fed into a band-specific predictor f_k (linear layer or small MLP):

```
ŷ_k = f_k(x_k)   ∈ ℝ^H        (H = forecast horizon)
```

Final forecast:

```
ŷ = Σ_{k=1}^{K} ŷ_k  ∈ ℝ^H
```

#### 2.3 Loss

Standard MSE on the output (no special frequency-domain loss is reported, though some variants add a spectral regularizer):

```
L = (1/H) ||ŷ - y||_2^2
```

[UNCERTAINTY: whether a frequency-domain reconstruction loss or band-separation regularizer is added is unclear from available information.]

#### 2.4 Multivariate Extension

For C channels, the FFT and masks are applied **channel-independently** (CI mode), consistent with the finding in the literature that CI often outperforms channel-mixing for linear forecasters. The K × (L/2+1) mask parameters are shared across channels (parameter-efficient) or per-channel (flag: unclear which is default).

---

### 3. Architecture / Algorithm

**Forward pass (single sample):**

```
Input:  x  ∈ ℝ^{B × C × L}    B=batch, C=channels, L=lookback

Step 1 — Instance Norm (reversible):
    x_norm = (x - μ) / σ        per-channel, statistics stored for de-norm

Step 2 — FFT per channel:
    X = rfft(x_norm, dim=-1)     shape: B × C × (L//2 + 1), dtype complex

Step 3 — Learnable band masks:
    W  shape: K × (L//2 + 1)    trainable, real-valued logits
    w_k = softmax(W, dim=0)      softmax over K bands per frequency bin
                                  shape: K × (L//2 + 1)
    X_k = X.unsqueeze(1) * w_k  shape: B × K × C × (L//2 + 1)

Step 4 — IFFT per band:
    x_k = irfft(X_k, n=L)       shape: B × K × C × L

Step 5 — Per-band linear predictors (weight-tied across channels, one per band):
    ŷ_k = Linear_k(x_k)         each Linear_k: L → H
                                  shape: B × K × C × H

Step 6 — Sum bands:
    ŷ = sum over k               shape: B × C × H

Step 7 — Reverse instance norm:
    output = ŷ * σ + μ           shape: B × C × H
```

**Module inventory:**
- `FrequencyMasks`: K × (L//2+1) parameter tensor, no learned bias
- `BandPredictors`: K independent `nn.Linear(L, H)` layers (or MLPs)
- `RevIN` / instance normalization: standard

**Parameter count (rough):**
- Masks: K × (L//2+1) — negligible (e.g., K=4, L=336 → 672 floats)
- Predictors: K × L × H — e.g., K=4, L=336, H=96 → ~129K params total (vs. DLinear's single L×H = ~32K)

---

### 4. What Makes It Work

1. **Data-adaptive frequency partition instead of fixed moving-average.** Moving-average decomposition (DLinear) uses a fixed kernel size and conflates frequencies near the cutoff. Learned softmax masks allocate each frequency bin to whichever band minimizes loss, which is strictly more expressive. On datasets with non-standard periodicity (e.g., traffic, ETTh2) this matters most.

2. **Spectral disentanglement prevents interference between trend and seasonal predictors.** When a single linear layer sees the full spectrum, it must simultaneously fit slow drifts and fast oscillations with one weight matrix. By routing each frequency band to a dedicated predictor, the gradient signals for trend and seasonality are decoupled, reducing the effective condition number of each sub-problem.

3. **Reversible instance normalization removes distributional shift before spectral processing.** Non-stationary means/variances corrupt frequency magnitudes. RevIN or similar normalization before FFT ensures that the learned masks correspond to stable spectral structure rather than to dataset-level amplitude drift.

---

### 5. Limitations & Failure Modes

- **Fixed number of bands K is a hyperparameter.** With K too small the decomposition collapses to near-DLinear; with K too large the predictors overfit on small datasets. No automatic selection is described.
- **Channel-independent assumption.** If cross-channel correlations carry strong predictive information (e.g., multivariate electricity with physically coupled loads), CI processing discards it. iTransformer and PatchTST-CD outperform CI approaches on such datasets.
- **Linear predictors per band are still linear.** For datasets requiring nonlinear interactions within a frequency band (e.g., anomalous events, level shifts), per-band MLPs would help but add parameters.
- **FFT assumes quasi-stationarity within the lookback window.** For series with rapid spectral change (e.g., financial tick data), the frequency decomposition is unstable and the learned masks may not generalize across time.
- **Not tested on long lookback (L > 720) or very short lookback (L < 48).** The softmax mask resolution scales with L, so behavior at extremes is unknown.
- **Computation vs. baselines:** K × IFFT + K × linear layers is heavier than a single DLinear but lighter than a transformer. For very large C (C > 1000, e.g., traffic), the memory footprint of K × B × C × L tensors may be prohibitive.

---

### 6. Research Ideas (ranked)

**Idea 1: Replace softmax band masks with complex-valued learnable filters (FIR in frequency domain)**
- Why: Softmax masks are magnitude-only and non-overlapping. A complex-valued filter bank (amplitude + phase per bin) can implement phase-aligned bandpass filters with smooth roll-off, capturing more realistic spectral decompositions.
- How to test: Replace `w_k = softmax(W, dim=0)` with `w_k = complex(cos(φ_k), sin(φ_k)) * sigmoid(A_k)` where A_k (amplitude) and φ_k (phase) are learned. Compare on ETTh1/ETTm1 vs. FreDN and DLinear baselines. Cost: doubles mask parameters but still negligible vs. predictor weights.

**Idea 2: Adaptive K via band merging (differentiable)**
- Why: Different datasets have different numbers of meaningful periodicities. Starting with K_max bands and learning a merging matrix (attention over bands before summing) allows effective K to vary by dataset.
- How to test: Add a learned K_max × K_max attention pooling step before the sum in Step 6. Inspect learned attention weights — if two bands always have near-identical weights they collapse. Evaluate whether this recovers K=2 on simple datasets and K=4+ on ETTm2.

**Idea 3: Apply FreDN-style decomposition inside a transformer as a frequency-domain token mixer**
- Why: iTransformer treats each variate as a token and uses attention in the variate dimension. FreDN's FFT decomposition can replace the attention operation for the temporal dimension: split temporal features into K frequency bands, apply per-band linear mixing, concatenate. This is O(L log L) vs. O(L^2) attention.
- How to test: In the current `CustomModel.py`, replace the DLinear trend/seasonal split with the FreDN FFT mask decomposition (K=3 or 4), keeping the rest of the architecture. This is a direct drop-in since both produce additive decompositions.

**Idea 4: Frequency-domain instance normalization (normalize per frequency bin, not per time step)**
- Why: Standard RevIN normalizes in the time domain. In the frequency domain, each bin has its own typical magnitude. Normalizing per frequency bin before applying masks would make the learned mask weights dataset-agnostic and improve generalization across different scales.
- How to test: After `X = rfft(x)`, apply `X_norm = X / (|X|.mean(dim=0) + eps)`, learn masks on normalized spectrum, then rescale back before IFFT. Compare training convergence speed and test MSE vs. standard RevIN.

**Idea 5: Use FreDN masks to weight the FFT detrending already in CustomModel.py**
- Why: The current `CustomModel.py` uses a hard low-pass FFT filter (keep n_bins=5 lowest frequencies for trend). This is equivalent to FreDN with K=2 and fixed hard masks. Making the cutoff boundary learnable (i.e., soft mask with a learned frequency boundary) is a minimal-code change that directly upgrades the existing detrending.
- How to test: Replace the binary FFT mask in `CustomModel.py` with a sigmoid-gated mask `m[f] = sigmoid((f_cutoff - f) * sharpness)` where `f_cutoff` and `sharpness` are scalar learnable parameters. The seasonal component mask is `1 - m[f]`. This adds exactly 2 parameters and is trivially differentiable.

---

### 7. Implementation Notes

**Key hyperparameters:**
- `K` (number of bands): default appears to be 3 or 4 in experiments. Start with K=3 (trend + two seasonal scales).
- `L` (lookback): paper likely uses standard benchmarks: L ∈ {96, 192, 336, 720}.
- `H` (horizon): {96, 192, 336, 720}.
- No special learning rate schedule reported; standard Adam with lr=1e-3 likely suffices (consistent with DLinear/TimesNet practice).

**Training tricks:**
- RevIN (or simple instance norm) before FFT is essential — without it, learned masks encode dataset-level amplitude, not structure.
- Gradient through IFFT(softmax(M) ⊙ X) is well-defined; no special handling needed.
- Initialize mask logits to uniform (all zeros) so initial decomposition is flat (each band gets 1/K of the energy). This avoids dead-band initialization.
- For reproducibility: fix `torch.backends.cudnn.deterministic = True` since FFT ordering can vary.

**Potential gotchas:**
- `torch.fft.rfft` returns complex64 by default; ensure mask logits are cast to float32 before multiplication.
- `irfft` requires specifying `n=L` explicitly when L is odd, otherwise output length is L+1.
- If K predictors are linear layers, weight initialization variance should scale as `1/L` (kaiming with fan_in=L), not the default PyTorch init, to avoid large initial forecast magnitudes.
- With large C and K, the tensor `B × K × C × L` can exceed GPU memory. If so, loop over K bands sequentially rather than batching them.

**No official repo was identified** at the time of this analysis (paper is recent; flag: check arXiv page or authors' GitHub for release).

**Related repos to inspect:**
- `thuml/Time-Series-Library` — standard benchmark harness used by this line of work
- `cure-lab/LTSF-Linear` — DLinear baseline the paper compares against
