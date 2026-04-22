# FITS: Modeling Time Series with 10k Parameters
**Paper:** arXiv 2307.03756 (Xu et al., 2023)

---

### 1. Core Idea (2–3 sentences)

FITS (Frequency Interpolation Time Series) represents a time series entirely in the frequency domain via DFT, discards high-frequency components above a cut-off c, and then applies a learned complex-valued linear layer to interpolate from the compressed spectrum back to the target prediction length. The key insight is that most time series energy is concentrated in low-frequency bins, so a drastic truncation (keeping only c << T/2 bins) loses little information while reducing parameters by orders of magnitude. The full forward pass is: FFT → low-pass truncation → complex linear → IFFT, with all learnable parameters in the single complex linear step.

---

### 2. Method — Mathematics

**Notation:**
- `x ∈ R^(B × T × N)` — input lookback window; B=batch, T=lookback length, N=variates
- `y ∈ R^(B × S × N)` — target horizon; S=prediction length
- `c` — cut-off frequency (hyperparameter, default 0.1·T/2 bins)
- `F(·)` — real FFT; `F^{-1}(·)` — IFFT
- `W ∈ C^(c × (c + S_c))` — complex weight matrix, where S_c corresponds to target length in freq domain

**Step 1 — DFT of input:**
```
X_f = F(x)  ∈ C^(B × (T//2+1) × N)
```

**Step 2 — Low-pass truncation:**
```
X_lp = X_f[:, :c, :]  ∈ C^(B × c × N)
```
Discards bins c..T//2. This is lossy compression; reconstruction error is bounded by the energy in truncated bins (negligible for smooth series).

**Step 3 — Complex linear interpolation:**
```
X_pred_f = W · X_lp  ∈ C^(B × S_c × N)
```
W is a learned complex matrix (real + imaginary parts stored separately). Equivalently:
```
Re(X_pred_f) = Re(W)·Re(X_lp) - Im(W)·Im(X_lp)
Im(X_pred_f) = Re(W)·Im(X_lp) + Im(W)·Re(X_lp)
```
This single matrix maps c input spectral bins to S_c output spectral bins, effectively learning to "stretch" or "interpolate" the spectrum from length T to length S.

**Step 4 — IFFT to prediction:**
```
y_hat = Re(F^{-1}(X_pred_f))  ∈ R^(B × S × N)
```
Only the real part is kept.

**Loss:**
```
L = MSE(y_hat, y) = (1/BSN) Σ (y_hat - y)²
```
Standard MSE on the time-domain prediction. No auxiliary losses.

**Parameter count:**
- Complex W has shape (c × S_c), stored as 2 real matrices → 2·c·S_c real parameters
- With c=0.1·T/2 and S_c≈S/2, for T=336, S=96: c≈17, S_c≈48 → ~1,600 params per variate
- For N variates with separate weights: N·2·c·S_c (shared across variates by default, so ≈10k total)

**Uncertainty flag:** The exact sharing scheme across variates (channel-independent vs. shared W) is not always explicit in descriptions of the method. The "10k parameters" claim implies W is shared across variates.

---

### 3. Architecture / Algorithm

**Forward pass (per sample, channel-independent):**

```
Input:  x  ∈ R^(B, T, N)

1. Normalize (RevIN or mean subtraction per channel):
   x_norm = (x - mean(x, dim=T)) / std(x, dim=T)
   → shape: (B, T, N)

2. FFT along time axis:
   X_f = torch.fft.rfft(x_norm, dim=1)
   → shape: (B, T//2+1, N),  dtype=complex64

3. Truncate to low-pass:
   X_lp = X_f[:, :c, :]
   → shape: (B, c, N)

4. Complex linear (learned):
   X_pred_f = complex_linear(X_lp)
   → shape: (B, S_c, N)
   where S_c = S//2+1 and complex_linear ≈ matmul with W∈C^(c×S_c)

5. IFFT to time domain:
   y_hat = torch.fft.irfft(X_pred_f, n=S, dim=1)
   → shape: (B, S, N),  dtype=float32

6. Denormalize:
   y_hat = y_hat * std + mean

Output: y_hat ∈ R^(B, S, N)
```

**Module structure:**
- `FITS` top-level model
  - `RevIN` normalization (optional but standard)
  - `FITSLayer` per channel (or shared):
    - `nn.Linear(c, S_c, dtype=complex)` implemented as two real linears
  - No attention, no convolution, no MLP trunk

**Computational complexity:**
- FFT: O(T log T · N · B)
- Complex linear: O(c · S_c · N · B)   ← dominant for large N
- IFFT: O(S log S · N · B)
- Total: effectively O(T log T · N · B), far cheaper than transformer O(T² · d · B)

---

### 4. What Makes It Work

**1. Low-frequency energy dominance (the key assumption).**
Real-world time series (weather, electricity, traffic) are band-limited in practice: 80–99% of signal energy lives in the lowest few percent of DFT bins. Truncating at c=0.1·(T/2) is nearly lossless for such signals. This is not a learned property — it is exploited structurally.

**2. Linear interpolation in frequency = global convolution in time.**
The complex linear W·X_lp applies a learned FIR filter to the signal. Frequency-domain interpolation is equivalent to sinc-weighted time-domain convolution across the full window, giving each output step access to the entire lookback without quadratic attention cost. The model learns which frequencies to amplify, attenuate, or phase-shift for each prediction step.

**3. Reversible Instance Normalization (RevIN).**
Without per-instance normalization, distribution shift across the batch causes the fixed frequency representation to fail. RevIN (subtract instance mean, divide by instance std, then undo at output) removes the non-stationary level and scale, leaving a stationary residual that the frequency model handles well. This is shared with PatchTST and others but is critical here given FITS's otherwise rigid spectral parameterization.

---

### 5. Limitations & Failure Modes

**What is not tested:**
- Non-stationary or highly volatile series (financial tick data, anomaly-heavy logs) — the low-frequency assumption breaks down.
- Very short lookback windows (T < 64) where c becomes too small (≤3 bins) to represent meaningful dynamics.
- Multi-resolution or hierarchical patterns (e.g., intraday + weekly seasonality at very different timescales) — a single cut-off c may miss one scale or include too much noise at another.
- Transfer learning / zero-shot generalization: the fixed W is dataset-specific.

**Where it would break:**
- Series with sharp discontinuities or spikes (Gibbs phenomenon inflates high-frequency content, making truncation lossy).
- Prediction horizons much longer than lookback (S >> T): spectral extrapolation requires the model to hallucinate frequency content outside the input support.
- Multivariate inter-channel dependencies: channel-independent mode ignores cross-variate correlations entirely; a shared W cannot capture them.
- Irregular sampling or missing data: DFT assumes uniform time steps; gaps corrupt spectral estimates.

**Theoretical concern:** The complex linear step is equivalent to a non-causal FIR filter with T+S taps — it can overfit to periodic artifacts in training data (e.g., dataset-specific recording cycles).

---

### 6. Research Ideas (ranked)

**Rank 1 — Adaptive cut-off per channel / per instance**
- *Idea:* Learn or predict c per variate (or per batch item) from signal statistics (spectral energy ratio, kurtosis), rather than a fixed global c.
- *Why it might work:* Different variates in multivariate datasets have different spectral profiles; a fixed c wastes capacity on low-variance channels and under-represents high-frequency ones.
- *How to test:* Add a lightweight gating network that predicts a soft mask over frequency bins (sigmoid over c bins) conditioned on per-channel energy. Compare MSE on ETTh1/Weather vs. fixed-c FITS.

**Rank 2 — FITS as a frequency prior / bottleneck inside a larger model**
- *Idea:* Replace or augment the MLP/linear projection in DLinear or iTransformer with a FITS layer as a structured frequency bottleneck.
- *Why it might work:* The inductive bias of smooth spectral interpolation acts as regularization; the rest of the model handles residuals. This is directly relevant to CustomModel.py in this repo.
- *How to test:* In CustomModel, replace the linear trend projection with a FITS-style complex linear operating on the FFT of the trend component; keep the detrended residual for standard processing.

**Rank 3 — Learnable frequency basis (beyond DFT)**
- *Idea:* Replace fixed DFT with a learned orthogonal basis (e.g., initialized to DCT or Legendre polynomials, then fine-tuned), allowing the model to adapt its "frequency" representation to data.
- *Why it might work:* DFT is optimal for stationary sinusoidal signals; real series may be better represented in wavelets or data-driven bases that the DFT cut-off approximates poorly.
- *How to test:* Parameterize the basis as an nn.Linear(T, c) initialized to the first c DFT rows; make it trainable. Measure whether learned basis diverges significantly from DFT.

**Rank 4 — Cross-variate frequency mixing**
- *Idea:* After per-channel FFT truncation, apply a small cross-channel mixing step (e.g., 1×1 conv or small MLP over the N-dim at each frequency bin) before IFFT.
- *Why it might work:* FITS ignores inter-channel phase/amplitude correlations; for datasets like Weather where temperature, pressure, humidity co-vary, a cheap cross-channel mix at c << T/2 bins is O(c·N²) — far cheaper than full attention.
- *How to test:* Add `nn.Linear(N, N)` applied at each of the c frequency bins (or shared across bins). Evaluate on Exchange-Rate (weakly correlated) vs. Weather (strongly correlated) to confirm the benefit is correlation-dependent.

**Rank 5 — Frequency-domain data augmentation**
- *Idea:* During training, randomly permute or drop frequency bins (spectral dropout) or add Gaussian noise to complex coefficients as regularization.
- *Why it might work:* FITS with so few parameters is unlikely to overfit on large datasets, but on small datasets (ETTm1 with <20k samples) it can. Spectral noise injection forces robust coefficient estimation.
- *How to test:* Apply complex Gaussian noise N(0, σ²) to X_lp during training only. Sweep σ ∈ {0.01, 0.1, 0.5} and measure val MSE vs. no-augmentation.

---

### 7. Implementation Notes

**Key hyperparameters:**
| Param | Default | Notes |
|-------|---------|-------|
| `cut_freq` c | 0.1 × (T//2+1) | Critical; too small = underfitting, too large = no compression gain |
| `T` (lookback) | 336 | Must match experiment setup |
| `S` (horizon) | 96/192/336/720 | Separate W per horizon in paper |
| RevIN | enabled | Disable only if data is already stationary |

**Training tricks:**
- Batch size 32–64 works; larger batches do not help (model is so small, gradient noise helps).
- Learning rate 0.001 with Adam; cosine decay is unnecessary given fast convergence (~5 epochs).
- No dropout needed (10k params on 10k+ samples = underparameterized regime).
- Use `torch.fft.rfft` (not `fft`) — rfft exploits real-valued input, halves complex output size.

**Implementation gotchas:**
- `irfft` requires specifying `n=S` explicitly; without it, output length is 2*(S_c-1) which may not equal S.
- Complex linear in PyTorch: use two `nn.Linear` layers for real and imaginary parts separately (complex dtype support is incomplete in older PyTorch versions). In PyTorch ≥ 1.9 you can use `dtype=torch.complex64` but gradient stability may differ.
- The cut-off c must satisfy c ≤ T//2+1 and c ≤ S//2+1; enforce this in model init.
- For channel-independence: apply the same W across all N variates (weight sharing) — this is what gives "10k params" for N=7 variates; with separate W per variate, params scale linearly with N.

**Reference implementation:**
- Official repo: https://github.com/FITS-Time-Series/FITS (check for updates; paper was under review during initial release)
- The core layer is ~20 lines of PyTorch; complexity is in the RevIN wrapper and data loading.

**Baseline context:**
FITS matches or slightly beats DLinear on ETT datasets and underperforms PatchTST on multivariate benchmarks — its strength is parameter efficiency, not raw accuracy. On ETTh1 (horizon=96), FITS MSE ≈ 0.376 vs. DLinear 0.386 vs. PatchTST 0.370 (approximate; varies by normalization choices).
