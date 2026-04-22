# WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting
**Source:** https://arxiv.org/abs/2412.17176  
**Analyzed:** 2026-04-22

---

### 1. Core Idea

WPMixer applies Wavelet Packet Decomposition (WPD) to decompose a multivariate time series into multiple frequency sub-bands at multiple resolutions, then mixes information across channels and time within each sub-band using lightweight MLP blocks. The key insight is that different frequency components benefit from separate mixing operations rather than joint processing in the original time domain. Reconstruction is done via inverse WPD, producing forecasts that are implicitly multi-resolution consistent.

---

### 2. Method — Mathematics

**Wavelet Packet Decomposition (WPD):**

Given input sequence `x ∈ R^{L×M}` (L = lookback length, M = variates), WPD at level `J` produces `2^J` sub-band tensors:

```
{W_n^j : j = 0, ..., J-1, n = 0, ..., 2^j - 1}
```

Each node `(j, n)` is computed by applying either a low-pass filter `h` or high-pass filter `g` to the parent node:

```
W_{2n}^{j+1}[k]   = sum_m h[m] * W_n^j[2k - m]   (approximation)
W_{2n+1}^{j+1}[k] = sum_m g[m] * W_n^j[2k - m]   (detail)
```

Unlike standard DWT which only recurses on the approximation branch, WPD recurses on both, giving a full binary tree of sub-bands. At level `J`, each sub-band has length `L / 2^J` (assuming dyadic).

**[Uncertainty: the paper may use a fixed J=3 or J=4; exact J value not confirmed — check ablations in the paper.]**

**Mixing within each sub-band:**

For sub-band tensor `Z ∈ R^{(L/2^J) × M}`, two MLP mixing steps are applied:

Channel mixing (variate interaction):
```
Z' = LayerNorm(Z + MLP_C(Z^T)^T)
```
where `MLP_C : R^M → R^M` is shared across time steps.

Temporal mixing (within sub-band time axis):
```
Z'' = LayerNorm(Z' + MLP_T(Z'))
```
where `MLP_T : R^{L/2^J} → R^{L_out/2^J}` maps to the target sub-band length for horizon H.

**Loss:**

Standard MSE on the reconstructed time-domain output:
```
L = (1/HM) * ||y_hat - y||_F^2
```

No wavelet-domain loss is used (reconstruction is done before computing loss). **[Uncertainty: some variants add a frequency-domain auxiliary loss — not confirmed for WPMixer.]**

**Inverse WPD:**

The mixed sub-band outputs `{Z''_n^J}` are recombined via inverse wavelet packet transform to produce `y_hat ∈ R^{H×M}`.

---

### 3. Architecture / Algorithm

**Forward pass, step by step:**

```
Input:  x ∈ R^{B × L × M}   (B=batch, L=lookback, M=variates)

Step 1 — Instance Normalization (RevIN or similar):
  x_norm = (x - mean(x)) / std(x)     # per-instance, per-variate

Step 2 — Wavelet Packet Decomposition:
  Apply WPD to x_norm along time axis → {Z_n : n=1..2^J}
  Each Z_n ∈ R^{B × (L/2^J) × M}

Step 3 — Sub-band Mixing (applied independently per sub-band):
  For each Z_n:
    a. Channel MLP:  Z_n' = Z_n + MLP_C(Z_n)        # mixes M dimension
    b. Temporal MLP: Z_n'' = Z_n' + MLP_T(Z_n')     # maps L/2^J → H/2^J
  Output: {Z_n'' ∈ R^{B × (H/2^J) × M} : n=1..2^J}

Step 4 — Inverse Wavelet Packet Transform:
  y_hat_norm = IWPT({Z_n''})   ∈ R^{B × H × M}

Step 5 — Denormalization:
  y_hat = y_hat_norm * std(x) + mean(x)

Output: y_hat ∈ R^{B × H × M}
```

**Module order:** RevIN → WPD → [ChannelMLP + TemporalMLP] × 2^J → IWPT → RevIN^{-1}

**Parameter count per sub-band:** Two MLPs, each typically 1-2 hidden layers. Total params scale as `O(2^J * (M^2 + (L/2^J)^2))`. For J=3, M=7, L=336: ~8 sub-bands, each with small MLPs. WPMixer claims parameter efficiency vs. Transformers by avoiding quadratic attention.

**[Uncertainty: whether channel and temporal MLPs share weights across sub-bands is not confirmed — likely independent per sub-band based on the "independent mixing" framing.]**

---

### 4. What Makes It Work

**1. Frequency-aware decomposition before mixing.**
Mixing in the wavelet packet domain separates frequency scales that have different temporal structures (trend vs. seasonality vs. noise). MLPs operating on individual sub-bands see a more stationary, narrowband signal that is easier to fit than the raw mixed-frequency time series. This is the primary inductive bias that differentiates WPMixer from TimeMixer or TSMixer which mix in the time domain.

**2. Multi-resolution temporal lengths reduce MLP input size.**
At decomposition level J, each sub-band has length L/2^J. The temporal MLP input shrinks exponentially, reducing parameter count and avoiding the overfitting risk that plagues large MLP-Mixer models on short datasets. This is why WPMixer can use deeper decompositions without blowing up model size.

**3. Lossless reconstruction constraint via IWPT.**
Because IWPT is an exact inverse of WPD (for orthogonal wavelets), the architecture enforces that the model's output lives in a consistent signal space. There is no information bottleneck from the decomposition itself — all capacity is in the MLP weights.

---

### 5. Limitations & Failure Modes

- **Dyadic length requirement:** WPD requires `L` divisible by `2^J`. Arbitrary lookback lengths require padding or truncation, which introduces boundary artifacts.
- **Wavelet choice is a hyperparameter with little guidance:** Daubechies db1–db8, Symlets, Coiflets all give different filter shapes. The paper likely uses a fixed choice (probably Haar or db4); transferability to new domains is unclear.
- **Channel mixing assumes inter-variate structure:** On datasets where variates are independent (e.g., randomly shuffled channel order), channel MLP mixing may hurt or at best be neutral. Not tested.
- **Not tested on irregular or missing-data time series:** WPD assumes uniform sampling; gaps break the convolution structure.
- **Long horizon degradation untested at very long H:** Standard benchmarks use H ∈ {96, 192, 336, 720}. Performance at H > 720 is unknown.
- **Single-scale decomposition depth J fixed globally:** Different variates may have different dominant frequencies; a fixed J is a one-size-fits-all choice.
- **Computational cost of IWPT at inference:** For embedded / low-latency applications, the repeated convolutions across 2^J sub-bands add non-trivial overhead vs. a direct MLP.

---

### 6. Research Ideas (ranked)

**Idea 1 — Learnable wavelet filters (highest priority)**
- **Idea:** Replace fixed orthogonal wavelet filters (h, g) with learnable lifting-scheme parameters, trained end-to-end with the MLP mixers.
- **Why it might work:** Fixed wavelets (Haar, Daubechies) are domain-agnostic. Learned filters can adapt the frequency partition to the dataset's actual spectral content (e.g., 24h electricity cycles, weekly patterns). Lifting-scheme parameterization guarantes perfect reconstruction during training.
- **How to test:** Replace `pywt.wavedec2` with a lifting-scheme layer with 4-8 learnable filter taps per level; add orthogonality regularization. Compare MSE on ETTh1/ETTm2 vs. fixed-filter WPMixer. Expected compute overhead: +10-20%.

**Idea 2 — Adaptive decomposition depth per variate**
- **Idea:** Use a different J per variate (or variate group), selected by a lightweight spectral analysis of the input (e.g., dominant frequency bin from FFT determines J).
- **Why it might work:** In Weather (21 variates), temperature and pressure have very different dominant frequencies. A single J wastes sub-band budget on variates that are already smooth or already high-frequency.
- **How to test:** Compute per-variate FFT on the lookback window, map dominant period to J ∈ {2,3,4}, route each variate to the corresponding decomposition branch. This is data-dependent routing with no learned parameters beyond the MLP mixers.

**Idea 3 — FFT detrending before WPD (directly relevant to current codebase)**
- **Idea:** Apply a low-pass FFT filter to separate trend before WPD, so WPD operates only on the detrended residual. Trend is forecasted by a separate linear layer.
- **Why it might work:** WPD's lowest-frequency sub-band still mixes trend and slow seasonality. Explicit trend separation (similar to what this repo already does in `CustomModel.py`) may prevent the trend from dominating the lowest sub-band and make all sub-bands more narrowband.
- **How to test:** Add a `FFTDetrend` module before WPD; pass trend to a linear projector; add WPD-mixed residual to trend prediction at output. Ablate: trend-only vs. residual-only vs. combined.

**Idea 4 — Sub-band dropout as regularization**
- **Idea:** During training, randomly zero out entire sub-bands (drop a full frequency level) with probability p=0.1-0.2, forcing the model to reconstruct from incomplete frequency information.
- **Why it might work:** Analogous to DropPath/Stochastic Depth in vision. Prevents the model from over-relying on any single frequency band. Should improve generalization on short datasets (ETTh1: 17k samples) where overfitting is common.
- **How to test:** Add a `SubbandDrop` layer after mixing, active only in training. Sweep p ∈ {0.05, 0.1, 0.2}. Measure val MSE at epoch 20 vs. no dropout.

**Idea 5 — Cross-sub-band attention (low priority)**
- **Idea:** After per-sub-band MLP mixing, add a single cross-sub-band attention layer that lets sub-bands attend to each other (2^J tokens, each of dim M * H/2^J).
- **Why it might work:** Frequency bands are not independent — e.g., trend modulates the amplitude of seasonal components. A cross-band attention layer can capture multiplicative interactions at low cost (2^J is at most 16 tokens for J=4).
- **How to test:** Add 1 multi-head attention layer (2^J × 2^J) after all sub-band MLPs, before IWPT. Monitor whether attention heads specialize by frequency pair.

---

### 7. Implementation Notes

**Key hyperparameters (from paper / standard practice):**
- `J` (decomposition depth): typically 3–4; larger J → more sub-bands, shorter per-band sequences; diminishing returns beyond J=4 for L=336.
- `wavelet`: Haar (db1) or Daubechies db4 are most common; db4 has better frequency localization at cost of boundary effects.
- `d_model` (MLP hidden dim per sub-band): 64–256; paper likely uses 128 or 256.
- `dropout`: 0.05–0.1 standard for MLP-Mixer family.
- `lr`: 1e-3 with OneCycleLR or cosine decay; WPMixer-family models tend to train fast (~10 epochs to convergence).
- RevIN (reversible instance normalization): almost certainly used; without it, distribution shift across train/test causes large degradation.

**Training tricks:**
- Patch the input length to the nearest power of 2 (or next multiple of 2^J) before WPD to avoid boundary artifacts — use reflection padding not zero padding.
- Use `pywt` (PyWavelets) for fast CPU/GPU WPD, or implement as 1D convolutions for full GPU utilization: filters become fixed conv1d kernels with stride=2.
- Orthogonal wavelets have exact perfect reconstruction; biorthogonal wavelets (e.g., bior2.2) do not — verify IWPT reconstruction error is ~1e-6 or below before training.

**Implementation via conv1d (recommended for speed):**
```python
# WPD as strided conv (Haar example)
h = torch.tensor([0.5, 0.5])    # low-pass
g = torch.tensor([0.5, -0.5])   # high-pass
# Apply as Conv1d(groups=M, kernel_size=2, stride=2, weight=h or g)
# Recurse J times on both branches
```

**Relevant repos:**
- Official WPMixer: likely at `https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer` (unconfirmed — search GitHub for "WPMixer time series").
- PyWavelets: `pip install PyWavelets` — `pywt.wavedec(x, 'db4', level=J, mode='periodization')` returns list of sub-bands.
- TSMixer (reference MLP-Mixer): `https://github.com/google-research/google-research/tree/master/tsmixer`

**Gotchas:**
- `mode='periodization'` in pywt is required for exact length `L/2^J` outputs (no boundary extension). `mode='reflect'` gives slightly longer arrays that are harder to batch.
- When adapting to this repo: `CustomModel.py` uses `(B, L, M)` layout (batch, time, channel); pywt expects `(B, M, L)` — transpose before/after WPD.
- IWPT after per-sub-band MLP mixing requires the sub-band lengths to be exactly `H/2^J`; ensure H is also divisible by `2^J` or pad H accordingly.
