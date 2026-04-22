# MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting
**Paper:** https://arxiv.org/abs/2405.16440  
**Authors:** Xiuding Cai et al., 2024

---

### 1. Core Idea

MambaTS adapts the Mamba selective SSM architecture for multivariate long-term time series forecasting by scanning variables rather than time steps (Channel-Independent Mamba, CIM), and introduces a Variable Permutation Training (VPT) strategy to break the artificial ordering imposed on unordered multivariate channels. It further adds Temporal Mamba Blocks (TMB) that capture intra-series temporal patterns while sharing parameters across channels to regularize learning.

---

### 2. Method — Mathematics

**Vanilla SSM (continuous):**
```
h'(t) = A h(t) + B x(t)
y(t)  = C h(t) + D x(t)
```
- `h(t) ∈ R^N`: hidden state; `A ∈ R^{N×N}`: state matrix; `B ∈ R^N`, `C ∈ R^N`: input/output projections; `D`: skip connection scalar.

**Discretization (ZOH, step size Δ):**
```
Ā = exp(ΔA)
B̄ = (ΔA)^{-1}(exp(ΔA) − I)ΔB
h_t = Ā h_{t-1} + B̄ x_t
y_t = C h_t + D x_t
```

**Selective scan (S6 — Mamba's key innovation):**  
B, C, Δ are *input-dependent* (functions of x_t), making A effectively input-selective:
```
Δ_t = softplus(Linear_Δ(x_t))       ∈ R^N
B_t  = Linear_B(x_t)                 ∈ R^N
C_t  = Linear_C(x_t)                 ∈ R^N
```
This breaks the LTI constraint and lets the model selectively remember or forget content.

**Variable-scan in MambaTS (CIM):**  
Input `X ∈ R^{L×M}` (L = look-back length, M = variables) is transposed to `X^T ∈ R^{M×L}`. The SSM then scans across the M-dimension (variables) rather than L (time), treating each time step as a "token" of length L and variables as the sequence:
```
scan direction: variable 1 → variable 2 → … → variable M
token at position m: x_m ∈ R^L
```
This is conceptually analogous to iTransformer's variable-token attention but using linear-time selective scans.

**Variable Permutation Training (VPT):**  
At each training step, randomly permute the variable order π ∈ Perm(M):
```
X_perm = X[:, π]    # shuffle columns
ŷ_perm = Model(X_perm)
ŷ      = ŷ_perm[:, π^{-1}]   # un-shuffle output
Loss   = MSE(ŷ, y)
```
Since the ground-truth variable ordering is arbitrary (channels have no natural sequence), VPT prevents the model from exploiting positional artifacts in the variable dimension.

**Temporal Mamba Block (TMB):**  
A secondary Mamba scan over the time dimension within each variable, with weights shared across all M channels:
```
for m in 1..M:
    h_m = MambaTime(x_m)    # same parameters θ_time for all m
```
Parameter sharing acts as a strong regularizer, preventing channel-specific overfitting.

**Loss:**
```
L = (1/T) Σ_{t=1}^{T} ||ŷ_t − y_t||_2^2
```
Standard MSE; no auxiliary losses reported.

---

### 3. Architecture / Algorithm

**Input:** `X ∈ R^{B×L×M}` (batch × look-back length × variables)

**Step 1 — Normalization:**  
Instance normalization (RevIN-style): subtract per-series mean, divide by std. Parameters stored for de-normalization at output.

**Step 2 — Patch/Token embedding (per-variable):**  
Each variable's time series `x_m ∈ R^L` is projected to `R^D` via a linear layer (no patching by default; the full L-length series is the token). Produces `Z ∈ R^{B×M×D}`.

**Step 3 — Variable Mamba Block (VMB) stack:**  
`N_v` layers of:
1. LayerNorm
2. Selective SSM scan over M-dimension (CIM): `Z ∈ R^{B×M×D}` → `Z' ∈ R^{B×M×D}`
3. Optional bidirectional scan (forward + reverse over variables, features concatenated)
4. Residual add

**Step 4 — Temporal Mamba Block (TMB) stack (interleaved or appended):**  
`N_t` layers of:
1. LayerNorm
2. Reshape to `R^{B×M×L}` (or sub-sequence patches)
3. Selective SSM scan over L-dimension, weights shared across M: `R^{B×M×L}` → `R^{B×M×L}`
4. Residual add

**Step 5 — Projection head:**  
`Z_final ∈ R^{B×M×D}` → Linear → `R^{B×M×T}` (T = forecast horizon).  
Transpose → `ŷ ∈ R^{B×T×M}`.

**Step 6 — De-normalization:**  
Apply stored mean/std from RevIN.

**Training:** VPT applied at step 2 (permute M before embedding). At inference, canonical ordering used.

---

### 4. What Makes It Work

**1. Variable-dimension scanning (CIM).**  
Scanning over variables rather than time is the single largest contributor. It mirrors iTransformer's insight that cross-variable dependencies are the signal worth modeling; the SSM's recurrence then captures those dependencies with O(M·L) cost rather than O(M²·L) for attention. Ablation in the paper shows CIM accounts for the majority of gains over a time-dimension Mamba baseline.

**2. Variable Permutation Training (VPT).**  
The Mamba scan is inherently ordered (causal left-to-right). Imposing a fixed arbitrary order on unordered multivariate channels leaks spurious positional information. VPT forces the model to be permutation-equivariant in the variable dimension, acting as data augmentation and a structural regularizer simultaneously. Ablation shows consistent improvement on all benchmarks.

**3. Shared-weight Temporal Mamba Block.**  
Parameter sharing across channels avoids per-channel overfitting, critical when M is large (e.g., 862 in Traffic). This distinguishes MambaTS from naive CI-Mamba baselines. Without weight sharing, temporal blocks hurt performance on high-dimensional datasets.

---

### 5. Limitations & Failure Modes

- **No long-range memory guarantee:** SSM hidden state dimension N is fixed; very long look-back windows (L > 720) may saturate the state and lose early-sequence context.
- **Variable ordering still matters at inference:** VPT randomizes during training but uses fixed order at test time. Truly permutation-invariant architectures (e.g., attention) may generalize better to datasets with changing channel availability.
- **Benchmarks are standard LTSF datasets** (ETTh1/h2, ETTm1/m2, Weather, Traffic, Solar-Energy, PEMS). Performance on irregular/missing-data or asynchronous multivariate series not evaluated.
- **Bidirectional scan over variables:** Non-causal in the variable dimension, which is fine for forecasting but means the model cannot be used for streaming/online settings without re-scanning from scratch.
- **Compute vs. Transformer:** MambaTS is O(M·L) but the selective scan has large constant factors (CUDA kernel overhead). On short L or small M, Transformer baselines may be faster in practice.
- **Uncertain:** Whether VPT helps when variables have a meaningful ordering (e.g., spatial grid data where neighbors matter) — the paper does not test spatially structured inputs.

---

### 6. Research Ideas (ranked)

**Idea 1 — VPT applied to temporal patches, not just variable order**  
*Why it might work:* If patch order is partially arbitrary (e.g., within non-overlapping windows), random patch permutation during training could similarly regularize temporal SSM blocks and prevent the model from over-relying on patch index.  
*How to test:* Add patch-order permutation to the TMB training pass on ETTm1 (patch size 16, horizon 96). Compare MSE vs. fixed-order baseline.

**Idea 2 — Learnable variable ordering via a differentiable sorting mechanism**  
*Why it might work:* VPT makes the model order-agnostic, but an optimal ordering might exist (e.g., by correlation structure). A Sinkhorn-based soft permutation learned end-to-end could find it, recovering the permutation-equivariance benefit while exploiting structure.  
*How to test:* Replace VPT with a Sinkhorn sort layer (tau-annealed) on top of a correlation matrix of X; compare to VPT and fixed-order on Traffic (M=862).

**Idea 3 — Hybrid VMB + cross-variable attention at bottleneck**  
*Why it might work:* SSM scans are sequential and asymmetric (early variables influence later ones more). A single multi-head attention layer at the midpoint of the VMB stack would allow symmetric all-pairs variable interaction without the O(M²) cost at every layer.  
*How to test:* Insert one attention layer between VMB blocks (N_v/2 + attention + N_v/2). Budget-match parameters to pure MambaTS. Evaluate on Weather and Solar.

**Idea 4 — FFT-based frequency conditioning on the SSM state**  
*Why it might work:* TMB with shared weights learns generic temporal dynamics. Conditioning the SSM's B/C projections on per-variable FFT features (dominant frequency, amplitude) allows variable-specific temporal adaptation without breaking weight sharing.  
*How to test:* Pre-compute top-k FFT coefficients per variable; concatenate as a conditioning vector to B_t, C_t projections in TMB. Test k ∈ {4, 8} on ETTh1.

**Idea 5 — Replacing linear projection head with a Mamba decoder**  
*Why it might work:* The linear head projects D-dimensional variable embeddings directly to horizon T, ignoring temporal structure in the output. An autoregressive Mamba decoder over the T steps could improve long-horizon coherence.  
*How to test:* Replace the Linear(D→T) head with a small TMB-style decoder (2 layers) that autoregressively generates T steps. Compare on horizon 720 across ETT datasets; check if training cost is acceptable.

---

### 7. Implementation Notes

**Key hyperparameters:**
| Param | Typical value | Notes |
|-------|--------------|-------|
| `d_model` (D) | 128 | Variable embedding dimension |
| `d_state` (N) | 16 | SSM hidden state size (Mamba default) |
| `d_conv` | 4 | Local conv in Mamba block |
| `expand` | 2 | Inner dimension = expand × d_model |
| `N_v` (VMB layers) | 1–2 | More layers often hurt |
| `N_t` (TMB layers) | 1 | Shared-weight temporal blocks |
| `L` (look-back) | 336–720 | Longer helps on most datasets |
| VPT | always on during training | Off at inference |

**Training tricks:**
- RevIN normalization is critical; without it, variable-scale differences corrupt the cross-variable SSM scan.
- VPT implementation: `torch.randperm(M)` per batch, applied before embedding, inverted before loss. Trivially parallelizable.
- Bidirectional variable scan: run forward and reverse separately, concatenate along D, project back to D. Doubles VMB parameter count.
- Gradient clipping at 1.0 recommended (SSM scans can produce large gradients on long sequences).
- Weight tying across variables in TMB: a single `nn.Module` instance applied in a loop over M (or equivalently, reshape to `(B*M, L, 1)` and pass through one Mamba block, then reshape back).

**Gotchas:**
- The selective scan (S6) requires the `mamba-ssm` CUDA package; pure-PyTorch fallback is ~10x slower and may OOM on L=720, M=862.
- VPT must be disabled at inference, or results will be stochastic across runs.
- For datasets where M > L (e.g., Traffic M=862, L=96 is possible), consider swapping VMB/TMB roles or capping M via PCA preprocessing.

**Reference implementation:**  
Official code: https://github.com/XiudingCai/MambaTS-pytorch  
Mamba CUDA kernel: https://github.com/state-spaces/mamba  
Close relatives to compare: iTransformer (ICLR 2024), S-Mamba (arxiv 2403), TimeMachine (arxiv 2403).
