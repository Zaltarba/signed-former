# Projection-Orthogonal Signed Attention — Implementation Guide

> This document is a self-contained specification for implementing the
> projection-orthogonal value transform inside `SignedAttention`.
> Follow it section by section. Every tensor shape, every index, every
> in-place operation is spelled out. No ambiguity should remain.

---

## 0. File and Class Map

All changes are in the **single file** that contains the model.

| Class | What changes |
|---|---|
| `SignedAttention` | Major rewrite: new `forward`, new helpers `_cross_orth_coeffs`, `_temporal_gs_window`. Mask change. |
| `SignedAttentionLayer` | Tiny: pass `n_channels` (= `N`, number of variates) down to `SignedAttention.__init__`. |
| `Stack` | Pass `enc_in` (= `N`) to `SignedAttentionLayer`. |
| `StackedEncoder` | Pass `enc_in` through to `Stack`. |
| `Model` | No change (already passes `enc_in`). |

Nothing outside `SignedAttention.forward` changes semantically. The rest is
plumbing the variate count `N` down to where it is needed.

---

## 1. Notation and Conventions

```
B  = batch size
H  = n_heads (head dimension after to_heads projection)
P  = n_patches
N  = number of variates (enc_in)
D  = patch_len (feature dimension inside attention)
W  = attention_window
L  = P * N  (total token count in attention)
```

Token layout in the `L` dimension: position `t * N + j` is **variate `j` at
patch-time `t`**. This is the layout produced by `TimeEmbedding`.

---

## 2. Mask Change

### Current

```python
mask = (diff <= 0) | (diff >= self.attention_window)
```

Self-attention (`diff == 0`) is **excluded**.

### New

```python
mask = (diff < 0) | (diff >= self.attention_window)
```

Self-attention (`diff == 0`) is **included**. Change the `<=` to `<`. That is
the only change to `_get_causal_mask`.

---

## 3. Score and Attention Matrix (unchanged logic)

Compute as before, using the updated mask:

```python
scores = Q @ K^T                           # (B, H, L, L)
scale  = log_scale.exp().clamp(1, 30) / sqrt(D)

cmask = self._get_causal_mask(N, device)    # (1, 1, L, L)
pos   = scores.masked_fill(cmask, -1e4)
neg   = (-scores).masked_fill(cmask, -1e4)

A = softmax(scale * pos, dim=-1) - softmax(scale * neg, dim=-1)
A = dropout(A)                              # (B, H, L, L)
```

**Important**: pass `N` (not `channels`) to `_get_causal_mask`. The current
code computes `channels = L // self.n_patches` — that already equals `N`, so
no logic change, just be aware.

---

## 4. Reshape Values to 5-D

```python
# V comes in as (B, H, L, D) from the to_heads projection.
V4 = V.reshape(B, H, P, N, D)
```

Dimension semantics: `V4[:, :, t, j, :]` = value vector for variate `j` at
patch-time `t`.

---

## 5. Phase 1 — Cross-Variate Orthogonalization Coefficients

### Goal

For query variate `i`, compute scalar coefficients `alpha[t, j]` such that:

```
V1_i[t, j] = V[t, j] - alpha[t, j] * V[t, i]     for j ≠ i
V1_i[t, i] = V[t, i]                               (untouched)
```

### Precompute Gram matrix (do once, before variate loop)

```python
# G[t, j1, j2] = V4[t, j1] · V4[t, j2]  (dot over D)
G = torch.einsum('bhpid, bhpjd -> bhpij', V4, V4)
    # shape: (B, H, P, N, N)

# Self-norms for every variate at every patch-time
norms2 = G.diagonal(dim1=-2, dim2=-1)              # (B, H, P, N)
norms2 = norms2.clamp(min=1e-8)
```

This replaces `N` separate dot-product calls with a single einsum.

### Per query variate `i`, the alpha vector is

```python
alpha_i = G[:, :, :, :, i] / norms2[:, :, :, i:i+1]
    # shape: (B, H, P, N)
    # alpha_i[:, :, :, j] = dot(V[t,j], V[t,i]) / ||V[t,i]||^2

alpha_i[:, :, :, i] = 0.0   # do not project self-variate
```

**Do NOT materialize `V1_i` as a full `(B, H, P, N, D)` tensor yet.**
Store `alpha_i` and reconstruct inside windows (phase 2).

---

## 6. Phase 2 — Windowed Value Construction + Temporal Gram-Schmidt

### 6.1 Create windows via unfold

```python
n_win = P - W + 1    # number of query patch-times that have a full window
                      # (queries at t < W-1 have partial windows — handle below)

# Unfold V4 along the patch-time axis:
V_win = V4.unfold(dimension=2, size=W, step=1)
    # V4 is (B, H, P, N, D), unfold dim=2 with size=W, step=1
    # result: (B, H, n_win, N, D, W)

V_win = V_win.permute(0, 1, 2, 5, 3, 4).contiguous()
    # → (B, H, n_win, W, N, D)

# Similarly unfold alpha_i (shape B, H, P, N) along dim=2:
alpha_win = alpha_i.unfold(dimension=2, size=W, step=1)
    # → (B, H, n_win, N, W)
alpha_win = alpha_win.permute(0, 1, 2, 4, 3)
    # → (B, H, n_win, W, N)

# And unfold ref_i = V4[:, :, :, i, :] (shape B, H, P, D) along dim=2:
ref_i = V4[:, :, :, i, :]                     # (B, H, P, D)
ref_win = ref_i.unfold(dimension=2, size=W, step=1)
    # → (B, H, n_win, D, W)
ref_win = ref_win.permute(0, 1, 2, 4, 3)
    # → (B, H, n_win, W, D)
```

### 6.2 Apply rank-1 cross-variate correction inside each window

```python
# V1_win[..., j, :] = V_win[..., j, :] - alpha_win[..., j, None] * ref_win[..., None, j_DROPPED, :]
# More precisely:

V1_win = V_win - alpha_win.unsqueeze(-1) * ref_win.unsqueeze(-2)
    # alpha_win: (B, H, n_win, W, N)     → unsqueeze(-1) → (B, H, n_win, W, N, 1)
    # ref_win:   (B, H, n_win, W, D)     → unsqueeze(-2) → (B, H, n_win, W, 1, D)
    # broadcast multiply: (B, H, n_win, W, N, D)
    # subtract from V_win: (B, H, n_win, W, N, D)

    # shape: (B, H, n_win, W, N, D)  — CORRECT
```

This is the materialization of `V1_i` but only inside windows.
Memory: `O(B × H × n_win × W × N × D)`. Since `n_win ≈ P` and `W` is small
(typically 10), this is `W×` the size of `V4`.

### 6.3 Temporal Gram-Schmidt backward loop inside each window

Window positions run from `0` (oldest in window) to `W-1` (the query
patch-time itself — the anchor).

```python
# The anchor is at s = W-1 (most recent = the query's own time).
# Orthogonalize backward: s = W-2, W-3, ..., 0.

for s in range(W - 2, -1, -1):
    ref_s = V1_win[:, :, :, s + 1, :, :]     # (B, H, n_win, N, D)
    cur_s = V1_win[:, :, :, s,     :, :]      # (B, H, n_win, N, D)

    dot  = (cur_s * ref_s).sum(dim=-1, keepdim=True)    # (B, H, n_win, N, 1)
    nrm  = (ref_s * ref_s).sum(dim=-1, keepdim=True).clamp(min=1e-8)

    V1_win[:, :, :, s, :, :] = cur_s - (dot / nrm) * ref_s     # in-place update
```

**This loop is W-1 iterations (e.g. 9 for W=10).** Each iteration is a
batched element-wise op over `(B, H, n_win, N, D)` — very fast on GPU.

After this loop, `V1_win` is now `V2_win` — the fully orthogonalized values
inside each window for query variate `i`.

---

## 7. Phase 3 — Windowed Attention Aggregation

### 7.1 Extract attention weights for query variate `i`

Query variate `i` occupies positions `i, i+N, i+2N, ...` in the `L`
dimension (i.e. every `N`-th position starting from `i`).

For query position `q = T*N + i` (query variate `i` at patch-time `T`),
the active keys are all `(j, t')` where `0 <= T - t' < W`, i.e.:

```
key positions: { t'*N + j  |  t' ∈ [T-W+1, T],  j ∈ [0, N) }
```

This is a contiguous block of `W * N` positions in `L`, starting at
`(T - W + 1) * N`.

```python
# Extract query-variate-i rows of A:
q_indices = torch.arange(i, L, N, device=A.device)      # P positions
# Only the last n_win queries have full windows (T >= W-1):
q_indices = q_indices[W - 1:]                            # n_win positions

A_rows = A[:, :, q_indices, :]                           # (B, H, n_win, L)
```

Now extract the W*N key columns for each query. Build a gather index:

```python
# For query at patch-time T (T = W-1, W, ..., P-1):
# The local window key indices are (T-W+1)*N .. (T+1)*N - 1 → W*N consecutive
# T_offset = T - (W - 1)  → 0, 1, ..., n_win-1

offsets = torch.arange(n_win, device=A.device) * N       # start of each window in L
key_local = torch.arange(W * N, device=A.device)         # 0..W*N-1
key_idx = offsets.unsqueeze(1) + key_local.unsqueeze(0)  # (n_win, W*N)

A_win = A_rows.gather(dim=-1, index=key_idx.expand(B, H, -1, -1))
    # (B, H, n_win, W*N)
```

### 7.2 Aggregate

```python
# Flatten V2_win from (B, H, n_win, W, N, D) to (B, H, n_win, W*N, D)
V2_flat = V1_win.reshape(B, H, n_win, W * N, D)

# Weighted sum:
out_i = torch.einsum('bhqk, bhqkd -> bhqd', A_win, V2_flat)
    # (B, H, n_win, D)
```

### 7.3 Scatter into output

```python
output = torch.zeros(B, H, L, D, device=V.device)

# Variate i's query positions with full windows:
scatter_idx = q_indices     # positions i + (W-1)*N, i + W*N, ...
output[:, :, scatter_idx, :] = out_i
```

### 7.4 Handling early queries (T < W-1)

The first `W - 1` query patch-times don't have a full backward window.
Two options (option A is recommended for simplicity):

**Option A — Pad with zeros (recommended).**
Pad `V4` with `W - 1` zero patch-times at the front before unfolding.
Then `n_win = P` and every query has a full window. The zero-padded
positions contribute nothing after orthogonalization (they are zero vectors,
so projections onto them are zero). This is consistent with the causal
semantics (nothing before t=0).

```python
V4_padded = F.pad(V4, (0, 0, 0, 0, W - 1, 0))    # pad dim=2 on left
    # (B, H, P + W - 1, N, D)
# Then unfold on the padded tensor: n_win = P
```

Do the same for `alpha_i` and `ref_i` before unfolding. Attention matrix `A`
does not need padding — the mask already zeros out positions before `t=0`.

With this option, `n_win = P`, `scatter_idx = arange(i, L, N)` (all
positions), and no separate handling is needed.

**Option B — Fallback to standard attention for early queries.**
Use the un-orthogonalized values for query patch-times `0..W-2`. Simpler
to reason about but creates a discontinuity.

---

## 8. Phase 4 — Variate Loop and Output Assembly

The full forward wraps phases 1–3 in a loop over `i`:

```python
def forward(self, queries, keys, values, attn_mask=None):
    B, H, L, D = queries.shape
    N = self.n_channels                    # stored at __init__
    P = self.n_patches
    W = self.attention_window

    # --- Scores and attention (unchanged except mask) ---
    scores = queries @ keys.transpose(-1, -2)
    scale  = self.log_scale.exp().clamp(1, 30) / sqrt(D)
    cmask  = self._get_causal_mask(N, queries.device)
    pos    = scores.masked_fill(cmask, -1e4)
    neg    = (-scores).masked_fill(cmask, -1e4)
    A      = softmax(scale * pos, -1) - softmax(scale * neg, -1)
    A      = self.dropout(A)

    # --- Reshape values ---
    V4 = values.reshape(B, H, P, N, D)

    # --- Gram matrix (once) ---
    G      = torch.einsum('bhpid, bhpjd -> bhpij', V4, V4)    # (B,H,P,N,N)
    norms2 = G.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8)     # (B,H,P,N)

    # --- Pad for early queries ---
    V4_pad = F.pad(V4, (0,0, 0,0, W-1,0))                     # (B,H,P+W-1,N,D)

    # --- Allocate output ---
    output = values.new_zeros(B, H, L, D)

    for i in range(N):
        # Phase 1: alpha coefficients
        alpha_i = G[:, :, :, :, i] / norms2[:, :, :, i:i+1]   # (B,H,P,N)
        alpha_i[:, :, :, i] = 0.0

        # Pad alpha_i and ref_i
        alpha_pad = F.pad(alpha_i, (0,0, W-1,0))               # (B,H,P+W-1,N)
        ref_i     = V4_pad[:, :, :, i, :]                      # (B,H,P+W-1,D)

        # Phase 2: unfold + correct + Gram-Schmidt
        V_win     = V4_pad.unfold(2, W, 1).permute(0,1,2,5,3,4)  # (B,H,P,W,N,D)
        alpha_win = alpha_pad.unfold(2, W, 1).permute(0,1,2,4,3) # (B,H,P,W,N)
        ref_win   = ref_i.unfold(2, W, 1).permute(0,1,2,4,3)     # (B,H,P,W,D)

        V1_win = V_win - alpha_win.unsqueeze(-1) * ref_win.unsqueeze(-2)
        V1_win = V1_win.contiguous()

        for s in range(W - 2, -1, -1):
            ref_s = V1_win[:, :, :, s+1, :, :]
            cur_s = V1_win[:, :, :, s,   :, :]
            dot   = (cur_s * ref_s).sum(-1, keepdim=True)
            nrm   = (ref_s * ref_s).sum(-1, keepdim=True).clamp(min=1e-8)
            V1_win[:, :, :, s, :, :] = cur_s - (dot / nrm) * ref_s

        # Phase 3: aggregate
        V2_flat = V1_win.reshape(B, H, P, W * N, D)

        q_idx   = torch.arange(i, L, N, device=A.device)         # (P,)
        offsets = torch.arange(P, device=A.device) * N            # (P,)
        k_local = torch.arange(W * N, device=A.device)            # (W*N,)
        k_idx   = offsets.unsqueeze(1) + k_local.unsqueeze(0)     # (P, W*N)

        A_win = A[:, :, q_idx].gather(-1, k_idx.expand(B, H, -1, -1))
        out_i = torch.einsum('bhqk, bhqkd -> bhqd', A_win, V2_flat)

        output[:, :, q_idx, :] = out_i

    return output.contiguous(), (A if self.output_attention else None)
```

---

## 9. Complexity Analysis

### Time

| Step | Cost per variate | Total (× N) |
|---|---|---|
| Gram matrix (once) | — | `O(B H P N² D)` |
| Alpha + unfold | `O(B H P N)` | `O(B H P N²)` |
| Rank-1 correction | `O(B H P W N D)` | `O(B H P W N² D)` |
| Gram-Schmidt (W-1 iters) | `O(B H P W N D)` | `O(B H P W N² D)` |
| Attention gather + einsum | `O(B H P W N D)` | `O(B H P W N² D)` |

**Dominant term**: `O(B H P W N² D)` — exactly `W` times the cost of
standard windowed attention `O(B H P W N D)` per variate, summed over `N`
variates. In practice `W ≈ 10` and `N` is moderate (7–21), so the constant
is manageable.

### Memory peak

`V1_win` is the largest tensor: `(B, H, P, W, N, D)`.

With `B=32, H=8, P=15, W=10, N=7, D=64` this is
`32 × 8 × 15 × 10 × 7 × 64 = 15.4M floats ≈ 59 MB` — comfortably fits.

Only **one** `V1_win` exists at a time (one per variate `i`). The loop
naturally bounds memory.

---

## 10. Memory Optimization Tricks

### 10.1 In-place Gram-Schmidt

The backward loop already operates in-place on `V1_win`. Make sure
`.contiguous()` is called after the rank-1 correction and before the loop so
the in-place writes don't alias unfold memory.

### 10.2 Reuse buffers across variate iterations

Pre-allocate `V1_win` once before the loop and overwrite each iteration:

```python
V1_win_buf = torch.empty(B, H, P, W, N, D, device=V.device, dtype=V.dtype)

for i in range(N):
    # ... compute V_win, correction ...
    V1_win_buf.copy_(V_win - alpha_win.unsqueeze(-1) * ref_win.unsqueeze(-2))
    # ... Gram-Schmidt on V1_win_buf ...
```

This avoids `N` separate allocations.

### 10.3 Chunked variate loop (large N)

If `N > 32`, process variates in chunks of `C`:

```python
for i_start in range(0, N, C):
    i_end = min(i_start + C, N)
    # Compute alpha for variates i_start..i_end simultaneously
    # V1_win shape becomes (B, H, P, W, N, D, C) — trade C× memory for N/C loops
```

For the typical `N ≤ 21` in the current codebase, the simple loop is fine.

### 10.4 Gradient checkpointing

The Gram-Schmidt loop creates a chain of `W-1` dependencies. If memory is
tight during backward, wrap each variate iteration in
`torch.utils.checkpoint.checkpoint`:

```python
for i in range(N):
    out_i = checkpoint(self._process_variate, i, V4, V4_pad, G, norms2, A, use_reentrant=False)
    output[:, :, q_idx_i, :] = out_i
```

---

## 11. Numerical Stability Notes

1. **Clamp norms**: every `||ref||²` divisor must be clamped to `≥ 1e-8`.
   A zero-norm reference means "nothing to project out", so clamping is
   semantically correct (the subtracted term becomes zero).

2. **Gram-Schmidt ordering**: the backward direction (from anchor at `W-1`
   to oldest at `0`) is deliberate. It mirrors PACF: the most recent lag is
   the reference; earlier lags are stripped of the most-recent component
   first. Reversing the order gives a different (and less interpretable)
   decomposition.

3. **Detach or not**: the orthogonalization should be **fully differentiable**.
   Do not detach any intermediate. Gradients will flow through the projection
   scalars `alpha` and through the Gram-Schmidt chain, allowing the model to
   learn representations that are easy to orthogonalize.

4. **Half-precision**: the dot-product / norm² division can lose precision
   in fp16. If using mixed precision, keep the Gram-Schmidt loop in fp32:

   ```python
   with torch.amp.autocast('cuda', enabled=False):
       V1_win_f32 = V1_win.float()
       # ... Gram-Schmidt loop in fp32 ...
       V1_win = V1_win_f32.to(V.dtype)
   ```

---

## 12. `__init__` Changes to SignedAttention

```python
class SignedAttention(nn.Module):
    def __init__(self, n_patches, patch_len, n_heads,
                 n_channels,                          # ← NEW: number of variates N
                 attention_dropout=0.1,
                 output_attention=False,
                 attention_window=10):
        super().__init__()
        self.n_patches = n_patches
        self.n_channels = n_channels                  # ← store N
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.attention_window = attention_window
        self._cached_mask = None
        self._cached_key = None
        self.log_scale = nn.Parameter(torch.tensor(log(2.0)))
```

No new `nn.Parameter` or `nn.Module` is added — the orthogonalization is
parameter-free.

---

## 13. Plumbing Changes

### SignedAttentionLayer.__init__

Add `n_channels` argument, pass to `SignedAttention`:

```python
class SignedAttentionLayer(nn.Module):
    def __init__(self, n_patches, patch_len, d_model, n_heads=None,
                 n_channels=1,                                        # ← NEW
                 dropout=0.1, output_attention=False, attention_window=10):
        ...
        self.attn = SignedAttention(n_patches, patch_len, self.n_heads,
                                    n_channels=n_channels,            # ← pass through
                                    ...)
```

### Stack.__init__

Pass `enc_in` to each `SignedAttentionLayer`:

```python
self.layers = nn.ModuleList([
    SignedAttentionLayer(n_patches, patch_len, d_model, n_heads,
                         n_channels=enc_in,                           # ← NEW
                         dropout=dropout, ...)
    for _ in range(e_layers)
])
```

Add `enc_in` to `Stack.__init__` signature and store it.

### StackedEncoder.__init__

Pass `enc_in` through to each `Stack`:

```python
Stack(..., enc_in=enc_in)
```

This is already available in `StackedEncoder.__init__` as a parameter but
currently unused by `Stack`. Just wire it through.

---

## 14. Testing Checklist

1. **Shape test**: for `B=2, H=4, P=8, N=3, D=16, W=4`, verify that
   `forward` produces output of shape `(B, H, L, D)` with `L = P * N = 24`.

2. **Self-attention included**: verify `A[:, :, k, k] != 0` for diagonal
   entries (the mask no longer blocks self).

3. **Orthogonality spot-check**: for a random query `(i, T)`, extract the
   windowed `V2` values. Check:
   - `V2[T, j] · V2[T, i_ref]` ≈ 0 for `j ≠ i` (cross-variate orth).
   - `V2[s, j] · V2[s+1, j]` ≈ 0 for consecutive `s` (temporal orth).

4. **Gradient flow**: verify `V4.requires_grad_(True)` and
   `output.sum().backward()` runs without error and `V4.grad` is non-None.

5. **Numerical equivalence with W=1**: when `attention_window=1`, the window
   contains only self. No temporal Gram-Schmidt runs (loop range is empty).
   Cross-variate step produces `alpha_i[:,:,:,i]=0` and only self contributes.
   Output should equal `diag(A) * V` (element-wise).

6. **Speed benchmark**: time the new `forward` vs old on typical shapes.
   Expected ~3–5× slower in wall time, ~2× more memory.

---

## 15. Summary of Operations in Forward Pass

```
┌─────────────────────────────────────────────────────────────┐
│ Scores + Signed Attention A  (same as before, new mask)     │
│   Q @ K^T → mask → softmax(+) - softmax(-) → A             │
│   shapes: (B,H,L,L)                                        │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ Reshape V → V4: (B, H, P, N, D)                            │
│ Gram matrix G = V4 · V4^T over D: (B, H, P, N, N)  [once]  │
│ Pad V4 on patch-time: (B, H, P+W-1, N, D)          [once]  │
└─────────────────┬───────────────────────────────────────────┘
                  │
       ┌──────────▼──────────┐
       │  for i in range(N): │
       │  ┌──────────────────┴────────────────────────────┐   │
       │  │ Step 1: alpha_i = G[:,:,:,:,i] / norms2_i     │   │
       │  │         zero out alpha_i[:,:,:,i]              │   │
       │  │         pad, unfold to windows                 │   │
       │  │         rank-1 correction → V1_win             │   │
       │  │         shape: (B, H, P, W, N, D)              │   │
       │  ├───────────────────────────────────────────────┤   │
       │  │ Step 2: for s = W-2 down to 0:                │   │
       │  │           project V1_win[:,s] against V1_win[:,s+1]│
       │  │           in-place update                      │   │
       │  ├───────────────────────────────────────────────┤   │
       │  │ Step 3: gather A rows for variate i            │   │
       │  │         einsum(A_win, V2_flat) → out_i         │   │
       │  │         scatter out_i into output[:,:,i::N,:]  │   │
       │  └───────────────────────────────────────────────┘   │
       └──────────────────────┘
                  │
                  ▼
           output: (B, H, L, D)
```
