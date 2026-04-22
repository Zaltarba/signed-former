# Projection-Orthogonal Signed Attention — Design Document

---

## 1. Motivation

Standard attention aggregates values from a context window, but the resulting output conflates two distinct sources of information:

- **Cross-series correlation**: variate `j` is correlated with variate `i` partly because both share a common contemporaneous signal, not because `j` causally informs `i`.
- **Temporal autocorrelation**: patch at time `t-1` is correlated with patch at time `t` partly through the chain `t-2 → t-1 → t`, not through a direct lag-1 relationship.

The goal is to make each value token carry only the **unique, partial** information it holds — controlling for both cross-series contemporaneous correlation (step 1) and intra-series temporal redundancy (step 2). This mirrors the statistical concept of **Partial Autocorrelation (PACF)**, which isolates the direct effect of lag `k` by removing all intermediate lag effects.

Additionally, self-attention (token attending to itself) is re-enabled to let the model use its own representation as an explicit reference point.

---

## 2. Current Code — SignedAttention

```python
class SignedAttention(nn.Module):
    """A = softmax(+s) - softmax(-s), row-normalised.

    Causal mask: query patch t attends only to key patches t' where
    0 < (t - t') < attention_window (excludes self).
    """

    def __init__(self, n_patches: int, patch_len: int, n_heads: int,
                 attention_dropout: float = 0.1,
                 output_attention: bool = False,
                 attention_window: int = 10):
        super().__init__()
        self.n_patches = n_patches
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.attention_window = attention_window
        self._cached_mask = None
        self._cached_key = None
        self.log_scale = nn.Parameter(torch.tensor(log(2.0)))

    def _get_causal_mask(self, channels: int, device: torch.device):
        key = (self.n_patches, channels, device)
        if self._cached_key != key or self._cached_mask is None:
            pid = torch.arange(self.n_patches, device=device).repeat_interleave(channels)
            diff = pid.unsqueeze(0) - pid.unsqueeze(1)
            mask = (diff <= 0) | (diff >= self.attention_window)
            self._cached_mask = mask.unsqueeze(0).unsqueeze(0)
            self._cached_key = key
        return self._cached_mask

    def forward(self, queries, keys, values, attn_mask=None):
        B, H, L, D = queries.shape
        channels = L // self.n_patches

        scores = torch.matmul(queries, keys.transpose(-1, -2))
        scale = self.log_scale.exp().clamp(1, 30.0) / sqrt(D)

        cmask = self._get_causal_mask(channels, queries.device)
        pos = scores.masked_fill(cmask, -1e4)
        neg = (-scores).masked_fill(cmask, -1e4)

        A = torch.softmax(scale * pos, dim=-1) - torch.softmax(scale * neg, dim=-1)
        A = self.dropout(A)

        V = torch.matmul(A, values)
        return V.contiguous(), (A if self.output_attention else None)
```

### What it does

1. **Signed scores**: `A = softmax(+s) - softmax(-s)`. Allows negative attention weights, expressing inhibitory relationships. Scaled by a learned `log_scale` clamped to `[1, 30]`.
2. **Causal window mask**: token `(i, t)` attends to token `(j, t')` only if `0 < t - t' < attention_window`. Both self (`diff=0`) and tokens outside the window are masked to `-1e4` before softmax.
3. **Aggregation**: standard `matmul(A, values)`.

### Current mask logic

```
mask = (diff <= 0) | (diff >= attention_window)
```

`diff = t_query - t_source`. Masked-out positions: self and future (`diff <= 0`), and distant past (`diff >= attention_window`). Active positions: `1 <= diff < attention_window`, i.e. the `attention_window - 1` most recent past patches.

---

## 3. Data Structures and Tensor Shapes

### 3.1 Input to the model

| Tensor | Shape | Description |
|--------|-------|-------------|
| `x` | `(B, seq_len, N)` | Raw multivariate time series |
| `x_mark` | `(B, seq_len, F)` | Temporal features (hour, weekday, …) |

### 3.2 After TimeEmbedding (patchification)

```
x: (B, seq_len, N)
    ↓  per-variate causal conv: CausalConv(1 → d_model)
    ↓  refine: ConvBlock(d_model)
    ↓  causal unfold: (seq_len) → (n_patches, patch_len)
    ↓  reshape + permute

tokens: (B, d_model, n_patches * N, patch_len)
```

- `n_patches = (seq_len - patch_len) // stride + 1`
- `L = n_patches * N` is the sequence length in attention space
- Position `t * N + j` in `L` corresponds to variate `j` at patch time `t`

### 3.3 Inside SignedAttentionLayer (channel mixing → attention)

```
tokens: (B, d_model, L, patch_len)     # H = d_model, D = patch_len
    ↓  to_heads: Linear(d_model → n_heads), applied per (L, D) position

queries / keys / values: (B, n_heads, L, patch_len)
```

The `to_heads` projection mixes the `d_model` channel dimension into `n_heads`. All downstream operations (mask, scores, orthogonalization) work in this projected space with shape `(B, H, L, D)`.

### 3.4 Attention score matrix

```
scores = Q @ K^T : (B, H, L, L)
mask             : (1, 1, L, L)    — broadcast over B and H
A                : (B, H, L, L)    — after signed softmax + mask
```

The mask is built from patch-time indices `pid` of length `L`, where each patch-time index is repeated `N` times (one per variate). `diff[q, k] = pid[q] - pid[k]` captures the temporal distance between query and key regardless of which variate they belong to.

### 3.5 Value tensor — logical decomposition

```
V: (B, H, L, D)

reshaped as V4: (B, H, n_patches, N, D)
    dim 2: patch time t  ∈ [0, n_patches)
    dim 3: variate    j  ∈ [0, N)
    dim 4: patch content ∈ R^D     (D = patch_len)
```

Token `(j, t)` = `V4[:, :, t, j, :]` — a vector in `R^D` per head.

---

## 4. Proposed Mechanism — Projection-Orthogonal Values

### 4.1 Mask change

```
# Before:
mask = (diff <= 0) | (diff >= attention_window)   # self excluded

# After:
mask = (diff < 0) | (diff >= attention_window)    # self included
```

Self-token `(i,t) → (i,t)` is now unmasked. Since step 1 leaves self-variate values unchanged, the self-token contributes its original representation to the output.

### 4.2 Step 1 — Cross-variate orthogonalization (query-variate-dependent)

**Purpose**: when computing the output for a query at variate `i`, remove from every source token `(j, t')` the component it shares with variate `i` at the same time `t'`. What remains is the information in `j` that is genuinely orthogonal to `i`'s own signal at that instant.

**Per query variate `i`**, applied on `V4: (B, H, n_patches, N, D)`:

```
ref  = V4[:, :, :, i, :]               # (B, H, n_patches, D)  — query variate at each t'
ref_exp = ref.unsqueeze(3)             # (B, H, n_patches, 1, D) for broadcast over j

dot   = sum_D(V4 * ref_exp)            # (B, H, n_patches, N)
norm2 = sum_D(ref_exp ** 2) + ε        # (B, H, n_patches, 1)

V1_i = V4 - (dot / norm2).unsqueeze(-1) * ref_exp    # (B, H, n_patches, N, D)

# Exception: j == i is left unchanged
V1_i[:, :, :, i, :] = V4[:, :, :, i, :]
```

**Result**: `V1_i` is specific to query variate `i`. N distinct value matrices, one per variate.

**Tensor shapes at this step**:

| | Shape | Note |
|-|-------|------|
| `V4` | `(B, H, n_patches, N, D)` | original values |
| `ref_exp` | `(B, H, n_patches, 1, D)` | broadcast reference |
| `dot` | `(B, H, n_patches, N)` | scalar projection per source |
| `V1_i` | `(B, H, n_patches, N, D)` | orthogonalized values for query variate `i` |

### 4.3 Step 2 — Temporal orthogonalization (query-time-dependent, chained Gram-Schmidt backward)

**Purpose**: for a query at time `T`, remove from each past patch of every source variate the component explained by the immediately more recent patch of that same variate. The chain goes backward from `T`, so intermediate patches are already orthogonalized when used as references. This mimics PACF: the value at lag `k` carries only the direct effect, not the indirect path through lags `1, ..., k-1`.

**Applied on `V1_i` for a query at time `T`**, independently per variate `j`:

```
V2[j, T]   = V1_i[j, T]                                              # anchor — unchanged

for s = T-1 down to 0:
    ref_s = V2[j, s+1]                                               # already orthogonalized
    dot   = sum_D(V1_i[j, s] * ref_s)                               # scalar
    norm2 = sum_D(ref_s ** 2) + ε
    V2[j, s] = V1_i[j, s] - (dot / norm2) * ref_s
```

The anchor at `T` is shared across all variates `j`. Tokens at time `t > T` are never needed (masked out), so only `t = 0, ..., T` matter.

**Result**: `V2_{i,T}` is specific to query `(i, T)`. In principle `N × n_patches` distinct value matrices — one per unique query.

**Tensor shapes at this step**:

| | Shape | Note |
|-|-------|------|
| `V1_i` | `(B, H, n_patches, N, D)` | input (from step 1) |
| `ref_s` | `(B, H, N, D)` | single time-slice, all variates |
| `V2[j, s]` | `(B, H, N, D)` | one time-slice after orthogonalization |
| `V2` (full) | `(B, H, T+1, N, D)` | only past of query time T needed |

### 4.4 Attention computation

```
# For query (i, T):
V2_flat = V2_{i,T}.reshape(B, H, L, D)     # flatten back to (B, H, L, D)

output[(i,T)] = sum_k  A[(i,T), k] * V2_flat[k]
```

`A` is computed from queries and keys unchanged — the orthogonalization touches only values.

---

## 5. PACF Interpretation

For a query at `(i, T)`, the value matrix `V2_{i,T}[j, t']` carries:

1. The part of variate `j` at patch `t'` that is **not linearly explained by variate `i` at `t'`** — cross-series partial (step 1).
2. The part of that residual that is **not linearly explained by variate `j`'s own patches at `t'+1, ..., T`** — temporal partial (step 2).

The attention weight `A[(i,T), (j,t')]` scaled by this doubly-orthogonalized value therefore measures the **unique, direct contribution** of `(j, t')` to `(i, T)`, analogous to the partial autocorrelation at lag `T - t'` between series `j` and series `i`.

---

## 6. Summary of Changes vs. Current Code

| Aspect | Current | Proposed |
|--------|---------|----------|
| Self-attention | excluded (`diff <= 0` masked) | included (`diff < 0` masked) |
| Value matrix | single shared `V` | `N × n_patches` query-specific matrices |
| Cross-variate correction | none | step 1: project out query variate at each t' |
| Temporal correction | none | step 2: chained Gram-Schmidt backward from T |
| Computation per layer | `O(L² D)` | `O(N · n_patches · L · D)` before optimization |
| Memory | `O(B H L D)` | `O(B H N · n_patches · L · D)` before optimization |
