# SignedAttention — Optimization Plan

## Context

`SignedAttention` in `model/CustomModel.py` is the core bottleneck of the architecture.
This document describes the performance issues identified and the directions to explore for fixing them.
It is intended as a guide for implementing the changes — not a rigid specification.
The implementing model should feel free to adapt the approach as it sees fit, as long as the goals are met.

---

## Tensor layout reminder

```
L = P * N   (total tokens)
Token order: patch-major → [p0·v0, p0·v1, …, p0·v(N-1),  p1·v0, …,  p(P-1)·v(N-1)]

P = n_patches,  N = n_channels (variates),  W = attention_window  (always < 5 in practice)
```

---

## Problem statement

The `forward` method of `SignedAttention` has two structural bottlenecks:

### Bottleneck A — Full L×L attention matrix (Phase 1)

```
scores = queries @ keys.T    # (B, H, L, L)
```

This computes **all L² pairs** then immediately masks out everything outside a local window of W patches.
Only `W × N` keys are actually used per query.
The full matrix is computed, stored, and operated on — O(L²) work for O(L × W × N) useful values.

### Bottleneck B — Python loop over N variates (Phase 4)

```python
output = zeros(B, H, L, D)
for i in range(N):                          # N Python iterations
    V1 = V_win - alpha_i * ref_i            # step 8 — cross-variate orthogonalization
    for s in range(W-2, -1, -1):            # W-2 Python iterations (≤ 3)
        slices[s] -= proj(slices[s], slices[s+1])   # step 9 — temporal GS
    A_win = A[q_idx].gather(k_idx)          # step 10 — gather attention weights
    output[q_idx] = einsum(A_win, V2_flat)  # step 11 — weighted sum
```

Total: `N × (W-2)` Python kernel dispatches for the GS, plus `N` separate gather + einsum calls.
For large N (e.g. 100+ variates) this is the dominant runtime cost.

---

## Optimization goals

1. **Eliminate or drastically reduce the Python loop over N** in Phase 4
2. **Avoid computing the full L×L attention matrix** when only W×N entries per row are needed
3. **Preserve the semantic intent** of the attention: signed softmax weights, windowed causality, and orthogonalized value vectors
4. Keep the model's output shape and interface unchanged

---

## Ideas to explore

### Idea A — Replace the inner W-loop with `torch.linalg.qr`

**Problem addressed:** the inner `for s in range(W-2, -1, -1)` loop (step 9) is sequential Python, called N times.

**Direction:** the inner loop is classical sequential Gram-Schmidt over W vectors.
`torch.linalg.qr` does the same computation in a single cuBLAS call.
For each query variate `i`, the W time-slot vectors (each of shape `(N, D)` or similar) can be stacked into a small matrix and decomposed.
The Q factor from QR gives a valid orthogonal basis with the same span as the GS result.

Since W < 5 always, the matrices involved are tiny (at most 4 vectors), so QR is cheap per call.
The key is batching it over (B, H, P, i) so that it replaces the inner Python loop with a single batched GPU call.

Note: QR produces a *different but equivalent* orthogonal basis compared to backward GS.
This is acceptable — the goal is decorrelated values, not a specific GS ordering.

---

### Idea B — Hard-unroll the W-loop for small W

**Problem addressed:** same as Idea A but simpler to implement.

**Direction:** since W is always < 5 at runtime (and fixed at construction time), the inner loop body can be written out as 1, 2, or 3 explicit consecutive tensor operations, chosen at `__init__` time based on `attention_window`.

No Python loop at inference time — just direct tensor ops.
Zero memory overhead, zero semantic change.

This can be combined with Idea A (use unrolled ops for W ≤ 4, fall back to QR for larger W).

---

### Idea C — Vectorize the gather + einsum over all N variates at once (steps 10-11)

**Problem addressed:** the N separate `gather` + `einsum` + `output[q_idx] = ...` calls.

**Direction:** the N query-index sets `q_idx` for each variate `i` are just strided slices that together tile the full sequence `L`. They can be computed jointly.

Once V2_win is available for all variates (shape `(B, H, P, W, N, D)`), reshape/permute it so that the key dimension is `W*N` and the variate dimension is explicit. Then do a single batched einsum over all N variates simultaneously, writing all output positions at once.

The output scatter (`output[:,:,q_idx,:] = out_i`) repeated N times becomes a single tensor assignment.

This idea has the highest leverage because it targets the `N` gather+einsum calls directly, and can be pursued independently of how step 9 is handled.

---

### Idea D — Local / windowed attention (replace the full L×L matrix)

**Problem addressed:** Bottleneck A — the O(L²) dense attention matrix.

**Direction:** instead of computing all L² scores and masking most of them to -inf, compute only the `W*N` relevant key scores per query directly.

Concretely: for each query token, gather its `W*N` key vectors first, then compute a small dot product of shape `(B, H, L, W*N)`. This avoids materializing the full `(B, H, L, L)` matrix entirely.

The resulting local attention tensor has shape `(B, H, P, N, W*N)` (query organized by patch and variate), which also naturally fits the structure needed by Idea C — the gather step in Phase 4 becomes trivial since the attention weights are already in the right shape.

This idea has a nice synergy with Idea C: if the attention matrix is already local, steps 10-11 can be vectorized without any additional gather.

**Implementation note:** gathering key vectors before the matmul requires an index construction step (similar to the `k_idx` already computed in the current code). The signed softmax is then applied to this smaller `(B, H, L, W*N)` score matrix.

---

## Suggested approach

A reasonable implementation order:

1. **Start with Idea C** (vectorize gather+einsum) — high impact, self-contained, does not touch the GS logic.
2. **Then Idea B or A** (unroll or QR for the W-loop) — eliminates the remaining inner Python iterations.
3. **Then Idea D** (local attention) — eliminates Bottleneck A and simplifies the gather in step 10 as a bonus.

Ideas B and D together would eliminate both Python loops and the quadratic attention cost.
The model should feel free to combine ideas in a single pass if a cleaner unified design is apparent.

---

## Constraints and invariants to preserve

- Only `model/CustomModel.py` is modified
- Output shape of the full `Model.forward` must remain `(B, pred_len, N)`
- The signed attention formula `softmax(+s) - softmax(-s)` should be preserved
- The windowed causality (queries attend to at most W older patches) should be preserved
- The orthogonalization intent (decorrelate values across variates and time) should be preserved — the exact GS ordering is not sacred
