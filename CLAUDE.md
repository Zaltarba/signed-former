# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Auto-research loop for multivariate time series forecasting. Claude runs as an autonomous agent that iteratively proposes and evaluates architectural changes to `model/CustomModel.py`, using git history and `results.tsv` as memory.

## Setup

```bash
uv sync          # install dependencies
```

## Auto-Research Loop

Invoke with `/loop`. Each iteration:

1. Read `program.md` for dataset config and constraints
2. Read `ideas_log.txt` (not git log) to understand what has been tried and what showed promise — this is the primary context memory
3. Read `model/CustomModel.py` for the current architecture
4. Propose one architectural change based on what showed promise
5. Edit `model/CustomModel.py`
6. Run the experiment with output captured — use exactly these two commands:
   ```bash
   bash experiment.sh > run.log 2>&1; echo "exit:$?"
   grep -E "mse:|mae:|Epoch" run.log | tail -10
   ```
   All training logs go to `run.log`; only the filtered tail is read into context.
7. Parse MSE from the grep output: look for the line `mse:{value}, mae:{value}` printed by `test()`. If `exit:1` was returned → crash: log failure, `git checkout model/CustomModel.py`, retry with a different change. This MSE is evaluated on the test set using the checkpoint from the epoch with the best validation loss — not the last epoch.
8. If MSE improved vs. best in `results.tsv`: keep the commit
9. If MSE did not improve: `git reset --hard HEAD~1`
10. Append one row to `results.tsv` — **the agent must write this row on success**, as `experiment.sh` only logs crash rows automatically
11. **Append one entry to `ideas_log.txt`** using the format defined below — always, whether success or failure
12. Run `bash scripts/push_tracking.sh` — pushes `results.tsv` and `ideas_log.txt` to the `results-tracking` branch so progress is visible remotely; this never affects the current branch or working tree

## ideas_log.txt Format and Rules

Each entry follows this exact format:

```
## <commit_hash> — <short title>
**Hypothesis:** <why this change might help, in one sentence>
**Change:** <what was changed conceptually — no code, just the idea>
**Result:** MSE <before> → <after> <✓ improved | ✗ no improvement | ✗ crash>
---
```

**Context discipline — strictly follow these rules:**
- Do **not** read `git log` at any point; `ideas_log.txt` is the sole history source
- Do **not** run `git show`, `git diff`, or any command that retrieves past code — the only code context is the current `model/CustomModel.py`
- Read **all entries** in `ideas_log.txt` in full each iteration to avoid re-trying ideas and to identify patterns across the full experiment history
- Check the **Status** field of recent entries: `open` means the idea has untried parameter values and should be continued before moving on; `exhausted` means all variants failed and it should not be revisited
- Keep entries short: hypothesis and change descriptions must each fit in one sentence
- Never rewrite or delete past entries — append only

## File Roles

| File | Purpose |
|------|---------|
| `program.md` | Read each iteration — dataset config, allowed modifications, fixed settings |
| `model/CustomModel.py` | The **only** file the agent modifies |
| `experiment.sh` | Launches timed training run |
| `results.tsv` | Append-only experiment log |
| `ideas_log.txt` | Append-only semantic memory — hypothesis, conceptual change, outcome per iteration |
| `model/iTransformer.py` | Reference baseline — never modified |
| `run.log` | Full training stdout/stderr — never read directly; only via `grep \| tail` |
| `scripts/push_tracking.sh` | Pushes `results.tsv` + `ideas_log.txt` to `results-tracking` branch — run after every iteration |

## results.tsv Format

Tab-separated columns: `commit_hash`, `MSE`, `gpu_mem_gb`, `status`, `description`

`status` is either `ok` or `crash`.

## Git Workflow

- One commit per attempted change, with a short message describing the hypothesis
- On improvement: commit stays
- On no improvement or crash: `git reset --hard HEAD~1` — never amend, never force-push
- Read `ideas_log.txt` (all entries) to calibrate next move — never `git log`; early iterations explore broadly, later ones exploit what worked
