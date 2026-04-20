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
2. Read `results.tsv` and `git log --oneline -20` to understand what has been tried
3. Read `model/CustomModel.py` for the current architecture
4. Propose one architectural change based on what showed promise (see `program.md`)
5. Edit `model/CustomModel.py`
6. Run `bash experiment.sh` (trains on first `keep_ratio` variates; stops at epoch boundary when `time_budget` is hit or early stopping triggers — whichever comes first)
7. Parse MSE from output; if crash → log failure, `git checkout model/CustomModel.py`, retry with a different change
8. If MSE improved vs. best in `results.tsv`: keep the commit
9. If MSE did not improve: `git reset HEAD~1`
10. Append one row to `results.tsv`

## File Roles

| File | Purpose |
|------|---------|
| `program.md` | Read each iteration — dataset config, allowed modifications, fixed settings |
| `model/CustomModel.py` | The **only** file the agent modifies |
| `experiment.sh` | Launches timed training run |
| `results.tsv` | Append-only experiment log |
| `model/iTransformer.py` | Reference baseline — never modified |

## results.tsv Format

Tab-separated columns: `commit_hash`, `MSE`, `gpu_mem_gb`, `status`, `description`

`status` is either `ok` or `crash`.

## Git Workflow

- One commit per attempted change, with a short message describing the hypothesis
- On improvement: commit stays
- On no improvement or crash: `git reset HEAD~1` — never amend, never force-push
- Read `git log` and `results.tsv` together to calibrate next move; early iterations explore broadly, later ones exploit what worked
