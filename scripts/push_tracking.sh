#!/bin/bash
# Push results.tsv and ideas_log.txt to the results-tracking branch using git
# plumbing — does not touch the working tree or current branch.
set -e

BLOB_R=$(git hash-object -w results.tsv)
BLOB_I=$(git hash-object -w ideas_log.txt)

TREE=$(printf "100644 blob %s\tresults.tsv\n100644 blob %s\tideas_log.txt\n" \
  "$BLOB_R" "$BLOB_I" | git mktree)

PARENT=$(git ls-remote origin results-tracking 2>/dev/null | cut -f1)

if [ -n "$PARENT" ]; then
  COMMIT=$(git commit-tree "$TREE" -p "$PARENT" -m "tracking: $(date +%Y%m%dT%H%M)")
else
  COMMIT=$(git commit-tree "$TREE" -m "tracking: $(date +%Y%m%dT%H%M)")
fi

git push origin "$COMMIT":refs/heads/results-tracking
echo "Tracking pushed: $COMMIT"
