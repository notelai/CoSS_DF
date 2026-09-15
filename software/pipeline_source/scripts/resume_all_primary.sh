#!/usr/bin/env bash
set -euo pipefail
for DATASET in kth_tips2b fmd kylberg; do
  echo "===== $DATASET ====="
  cossdf all --dataset "$DATASET"
done
