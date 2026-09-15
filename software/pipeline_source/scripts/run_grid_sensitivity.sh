#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:-kth_tips2b}"
cossdf grid --dataset "$DATASET"
