#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:-kth_tips2b}"
export COSSDF_DRIVE_ROOT="${COSSDF_DRIVE_ROOT:-/content/drive/MyDrive/CoSS_DF}"
export COSSDF_LOCAL_ROOT="${COSSDF_LOCAL_ROOT:-/content/cossdf_work}"
cossdf all --dataset "$DATASET"
