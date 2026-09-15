#!/usr/bin/env bash
set -euo pipefail
for d in kth_tips2b fmd kylberg; do cossdf directional_fusion_controls --dataset "$d"; done
cossdf directional_fusion_controls_collect
