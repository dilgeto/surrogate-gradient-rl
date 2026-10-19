#!/bin/bash
# ¿En qué paso se estabiliza CarTrack? 3 semillas en paralelo, 6M pasos (el preset usa 2M). CarTrack no tiene shaping.
#   scripts/stab_carrace.sh [flags extra de sgrl.train]      Variables: ver scripts/_stab_common.sh
set -euo pipefail
source "$(dirname "$0")/_stab_common.sh"
stab_run CarTrack est-carrace 6000000 12 - "$@"
