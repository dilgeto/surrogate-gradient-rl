#!/bin/bash
# ¿En qué paso se estabiliza MountainCar? 3 semillas en paralelo, 3M pasos (el preset usa 1M), con shaping 10 como wann-cpp.
#   scripts/stab_mountaincar.sh [flags extra de sgrl.train]  Variables: ver scripts/_stab_common.sh
set -euo pipefail
source "$(dirname "$0")/_stab_common.sh"
stab_run MountainCarDiscrete est-mountaincar 3000000 12 10 "$@"
