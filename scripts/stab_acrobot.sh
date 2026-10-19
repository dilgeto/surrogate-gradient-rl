#!/bin/bash
# ¿En qué paso se estabiliza Acrobot? 3 semillas en paralelo, 3M pasos (el preset usa 1M), con shaping 10 como wann-cpp.
#   scripts/stab_acrobot.sh [flags extra de sgrl.train]      Variables: ver scripts/_stab_common.sh
set -euo pipefail
source "$(dirname "$0")/_stab_common.sh"
stab_run Acrobot est-acrobot 3000000 25 10 "$@"
