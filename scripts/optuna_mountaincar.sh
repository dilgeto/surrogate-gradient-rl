#!/bin/bash
# Búsqueda de hiperparámetros en dos fases para MountainCarDiscrete (3 semillas por trial y sin poda, ver TASK_DEFAULTS).
#   scripts/optuna_mountaincar.sh [flags extra de sgrl.train, p. ej. --steps 500000]
# Variables de entorno: ver scripts/_optuna_common.sh
set -euo pipefail
source "$(dirname "$0")/_optuna_common.sh"
optuna_two_phase mountaincar --env MountainCarDiscrete "$@"
