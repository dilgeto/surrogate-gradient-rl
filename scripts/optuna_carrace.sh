#!/bin/bash
# Búsqueda de hiperparámetros en dos fases para CarTrack, el coche de carreras (no poda hasta la 5.ª evaluación, ver TASK_DEFAULTS).
#   scripts/optuna_carrace.sh [flags extra de sgrl.train, p. ej. --steps 1000000]
# Variables de entorno: ver scripts/_optuna_common.sh
set -euo pipefail
source "$(dirname "$0")/_optuna_common.sh"
optuna_two_phase carrace --env CarTrack "$@"
