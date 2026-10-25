#!/bin/bash
# Búsqueda estilo wann-cpp para MountainCar discreto (shaping 10, 500 000 pasos a presupuesto completo).
#   [JOBS=8 JOBS2=4 JOBS3=0 N2=20 HIDDEN="3 2" SCR_DIR=screening RUNS_DIR=runs] scripts/screening_mountaincar.sh [flags extra de scripts/screening.py]
# Resultados en screening/disc_mc/ y runs/. La salida de la terminal también va a screening/mountaincar.out.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
SCR_DIR="${SCR_DIR:-screening}"; RUNS_DIR="${RUNS_DIR:-runs}"
mkdir -p "$SCR_DIR"
python scripts/screening.py --task disc_mc --mode all --jobs "${JOBS:-8}" --jobs2 "${JOBS2:-4}" --jobs3 "${JOBS3:-0}" --n2 "${N2:-20}" ${HIDDEN:+--hidden $HIDDEN} --dir "$SCR_DIR" --runs-dir "$RUNS_DIR" "$@" 2>&1 | tee "$SCR_DIR/mountaincar.out"
