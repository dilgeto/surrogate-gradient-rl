#!/bin/bash
# Búsqueda estilo wann-cpp para MountainCar discreto (shaping 10, 500 000 pasos a presupuesto completo).
#   scripts/screening_mountaincar.sh [flags extra de scripts/screening.py]
# Resultados en screening/disc_mc/ y runs/. La salida de la terminal también va a screening/mountaincar.out.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
mkdir -p screening
python scripts/screening.py --task disc_mc --mode all --jobs "${JOBS:-8}" --jobs2 "${JOBS2:-4}" "$@" 2>&1 | tee screening/mountaincar.out
