#!/bin/bash
# Búsqueda estilo wann-cpp para Acrobot: fase 1 (reducción del espacio, 25 % de los pasos), fase 2 y fase 3 (top 3 x 11 semillas).
#   [JOBS=8 JOBS2=4 JOBS3=0 N2=20 HIDDEN="3 2"] scripts/screening_acrobot.sh [flags extra de scripts/screening.py, p. ej. --jobs 12]
# Resultados en screening/acrobot/ y runs/. La salida de la terminal también va a screening/acrobot.out.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
mkdir -p screening
python scripts/screening.py --task acrobot --mode all --jobs "${JOBS:-8}" --jobs2 "${JOBS2:-4}" --jobs3 "${JOBS3:-0}" --n2 "${N2:-20}" ${HIDDEN:+--hidden $HIDDEN} "$@" 2>&1 | tee screening/acrobot.out
