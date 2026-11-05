#!/bin/bash
# Variante de screening_all.sh con la red del preset: 64x64 totalmente conectada (4544 conexiones en Acrobot, 4800 en
# CarTrack, 4416 en MountainCar), mismo procedimiento de búsqueda que las variantes por conexiones y por neuronas.
#   scripts/screening_all_64.sh                   # MountainCar, Acrobot y CarTrack, una tras otra, para un PC de 12 hebras
#   JOBS=10 JOBS2=10 JOBS3=11 N2=40 scripts/screening_all_64.sh      (valores por defecto)
# Resultados en screening_64x64/ y runs_64x64/ (runs_default/ son las redes 64x64 de la búsqueda con Optuna: no se tocan).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/_screening_nets.sh
[ -f .venv/bin/activate ] && . .venv/bin/activate
python scripts/check_env.py --quiet || { echo "!! el entorno no está listo (ver arriba); no lanzo nada" >&2; exit 1; }
export JOBS="${JOBS:-10}" JOBS2="${JOBS2:-10}" JOBS3="${JOBS3:-11}" N2="${N2:-40}"
export SCR_DIR="${SCR_DIR:-screening_64x64}" RUNS_DIR="${RUNS_DIR:-runs_64x64}"
run() {  # <script> <hidden>
  echo; echo "################ $1  (hidden $2)  $(date '+%F %T')"
  HIDDEN="$2" "scripts/$1" || echo "!! $1 terminó con error; sigo con la siguiente tarea"
}
run screening_mountaincar.sh "${HID_MC:-$HID_64_MC}"
run screening_acrobot.sh "${HID_ACROBOT:-$HID_64_ACROBOT}"
run screening_carrace.sh "${HID_CAR:-$HID_64_CAR}"
echo; echo "################ terminado $(date '+%F %T')"
