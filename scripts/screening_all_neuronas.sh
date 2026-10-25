#!/bin/bash
# Variante de screening_all.sh con redes totalmente conectadas con tantas NEURONAS como la red de wann-cpp de cada tarea
# (en vez de tantas conexiones). Cuenta como neuronas las entradas + ocultas + salidas de nuestro actor, es decir, los
# nodos de WANN sin su nodo de sesgo: CarTrack 26 (WANN 27 nodos), Acrobot 34 (35), MountainCar 36 (37).
# Una red densa con esas neuronas tiene muchas más conexiones: 142, 279 y 317 (WANN: 35, 44 y 79).
#   scripts/screening_all_neuronas.sh             # MountainCar, Acrobot y CarTrack, una tras otra, para un PC de 12 hebras
#   HID_CAR="8 7" HID_ACROBOT="14 13" HID_MC="16 15" JOBS=10 JOBS2=10 JOBS3=11 N2=40 scripts/screening_all_neuronas.sh   (por defecto)
# Resultados en screening_neuronas/ y runs_neuronas/ para no mezclarse con los de la variante por conexiones.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
python scripts/check_env.py --quiet || { echo "!! el entorno no está listo (ver arriba); no lanzo nada" >&2; exit 1; }
export JOBS="${JOBS:-10}" JOBS2="${JOBS2:-10}" JOBS3="${JOBS3:-11}" N2="${N2:-40}"
export SCR_DIR="${SCR_DIR:-screening_neuronas}" RUNS_DIR="${RUNS_DIR:-runs_neuronas}"
run() {  # <script> <hidden>
  echo; echo "################ $1  (hidden $2)  $(date '+%F %T')"
  HIDDEN="$2" "scripts/$1" || echo "!! $1 terminó con error; sigo con la siguiente tarea"
}
run screening_mountaincar.sh "${HID_MC:-16 15}"
run screening_acrobot.sh "${HID_ACROBOT:-14 13}"
run screening_carrace.sh "${HID_CAR:-8 7}"
echo; echo "################ terminado $(date '+%F %T')"
