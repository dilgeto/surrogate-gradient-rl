#!/bin/bash
# Las tres búsquedas estilo wann-cpp, UNA TRAS OTRA, para un PC de 12 hebras. Redes totalmente conectadas con tantas
# CONEXIONES como la red de wann-cpp de cada tarea (35 en CarTrack, 44 en Acrobot, 79 en MountainCar); fase 2 de 40 trials.
#   scripts/screening_all.sh                      # en este orden: MountainCar, Acrobot, CarTrack
#   JOBS=10 JOBS2=10 JOBS3=11 scripts/screening_all.sh   (valores por defecto)
# 10 / 10 / 11 procesos a la vez dan oleadas exactas (30 trials = 3x10, 40 = 4x10, 33 corridas = 3x11) y dejan 1 hebra
# libre para el sistema; con 12 saldrían oleadas medio vacías (12, 12, 6) y tardaría lo mismo.
set -uo pipefail
cd "$(dirname "$0")/.."
export JOBS="${JOBS:-10}" JOBS2="${JOBS2:-10}" JOBS3="${JOBS3:-11}" N2="${N2:-40}"
run() {  # <script> <hidden>
  echo; echo "################ $1  (hidden $2)  $(date '+%F %T')"
  HIDDEN="$2" "scripts/$1" || echo "!! $1 terminó con error; sigo con la siguiente tarea"
}
#run screening_mountaincar.sh "14 3"
#run screening_acrobot.sh "4 4"
run screening_carrace.sh "3 2"
echo; echo "################ terminado $(date '+%F %T')"
