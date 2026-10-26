#!/bin/bash
# Lanza las búsquedas estilo wann-cpp EN PARALELO, cada una en su propia ventana de tmux.
#   scripts/screening_tmux.sh [todo|conexiones|neuronas]     (por defecto: todo = las 6 a la vez)
# Ventanas: con-mountaincar, con-acrobot, con-carrace (red con las conexiones de wann-cpp, resultados en screening/ y runs/)
#           neu-mountaincar, neu-acrobot, neu-carrace (red con las neuronas de wann-cpp, en screening_neuronas/ y runs_neuronas/)
# Entrar: tmux attach -t screening   (Ctrl-b w: lista de ventanas; Ctrl-b d: salir sin parar nada)
#
# Reparte las hebras de la máquina entre las búsquedas lanzadas: procesos por búsqueda = hebras / nº de búsquedas, con tope
# de 10 en las fases 1 y 2 (TPE aprende peor con más trials simultáneos) y de 33 en la fase 3 (son 33 corridas). Antes de
# lanzar comprueba que cabe en la RAM (≈1.5 GB por proceso) y que no hay ya otra búsqueda corriendo.
# Variables: JOBS JOBS2 JOBS3 (procesos por fase), N2 (trials de la fase 2, def. 40), SESSION (def. screening),
#            THREADS y MEM_GB (para forzar los recursos que se suponen), DRY=1 (sólo muestra lo que haría), FORCE=1.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_screening_nets.sh
VARIANT="${1:-todo}"; SESSION="${SESSION:-screening}"; N2="${N2:-40}"
case "$VARIANT" in todo|conexiones|neuronas) ;; *) echo "uso: $0 [todo|conexiones|neuronas]" >&2; exit 1 ;; esac
[ -f .venv/bin/activate ] && . .venv/bin/activate
python scripts/check_env.py --quiet || { echo "!! el entorno no está listo (ver arriba); no lanzo nada" >&2; exit 1; }
[ -n "${DRY:-}" ] || command -v tmux > /dev/null || { echo "!! tmux no está instalado" >&2; exit 1; }

entries=()  # etiqueta|script|carpeta de screening|carpeta de runs|hidden
add() {  # <prefijo> <script> <hidden> <screening> <runs>
  local lab="${2#screening_}"; lab="${lab%.sh}"
  entries+=("$1-$lab|$2|$4|$5|$3")
}
if [ "$VARIANT" != neuronas ]; then
  add con screening_mountaincar.sh "$HID_CON_MC" screening runs
  add con screening_acrobot.sh "$HID_CON_ACROBOT" screening runs
  add con screening_carrace.sh "$HID_CON_CAR" screening runs
fi
if [ "$VARIANT" != conexiones ]; then
  add neu screening_mountaincar.sh "$HID_NEU_MC" screening_neuronas runs_neuronas
  add neu screening_acrobot.sh "$HID_NEU_ACROBOT" screening_neuronas runs_neuronas
  add neu screening_carrace.sh "$HID_NEU_CAR" screening_neuronas runs_neuronas
fi
R=${#entries[@]}

# --- recursos ---
T="${THREADS:-$(python -c 'import os; print(len(os.sched_getaffinity(0)))')}"
MEM="${MEM_GB:-$(awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo)}"
clamp() { local v=$1 lo=$2 hi=$3; [ "$v" -lt "$lo" ] && v=$lo; [ "$v" -gt "$hi" ] && v=$hi; echo "$v"; }
PER=$(( T / R ))
JOBS="${JOBS:-$(clamp "$PER" 1 10)}"; JOBS2="${JOBS2:-$(clamp "$PER" 1 10)}"; JOBS3="${JOBS3:-$(clamp "$PER" 1 33)}"
MAXJ=$JOBS; [ "$JOBS2" -gt "$MAXJ" ] && MAXJ=$JOBS2; [ "$JOBS3" -gt "$MAXJ" ] && MAXJ=$JOBS3
PEAK=$(( R * MAXJ )); NEED=$(( PEAK * 3 / 2 ))
echo "búsquedas en paralelo: $R | hebras: $T | RAM libre: ${MEM} GB"
echo "procesos por búsqueda: fase 1 = $JOBS, fase 2 = $JOBS2, fase 3 = $JOBS3  (pico: hasta $PEAK procesos, ~${NEED} GB)"
if [ "$NEED" -gt $(( MEM * 9 / 10 )) ]; then
  echo "!! el pico (~${NEED} GB) no cabe en la RAM libre (${MEM} GB): arriesgas que el sistema mate procesos por falta de memoria." >&2
  echo "   Lanza menos búsquedas a la vez (variante 'conexiones' o 'neuronas', o una tarea con scripts/screening_<tarea>.sh)," >&2
  echo "   baja JOBS/JOBS2/JOBS3, o fuerza con FORCE=1 si sabes lo que haces." >&2
  [ -n "${FORCE:-}" ] || [ -n "${DRY:-}" ] || exit 1
fi
if [ "$PER" -lt 3 ]; then
  echo "AVISO: con $T hebras para $R búsquedas tocan $PER por búsqueda. Lanzarlas todas a la vez NO acelera el total (el trabajo" >&2
  echo "       es el mismo y está limitado por las hebras): sólo hace que terminen todas al final. En una máquina grande sí compensa." >&2
fi
if pgrep -f "scripts/screening.py --task" > /dev/null && [ -z "${FORCE:-}" ]; then
  echo "!! ya hay búsquedas corriendo (pgrep -af 'scripts/screening.py'); lanzar otra vez sobre las mismas carpetas las mezclaría." >&2
  echo "   Páralas primero, o usa FORCE=1." >&2
  [ -n "${DRY:-}" ] || exit 1
fi
if [ -z "${DRY:-}" ] && tmux has-session -t "$SESSION" 2> /dev/null; then
  echo "!! ya existe la sesión de tmux '$SESSION'. Ciérrala (tmux kill-session -t $SESSION) o usa SESSION=otro_nombre." >&2
  exit 1
fi

[ -n "${DRY:-}" ] || tmux new-session -d -s "$SESSION" -n info "bash"
for e in "${entries[@]}"; do
  IFS='|' read -r label script scr runs hid <<< "$e"
  cmd="cd '$PWD' && JOBS=$JOBS JOBS2=$JOBS2 JOBS3=$JOBS3 N2=$N2 SCR_DIR=$scr RUNS_DIR=$runs HIDDEN=\"$hid\" scripts/$script; echo; printf 'Terminado. Pulsa Enter para cerrar esta ventana... '; read x"
  if [ -n "${DRY:-}" ]; then
    echo; echo "[$label]"; echo "  $cmd"
  else
    tmux new-window -d -t "$SESSION" -n "$label" "$cmd"
    echo "lanzada: $label  (hidden $hid; resultados en $scr/ y $runs/)"
  fi
done
if [ -z "${DRY:-}" ]; then
  echo; echo "Entra con:  tmux attach -t $SESSION      (Ctrl-b w: ventanas, Ctrl-b d: salir sin parar nada)"
fi
