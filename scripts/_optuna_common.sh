# Funciones comunes de optuna_acrobot.sh, optuna_mountaincar.sh y optuna_carrace.sh (se cargan con `source`).
#
# Variables de entorno (todas opcionales):
#   WORKERS   trials simultáneos (procesos que comparten el estudio; def. 4, ver nota abajo)
#   TRIALS1   trials totales de la fase 1 (def. 50)      TRIALS2   trials totales de la fase 2 (def. 40)
#   DEVICE    cpu | cuda | cuda:N (def. cpu)             THREADS   hilos de torch por proceso (def. 1)
#   STUDY     prefijo de los estudios (def. el de la tarea): <STUDY>-f1 y <STUDY>-f2
#   SEEDS_PER_TRIAL  semillas que se promedian en cada trial (def. el de la tarea: 3 en MountainCar, 1 en las demás)
#   VALIDATE  K > 0: al final reentrena el top K de la fase 2 con 11 semillas nuevas (def. 0)
#   JOBS      corridas de validación simultáneas (def. 0 = el máximo que cabe: núcleos y RAM libre)
#   AIM_REPO  repositorio de Aim (aim://host:53800 o ruta local); si se omite no se usa Aim. Hace falta `pip install aim`.
# Los argumentos del script se añaden a la config base (p. ej. --steps 300000) y ganan a los presets.

# Nota sobre WORKERS: la búsqueda (TPE) y la poda aprenden de los trials TERMINADOS, así que más trials simultáneos
# = decisiones con menos información (con 4 a la vez, cada trial nuevo se elige sin conocer, como mucho, los 3 en curso).
# Además cada proceso adicional rinde menos al compartir núcleos. En cambio la validación son corridas totalmente
# independientes y sí conviene paralelizarla al máximo (JOBS).
WORKERS="${WORKERS:-4}"
TRIALS1="${TRIALS1:-50}"
TRIALS2="${TRIALS2:-40}"
DEVICE="${DEVICE:-cpu}"
THREADS="${THREADS:-1}"
VALIDATE="${VALIDATE:-0}"
JOBS="${JOBS:-0}"
SEEDS_PER_TRIAL="${SEEDS_PER_TRIAL:-}"
AIM_REPO="${AIM_REPO:-}"

# Un hilo por proceso en TODAS las librerías: la paralelización es por procesos. Sin esto, en un servidor grande cada
# proceso (numpy/OpenBLAS/MKL/OpenMP) puede abrir tantos hilos como núcleos tenga la máquina y se estorban entre sí.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f .venv/bin/activate ] && . .venv/bin/activate
mkdir -p optuna/logs

# Falla pronto y con un mensaje claro, en vez de lanzar 0 procesos o morir dentro de los logs.
if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "!! WORKERS debe ser un entero >= 1 (vale '$WORKERS'). OJO: nproc obedece OMP_NUM_THREADS; usa nproc --all" >&2
  exit 1
fi
python -c "import optuna" 2>/dev/null || { echo "!! no se puede importar optuna con $(command -v python); ejecuta: python -c 'import optuna'" >&2; exit 1; }

# run_phase <estudio> <fase> <trials totales> <flags base...>
# Lanza WORKERS procesos con el mismo estudio y espera a que acaben. --n-trials es por proceso.
run_phase() {
  local study=$1 phase=$2 total=$3; shift 3
  local per=$(( (total + WORKERS - 1) / WORKERS )) pids=() i rc=0
  local aim=() seeds=()
  [ -n "$AIM_REPO" ] && aim=(--aim-repo "$AIM_REPO")
  [ -n "$SEEDS_PER_TRIAL" ] && seeds=(--seeds-per-trial "$SEEDS_PER_TRIAL")
  echo ">> fase $phase · estudio $study · $WORKERS procesos x $per trials · logs: optuna/logs/$study-w*.log"
  for ((i = 0; i < WORKERS; i++)); do
    python scripts/optuna_search.py --study "$study" --phase "$phase" --n-trials "$per" "${seeds[@]}" "${aim[@]}" -- "$@" \
      > "optuna/logs/$study-w$i.log" 2>&1 &
    pids+=($!)
    sleep 5  # evita que los procesos creen la base SQLite del estudio a la vez
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" -eq 0 ] || echo "!! algún proceso de $study falló; revisa optuna/logs/$study-w*.log" >&2
  return "$rc"
}

# best_argv <estudio>: imprime, una por línea, la config base del estudio + los flags del mejor trial.
best_argv() {
  python - "$1" <<'PY'
import shlex, sys
import optuna
s = optuna.load_study(study_name=sys.argv[1], storage=f"sqlite:///optuna/{sys.argv[1]}.db")
print("\n".join(shlex.split(s.user_attrs["base_argv"] + " " + s.best_trial.user_attrs["flags"])))
PY
}

# optuna_two_phase <prefijo por defecto> <flags base...>
# Fase 1 (PPO + beta) -> fija el mejor trial como base -> fase 2 (surrogado, ganancias, T, ...) -> reportes.
optuna_two_phase() {
  local prefix="${STUDY:-$1}"; shift
  local base=(--device "$DEVICE" --threads "$THREADS" "$@")
  local f1="$prefix-f1" f2="$prefix-f2"

  run_phase "$f1" 1 "$TRIALS1" "${base[@]}"
  python scripts/optuna_report.py "$f1" --top 5 --plot

  local best
  mapfile -t best < <(best_argv "$f1")
  run_phase "$f2" 2 "$TRIALS2" "${best[@]}"
  python scripts/optuna_report.py "$f2" --top 5 --plot

  if [ "$VALIDATE" -gt 0 ]; then
    python scripts/optuna_report.py "$f2" --validate "$VALIDATE" --jobs "$JOBS"
  else
    echo ">> para validar el top con 11 semillas nuevas (todas las corridas en paralelo): python scripts/optuna_report.py $f2 --validate 3"
  fi
}
