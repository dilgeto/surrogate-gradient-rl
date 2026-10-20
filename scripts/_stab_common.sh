# Funciones comunes de stab_acrobot.sh, stab_mountaincar.sh y stab_carrace.sh (se cargan con `source`).
# Entrenan varias semillas EN PARALELO con un presupuesto mayor que el del preset y luego ejecutan
# scripts/plateau.py para ver en qué paso se estabiliza la curva de evaluación.
#
# Variables de entorno (todas opcionales):
#   DEVICE    cpu | cuda | cuda:N (def. cpu)        SEEDS   semillas separadas por espacios (def. "0 1 2")
#   STEPS     pasos por corrida (def. el de la tarea) SHAPING escala de --reward-shaping (def. 10 en Acrobot y
#                                                      MountainCar, como wann-cpp; 0 = sin shaping)
#   EVAL_EVERY iteraciones entre evaluaciones (def. ~50k pasos)   EVAL_EPISODES episodios por evaluación (def. 20)
#   OUT       directorio de resultados (def. runs)  THREADS hilos de torch por proceso (def. 1)
# Los argumentos del script se añaden a los flags de sgrl.train (y ganan).

DEVICE="${DEVICE:-cpu}"
SEEDS="${SEEDS:-0 1 2}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
OUT="${OUT:-runs}"
THREADS="${THREADS:-1}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f .venv/bin/activate ] && . .venv/bin/activate

# stab_run <env de sgrl> <prefijo de nombre> <pasos por defecto> <eval-every por defecto> <shaping por defecto, - si no aplica> [flags...]
stab_run() {
  local env=$1 prefix=$2 steps=${STEPS:-$3} every=${EVAL_EVERY:-$4} shaping=${SHAPING:-$5}; shift 5
  local flags=() pids=() rc=0 s
  if [ "$shaping" = "-" ]; then shaping="no aplica"  # tarea sin potencial (CarTrack)
  elif [ "$shaping" != "0" ]; then flags+=(--reward-shaping "$shaping")
  else prefix="$prefix-ns"; fi
  mkdir -p "$OUT/_logs"
  # falla pronto si se pidió GPU y torch no la ve (sgrl.train también aborta, pero así se avisa antes de lanzar)
  case "$DEVICE" in cuda*) python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "!! DEVICE=$DEVICE pero torch no ve ninguna GPU" >&2; return 1; };; esac
  echo ">> $env · $steps pasos · semillas: $SEEDS · $DEVICE · shaping ${shaping} · resultados en $OUT/$prefix-s*"
  for s in $SEEDS; do
    python -m sgrl.train --env "$env" "${flags[@]}" --steps "$steps" --eval-every "$every" \
      --eval-episodes "$EVAL_EPISODES" --device "$DEVICE" --threads "$THREADS" --seed "$s" \
      --name "$prefix-s$s" --out "$OUT" "$@" > "$OUT/_logs/$prefix-s$s.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" -eq 0 ] || { echo "!! alguna corrida falló; revisa $OUT/_logs/$prefix-s*.log" >&2; return 1; }
  python scripts/plateau.py "$OUT/$prefix-s*"
}
