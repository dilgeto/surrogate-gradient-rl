"""Estado de los estudios de Optuna mientras corren (sólo lee las bases de datos; no interfiere con la búsqueda).

    python scripts/optuna_status.py                       # todos los optuna/*.db
    python scripts/optuna_status.py acrobot-f1 acrobot-f2
    python scripts/optuna_status.py --watch 30            # se actualiza cada 30 s
    python scripts/optuna_status.py acrobot-f1 --total 160  # añade % hecho y tiempo restante estimado

Por estudio: trials por estado, mejor valor, duración media de un trial completo, ritmo y cuánto hace que acabó el último.
`--total` es el nº de trials previsto del estudio (TRIALS1/TRIALS2); si hay varios estudios, usa el mismo para todos.
Un trial RUNNING que lleva horas sin cambiar suele ser de un proceso muerto (Optuna no lo marca solo como fallido).
"""
import argparse
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)
S = optuna.trial.TrialState


def fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def status(name: str, storage_dir: Path, total: int | None) -> str:
    try:
        study = optuna.load_study(study_name=name, storage=f"sqlite:///{storage_dir / (name + '.db')}")
    except Exception as e:  # noqa: BLE001
        return f"{name}: no se puede leer ({type(e).__name__}: {e})"
    trials = study.trials
    c = Counter(t.state for t in trials)
    done = [t for t in trials if t.state == S.COMPLETE]
    finished = [t for t in trials if t.state in (S.COMPLETE, S.PRUNED, S.FAIL)]
    line = (f"{name:22s} completos {c[S.COMPLETE]:4d} · podados {c[S.PRUNED]:4d} · en curso {c[S.RUNNING]:3d}"
            f" · fallidos {c[S.FAIL]:3d}")
    if done:
        line += f" · mejor {study.best_value:9.1f} (trial {study.best_trial.number})"
    now = datetime.now()
    if finished:
        durs = [(t.datetime_complete - t.datetime_start).total_seconds() for t in finished if t.datetime_complete]
        started = min(t.datetime_start for t in trials)
        last = max(t.datetime_complete for t in finished if t.datetime_complete)
        elapsed = (now - started).total_seconds()
        rate = len(finished) / max(elapsed, 1)
        line += f"\n{'':22s} trial medio {fmt(sum(durs) / len(durs))} · {rate * 3600:.1f} trials/h · último acabó hace {fmt((now - last).total_seconds())}"
        if total:
            left = max(total - len(finished), 0)
            line += f" · {100 * len(finished) / total:.0f}% de {total}" + (f" · quedan ~{fmt(left / rate)}" if left else " · COMPLETO")
    return line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("studies", nargs="*", help="nombres de estudio (def.: todos los de la carpeta)")
    ap.add_argument("--dir", default="optuna", help="carpeta de las bases de datos")
    ap.add_argument("--total", type=int, default=None, help="trials previstos por estudio (para % y tiempo restante)")
    ap.add_argument("--watch", type=float, default=0, metavar="SEG", help="repetir cada SEG segundos")
    a = ap.parse_args()
    d = Path(a.dir)
    while True:
        names = a.studies or sorted(p.stem for p in d.glob("*.db"))
        if not names:
            raise SystemExit(f"no hay bases de datos en {d}/")
        if a.watch:
            print("\033[2J\033[H", end="", flush=True)  # limpia la pantalla
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        for n in names:
            print(status(n, d, a.total), flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
