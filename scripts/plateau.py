"""¿Cuándo se estabiliza el entrenamiento? Lee `eval_return` de varias corridas (semillas) de una misma tarea.

    python scripts/plateau.py 'runs/est-acrobot-s*'
    python scripts/plateau.py 'runs/est-carrace-s*' --last 8 --fracs 0.9 0.95 0.99

Con la curva media entre semillas calcula el nivel final (media de las últimas `--last` evaluaciones) y el
primer paso a partir del cual la curva se queda por encima de `inicio + f·(final - inicio)` para cada f de
`--fracs`. Avisa si la curva sigue subiendo al final (el presupuesto se quedó corto). Los retornos son los
originales del entorno, también si se entrenó con --reward-shaping.
"""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np


def read_eval(path: Path):
    steps, vals = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("eval_return") not in (None, ""):
                steps.append(int(float(row["steps"])))
                vals.append(float(row["eval_return"]))
    return np.array(steps), np.array(vals)


def first_stable(steps, curve, level, rising: bool):
    """Primer paso desde el que la curva no vuelve a estar por debajo (o por encima) de `level`."""
    ok = curve >= level if rising else curve <= level
    bad = np.flatnonzero(~ok)
    i = 0 if bad.size == 0 else bad[-1] + 1
    return int(steps[i]) if i < len(steps) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", help="glob de directorios de corrida, p. ej. 'runs/est-acrobot-s*'")
    ap.add_argument("--last", type=int, default=5, help="nº de evaluaciones finales que definen el nivel final")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.9, 0.95, 0.99])
    a = ap.parse_args()

    runs = sorted(Path(p) for p in glob.glob(a.pattern))
    data = {r.name: read_eval(r / "log.csv") for r in runs if (r / "log.csv").exists()}
    data = {k: v for k, v in data.items() if len(v[0])}
    if not data:
        raise SystemExit(f"sin evaluaciones en {a.pattern!r} (¿--eval-every > 0?)")
    n = min(len(v[0]) for v in data.values())  # corridas aún en curso: se compara el tramo común
    steps = next(iter(data.values()))[0][:n]
    M = np.stack([v[1][:n] for v in data.values()])
    mean = M.mean(0)
    k = min(a.last, n)
    start, final = float(mean[0]), float(mean[-k:].mean())
    rising = final >= start
    print(f"{len(data)} corridas, {n} evaluaciones hasta el paso {steps[-1]:,}")
    for name, (s, v) in data.items():
        print(f"  {name:28s} inicio {v[0]:9.1f}  final {v[:n][-k:].mean():9.1f}  (máx {v[:n].max():9.1f})")
    print(f"media: inicio {start:.1f} -> final {final:.1f}")
    for f in a.fracs:
        st = first_stable(steps, mean, start + f * (final - start), rising)
        print(f"  {int(f * 100):3d}% de la mejora desde el paso {st:,}" if st else f"  {int(f * 100):3d}%: no se alcanza")
    # ¿sigue mejorando? compara el último quinto de la curva con el quinto anterior
    w = max(2, n // 5)
    if n >= 2 * w:
        gain = float(mean[-w:].mean() - mean[-2 * w:-w].mean()) * (1 if rising else -1)
        total = abs(final - start) or 1.0
        print(f"último quinto vs anterior: {gain:+.1f} ({100 * gain / total:+.1f}% de la mejora total)  ->  "
              + ("SIGUE SUBIENDO: el presupuesto se queda corto" if gain > 0.05 * total else "estable"))
    spread = float(np.std(M[:, -k:].mean(1)))
    print(f"dispersión del nivel final entre semillas: {spread:.1f}")


if __name__ == "__main__":
    main()
