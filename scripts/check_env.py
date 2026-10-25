"""Comprueba, ANTES de lanzar una búsqueda larga, que este Python tiene todo lo que necesita el pipeline.

    python scripts/check_env.py            # tabla completa (OK / FALTA) y sale con código 1 si algo falla
    python scripts/check_env.py --quiet    # sólo imprime si hay problemas

No basta con que un paquete "esté instalado": se importa con el MISMO intérprete que ejecutará el script y se prueba
lo que el pipeline usa de él (SQLite a través de Optuna, el test de Mann-Whitney, matplotlib sin pantalla) y que el módulo
compilado de entornos es el actual (no uno viejo: compara con valores de referencia de wann-cpp).
Lo llaman scripts/screening.py y los lanzadores screening_all*.sh nada más empezar.
"""
from __future__ import annotations

import argparse
import importlib
import math
import os
import sys

REQUIRED = [  # (módulo, paquete de pip)
    ("numpy", "numpy"), ("torch", "torch"), ("snntorch", "snntorch"), ("matplotlib", "matplotlib"),
    ("optuna", "optuna"), ("scipy", "scipy"), ("sqlalchemy", "sqlalchemy"),
]
OPTIONAL = [("pytest", "pytest"), ("aim", "aim (sólo con --aim-repo en los scripts antiguos)")]


def _version(mod, pkg: str | None = None) -> str:
    try:
        from importlib.metadata import version
        return version((pkg or mod.__name__).split()[0])
    except Exception:  # noqa: BLE001
        v = getattr(mod, "__version__", "?")
        return v if isinstance(v, str) else "?"


def run(quiet: bool = False) -> bool:
    rows, problems, pip_missing = [], [], []

    def ok(name, detail=""):
        rows.append(("OK   ", name, detail))

    def bad(name, detail, fix=None):
        rows.append(("FALTA", name, detail))
        problems.append(f"{name}: {detail}" + (f"\n      arreglo: {fix}" if fix else ""))

    ok("python", f"{sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 10):
        bad("python", "se necesita Python >= 3.10")

    mods = {}
    for mod, pkg in REQUIRED:
        try:
            mods[mod] = importlib.import_module(mod)
            ok(mod, _version(mods[mod], pkg))
        except Exception as e:  # noqa: BLE001
            pip_missing.append(pkg)
            bad(mod, f"no se puede importar ({type(e).__name__}: {e})")
    for mod, pkg in OPTIONAL:
        try:
            m = importlib.import_module(mod)
            rows.append(("opc. ", mod, _version(m, pkg)))
        except Exception:  # noqa: BLE001
            rows.append(("opc. ", mod, "no instalado (no hace falta)"))

    try:
        import sqlite3
        sqlite3.connect(":memory:").execute("select 1")
        ok("sqlite3", sqlite3.sqlite_version)
    except Exception as e:  # noqa: BLE001
        bad("sqlite3", f"{type(e).__name__}: {e}",
            "Python compilado sin SQLite: instala las cabeceras (p. ej. sqlite-devel) y recompílalo (pyenv install -f <versión>)")

    if "optuna" in mods:  # Optuna + SQLite de verdad (el pipeline guarda todos sus estudios así)
        try:
            optuna = mods["optuna"]
            optuna.logging.set_verbosity(optuna.logging.ERROR)
            st = optuna.create_study(storage="sqlite:///:memory:", sampler=optuna.samplers.TPESampler(
                constant_liar=True, multivariate=True))
            t = st.ask()
            t.suggest_float("x", 0, 1)
            st.tell(t, 1.0)
            ok("optuna+sqlite", "estudio ask/tell con TPE (constant_liar, multivariante)")
        except Exception as e:  # noqa: BLE001
            bad("optuna+sqlite", f"{type(e).__name__}: {e}")

    if "scipy" in mods:
        try:
            from scipy.stats import mannwhitneyu
            mannwhitneyu([1, 2, 3, 4], [5, 6, 7, 8], alternative="less")
            ok("scipy.mannwhitneyu", "funciona")
        except Exception as e:  # noqa: BLE001
            bad("scipy.mannwhitneyu", f"{type(e).__name__}: {e}")

    if "matplotlib" in mods:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure().savefig(os.devnull, format="png")
            plt.close("all")
            ok("matplotlib (sin pantalla)", "backend Agg")
        except Exception as e:  # noqa: BLE001
            bad("matplotlib (sin pantalla)", f"{type(e).__name__}: {e}")

    if "torch" in mods and "snntorch" in mods:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
            from sgrl.snn import SNNActor  # noqa: F401
            ok("sgrl (actor SNN)", "importa")
        except Exception as e:  # noqa: BLE001
            bad("sgrl (actor SNN)", f"{type(e).__name__}: {e}", "pip install -e . --no-deps   (desde la raíz del repo)")

    # módulo compilado de entornos: que exista y sea el ACTUAL (valores de referencia de wann-cpp, ver tests/test_wann_envs.py)
    if "numpy" in mods:
        try:
            import numpy as np
            from sgrl.envs import VecEnv
            stale = []
            mc = VecEnv("MountainCarDiscrete", 1)
            if mc.episode_step_limit != 200:
                stale.append(f"MountainCarDiscrete dura {mc.episode_step_limit} pasos (debe ser 200)")
            car = VecEnv("CarTrack", 1, seed=0)
            o = car.reset(12345)[0][0]
            if not math.isclose(o, 0.6002231392859763, rel_tol=1e-6):
                stale.append(f"CarTrack: obs[0]={o:.4f} (debe ser 0.6002; con 0.2401 la pista es de 20 mm, la antigua)")
            ac = VecEnv("Acrobot", 1, seed=0)
            ac.reset(12345)
            fo = ac.step(np.array([[math.sin(0.5)]]))[4][0]
            if not math.isclose(fo[4], -0.01736849781899555, rel_tol=1e-4):
                stale.append(f"Acrobot: dθ1={fo[4]:.4f} tras un paso (debe ser -0.0174; con torque ±5 sale otro valor)")
            if stale:
                bad("entornos compilados", "; ".join(stale), "scripts/build_envs.sh   (con el venv activo; si no cambia, rm -rf build antes)")
            else:
                ok("entornos compilados", "Acrobot, MountainCarDiscrete y CarTrack = los de wann-cpp")
        except ImportError as e:
            bad("entornos compilados", f"no se puede importar el módulo ({e})", "scripts/build_envs.sh (necesita pybind11, cmake y g++ >= 10)")
        except Exception as e:  # noqa: BLE001
            bad("entornos compilados", f"{type(e).__name__}: {e}", "scripts/build_envs.sh")

    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        kb = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
        mem = f"{kb / 1024 / 1024:.1f} GB libres"
    except (OSError, StopIteration, ValueError):
        mem = "RAM desconocida"
    ok("recursos", f"{cpus} hebras, {mem}")

    if not quiet or problems:
        w = max(len(r[1]) for r in rows)
        for flag, name, detail in rows:
            print(f"  [{flag}] {name:<{w}s}  {detail}", flush=True)
    if problems:
        print("\nHAY PROBLEMAS:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        if pip_missing:
            print(f"\n  Instala lo que falta con ESTE intérprete:\n      {sys.executable} -m pip install {' '.join(sorted(set(pip_missing)))}",
                  file=sys.stderr)
        return False
    if not quiet:
        print("\nTodo en orden.")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true", help="sólo imprime si hay problemas")
    a = ap.parse_args()
    return 0 if run(a.quiet) else 1


if __name__ == "__main__":
    sys.exit(main())
