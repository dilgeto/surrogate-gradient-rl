"""Búsqueda de hiperparámetros con el mismo procedimiento que wann-cpp (screening_reduce.py + screening_full.py).

  Fase 1 (reduce)  Rondas de N trials con TPE a BAJA fidelidad (una fracción de los pasos). Tras cada ronda cada
                   parámetro se divide en K tramos y se eliminan los significativamente peores (Mann-Whitney U
                   unilateral, p < alpha y mediana del tramo < mediana global, con los resultados ACUMULADOS de todas
                   las rondas); el rango se encoge a la caja envolvente de los tramos que sobreviven. Se repite hasta
                   que no cae ningún tramo o se agotan las rondas.
  Fase 2 (phase2)  TPE a presupuesto COMPLETO en el espacio ya reducido, sin poda.
  Fase 3 (phase3)  Los mejores K de la fase 2, cada uno con varias semillas nuevas: media ± desviación.

Se buscan los 13 hiperparámetros a la vez (PPO + SNN), con rangos de partida amplios que la reducción estrecha.
Cada trial es UNA corrida (como en wann-cpp). Los trials de una ronda se reparten entre `--jobs` procesos que
comparten el estudio (SQLite); TPE usa constant_liar para tener en cuenta los trials en curso.

    python scripts/screening.py --task acrobot --mode all                  # fases 1, 2 y 3
    python scripts/screening.py --task car --mode reduce --rounds 4 --n 30 --jobs 8
    python scripts/screening.py --task car --mode all --hidden 3 2 --n2 40       # red pequeña (equivalente a una de wann-cpp)
    python scripts/screening.py --task disc_mc --mode both --n2 40 --top 3 --seeds 11 --jobs2 10 --jobs3 11
    python scripts/screening.py --task acrobot --mode analyse
    python scripts/screening.py --task acrobot --mode all -- --steps 2000000      # lo que va tras -- son flags de sgrl.train

Tareas: acrobot (Acrobot, shaping 10), disc_mc (MountainCarDiscrete, shaping 10, 500k pasos), car (CarTrack).
Resultados: screening/<tarea>/{results.csv, space_log.jsonl, p2_results.csv, p3_summary.json, studies.db} y
runs/<tarea>-p3-rank<R>-s<semilla>/ (las corridas de validación).
"""
from __future__ import annotations

import os

# Un hilo por proceso en todas las librerías (antes de importar numpy/torch): la paralelización es por procesos.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import csv
import fcntl
import json
import math
import multiprocessing as mp
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna

TASKS = {  # tarea -> entorno, shaping de wann-cpp (p/*.json), pasos por defecto (None = los del preset)
    "acrobot": dict(env="Acrobot", shaping=10, steps=None),
    "disc_mc": dict(env="MountainCarDiscrete", shaping=10, steps=500_000),
    "car": dict(env="CarTrack", shaping=None, steps=None, fidelity=0.5),  # sus curvas despegan tarde: 25 % se queda corto
}

# nombre -> (tipo, mínimo, máximo, paso). Tipos: log (log-uniforme), float (rejilla de `paso`), int, pow2 (entero k,
# el valor real es 2**k). Rangos amplios a propósito: en la búsqueda anterior los mejores valores se pegaron a los
# bordes. Se acota el coste de un trial: minibatch >= 128 y epochs <= 15 (el máximo sería 16 x 15 = 240 pasos de
# gradiente por iteración, unas 3 veces más que 256 x 10).
INITIAL_SPACE: dict[str, tuple] = {
    "actor_lr":       ("log", 1e-4, 1e-2, None),
    "critic_lr":      ("log", 1e-4, 1e-2, None),
    "ent_coef":       ("log", 1e-5, 0.2, None),
    "minibatch":      ("pow2", 7, 12, None),           # 128 .. 4096
    "epochs":         ("int", 2, 15, None),
    "beta":           ("float", 0.50, 0.99, 0.01),
    "alpha":          ("log", 5.0, 100.0, None),
    "sim_steps":      ("pow2", 3, 5, None),            # 8 .. 32
    "in_gain_scale":  ("log", 0.25, 4.0, None),
    "hid_gain_scale": ("log", 0.25, 4.0, None),
    "gae_lambda":     ("float", 0.90, 0.99, 0.01),
    "init_std":       ("float", 0.20, 1.00, 0.01),
    "gamma":          ("float", 0.900, 0.999, 0.001),
}


def real_value(name: str, kind: str, v):
    if kind == "pow2":
        return 2 ** int(round(v))
    return int(round(v)) if kind == "int" else v  # los enteros pueden volver de un CSV como 14.0


def to_flags(params: dict, space: dict) -> list[str]:
    from sgrl.snn import DEFAULT_GAIN
    p = {n: real_value(n, space[n][0], v) for n, v in params.items()}
    g = lambda x: f"{x:.6g}"
    flags = []
    if "actor_lr" in p: flags += ["--actor-lr", g(p["actor_lr"])]
    if "critic_lr" in p: flags += ["--critic-lr", g(p["critic_lr"])]
    if "ent_coef" in p: flags += ["--ent-coef", g(p["ent_coef"])]
    if "minibatch" in p: flags += ["--minibatch", str(p["minibatch"])]
    if "epochs" in p: flags += ["--epochs", str(p["epochs"])]
    if "beta" in p: flags += ["--beta", g(p["beta"])]
    if "alpha" in p: flags += ["--alpha", g(p["alpha"])]
    if "sim_steps" in p: flags += ["--sim-steps", str(p["sim_steps"])]
    if "in_gain_scale" in p: flags += ["--in-gain", g(DEFAULT_GAIN[0] * p["in_gain_scale"])]
    if "hid_gain_scale" in p: flags += ["--hid-gain", g(DEFAULT_GAIN[1] * p["hid_gain_scale"])]
    if "gae_lambda" in p: flags += ["--gae-lambda", g(p["gae_lambda"])]
    if "init_std" in p: flags += ["--init-std", g(p["init_std"])]
    if "gamma" in p: flags += ["--gamma", g(p["gamma"])]
    return flags


def suggest(trial: optuna.Trial, space: dict) -> dict:
    out = {}
    for name, (kind, lo, hi, step) in space.items():
        if kind == "log":
            out[name] = trial.suggest_float(name, lo, hi, log=True)
        elif kind == "float":
            out[name] = round(trial.suggest_float(name, lo, hi, step=step), 6)
        else:
            out[name] = trial.suggest_int(name, int(lo), int(hi))
    return out


# --- reducción del espacio (port de screening_reduce.reduce_space) ---------------------------------------------
def _edges(kind: str, lo: float, hi: float, K: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(lo), np.log(hi), K + 1)) if kind == "log" else np.linspace(lo, hi, K + 1)


def _round_out(v: float, step: float, up: bool) -> float:
    q = v / step
    return round((math.ceil(q - 1e-9) if up else math.floor(q + 1e-9)) * step, 6)


def reduce_space(rows: list[dict], space: dict, K: int = 5, alpha: float = 0.05, min_per_bin: int = 3):
    from scipy.stats import mannwhitneyu
    ok = [r for r in rows if r.get("score") is not None and not (isinstance(r["score"], float) and math.isnan(r["score"]))]
    y = np.array([r["score"] for r in ok], dtype=float)
    gmed = float(np.median(y))
    new_space, report = {}, {}
    for name, (kind, lo, hi, step) in space.items():
        if name not in ok[0]:
            new_space[name] = (kind, lo, hi, step)
            continue
        x = np.array([r[name] for r in ok], dtype=float)
        edges = _edges(kind, lo, hi, K)
        bin_of = np.clip(np.digitize(x, edges[1:-1]), 0, K - 1)
        eliminated = set()
        for b in range(K):
            m = bin_of == b
            if m.sum() < min_per_bin or (~m).sum() < 2:
                continue  # pocos datos: se conserva
            _, p = mannwhitneyu(y[m], y[~m], alternative="less")
            if p < alpha and float(np.median(y[m])) < gmed:
                eliminated.add(b)
        surviving = [b for b in range(K) if b not in eliminated]
        if not surviving:  # seguridad: al menos el tramo de mejor mediana
            meds = [float(np.median(y[bin_of == b])) if (bin_of == b).any() else -np.inf for b in range(K)]
            surviving = [int(np.argmax(meds))]
        nlo = lo if min(surviving) == 0 else float(edges[min(surviving)])  # borde sin recorte: exacto (sin ruido de log/exp)
        nhi = hi if max(surviving) == K - 1 else float(edges[max(surviving) + 1])
        if kind in ("int", "pow2"):
            nlo, nhi = max(lo, float(math.ceil(nlo))), min(hi, float(math.floor(nhi)))
            if nlo > nhi:
                nlo, nhi = lo, hi
        elif kind == "float":  # conservador: hacia fuera hasta la rejilla, sin salirse del rango actual
            nlo, nhi = max(lo, _round_out(nlo, step, False)), min(hi, _round_out(nhi, step, True))
        else:
            nlo, nhi = max(lo, nlo), min(hi, nhi)
        new_space[name] = (kind, nlo, nhi, step)
        report[name] = dict(old=(lo, hi), new=(nlo, nhi), eliminated=sorted(eliminated),
                            counts=[int((bin_of == b).sum()) for b in range(K)])
    return new_space, report


# --- ejecución de trials ---------------------------------------------------------------------------------------
class _Lock:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.f = open(self.path, "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def worker(cfg: dict) -> int:
    """Un proceso: pide trials al estudio compartido hasta que haya `n`, y los ejecuta (ask/tell con cerrojo: exacto)."""
    import optuna_search as osr
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = {k: tuple(v) for k, v in cfg["space"].items()}
    storage = optuna.storages.RDBStorage(cfg["storage"], engine_kwargs={"connect_args": {"timeout": 60}})
    pruner = (optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1) if cfg["prune"]
              else optuna.pruners.NopPruner())
    sampler = optuna.samplers.TPESampler(seed=cfg["sampler_seed"], constant_liar=True, multivariate=True,
                                         n_startup_trials=max(5, cfg["n"] // 4))
    study = optuna.load_study(study_name=cfg["study"], storage=storage, sampler=sampler, pruner=pruner)
    ns = SimpleNamespace(score="last-k", last_k=3, dead_check=True, dead_check_iter=3)
    log = open(cfg["log"], "a")
    done = 0
    while True:
        with _Lock(cfg["lock"]):
            if len(study.get_trials(deepcopy=False)) >= cfg["n"]:
                break
            trial = study.ask()
        t0 = time.time()
        try:
            params = suggest(trial, space)
            flags = to_flags(params, space)
            argv = [*cfg["base_argv"], *flags, "--steps", str(cfg["steps"]), "--eval-every", str(cfg["eval_every"])]
            with contextlib.redirect_stdout(log):
                score, curve = osr.run_seed(trial, argv, cfg["seed"] + cfg["round"] * 100_000 + trial.number, ns,
                                            report=cfg["prune"])
            trial.set_user_attr("flags", shlex.join(flags))
            trial.set_user_attr("elapsed_s", round(time.time() - t0, 1))
            study.tell(trial, score)
            best = study.best_value
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} · {cfg['tag']} · trial {trial.number} · valor {score:.4g} · "
                  f"mejor {best:.4g} · {{{', '.join(f'{k}: {v:.4g}' for k, v in params.items())}}}", flush=True)
            done += 1
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except Exception as e:  # noqa: BLE001
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f"!! {cfg['tag']} trial {trial.number} falló: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    return done


def run_trials(a, tag: str, study_name: str, space: dict, n: int, round_idx: int, steps: int, prune: bool,
               base_argv: list[str], out: Path, jobs: int):
    """Lanza `jobs` procesos sobre un estudio nuevo y devuelve las filas (completados) de este estudio."""
    from sgrl.train import parse_args
    base = parse_args(base_argv)
    n_iter = max(1, math.ceil(steps / (base.num_envs * base.rollout_steps)))
    every = max(1, n_iter // 8)  # ~8 evaluaciones por trial (wann-cpp: un punto de control cada maxGen/8)
    url = f"sqlite:///{(out / 'studies.db').resolve()}"
    storage = optuna.storages.RDBStorage(url, engine_kwargs={"connect_args": {"timeout": 60}})
    optuna.create_study(study_name=study_name, storage=storage, direction="maximize", load_if_exists=True)
    (out / "logs").mkdir(exist_ok=True)
    jobs = max(1, min(jobs, n))
    cfgs = [dict(storage=url, study=study_name, space={k: list(v) for k, v in space.items()}, n=n, round=round_idx,
                 prune=prune, sampler_seed=a.seed + round_idx + 1000 * w, base_argv=base_argv, steps=steps,
                 eval_every=every, seed=a.seed, tag=tag, lock=str(out / f".lock-{study_name}"),
                 log=str(out / "logs" / f"{study_name}-w{w}.log")) for w in range(jobs)]
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
        for f in [ex.submit(worker, c) for c in cfgs]:
            f.result()
    study = optuna.load_study(study_name=study_name, storage=storage)
    rows = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            rows.append(dict(round=round_idx, trial=t.number, score=t.value, elapsed_s=t.user_attrs.get("elapsed_s"),
                             **t.params))
    states = [t.state.name for t in study.trials]
    print(f"  {tag}: {states.count('COMPLETE')} completos, {states.count('PRUNED')} podados, "
          f"{states.count('FAIL')} fallidos" + (f" · mejor {max(r['score'] for r in rows):.4g}" if rows else ""),
          flush=True)
    return rows


# --- utilidades de E/S -----------------------------------------------------------------------------------------
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [{k: (float(v) if v not in ("", None) and k != "tag" else None) for k, v in r.items()} for r in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict]):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def fmt_space(space: dict) -> str:
    return "\n".join(f"    {n:<15s} [{lo:.4g}, {hi:.4g}]  ({kind}{', 2**k' if kind == 'pow2' else ''})"
                     for n, (kind, lo, hi, _s) in space.items())


def last_space(out: Path):
    log = out / "space_log.jsonl"
    last = None
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                last = json.loads(line)
    return last


# --- fases -----------------------------------------------------------------------------------------------------
def check_deps():
    """Falla al arrancar (no tras una hora de trials) si falta scipy en ESTE intérprete."""
    try:
        from scipy.stats import mannwhitneyu  # noqa: F401
    except ImportError:
        raise SystemExit(f"falta scipy en el Python que ejecuta el script ({sys.executable}).\n"
                         f"Instálalo con ese mismo intérprete:  {sys.executable} -m pip install scipy")


def merge_round(rows: list[dict], r: int, new: list[dict]) -> list[dict]:
    """Resultados acumulados con los de la ronda r sustituidos por `new`. Si una ejecución anterior se cortó después de
    guardar results.csv y antes de reducir el espacio, esa ronda ya está en `rows`: añadirla otra vez duplicaría datos."""
    return [x for x in rows if int(x["round"]) != r] + new


def phase_reduce(a, base_argv, full_steps, out):
    check_deps()
    steps = max(1, int(round(a.fidelity * full_steps)))
    rows = read_csv(out / "results.csv")
    space = dict(INITIAL_SPACE)
    start = 0
    last = last_space(out)
    if last:
        space = {k: tuple(v) for k, v in last["space"].items()}
        start = last["round"] + 1
        print(f"Reanudando: espacio restaurado de la ronda {last['round']}, {len(rows)} resultados previos.")
        if not any(last["eliminated"].values() if last["eliminated"] else []) and last["round"] >= 0:
            print("La reducción ya había convergido.")
            return
    print(f"\n== {a.task} · fase 1 (reducción) · {a.rounds} rondas x {a.n} trials · fidelidad {a.fidelity:.0%} "
          f"({steps:,} de {full_steps:,} pasos) · {a.jobs} procesos ==", flush=True)
    for r in range(start, a.rounds):
        print(f"\nRonda {r} · espacio:\n{fmt_space(space)}", flush=True)
        new = run_trials(a, f"r{r}", f"{a.task}-r{r:02d}", space, a.n, r, steps, True, base_argv, out, a.jobs)
        rows = merge_round(rows, r, new)
        write_csv(out / "results.csv", rows)
        if len(rows) < 3:
            print("  Muy pocos trials completados para reducir el espacio.")
            continue
        space_before = dict(space)
        space, report = reduce_space(rows, space, K=a.K, alpha=a.alpha)
        with open(out / "space_log.jsonl", "a") as f:
            f.write(json.dumps(dict(round=r, n_trials=len(new), space={k: list(v) for k, v in space.items()},
                                    eliminated={k: v["eliminated"] for k, v in report.items() if v["eliminated"]})) + "\n")
        print(f"\n── Espacio tras la ronda {r} (K={a.K} tramos, α={a.alpha}) ──")
        changed = False
        for n_, info in report.items():
            if info["eliminated"]:
                changed = True
                (lo0, hi0), (lo1, hi1) = info["old"], info["new"]
                pct = 100 * (1 - (hi1 - lo1) / (hi0 - lo0 + 1e-12))
                print(f"  {n_:<15s} [{lo0:.4g}, {hi0:.4g}] → [{lo1:.4g}, {hi1:.4g}]  (-{pct:.0f}%)  tramos eliminados: {info['eliminated']}")
        if not changed:
            print(f"  Ningún tramo eliminado: converge en la ronda {r}.")
            break


def phase2(a, base_argv, full_steps, out):
    last = last_space(out)
    space = {k: tuple(v) for k, v in last["space"].items()} if last else dict(INITIAL_SPACE)
    src = f"ronda {last['round']} de space_log.jsonl" if last else "INITIAL_SPACE (no hay reducción)"
    n = a.n2
    print(f"\n== {a.task} · fase 2 · {n} trials a presupuesto completo ({full_steps:,} pasos) · {min(a.jobs2, n)} procesos ==\n"
          f"  espacio: {src}\n{fmt_space(space)}", flush=True)
    study_name = f"{a.task}-p2"
    rows = run_trials(a, "p2", study_name, space, n, 0, full_steps, False, base_argv, out, a.jobs2)
    rows = sorted(rows, key=lambda r: -r["score"])
    write_csv(out / "p2_results.csv", rows)
    (out / "p2_space.json").write_text(json.dumps({k: list(v) for k, v in space.items()}))
    print("  mejores:")
    for r in rows[:5]:
        print(f"    trial {int(r['trial']):3d}  valor {r['score']:9.4g}  " + " ".join(f"{k}={r[k]:.4g}" for k in space))


def phase3(a, base_argv, full_steps, out):
    import optuna_report as rep
    from sgrl.train import parse_args
    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows = read_csv(out / "p2_results.csv")
    if len(rows) < a.top:
        raise SystemExit(f"hacen falta {a.top} resultados en {out}/p2_results.csv (hay {len(rows)}); ejecuta la fase 2")
    space = {k: tuple(v) for k, v in json.loads((out / "p2_space.json").read_text()).items()}
    top = sorted(rows, key=lambda r: -r["score"])[:a.top]
    base = parse_args(base_argv)
    every = max(1, math.ceil(full_steps / (base.num_envs * base.rollout_steps)) // 8)
    work = []
    for rank, r in enumerate(top):
        flags = to_flags({k: r[k] for k in space}, space)
        for si in range(a.seeds):
            seed = 100 + si
            argv = [*base_argv, *flags, "--steps", str(full_steps), "--eval-every", str(every), "--seed", str(seed)]
            work.append((rank, seed, f"{a.task}-p3-rank{rank}-s{seed}", argv))
    jobs = a.jobs3 if a.jobs3 > 0 else rep.auto_jobs(len(work), "cpu")
    jobs = max(1, min(jobs, len(work)))
    print(f"\n== {a.task} · fase 3 · top {a.top} x {a.seeds} semillas = {len(work)} corridas, {jobs} a la vez ==", flush=True)
    finals = {rk: {} for rk in range(len(top))}
    failed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(rep.run_one, argv, name, a.runs_dir): (rk, sd, name) for rk, sd, name, argv in work}
        for i, fut in enumerate(as_completed(futs), 1):
            rk, sd, name = futs[fut]
            try:
                finals[rk][sd] = fut.result()["return_mean"]
                print(f"  [{i}/{len(work)}] {name}: {finals[rk][sd]:.1f}", flush=True)
            except Exception as e:  # noqa: BLE001
                failed.append(name)
                print(f"  [{i}/{len(work)}] {name}: FALLÓ ({e})", file=sys.stderr, flush=True)
    summary = {}
    for rk, r in enumerate(top):
        v = [finals[rk][sd] for sd in sorted(finals[rk])]
        if v:
            summary[rk] = dict(search_value=r["score"], finals=v, mean=float(np.mean(v)), std=float(np.std(v)),
                               trial=int(r["trial"]), flags=shlex.join(to_flags({k: r[k] for k in space}, space)))
            print(f"rank {rk} (trial {int(r['trial'])}): fase 2 {r['score']:.1f} -> validación {np.mean(v):.1f} ± {np.std(v):.1f} "
                  f"({len(v)}/{a.seeds} semillas)")
    (out / "p3_summary.json").write_text(json.dumps(summary, indent=2))
    if summary:
        rep.plot_validation(f"{a.task}-p3", {str(v["trial"]): v for v in summary.values()}, f"{a.runs_dir}/{a.task}-p3-validation.png")
    if failed:
        raise SystemExit(f"fallaron {len(failed)} corridas: {', '.join(failed)}")


def analyse(a, out):
    log = out / "space_log.jsonl"
    if log.exists():
        print("Evolución del espacio:")
        for line in log.read_text().splitlines():
            e = json.loads(line)
            print(f"  ronda {e['round']}: {e['n_trials']} trials, tramos eliminados: {e['eliminated'] or 'ninguno'}")
        print("Espacio final:\n" + fmt_space({k: tuple(v) for k, v in json.loads(log.read_text().splitlines()[-1])["space"].items()}))
    p2 = read_csv(out / "p2_results.csv")
    if p2:
        print(f"\nFase 2: {len(p2)} trials, mejor {max(r['score'] for r in p2):.4g}")
    s3 = out / "p3_summary.json"
    if s3.exists():
        print("\nFase 3:")
        for k, v in json.loads(s3.read_text()).items():
            print(f"  rank {k} (trial {v['trial']}): fase 2 {v['search_value']:.1f} -> validación {v['mean']:.1f} ± {v['std']:.1f}")
            print(f"     {v['flags']}")


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--mode", default="all", choices=("reduce", "phase2", "phase3", "both", "all", "analyse"),
                    help="both = fases 2 y 3; all = 1, 2 y 3")
    ap.add_argument("--rounds", type=int, default=4, help="rondas máximas de reducción (def. 4, como wann-cpp)")
    ap.add_argument("--n", type=int, default=30, help="trials por ronda en la fase 1 (def. 30, como wann-cpp)")
    ap.add_argument("--n2", type=int, default=20, help="trials de la fase 2 (def. 20, como wann-cpp)")
    ap.add_argument("--hidden", type=int, nargs="+", default=None,
                    help="neuronas ocultas del actor, p. ej. --hidden 3 2 (def. el del preset: 64 64). Fijo, no se busca")
    ap.add_argument("--jobs", type=int, default=8, help="procesos en paralelo en la fase 1 (def. 8)")
    ap.add_argument("--jobs2", type=int, default=4,
                    help="procesos en paralelo en la fase 2 (def. 4): con solo 20 trials, más procesos dejan casi la mitad "
                         "de los trials sin guía de TPE (wann-cpp usa 3 en esa fase)")
    ap.add_argument("--jobs3", type=int, default=0, help="corridas simultáneas en la fase 3 (0 = el máximo que cabe)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=5, help="tramos por parámetro en la reducción")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fidelity", type=float, default=None, help="fracción de los pasos en la fase 1 (def. 0.25; 0.5 en car)")
    ap.add_argument("--top", type=int, default=3, help="trials de la fase 2 que se validan en la fase 3")
    ap.add_argument("--seeds", type=int, default=11, help="semillas por trial en la fase 3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dir", default="screening", help="carpeta de resultados")
    ap.add_argument("--runs-dir", default="runs", dest="runs_dir", help="carpeta de las corridas de la fase 3")
    a, extra = ap.parse_known_args()
    if "--" in extra:
        extra.remove("--")
    t = TASKS[a.task]
    if a.fidelity is None:
        a.fidelity = t.get("fidelity", 0.25)
    base_argv = ["--device", a.device, "--threads", "1", "--env", t["env"]]
    if t["shaping"]:
        base_argv += ["--reward-shaping", str(t["shaping"])]
    if t["steps"]:
        base_argv += ["--steps", str(t["steps"])]
    if a.hidden:
        base_argv += ["--hidden", *map(str, a.hidden)]
    base_argv += extra
    from sgrl.train import parse_args
    full_steps = parse_args(base_argv).steps
    if a.mode != "analyse":  # falla ya, y no tras horas de trials, si falta algo
        import check_env
        if not check_env.run(quiet=True):
            raise SystemExit(1)
    out = Path(a.dir) / a.task
    out.mkdir(parents=True, exist_ok=True)
    Path(a.runs_dir).mkdir(parents=True, exist_ok=True)
    (out / "base_argv.txt").write_text(shlex.join(base_argv) + "\n")
    print(f"tarea {a.task} · base: {shlex.join(base_argv)}", flush=True)
    if a.mode == "analyse":
        return analyse(a, out)
    if a.mode in ("reduce", "all"):
        phase_reduce(a, base_argv, full_steps, out)
    if a.mode in ("phase2", "both", "all"):
        phase2(a, base_argv, full_steps, out)
    if a.mode in ("phase3", "both", "all"):
        phase3(a, base_argv, full_steps, out)


if __name__ == "__main__":
    sys.exit(main())
