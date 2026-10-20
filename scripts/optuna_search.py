"""Búsqueda de hiperparámetros con Optuna (actor LIF, PPO, sin ruido de memristores).

    # fase 1 (por defecto): PPO + beta, con el resto de flags fijos como en sgrl.train
    python scripts/optuna_search.py --study pend-f1 --n-trials 50 -- \
        --env Pendulum --steps 300000 --gamma 0.9 --device cuda --threads 1

    # fase 2: fija lo mejor de la fase 1 como flags base y busca surrogado/ganancias/T
    python scripts/optuna_search.py --study pend-f2 --phase 2 --n-trials 40 -- \
        --env Pendulum --steps 300000 --gamma 0.9 --actor-lr 1e-3 ...

Todo lo que va tras `--` son flags de `sgrl.train` (la configuración base, que no se busca). Los
parámetros que sí se buscan se añaden después y ganan. Varios procesos pueden compartir el estudio
(mismo `--study` y `--storage`) para repartir trials; con la GPU de la RTX 2050 no pases de ~3.

La puntuación de un trial (`--score`) es la media de las últimas `--last-k` evaluaciones deterministas
(con pesos nominales; `last-k`) o la media de toda la curva de evaluación (`auc`, premia aprender
rápido y no se satura cuando varios trials llegan al máximo), promediada sobre `--seeds-per-trial`
semillas (las mismas en todos los trials). El pruner sólo actúa durante la primera semilla.
Algunas tareas traen valores por defecto propios para --score, --seeds-per-trial, --pruner y
--warmup-evals (TASK_DEFAULTS); los flags explícitos ganan. Resultados: scripts/optuna_report.py.

Con `--aim-repo aim://host:53800` cada trial es un run de Aim (experimento = nombre del estudio, nombre
`<study>/trial-<n>`, hparams = parámetros del trial); las semillas van como contexto {seed: s}. Si Aim
falla (servidor caído) se avisa una vez y la búsqueda sigue.
"""
from __future__ import annotations

import argparse
import gc
import math
import shlex
import sys
from pathlib import Path

import numpy as np
import optuna
import torch

from sgrl.ppo import evaluate
from sgrl.snn import DEFAULT_ALPHA, DEFAULT_GAIN
from sgrl.train import build, parse_args

# nombre del parámetro -> (flags de sgrl.train que produce, función que lo sortea)
# Los rangos están justificados en la conversación de diseño: fase 1 = lo que más mueve el resultado.
# Valores por defecto de la búsqueda según la tarea (los flags explícitos ganan).
#  Acrobot: la puntuación se satura (≈ -16) en cuanto resuelve, así que se mide la curva entera.
#  MountainCarDiscrete: el resultado es casi binario (encuentra la meta o no) -> 3 semillas, y sin poda
#     porque un trial que aún no la encontró puede hacerlo después.
#  CarTrack: las curvas despegan tarde (la 1.ª evaluación ronda 30-40 y luego sube a 500+).
TASK_DEFAULTS = {
    "Acrobot": dict(score="auc"),
    "MountainCarDiscrete": dict(seeds_per_trial=3, pruner="none"),
    "CarTrack": dict(warmup_evals=5),
}
GLOBAL_DEFAULTS = dict(score="last-k", seeds_per_trial=1, pruner="median", warmup_evals=2)
MINIBATCHES = (256, 512, 1024, 2048, 4096)  # 4096: los lotes grandes (64 entornos x 128 pasos = 8192)
ENT_COEF_RANGE = (1e-5, 5e-2)  # MountainCar usa 0.02 en su preset

PHASE1 = ("actor_lr", "critic_lr", "ent_coef", "minibatch", "epochs", "beta")
PHASE2 = ("alpha", "sim_steps", "in_gain_scale", "hid_gain_scale", "gae_lambda", "init_std", "gamma")


def _suggest(trial: optuna.Trial, name: str) -> list[str]:
    f = trial.suggest_float
    if name == "actor_lr":
        return ["--actor-lr", f"{f(name, 1e-4, 3e-3, log=True):.6g}"]
    if name == "critic_lr":
        return ["--critic-lr", f"{f(name, 3e-4, 3e-3, log=True):.6g}"]
    if name == "ent_coef":
        return ["--ent-coef", f"{f(name, *ENT_COEF_RANGE, log=True):.6g}"]
    if name == "minibatch":
        return ["--minibatch", str(trial.suggest_categorical(name, list(MINIBATCHES)))]
    if name == "epochs":
        return ["--epochs", str(trial.suggest_int(name, 2, 10))]
    if name == "beta":
        return ["--beta", f"{f(name, 0.7, 0.97):.6g}"]
    if name == "alpha":
        return ["--alpha", f"{f(name, 5.0, 100.0, log=True):.6g}"]
    if name == "sim_steps":
        return ["--sim-steps", str(trial.suggest_categorical(name, [8, 16, 32]))]
    if name == "in_gain_scale":
        return ["--in-gain", f"{DEFAULT_GAIN[0] * f(name, 0.5, 2.0, log=True):.6g}"]
    if name == "hid_gain_scale":
        return ["--hid-gain", f"{DEFAULT_GAIN[1] * f(name, 0.5, 2.0, log=True):.6g}"]
    if name == "gae_lambda":
        return ["--gae-lambda", f"{f(name, 0.9, 0.99):.6g}"]
    if name == "init_std":
        return ["--init-std", f"{f(name, 0.2, 1.0):.6g}"]
    if name == "gamma":
        return ["--gamma", str(trial.suggest_categorical(name, [0.9, 0.95, 0.99]))]
    raise ValueError(f"parámetro desconocido: {name}")


def _baseline(name: str, base) -> object | None:
    """Valor de la config base para encolarlo como primer trial (None si no cabe en el rango)."""
    clamp = lambda v, lo, hi: min(max(v, lo), hi)
    if name == "actor_lr":
        return clamp(base.actor_lr, 1e-4, 3e-3)
    if name == "critic_lr":
        return clamp(base.critic_lr, 3e-4, 3e-3)
    if name == "ent_coef":
        return clamp(base.ent_coef, *ENT_COEF_RANGE)
    if name == "minibatch":
        return base.minibatch if base.minibatch in MINIBATCHES else None
    if name == "epochs":
        return int(clamp(base.epochs, 2, 10))
    if name == "beta":
        return clamp(base.beta, 0.7, 0.97)
    if name == "alpha":
        d = base.alpha if base.alpha is not None else DEFAULT_ALPHA[base.surrogate]
        return None if d is None else clamp(d, 5.0, 100.0)
    if name == "sim_steps":
        return base.sim_steps if base.sim_steps in (8, 16, 32) else None
    if name in ("in_gain_scale", "hid_gain_scale"):
        return 1.0 if (base.in_gain if name[0] == "i" else base.hid_gain) is None else None
    if name == "gae_lambda":
        return clamp(base.gae_lambda, 0.9, 0.99)
    if name == "init_std":
        return clamp(base.init_std, 0.2, 1.0)
    if name == "gamma":
        return base.gamma if base.gamma in (0.9, 0.95, 0.99) else None
    return None


class _Recorder:
    """Sustituye a optuna.Trial para listar el rango de cada parámetro sin duplicar las definiciones de _suggest."""

    spec = ""

    def suggest_float(self, name, low, high, log=False):
        self.spec = f"{low:g} … {high:g}" + ("  (escala log)" if log else "")
        return low

    def suggest_int(self, name, low, high):
        self.spec = f"{low} … {high}  (entero)"
        return low

    def suggest_categorical(self, name, choices):
        self.spec = "{" + ", ".join(str(c) for c in choices) + "}"
        return choices[0]


def print_plan(args, base, base_argv: list[str], params: tuple[str, ...]):
    """Qué se va a probar: parámetros con su rango, configuración base y ajustes de cada trial."""
    print(f"  Base fija: {shlex.join(base_argv)}")
    print(f"  Se prueba ({len(params)} parámetros):")
    for p in params:
        rec = _Recorder()
        _suggest(rec, p)
        note = "  [multiplica la ganancia por defecto]" if p.endswith("_gain_scale") else ""
        print(f"    {p:<15s} {rec.spec}{note}")
    print(f"  Cada trial: {base.env}, {base.steps:,} pasos · {args.seeds_per_trial} semilla(s) · puntuación {args.score}"
          f"{'' if args.score == 'auc' else f' (últimas {args.last_k} evaluaciones)'} · poda {args.pruner}")
    if not args.no_baseline:
        vals = {p: _baseline(p, base) for p in params}
        txt = ", ".join(f"{p}={v:.4g}" for p, v in vals.items() if v is not None)
        print(f"  Trial 0 (si el estudio es nuevo): la configuración base como referencia -> {txt}")
    print(f"  Los primeros {args.n_startup_trials} trials terminados se sortean al azar; después TPE propone según los resultados.")


class DeadNetwork(optuna.TrialPruned):
    pass


class AimTrial:
    """Run de Aim de un trial de Optuna. Todo error de red se degrada a un aviso: no tumba el trial."""

    def __init__(self, repo: str, experiment: str, trial: optuna.Trial, base_argv: list[str]):
        self.run = None
        try:
            import aim  # import tardío: sin --aim-repo no hace falta tenerlo
            self.run = aim.Run(repo=repo, experiment=experiment, system_tracking_interval=None,
                               log_system_params=False)
            self.run.name = f"{experiment}/trial-{trial.number}"
            self.run["hparams"] = {**trial.params, "trial": trial.number, "base": shlex.join(base_argv)}
            self.run.add_tag(f"study:{experiment}")
        except Exception as e:  # noqa: BLE001
            self._warn(e)

    def _warn(self, e: Exception):
        print(f"  [aim] desactivado: {type(e).__name__}: {e}", flush=True)
        self.run = None

    def track(self, seed: int, step: int, **metrics: float):
        if self.run is None:
            return
        try:
            for k, v in metrics.items():
                if v is not None and math.isfinite(v):
                    self.run.track(float(v), name=k, step=step, context={"seed": seed})
        except Exception as e:  # noqa: BLE001
            self._warn(e)

    def close(self, state: str, score: float | None = None):
        if self.run is None:
            return
        try:
            self.run["state"] = state
            if score is not None:
                self.run["score"] = score
            self.run.close()
        except Exception as e:  # noqa: BLE001
            self._warn(e)


def run_seed(trial: optuna.Trial, argv: list[str], seed: int, args, report: bool, aim: AimTrial | None = None) -> tuple[float, list[float]]:
    """Entrena una semilla y devuelve (puntuación, curva de evaluación)."""
    a = parse_args([*argv, "--seed", str(seed)])
    env, agent, ppo, cfg, _ = build(a)
    per_iter = cfg.rollout_steps * cfg.num_envs
    n_iter = max(1, math.ceil(a.steps / per_iter))
    evals: list[float] = []

    def log(info: dict):
        it = info["iteration"]
        if not all(math.isfinite(info[k]) for k in ("pg", "vf", "kl")):
            raise optuna.TrialPruned(f"NaN/inf en la pérdida (iter {it})")
        rates = info["rates"]
        if aim:
            aim.track(seed, info["steps"], ep_return=info["ep_return"], pg=info["pg"], vf=info["vf"], kl=info["kl"],
                      ent=info["ent"], std=info["std"], grad_norm=info["grad_norm"], sps=info["sps"],
                      rate_mean=float(np.mean(rates)) if rates else None)
        if args.dead_check and it == args.dead_check_iter and rates and (min(rates) < 1e-3 or max(rates) > 0.98):
            raise DeadNetwork(f"red muerta/saturada, tasas por capa = {[round(r, 3) for r in rates]}")
        if it % a.eval_every == 0 or it == n_iter:
            ev = evaluate(agent, a.env, ppo.norm, a.eval_episodes, a.seed + 10_000, ppo.device)["return_mean"]
            evals.append(ev)
            if aim:
                aim.track(seed, info["steps"], eval_return=ev)
            print(f"  [trial {trial.number} seed {seed}] it {it:4d}/{n_iter} steps {info['steps']:8d} "
                  f"train {info['ep_return']:9.1f} eval {ev:9.1f}", flush=True)
            if report:
                trial.report(ev, len(evals))
                if trial.should_prune():
                    raise optuna.TrialPruned()

    try:
        ppo.learn(a.steps, log)
    except ValueError as e:  # parámetros NaN en la distribución de la política
        if "invalid values" in str(e) or "nan" in str(e).lower():
            raise optuna.TrialPruned(f"política con NaN: {e}") from e
        raise
    finally:
        del ppo, agent, env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return float(np.mean(evals if args.score == "auc" else evals[-args.last_k:])), evals


def make_objective(base_argv: list[str], params: tuple[str, ...], args):
    def objective(trial: optuna.Trial) -> float:
        flags: list[str] = []
        for p in params:
            flags += _suggest(trial, p)
        trial.set_user_attr("flags", shlex.join(flags))
        argv = [*base_argv, *flags]
        aim = AimTrial(args.aim_repo, args.aim_experiment or args.study, trial, base_argv) if args.aim_repo else None
        state, score = "failed", None
        try:
            scores = []
            for i in range(args.seeds_per_trial):
                seed = args.seed + i
                s, curve = run_seed(trial, argv, seed, args, report=(i == 0), aim=aim)
                trial.set_user_attr(f"curve_seed{seed}", curve)
                scores.append(s)
                trial.set_user_attr("scores", scores)
            state, score = "complete", float(np.mean(scores))
            return score
        except optuna.TrialPruned as e:
            state = "dead" if isinstance(e, DeadNetwork) else "pruned"
            raise
        finally:
            if aim:
                aim.close(state, score)

    return objective


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True, help="nombre del estudio (varios procesos con el mismo nombre lo comparten)")
    ap.add_argument("--storage", default=None, help="URL de Optuna; por defecto sqlite:///optuna/<study>.db")
    ap.add_argument("--n-trials", type=int, default=50, help="trials de ESTE proceso")
    ap.add_argument("--timeout", type=float, default=None, help="segundos máximos de este proceso")
    ap.add_argument("--phase", choices=("1", "2", "all"), default="1")
    ap.add_argument("--params", default=None, help="lista separada por comas; sustituye a --phase "
                    f"(opciones: {', '.join(PHASE1 + PHASE2)})")
    ap.add_argument("--seeds-per-trial", type=int, default=None, help="def.: 1, o el de la tarea")
    ap.add_argument("--seed", type=int, default=0, help="primera semilla de entrenamiento (las demás son seed+1, ...)")
    ap.add_argument("--sampler-seed", type=int, default=None)
    ap.add_argument("--score", choices=("last-k", "auc"), default=None,
                    help="last-k: media de las últimas k evaluaciones; auc: media de toda la curva (def.: last-k, o el de la tarea)")
    ap.add_argument("--last-k", type=int, default=3, help="k de --score last-k")
    ap.add_argument("--pruner", choices=("median", "hyperband", "none"), default=None, help="def.: median, o el de la tarea")
    ap.add_argument("--n-startup-trials", type=int, default=8, help="trials completos antes de podar y de usar TPE")
    ap.add_argument("--warmup-evals", type=int, default=None, help="evaluaciones sin podar (pruner median; def.: 2, o el de la tarea)")
    ap.add_argument("--describe", action="store_true", help="imprime qué se va a probar y sale (no entrena ni crea el estudio)")
    ap.add_argument("--no-baseline", action="store_true", help="no encolar la config base como primer trial")
    ap.add_argument("--no-dead-check", dest="dead_check", action="store_false",
                    help="no podar redes sin disparos / saturadas a la iteración --dead-check-iter")
    ap.add_argument("--dead-check-iter", type=int, default=3)
    ap.add_argument("--aim-repo", default=None, help="repositorio de Aim (aim://host:53800 o ruta local); "
                    "si se omite no se usa Aim")
    ap.add_argument("--aim-experiment", default=None, help="experimento de Aim (por defecto, el nombre del estudio)")
    args, base_argv = ap.parse_known_args()
    if "--" in base_argv:
        base_argv.remove("--")

    base = parse_args(base_argv)
    for k, v in {**GLOBAL_DEFAULTS, **TASK_DEFAULTS.get(base.env, {})}.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if base.neuron == "ann":
        raise SystemExit("este script busca hiperparámetros de la SNN LIF (usa --neuron lif)")
    if base.train_noise_model != "none":
        raise SystemExit("la búsqueda es sin ruido: quita --train-noise-model")
    if base.eval_every <= 0:
        base_argv += ["--eval-every", "10"]
        base = parse_args(base_argv)
    params = tuple(args.params.split(",")) if args.params else {"1": PHASE1, "2": PHASE2, "all": PHASE1 + PHASE2}[args.phase]
    for p in params:
        if p not in PHASE1 + PHASE2:
            raise SystemExit(f"parámetro desconocido {p!r}; opciones: {PHASE1 + PHASE2}")

    if args.describe:
        print_plan(args, base, base_argv, params)
        return

    storage = args.storage
    if storage is None:
        Path("optuna").mkdir(exist_ok=True)
        storage = optuna.storages.RDBStorage(f"sqlite:///optuna/{args.study}.db",
                                             engine_kwargs={"connect_args": {"timeout": 60}})
    n_evals = max(1, math.ceil(base.steps / (base.num_envs * base.rollout_steps)) // base.eval_every)
    pruner = {
        "median": optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=args.warmup_evals),
        "hyperband": optuna.pruners.HyperbandPruner(min_resource=2, max_resource=max(n_evals, 3), reduction_factor=3),
        "none": optuna.pruners.NopPruner(),
    }[args.pruner]
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True, n_startup_trials=args.n_startup_trials)
    study = optuna.create_study(study_name=args.study, storage=storage, sampler=sampler, pruner=pruner,
                                direction="maximize", load_if_exists=True)

    # La config base se guarda en el estudio: así el reporte sabe qué se fijó, y se evita mezclar
    # trials de procesos lanzados con flags distintos.
    meta = dict(base_argv=shlex.join(base_argv), params=",".join(params), seeds_per_trial=args.seeds_per_trial,
                seed=args.seed, last_k=args.last_k, score=args.score)
    for k, v in meta.items():
        old = study.user_attrs.get(k)
        if old is None:
            study.set_user_attr(k, v)
        elif old != v:
            raise SystemExit(f"el estudio {args.study!r} se creó con {k}={old!r} y ahora pasas {v!r}; "
                             "usa otro --study (o los mismos flags) para no mezclar trials")

    if not args.no_baseline and not study.trials:
        vals = {p: _baseline(p, base) for p in params}
        study.enqueue_trial({p: v for p, v in vals.items() if v is not None})

    print(f"estudio {args.study}: buscando {list(params)}\n  base: {shlex.join(base_argv)}\n"
          f"  {base.env} · {base.steps} pasos/trial · {args.seeds_per_trial} semilla(s) · pruner {args.pruner} "
          f"(warmup {args.warmup_evals}) · puntuación {args.score}", flush=True)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(make_objective(base_argv, params, args), n_trials=args.n_trials, timeout=args.timeout,
                   catch=(), gc_after_trial=True)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if done:
        print(f"\nmejor trial #{study.best_trial.number}: {study.best_value:.1f}  {study.best_trial.user_attrs['flags']}")


if __name__ == "__main__":
    sys.exit(main())
