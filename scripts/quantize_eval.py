"""Cuantiza (por redondeo) los pesos de las redes entrenadas y mide cuánto empeoran.

Esquema por defecto `odin`: 3 bits de magnitud (0..7) + signo, como las sinapsis de ODIN (sgrl/quantize.py).
Para cada corrida de validación (`<tarea>-p3-rank<k>-s<semilla>`) y cada anchura de bits se crea una red
NUEVA en `<out>/<esquema>/<carpeta de runs>/<corrida>/w<bits>/` (final.pt + config.json + quant.json) y se evalúa
con la política determinista. Las corridas originales sólo se leen, nunca se modifican. Además de las filas
cuantizadas hay una fila `bits=32` con la red original, evaluada con los mismos episodios, para comparar
por pares (misma semilla de entrenamiento, mismos estados iniciales).

    python scripts/quantize_eval.py --jobs 10                                  # ODIN, 3 bits, todas las corridas
    python scripts/quantize_eval.py --scheme symmetric --bits 8 6 5 4 3 2        # barrido de bits con signo
    python scripts/quantize_eval.py --runs-dir runs_default --jobs 10            # modelos 64x64
    python scripts/quantize_eval.py --runs-dir runs --task acrobot --model rank0  # un subconjunto
    python scripts/quantize_eval.py --summary-only                               # rehace el resumen desde el CSV

Escribe `<out>/<esquema>/quantization.csv` (una fila por red y bits) y `<out>/<esquema>/summary.csv` (agregado
entre semillas). `--episodes` y la semilla de evaluación (`semilla de entrenamiento +
20000`, la de `final_eval.json`) hacen que con `--episodes 20` la fila bits=32 reproduzca final_eval.json.
"""
import argparse
import csv
import json
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from sgrl.checkpoint import load
from sgrl.ppo import evaluate
from sgrl.quantize import SCHEMES, quantize_actor, weight_rmse

# <tarea>-<modelo>-s<semilla>: runs/ y runs_neuronas/ -> modelo `rank<k>` (p3-rank<k>); runs_default/ (64x64) -> `t<trial>` (f2-t<n>)
RUN_RE = re.compile(r"^(?P<task>.+?)-(?:p3-(?P<rank>rank\d+)|f2-(?P<trial>t\d+))-s(?P<seed>\d+)$")
FLOAT_BITS = 32
COLUMNS = ["family", "task", "model", "seed", "bits", "weight_rmse", "odin_neurons", "return_mean", "return_std", "length_mean"]


def discover(runs_dirs, task=None, model=None):
    runs = []
    for d in runs_dirs:
        for p in sorted(Path(d).iterdir()):
            m = RUN_RE.match(p.name)
            if not (m and (p / "final.pt").exists()):
                continue
            mod = m["rank"] or m["trial"]
            if (task and m["task"] != task) or (model and mod != model):
                continue
            runs.append((Path(d), p.name, m["task"], mod, int(m["seed"])))
    return runs


def quantize_run(job):
    """Una corrida -> filas (original + una por anchura de bits). Cada anchura parte del checkpoint original."""
    runs_dir, name, task, model, seed, bits_list, episodes, out, include_bias, device, scheme = job
    torch.set_num_threads(1)
    src = runs_dir / name
    eval_seed = seed + 20_000  # la de final_eval.json

    def row(agent, norm, env_name, bits, rmse, odin=0):
        ev = evaluate(agent, env_name, norm, episodes, eval_seed, torch.device(device))
        return dict(family=runs_dir.name, task=task, model=model, seed=seed, bits=bits, weight_rmse=rmse, odin_neurons=odin,
                    return_mean=ev["return_mean"], return_std=ev["return_std"], length_mean=ev["length_mean"])

    agent, norm, env_name = load(src / "final.pt", device)
    rows = [row(agent, norm, env_name, FLOAT_BITS, 0.0)]
    for bits in bits_list:
        agent, norm, env_name = load(src / "final.pt", device)  # copia fresca en memoria
        info = quantize_actor(agent.actor, bits, include_bias, scheme)
        dst = out / runs_dir.name / name / f"w{bits}"  # `out` ya incluye el esquema
        dst.mkdir(parents=True, exist_ok=True)
        ck = torch.load(src / "final.pt", map_location="cpu", weights_only=False)
        ck["agent"] = agent.state_dict()
        ck["quant"] = {"bits": bits, "scheme": scheme, "granularity": "per-layer", "include_bias": include_bias,
                       "source": str(src / "final.pt")}
        torch.save(ck, dst / "final.pt")
        shutil.copy(src / "config.json", dst / "config.json")  # `load` lee de aquí el normalizador
        rmse = weight_rmse(info)
        twins = sum(v.get("dale_twins", 0) for v in info.values())
        # neuronas físicas de ODIN: entradas + todas las capas + una gemela por neurona presináptica de signo mixto
        odin_neurons = agent.actor.fcs[0].in_features + sum(m.out_features for m in (*agent.actor.fcs, agent.actor.out)) + twins
        (dst / "quant.json").write_text(json.dumps({**ck["quant"], "weight_rmse": rmse, "dale_twins": twins,
                                                    "odin_neurons": odin_neurons, "tensors": info}, indent=2))
        agent, norm, env_name = load(dst / "final.pt", device)  # se evalúa lo guardado, no lo de memoria
        rows.append(row(agent, norm, env_name, bits, rmse, odin_neurons))
    return rows


def summarize(rows):
    """Agrega entre semillas de entrenamiento por (familia, tarea, modelo, bits); delta = cuantizada - original, por pares."""
    base = {(r["family"], r["task"], r["model"], r["seed"]): r["return_mean"] for r in rows if r["bits"] == FLOAT_BITS}
    groups = {}
    for r in rows:
        groups.setdefault((r["family"], r["task"], r["model"], r["bits"]), []).append(r)
    out = []
    for (fam, task, model, bits), rs in sorted(groups.items(), key=lambda kv: (kv[0][:3], -kv[0][3])):
        ret = np.array([r["return_mean"] for r in rs])
        delta = np.array([r["return_mean"] - base[(fam, task, model, r["seed"])] for r in rs])
        out.append(dict(family=fam, task=task, model=model, bits=bits, n_seeds=len(rs),
                        return_mean=ret.mean(), return_std_seeds=ret.std(), return_median=np.median(ret),
                        return_min=ret.min(), return_max=ret.max(), delta_mean=delta.mean(), delta_median=np.median(delta),
                        n_worse=int((delta < 0).sum()), weight_rmse=np.mean([r["weight_rmse"] for r in rs]),
                        odin_neurons_max=int(max(r["odin_neurons"] for r in rs)),
                        n_over_256=sum(r["odin_neurons"] > 256 for r in rs)))  # redes que no caben en el chip
    return out


def read_rows(path):
    """CSV -> filas tipadas. Acepta el formato antiguo (columna `rank` numérica en lugar de `model`)."""
    rows = []
    for r in csv.DictReader(open(path)):
        r["model"] = r.get("model") or f"rank{r['rank']}"
        rows.append({**r, **{k: float(r.get(k) or 0) for k in ("weight_rmse", "return_mean", "return_std", "length_mean", "odin_neurons")},
                     **{k: int(r[k]) for k in ("seed", "bits")}})
    return rows


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def print_summary(summary):
    print(f"\n{'familia':14s} {'tarea':11s} {'modelo':7s} bits  n | {'retorno medio ± std(semillas)':>31s} | {'mediana':>8s} {'Δ medio':>9s} {'Δ mediana':>9s} peor")
    for s in summary:
        b = "float" if s["bits"] == FLOAT_BITS else f"{s['bits']:5d}"
        print(f"{s['family']:14s} {s['task']:11s} {s['model']:7s} {b:>5s} {s['n_seeds']:2d} | "
              f"{s['return_mean']:14.1f} ± {s['return_std_seeds']:12.1f} | {s['return_median']:8.1f} "
              f"{s['delta_mean']:9.1f} {s['delta_median']:9.1f} {s['n_worse']:2d}/{s['n_seeds']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", nargs="+", default=["runs", "runs_neuronas"], help="carpetas con las corridas de validación")
    ap.add_argument("--out", type=Path, default=Path("runs_quant"))
    ap.add_argument("--scheme", choices=SCHEMES, default="odin", help="odin: --bits son bits de magnitud (signo aparte)")
    ap.add_argument("--bits", type=int, nargs="+", default=None, help="def.: 3 (odin) o 8 6 5 4 3 2 (symmetric)")
    ap.add_argument("--episodes", type=int, default=100, help="episodios por red y anchura de bits")
    ap.add_argument("--task", default=None, help="sólo esta tarea (acrobot, car, disc_mc; carrace y mountaincar en runs_default)")
    ap.add_argument("--model", default=None, help="sólo este modelo (rank0, rank1... en runs/; t0, t13... en runs_default/)")
    ap.add_argument("--include-bias", action="store_true", help="cuantizar también los sesgos (def.: digitales, en float)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)

    a.bits = a.bits or ([3] if a.scheme == "odin" else [8, 6, 5, 4, 3, 2])
    a.out = a.out / a.scheme
    a.out.mkdir(parents=True, exist_ok=True)
    csv_path = a.out / "quantization.csv"
    if a.summary_only:
        rows = read_rows(csv_path)
    else:
        runs = discover(a.runs_dir, a.task, a.model)
        if not runs:
            sys.exit("no hay corridas que cuantizar")
        print(f"{len(runs)} redes x bits {a.bits} ({a.scheme}) -> {a.out}/", file=sys.stderr)
        jobs = [(d, n, t, k, s, a.bits, a.episodes, a.out, a.include_bias, a.device, a.scheme) for d, n, t, k, s in runs]
        rows = []
        with ProcessPoolExecutor(min(a.jobs, len(jobs))) as ex:
            for i, rs in enumerate(ex.map(quantize_run, jobs), 1):
                rows += rs
                print(f"[{i}/{len(jobs)}] {rs[0]['family']}/{jobs[i-1][1]}", file=sys.stderr, flush=True)
        if csv_path.exists():  # conserva lo ya calculado de otras corridas (las nuevas reemplazan a las mismas redes)
            new = {(r["family"], r["task"], r["model"], r["seed"], r["bits"]) for r in rows}
            rows = [r for r in read_rows(csv_path) if (r["family"], r["task"], r["model"], r["seed"], r["bits"]) not in new] + rows
        write_csv(csv_path, rows, COLUMNS)
        print(f"-> {csv_path}", file=sys.stderr)
    summary = summarize(rows)
    write_csv(a.out / "summary.csv", summary, list(summary[0]))
    print_summary(summary)
    print(f"-> {a.out / 'summary.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
