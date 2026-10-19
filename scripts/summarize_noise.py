"""Resume los CSV de scripts/noise_eval.py (puerto de wann-cpp `summarize_car_noise.py`).

No corre ningún modelo: sólo agrega filas ya calculadas. Cada fila del CSV es una lectura de un chip
(seed, chip, read, t_inference). Agrupa por (tag, noise_model, t_inference, prog/read/drift scale,
n_levels) y reporta del retorno (`return_mean`, ya promediado sobre los episodios de la lectura):

  * media, desviación y error estándar sobre filas;
  * desviación ENTRE chips (de las medias por chip: mala suerte de programación) y DENTRO de chip
    (entre lecturas del mismo chip: ruido de lectura puro; NaN con `--reads-per-chip 1`);
  * percentiles 10/50/90 de las medias por chip;
  * IC 95 % por bootstrap de la media, remuestreando CHIPS (no filas: lecturas del mismo chip no son
    independientes, remuestrear filas subestimaría la incertidumbre);
  * `frac_chips_above_<umbral>` con `--success-threshold`;
  * con filas `noise_model == none` (o `--baseline R`): delta_mean y delta_pct frente al retorno sin
    ruido. delta_pct = (media - base) / |base| · 100, así que negativo = peor también con retornos < 0.

    python scripts/summarize_noise.py --csv noise_pcm.csv [--out summary.csv] [--success-threshold -200]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_GROUP_COLS = ["noise_model", "t_inference", "prog_noise_scale", "read_noise_scale", "drift_scale", "n_levels"]
CHIP_COLS = ["seed", "chip"]  # identidad de un chip: (seed, chip)
REWARD = "return_mean"


def load_rows(csv_paths) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        if not Path(p).is_file():
            sys.exit(f"Error: no existe {p}")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    missing = set(BASE_GROUP_COLS + CHIP_COLS + [REWARD, "weight_rmse"]) - set(df.columns)
    if missing:
        sys.exit(f"Error: al CSV le faltan columnas de noise_eval.py: {sorted(missing)}")
    if "tag" in df.columns:
        df["tag"] = df["tag"].fillna("")
    return df


def group_cols(df: pd.DataFrame):
    return (["tag"] if "tag" in df.columns else []) + BASE_GROUP_COLS


def percentile_ci(x: np.ndarray, ci: float = 0.95):
    alpha = 1.0 - ci
    lo, hi = np.percentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_mean_ci(chip_means: np.ndarray, n_resamples: int, rng: np.random.Generator, ci: float = 0.95):
    """IC por percentiles de la media remuestreando chips con reemplazo."""
    n = len(chip_means)
    if n < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_resamples, n))
    return percentile_ci(chip_means[idx].mean(axis=1), ci=ci)


def summarize(df: pd.DataFrame, n_resamples: int = 10000, bootstrap_seed: int = 0,
              success_threshold=None, baseline=None) -> pd.DataFrame:
    cols = group_cols(df)
    rng = np.random.default_rng(bootstrap_seed)
    if baseline is None:
        base_rows = df[df["noise_model"] == "none"]
        baseline = float(base_rows[REWARD].mean()) if not base_rows.empty else None

    rows = []
    for keys, group in df.groupby(cols, dropna=False):
        row = dict(zip(cols, keys))
        per_chip = group.groupby(CHIP_COLS)
        row["n"], row["n_chips"] = len(group), per_chip.ngroups
        row["reads_per_chip"] = len(group) / per_chip.ngroups
        row["weight_rmse_mean"] = group["weight_rmse"].mean()

        mean = group[REWARD].mean()
        std = group[REWARD].std(ddof=1) if len(group) > 1 else float("nan")
        row["ret_mean"], row["ret_std"] = mean, std
        row["ret_sem"] = std / len(group) ** 0.5 if len(group) > 1 else float("nan")

        chip_mean = per_chip[REWARD].mean()  # una fila por (seed, chip)
        row["ret_between_chip_std"] = chip_mean.std(ddof=1) if len(chip_mean) > 1 else float("nan")
        row["ret_within_chip_std"] = per_chip[REWARD].std(ddof=1).mean(skipna=True)
        row["ret_p10"], row["ret_p50"], row["ret_p90"] = chip_mean.quantile([0.10, 0.50, 0.90])
        row["ret_ci_lo"], row["ret_ci_hi"] = bootstrap_mean_ci(chip_mean.to_numpy(), n_resamples, rng)
        if success_threshold is not None:
            row[f"ret_frac_chips_above_{success_threshold:g}"] = float((chip_mean >= success_threshold).mean())
        if baseline is not None and row["noise_model"] != "none":
            row["ret_delta_mean"] = mean - baseline
            row["ret_delta_pct"] = (mean - baseline) / abs(baseline) * 100.0 if baseline != 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(cols).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True, help="uno o más CSV de noise_eval.py (se concatenan)")
    ap.add_argument("--out", default=None, help="además guarda el resumen en este CSV")
    ap.add_argument("--resamples", type=int, default=10000, help="remuestreos del bootstrap")
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--success-threshold", type=float, default=None,
                    help="agrega la fracción de chips cuya media de retorno queda por encima")
    ap.add_argument("--baseline", type=float, default=None, help="retorno sin ruido (def.: filas noise_model=none)")
    a = ap.parse_args(argv)

    summary = summarize(load_rows(a.csv), a.resamples, a.bootstrap_seed, a.success_threshold, a.baseline)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(summary.to_string(index=False))
    if a.out:
        summary.to_csv(a.out, index=False)
        print(f"\nGuardado en {a.out}")


if __name__ == "__main__":
    main()
