"""Envía `runs/<nombre>/log.csv` a un servidor de Aim (casi en tiempo real con --follow).

Corre aparte del entrenamiento y sólo lee los archivos de la corrida, así que `aim` puede vivir en otro
venv (p. ej. Python 3.12: `aim` aún no instala en 3.14, `aimrocks` no tiene distribución). No importa
sgrl ni torch; sólo la biblioteca estándar y `aim`.

  python scripts/aim_sync.py runs/Pendulum-lif-s0 --repo aim://192.168.1.50:53800 --follow
  python scripts/aim_sync.py runs/a runs/b --repo aim://host:53800 --experiment pendulum --follow 5
  python scripts/aim_sync.py runs/a --dry-run          # sin `aim` ni servidor: muestra qué enviaría

Cada corrida se sube como un run de Aim con el nombre del directorio; el eje x (`step`) son los pasos de
entorno. Las métricas por capa (`rates`) llevan el contexto {layer: j}. `config.json` va como hparams y
`final_eval.json` como `final_eval`. El estado (hash del run de Aim e última iteración enviada) se guarda
en `runs/<nombre>/.aim_sync.json`: volver a lanzar el script retoma el mismo run sin duplicar puntos.
Con --follow SEG se queda sondeando cada SEG s (por defecto 10) y termina cuando existe final_eval.json.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

SKIP = {"iteration", "steps", "rates"}


def read_rows(path: Path):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def metrics_of(row: dict):
    """(nombre, valor, contexto) de cada punto de una fila del log; omite vacíos y nan."""
    for k, v in row.items():
        if k in SKIP or v in (None, ""):
            continue
        x = float(v)
        if not math.isnan(x):
            yield k, x, {}
    for j, r in enumerate((row.get("rates") or "").split()):
        yield "rate", float(r), {"layer": j}


class Sync:
    def __init__(self, path: Path, args):
        self.dir = path if path.is_dir() else path.parent
        self.state_file = self.dir / ".aim_sync.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        self.args = args
        self.run = None
        self.done = False

    def _open(self):
        import aim  # import tardío: --dry-run funciona sin aim instalado

        h = self.state.get("run_hash")
        kw = dict(repo=self.args.repo, experiment=self.args.experiment, system_tracking_interval=None,
                  log_system_params=False)
        self.run = aim.Run(run_hash=h, **kw) if h else aim.Run(**kw)
        self.state["run_hash"] = self.run.hash
        if not h:
            self.run.name = self.dir.name
            cfg = self.dir / "config.json"
            if cfg.exists():
                self.run["hparams"] = json.loads(cfg.read_text())

    def step(self) -> int:
        """Envía las filas nuevas; devuelve cuántas. Marca `done` si la corrida ya terminó."""
        rows = read_rows(self.dir / "log.csv")
        last = self.state.get("last_iteration", 0)
        new = [r for r in rows if int(r["iteration"]) > last]
        final = self.dir / "final_eval.json"
        if new and not self.args.dry_run and self.run is None:
            self._open()
        for r in new:
            pts = list(metrics_of(r))
            if self.args.dry_run:
                if int(r["iteration"]) <= self.args.dry_run_rows:
                    print(f"{self.dir.name} step={r['steps']}: " + ", ".join(
                        f"{n}{c or ''}={v:.4g}" for n, v, c in pts))
            else:
                for name, v, ctx in pts:
                    self.run.track(v, name=name, step=int(r["steps"]), context=ctx)
            self.state["last_iteration"] = int(r["iteration"])
        if final.exists() and not new:
            if not self.args.dry_run:
                if self.run is None:
                    self._open()
                self.run["final_eval"] = json.loads(final.read_text())
                self.run.close()
            self.done = True
        if not self.args.dry_run:
            self.state_file.write_text(json.dumps(self.state))
        return len(new)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="directorios de corrida (runs/<nombre>)")
    p.add_argument("--repo", help="repositorio de Aim: ruta local o aim://host:53800")
    p.add_argument("--experiment", default=None, help="experimento de Aim (por defecto, el de Aim)")
    p.add_argument("--follow", nargs="?", const=10.0, type=float, default=0, metavar="SEG",
                   help="seguir sondeando cada SEG s (10 si se omite) hasta que termine la corrida")
    p.add_argument("--dry-run", action="store_true", help="no conecta con Aim; imprime lo que enviaría")
    p.add_argument("--dry-run-rows", type=int, default=3, help="filas a mostrar en --dry-run")
    args = p.parse_args(argv)
    if not args.repo and not args.dry_run:
        p.error("falta --repo (p. ej. aim://192.168.1.50:53800)")
    syncs = [Sync(Path(r), args) for r in args.runs]
    try:
        while True:
            for s in syncs:
                if not s.done:
                    n = s.step()
                    print(f"{s.dir.name}: +{n} iteraciones (hasta la {s.state.get('last_iteration', 0)})"
                          + (" — terminada" if s.done else ""), flush=True)
            if not args.follow or all(s.done for s in syncs) or args.dry_run:
                break
            time.sleep(args.follow)
    except KeyboardInterrupt:
        pass
    finally:
        for s in syncs:
            if s.run is not None and not s.done:
                s.run.close()


if __name__ == "__main__":
    main()
