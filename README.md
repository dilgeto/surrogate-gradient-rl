# surrogate-gradient-rl

Entrenamiento por **gradientes surrogados** de políticas espiguiformes **LIF**
(librería [snnTorch](https://github.com/jeshraghian/snntorch)) en tareas de RL cuyos entornos vienen de
[**RLTools**](https://github.com/rl-tools/rl-tools) (`../snn-simulator/external/rl-tools`), con
**ruido de memristores** inyectable en las capas sinápticas del actor (entrenamiento y evaluación).

```
 obs ─ normalizar ─ NoisyLinear ─┬─ LIF ─ NoisyLinear ─ LIF ─ NoisyLinear ─ LI ─ media ─► N(μ, σ)
     (RLTools, C++)              │   T pasos de simulación, BPTT con gradiente surrogado   (acción)
                                 └─ corriente constante en los T pasos (codificación directa)
```

* **Entornos**: RLTools compilado como módulo Python (`csrc/rlt_envs.cpp`, pybind11): N entornos
  en un solo objeto, autoreset, OpenMP opcional. Disponibles: `Pendulum`, `Acrobot`,
  `AcrobotSwingup`, `MountainCarContinuous`, `BipedalWalker`, `BipedalWalkerHardcore`
  (los dos últimos son el port sin Box2D de `snn-simulator/include/rl/environments`).
  `Acrobot`, `MountainCarDiscrete` y `CarTrack` son **exactamente** los de `../wann-cpp` para que los
  resultados de los dos repos sean comparables: Acrobot con torque ±1 (RLTools usa ±5), MountainCar como
  MountainCar-v0 de Gymnasium (fuerza 0.001, meta 0.5, -1 por paso, 200 pasos) y la pista del coche con
  celdas de 50 mm (5 x 5 m). `tests/test_wann_envs.py` fija sus trayectorias de referencia.
* **Reward shaping** (opcional, `--reward-shaping ESCALA`, 0 por defecto): el mismo shaping basado en potencial
  que wann-cpp usa para entrenar, `escala·(φ(s') - φ(s))` sin descuento, con φ = altura de la punta en Acrobot y
  sin(3·posición) en MountainCarDiscrete (wann-cpp: escala 10 en ambas; CarTrack no tiene y el flag falla).
  Sólo cambia la recompensa que ve el aprendizaje: los retornos de evaluación, los logs y la puntuación de
  Optuna son siempre los originales del entorno. Para igualar la señal de entrenamiento de wann-cpp:
  `python -m sgrl.train --env Acrobot --reward-shaping 10 ...` o `scripts/optuna_acrobot.sh --reward-shaping 10`.
  (`scripts/sb3_baseline.py` no lo aplica.)
* **Algoritmo**: PPO propio (`sgrl/ppo.py`) en PyTorch. RLTools aporta los entornos; su PPO
  usa sus propias redes en C++ y no admite una SNN. El crítico es un MLP analógico y digital
  (sin ruido: sólo el actor se despliega).
* **Actor**: `sgrl/snn.py`, neuronas `snn.Leaky` de snnTorch (LIF, reset a cero) con estado explícito
  que arranca en reposo en cada paso del entorno; lectura con un integrador leaky sin espigas.
  `--neuron ann` da una línea base MLP con el mismo PPO y las mismas capas `NoisyLinear`.
* **Ruido**: `sgrl/noise.py`, puerto a PyTorch de los modelos de `../wann-cpp/memristor_noise.hpp`
  (a su vez puerto de aihwkit): `simple`, `pcm`, `reram` (Wan 2022) y `reramcmo`.

## Uso

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # o la build CUDA
pip install snntorch numpy pybind11 matplotlib pytest && pip install -e . --no-deps
scripts/build_envs.sh        # compila sgrl/_rlt_envs*.so (usa ../snn-simulator por defecto)

# Pendulum (hiperparámetros tipo rl-zoo: gamma 0.9, 10 épocas)
python -m sgrl.train --env Pendulum --neuron lif --steps 300000 --gamma 0.9 --epochs 10 --minibatch 256 --actor-lr 1e-3
python -m sgrl.train --env Pendulum --neuron ann --steps 300000 --gamma 0.9 --epochs 10 --minibatch 256 --actor-lr 1e-3
```

Cada corrida escribe en `runs/<nombre>/`: `config.json`, `log.csv` (retorno, KL, tasas de
disparo por capa, norma del gradiente…), `best.pt`, `final.pt` y `final_eval.json`.
`python -m pytest tests` corre las pruebas.

Para ver cómo va el entrenamiento (también mientras corre):

```bash
python scripts/plot_training.py runs/Pendulum-lif-s0                  # resumen + runs/Pendulum-lif-s0/training.png
python scripts/plot_training.py runs/a runs/b --out cmp.png           # superpone corridas
python scripts/plot_training.py runs/Pendulum-lif-s0 --watch 30       # se actualiza cada 30 s
```

Sin `snn-simulator` (BipedalWalker): `BIPEDAL_INCLUDE= scripts/build_envs.sh`.
Rutas alternativas: `SNN_SIMULATOR_DIR`, `RLTOOLS_INCLUDE`, `BIPEDAL_INCLUDE`.

Los checkpoints anteriores al cambio de Norse a snnTorch (y a Izhikevich, ya no soportado) no se pueden cargar.

## Qué hay que saber al elegir hiperparámetros SNN

| | LIF (snnTorch `Leaky`) |
|---|---|
| paso de tiempo | 1 ms; `--beta 0.9` = exp(-dt/τ) con τ = 10 ms |
| pasos de simulación por paso de entorno (`--sim-steps`) | 16 |
| surrogado | `fast_sigmoid` (el derivado de SuperSpike), pendiente 25; `--surrogate atan\|triangular\|straight_through` |
| ganancia entrada / capas ocultas | 10 / 40 |
| tasa de disparo inicial por paso | ~0.17 |

* Las ganancias (`--in-gain`, `--hid-gain`) fijan la corriente que reciben las neuronas (se
  multiplican por 1-β para que el potencial de reposo sea la corriente); con valores malos la red
  nace silenciosa y no llega gradiente. Se eligieron con `scripts/probe_rates.py`, que imprime tasas
  de disparo y normas de gradiente sin entrenar. Si cambias tamaño de red, entorno, `--beta` o
  `--sim-steps`, vuelve a mirarlo.
* No se ofrece el surrogado `sigmoid` de snnTorch: su backward da NaN cuando `exp(-slope·x)` desborda.
* El PPO recorta actor y crítico por separado para que un gradiente grande del actor no aplaste el
  del crítico.
* La política es *sin estado entre pasos del entorno*: el potencial de membrana se reinicia
  en cada decisión. Eso permite minilotes aleatorios en PPO; una SNN con memoria entre pasos
  exigiría minilotes de secuencias (no implementado).

## Búsqueda de hiperparámetros (Optuna, sin ruido)

```bash
pip install optuna      # o pip install -e .[optuna]
# fase 1: actor_lr, critic_lr, ent_coef, minibatch, epochs, beta (lo que va tras `--` es la config base fija)
python scripts/optuna_search.py --study pend-f1 --n-trials 50 --seeds-per-trial 1 -- \
    --env Pendulum --steps 300000 --gamma 0.9 --device cuda --threads 1
# fase 2: con lo mejor de la fase 1 en la base, busca alpha, sim_steps, ganancias, gae_lambda, init_std
python scripts/optuna_search.py --study pend-f2 --phase 2 --n-trials 40 -- --env Pendulum --steps 300000 ...
python scripts/optuna_report.py pend-f1 --plot                          # top, importancias, comando del mejor
python scripts/optuna_report.py pend-f1 --validate 3                 # reentrena el top 3 con 11 semillas nuevas (100-110)
```

El estudio vive en `optuna/<study>.db` (SQLite): lanza 2-3 procesos con el mismo `--study` para repartir
trials. El primer trial es la config base (referencia). La puntuación es la media de las últimas 3
evaluaciones deterministas; el `MedianPruner` actúa sobre la primera semilla y se podan los trials con red
muerta/saturada o NaN. Con `--seeds-per-trial 1` Optuna puede premiar semillas con suerte: valida el top.

Algunas tareas traen valores propios de búsqueda (`TASK_DEFAULTS` en el script; los flags los sobrescriben):
Acrobot puntúa con la curva entera (`--score auc`, porque el retorno se satura al resolver), MountainCarDiscrete
usa 3 semillas por trial y no poda (resultado casi binario), y CarTrack espera 5 evaluaciones antes de podar
(las curvas despegan tarde). `minibatch` llega a 4096 y `ent_coef` a 5e-2 (el preset de MountainCar usa 0.02);
para lotes grandes pasa `--num-envs`/`--rollout-steps` en la config base, p. ej. `--num-envs 64 --rollout-steps 128`.

### Búsqueda como en wann-cpp (`scripts/screening.py`)

Mismo procedimiento que `screening_reduce.py` + `screening_full.py` de wann-cpp, con los 13 hiperparámetros (PPO + SNN)
a la vez (12 en Acrobot y MountainCar: `sim_steps` fijo en 20, como la ventana de wann-cpp): **fase 1** rondas de N trials a baja fidelidad (25 % de los pasos) con TPE (`constant_liar`, multivariante) y
poda por mediana; tras cada ronda se divide cada parámetro en 5 tramos y se eliminan los significativamente peores
(Mann-Whitney U unilateral, α = 0.05, resultados acumulados) hasta que no cae ninguno; **fase 2** TPE a presupuesto
completo en el espacio reducido; **fase 3** los mejores K con varias semillas nuevas (todas en paralelo).

```bash
pip install -e .[optuna]       # incluye scipy
python scripts/screening.py --task acrobot --mode all --jobs 8          # 4 rondas x 30, fase 2 de 20, top 3 x 11 semillas
python scripts/screening.py --task acrobot --mode analyse               # evolución del espacio y resultados
```
Tareas: `acrobot`, `disc_mc` (MountainCarDiscrete), `car` (CarTrack). Resultados en `screening/<tarea>/`.
`--n2 40` fija los trials de la fase 2 y `--hidden 3 2` el tamaño del actor (fijo, no se busca). `scripts/screening_all.sh` lanza
las tres tareas una tras otra con redes totalmente conectadas de tantas conexiones como la red de wann-cpp de cada tarea.

## Ruido de memristores

Cada peso de las capas del actor (`NoisyLinear`, incluidas las del baseline `--neuron ann`) se mapea a un
par diferencial de conductancias `w = (G+ - G-)·w_max/g_max` (`w_max` = max|W| de la capa, o `--w-max`).
Hay dos mecanismos de distinta naturaleza temporal, igual que en `wann-cpp`:

* **Programación** (escritura): se muestrea una vez por *chip* y queda fija; en PCM también se sortea ahí
  el coeficiente de deriva de cada dispositivo. `--n-levels N` cuantiza la conductancia a N niveles.
* **Lectura**: ruido nuevo en cada lectura más la deriva a `t` segundos de la programación. Una lectura
  fija los pesos efectivos hasta la siguiente (un chip se lee y luego corre episodios).

Los sesgos son digitales. Modelos y parámetros: `sgrl/noise.py` (misma notación y defectos que
`memristor_noise.hpp`: `--noise-scale` escala todo; `--prog-noise-scale`, `--read-noise-scale` y
`--drift-scale` lo desglosan para ablación). `reram` sólo admite t ∈ {1 s, 1 día, 2 días};
`reramcmo` usa `g_max=25, g_min=1` µS que **no** son de Falcone et al. (aihwkit no trae valores).

**Evaluar** un checkpoint (protocolo de `wann_car_noise`: por semilla y chip, programar una vez y leer
`--reads-per-chip` veces para cada `--t-inference`; una fila de CSV por lectura):

```bash
python scripts/noise_eval.py runs/Pendulum-lif-s0/final.pt --noise-model pcm --noise-scales 0,1,2,4 \
    --t-inference 0,3600,86400 --n-chips 20 --episodes 10 --out noise_pcm.csv
python scripts/summarize_noise.py --csv noise_pcm.csv --out summary.csv   # requiere pandas
```

`summarize_noise.py` (puerto de `summarize_car_noise.py`) separa la varianza en **entre chips** (mala suerte
de programación) y **dentro de chip** (ruido de lectura; necesita `--reads-per-chip` > 1), da percentiles
10/50/90 de las medias por chip, un IC 95 % por bootstrap remuestreando chips (no filas), la caída frente
al retorno sin ruido (filas `--noise-model none` o `--baseline`; negativo = peor también con retornos < 0)
y, con `--success-threshold R`, la fracción de chips cuya media supera R.

**Barridos y figuras** (`pip install pandas`; los CSV y PNG van a `<carpeta del checkpoint>/noise/`):

```bash
python scripts/run_noise_sweeps.py runs/Pendulum-lif-s0/final.pt --n-chips 20 --jobs 3 --grid --ablation
python scripts/run_noise_sweeps.py runs/Pendulum-lif-s0/final.pt --dry-run     # sólo imprime los comandos
```

Genera retorno vs intensidad (escala de ruido) y vs error efectivo de pesos, retorno vs tiempo desde la
programación (60 s a 1 año, escala symlog), mapas de calor escala×tiempo (`--grid`) y la ablación por
mecanismo (`--ablation`: sólo programación, sólo lectura o sólo deriva, en `pcm` y `reramcmo`, que los
separan; a t=0 la deriva de PCM no actúa, por eso la ablación usa `--ablation-t 86400`). Cada figura se
puede rehacer a mano con `scripts/plot_noise.py --kind intensity|time|heatmap|rmse|ablation`; las
bandas son el IC bootstrap de la media y los percentiles 10-90 de las medias por chip.

`weight_rmse` es el error RMS de los pesos realizados frente a los nominales, normalizado por `w_max` de
cada capa, y sirve de eje común entre modelos. `--noise-model none` es el chequeo de regresión
(coincide con `scripts/eval.py`).

**Entrenar con ruido** (hardware-aware, Joshi et al. 2020): `--train-noise-model pcm [--train-noise-scale 1
--train-noise-t 0 --train-noise-every 1]`. El ruido se suma al peso en el forward y el gradiente pasa
por el peso nominal. Se sortea un chip nuevo cada iteración de PPO (no en cada forward): así el rollout y
las épocas de actualización ven la misma política y la razón de PPO vale 1 al empezar. La evaluación
durante el entrenamiento (`eval_return`, `best.pt`) usa pesos nominales; el retorno con ruido se mide
después con `noise_eval.py`.

Validación: `tests/test_all.py::test_noise_models_match_cpp_port` fija media y desviación del peso
ruidoso medidas con el C++ de `wann-cpp` (40 000 chips, los cuatro modelos, con y sin
cuantización); el puerto coincide dentro del error estadístico.

## Validación del PPO contra stable-baselines3

El PPO es propio (`sgrl/ppo.py`), así que para poder usarlo como línea base hay un preset que
reproduce los valores por defecto de SB3 y un adaptador para correr SB3 sobre los mismos entornos:

```bash
pip install stable-baselines3
python -m sgrl.train --env Pendulum --neuron ann --sb3 --steps 100000 --gamma 0.9   # PPO propio, config SB3
python scripts/sb3_baseline.py --env Pendulum --steps 100000 --gamma 0.9           # SB3 real
```

`--sb3` (o `PPOConfig.sb3()` + `ActorCritic(..., ortho_init=True, init_std=1.0)`) fija minilote 64,
10 épocas, LR único 3e-4, sin `target_kl` ni annealing, recorte de gradiente global e inicialización
ortogonal; los flags explícitos siguen ganando. Ambos scripts escriben `log.csv` y `final_eval.json`
con el mismo protocolo de evaluación. `sgrl/sb3_compat.py` adapta `VecEnv` a SB3 (sólo tareas de un
agente). Comprobación hecha: Pendulum, ANN, 100 k pasos, semilla 0: retorno final -220.0 (propio)
vs -222.3 (SB3), mismas curvas de evaluación; es una sola semilla, no una prueba estadística.

Con un actor SNN la config SB3 hace ~20× más actualizaciones (cada una atraviesa los pasos de
simulación), por eso los presets por defecto del proyecto usan otra (minilote grande, pocas épocas).

## Rendimiento y paralelismo (medido en RTX 2050 4 GB + 12 hebras)

El coste del actor lo domina el despacho de operaciones pequeñas (16 pasos × capas), no los
FLOPs. En GPU el tiempo por llamada es casi constante hasta lotes de ~2 000 (≈5 ms forward,
≈12 ms forward+backward), así que **lo que escala es el tamaño del lote**, no el nº de procesos:

| `scripts/bench_actor.py`, LIF 64×64, lote 2048 | forward | forward+backward |
|---|---|---|
| CPU 4 hebras | 14 ms | 40 ms (51k muestras/s) |
| CPU 12 hebras | 14 ms | **413 ms** (sobresuscripción: no uses 12 hebras de torch) |
| GPU | 6 ms | 16 ms (132k muestras/s) |

Entrenamiento de extremo a extremo (Acrobot, LIF, `--device cuda --threads 1`):

| configuración | pasos/s | VRAM |
|---|---|---|
| por defecto: 16 entornos, minilote 512 | 1 470 | — |
| `--num-envs 256 --rollout-steps 64 --minibatch 4096` | **8 750** | 0.3 GB |
| 3 procesos así a la vez (semillas distintas) | 3 400 c/u (≈10 200 en total) | 0.8 GB |

* Con un solo proceso de lote grande la GPU ya está al 63 %; tres procesos la saturan (100 %) y
  sólo suman ~17 % más. La VRAM no limita: sobra por un factor >10. Pasar de ~4 000 por minilote
  no mejora el rendimiento (la curva se aplana a partir de ~2 000).
* Las hebras de CPU sólo ayudan en el **entorno** y sólo en BipedalWalker (256 entornos:
  38k pasos/s con 1 hebra → 116k con 8 → 159k con 12). El resto de entornos ya va a >1M pasos/s.
  Usa `--env-threads 8` en BipedalWalker y `--threads 1` para torch cuando entrenes en GPU.
* Cambiar `--num-envs`/`--minibatch` cambia el PPO (menos actualizaciones por paso de entorno):
  lo medido es el *rendimiento*, no que la eficiencia de muestras se conserve. Comprueba la curva.

Línea de partida para exprimir la GPU:

```bash
python -m sgrl.train --env Acrobot --neuron lif --device cuda --threads 1 \
    --num-envs 256 --rollout-steps 64 --minibatch 4096
python -m sgrl.train --env BipedalWalker --neuron lif --device cuda --threads 1 --env-threads 8 \
    --num-envs 256 --rollout-steps 64 --minibatch 4096
```

Para barridos (semillas, con/sin ruido) lanza varios procesos con `--seed`/`--name`; ayuda
poco con la GPU ya cargada, pero sí si repartes algunos en CPU (`--device cpu --threads 2`).
`torch.compile` funciona con Python 3.14 pero sólo da ~1.7× y recompila por tamaño de lote.

## Verificación hecha

Pendulum, 1 semilla, hiperparámetros del ejemplo de arriba, evaluación determinista (20 episodios,
retorno medio ± std; mayor es mejor, ≈-150 es "balancea y sostiene"):

| actor | pasos | retorno final |
|---|---|---|
| ANN (MLP 64×64) | 500k | -146 ± 95 |
| LIF | 500k | -207 ± 308 |
| Izhikevich (`tonic_spiking`) | 300k | -209 ± 181 |

Es una comprobación de que el pipeline aprende, no una comparación: una semilla, sin ajuste fino
y con desviaciones grandes (algunos arranques no logran el swing-up). BipedalWalker sólo se probó
como humo (corre y entrena unos pasos); no se ha entrenado hasta converger.
