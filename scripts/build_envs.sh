#!/bin/bash
# Compila sgrl/_rlt_envs*.so. Rutas por defecto: repo hermano ../snn-simulator.
#   RLTOOLS_INCLUDE   include/ de RLTools          (def. ../snn-simulator/external/rl-tools/include)
#   BIPEDAL_INCLUDE   include/ de snn-simulator, para BipedalWalker (def. ../snn-simulator/include;
#                     poner a "" para compilar sin BipedalWalker)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIM="${SNN_SIMULATOR_DIR:-$ROOT/../snn-simulator}"
RLTOOLS_INCLUDE="${RLTOOLS_INCLUDE:-$SIM/external/rl-tools/include}"
BIPEDAL_INCLUDE="${BIPEDAL_INCLUDE-$SIM/include}"
PY="${PYTHON:-python}"

cmake -S "$ROOT/csrc" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$($PY -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$($PY -c 'import sys;print(sys.executable)')" \
  -DRL_TOOLS_INCLUDE_DIR="$RLTOOLS_INCLUDE" -DBIPEDAL_INCLUDE_DIR="$BIPEDAL_INCLUDE"
cmake --build "$ROOT/build" -j"$(nproc)"
