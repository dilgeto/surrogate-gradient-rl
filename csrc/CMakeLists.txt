# Módulo Python `_rlt_envs`: entornos vectorizados de RLTools (Pendulum, Acrobot, MountainCar,
# BipedalWalker). Se compila con scripts/build_envs.sh, que fija las rutas de las cabeceras.
cmake_minimum_required(VERSION 3.10)
project(rlt_envs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

# Cabeceras de RLTools y del BipedalWalker (este último vive en el repo snn-simulator).
set(RL_TOOLS_INCLUDE_DIR "" CACHE PATH "Directorio include/ de RLTools")
set(BIPEDAL_INCLUDE_DIR "" CACHE PATH "Directorio include/ de snn-simulator (rl/environments/bipedal_walker)")

pybind11_add_module(_rlt_envs rlt_envs.cpp)
target_include_directories(_rlt_envs PRIVATE ${RL_TOOLS_INCLUDE_DIR} ${BIPEDAL_INCLUDE_DIR})
if(BIPEDAL_INCLUDE_DIR)
  target_compile_definitions(_rlt_envs PRIVATE SGRL_WITH_BIPEDAL=1)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(_rlt_envs PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(_rlt_envs PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../sgrl)
