// Entornos vectorizados de RLTools para Python.
//
//   env = _rlt_envs.make("Pendulum", num_envs=16, seed=0, num_threads=1)
//   obs = env.reset()                                   # (N, obs_dim) float64
//   obs, rew, terminated, truncated, final_obs = env.step(actions)   # actions: (N, act_dim)
//
// Cada entorno tiene su propio RNG y se reinicia solo al terminar (autoreset). `obs` ya es la
// observación del episodio nuevo; `final_obs` es la última observación del episodio que acaba de
// cerrarse (igual que `obs` en los entornos que siguen corriendo), necesaria para hacer bootstrap
// del valor cuando el episodio se corta por límite de tiempo (truncated) y no por terminación.
//
// Sólo usa la interfaz genérica de RLTools (sample_initial_parameters / sample_initial_state /
// step / reward / observe / terminated), la misma que usa su bucle PPO, así que cualquier entorno
// que la cumpla se registra con una línea en PYBIND11_MODULE.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/rl/environments/acrobot/operations_cpu.h>
#include <rl_tools/rl/environments/mountain_car/operations_cpu.h>
#include <rl_tools/rl/environments/car/operations_generic.h>
#include <rl_tools/rl/environments/multi_agent/bottleneck/operations_cpu.h>
#ifdef SGRL_WITH_BIPEDAL
#include "rl/environments/bipedal_walker/operations_generic.h"
#endif

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace rlt = rl_tools;

using T = double;
using TI = size_t;
using DEVICE = rlt::devices::DefaultCPU;
using RNG = rlt::devices::random::CPU::ENGINE<>;

class VecEnvBase {
public:
    virtual ~VecEnvBase() = default;
    virtual py::array_t<T> reset(py::object seed) = 0;
    virtual py::tuple step(py::array_t<T, py::array::c_style | py::array::forcecast> actions) = 0;
    virtual TI obs_dim() const = 0;
    virtual TI act_dim() const = 0;
    virtual TI episode_step_limit() const = 0;
    virtual TI num_envs() const = 0;
    virtual TI n_agents() const = 0;
};

// LIMIT (> 0) sustituye al EPISODE_STEP_LIMIT del entorno de RLTools; REWARD_SHIFT se suma a la recompensa
// de cada paso. Sólo los usa MountainCarDiscrete (200 pasos y -1 por paso, como en wann-cpp).
template <typename ENV, TI LIMIT = 0, int REWARD_SHIFT = 0>
class VecEnv : public VecEnvBase {
public:
    static constexpr TI OBS_DIM = ENV::Observation::DIM;
    static constexpr TI ACT_DIM = ENV::ACTION_DIM;
    // Multiagente (Bottleneck): obs y acciones conjuntas = N_AGENTS bloques contiguos iguales.
    static constexpr TI N_AGENTS = [] { if constexpr (requires { ENV::N_AGENTS; }) return (TI)ENV::N_AGENTS; else return (TI)1; }();
    // Car no define EPISODE_STEP_LIMIT; 1000 es el que usan sus ejemplos de RLTools.
    static constexpr TI DEFAULT_LIMIT = [] { if constexpr (requires { ENV::EPISODE_STEP_LIMIT; }) return (TI)ENV::EPISODE_STEP_LIMIT; else return (TI)1000; }();
    using ObsMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OBS_DIM, false>>;
    using ActMatrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ACT_DIM, false>>;

    VecEnv(TI n, unsigned seed, int threads, TI max_steps) : n_(n), threads_(threads), inst_(n) {
        max_steps_ = max_steps > 0 ? max_steps : (LIMIT > 0 ? LIMIT : DEFAULT_LIMIT);
        for (TI i = 0; i < n_; i++) {
            rlt::init(inst_[i].device, inst_[i].rng, seed + static_cast<unsigned>(i));
            rlt::init(inst_[i].device, inst_[i].env);
        }
    }

    py::array_t<T> reset(py::object seed) override {
        py::array_t<T> out({(py::ssize_t)n_, (py::ssize_t)OBS_DIM});
        T* o = out.mutable_data();
        for (TI i = 0; i < n_; i++) {
            if (!seed.is_none())
                rlt::init(inst_[i].device, inst_[i].rng, seed.cast<unsigned>() + static_cast<unsigned>(i));
            reset_one(inst_[i]);
            observe(inst_[i], o + i * OBS_DIM);
        }
        return out;
    }

    py::tuple step(py::array_t<T, py::array::c_style | py::array::forcecast> actions) override {
        if (actions.ndim() != 2 || (TI)actions.shape(0) != n_ || (TI)actions.shape(1) != ACT_DIM)
            throw std::invalid_argument("actions debe tener forma (" + std::to_string(n_) + ", " + std::to_string(ACT_DIM) + ")");
        py::array_t<T> obs({(py::ssize_t)n_, (py::ssize_t)OBS_DIM});
        py::array_t<T> final_obs({(py::ssize_t)n_, (py::ssize_t)OBS_DIM});
        py::array_t<T> reward(n_);
        py::array_t<bool> terminated(n_), truncated(n_);
        const T* a = actions.data();
        T* o = obs.mutable_data();
        T* fo = final_obs.mutable_data();
        T* r = reward.mutable_data();
        bool* te = terminated.mutable_data();
        bool* tr = truncated.mutable_data();
        {
            py::gil_scoped_release release;
#pragma omp parallel for num_threads(threads_) schedule(static) if (threads_ > 1)
            for (long i = 0; i < (long)n_; i++) {
                Instance& s = inst_[i];
                ActMatrix act;
                for (TI k = 0; k < ACT_DIM; k++) rlt::set(act, 0, k, a[i * ACT_DIM + k]);
                typename ENV::State next;
                rlt::step(s.device, s.env, s.params, s.state, act, next, s.rng);
                r[i] = rlt::reward(s.device, s.env, s.params, s.state, act, next, s.rng) + REWARD_SHIFT;
                s.state = next;
                s.steps++;
                te[i] = rlt::terminated(s.device, s.env, s.params, s.state, s.rng);
                tr[i] = !te[i] && s.steps >= max_steps_;
                observe(s, fo + i * OBS_DIM);
                if (te[i] || tr[i]) {
                    reset_one(s);
                    observe(s, o + i * OBS_DIM);
                } else {
                    std::copy(fo + i * OBS_DIM, fo + (i + 1) * OBS_DIM, o + i * OBS_DIM);
                }
            }
        }
        return py::make_tuple(obs, reward, terminated, truncated, final_obs);
    }

    TI obs_dim() const override { return OBS_DIM; }
    TI act_dim() const override { return ACT_DIM; }
    TI episode_step_limit() const override { return max_steps_; }
    TI num_envs() const override { return n_; }
    TI n_agents() const override { return N_AGENTS; }

private:
    struct Instance {
        DEVICE device;
        RNG rng;
        ENV env;
        typename ENV::Parameters params;
        typename ENV::State state;
        TI steps = 0;
    };

    void reset_one(Instance& s) {
        rlt::sample_initial_parameters(s.device, s.env, s.params, s.rng);
        rlt::sample_initial_state(s.device, s.env, s.params, s.state, s.rng);
        s.steps = 0;
    }

    void observe(Instance& s, T* out) {
        ObsMatrix obs;
        rlt::observe(s.device, s.env, s.params, s.state, typename ENV::Observation{}, obs, s.rng);
        for (TI i = 0; i < OBS_DIM; i++) out[i] = rlt::get(obs, 0, i);
    }

    TI n_;
    int threads_;
    TI max_steps_;
    std::vector<Instance> inst_;
};

using PendulumSpec = rlt::rl::environments::pendulum::Specification<T, TI>;
using AcrobotSpec = rlt::rl::environments::acrobot::Specification<T, TI>;
using MountainCarSpec = rlt::rl::environments::mountain_car::Specification<T, TI>;

// Las tres tareas (Acrobot, MountainCarDiscrete, CarTrack) deben ser EXACTAMENTE las de wann-cpp, o los
// resultados de los dos repos no son comparables. Copiado de wann-cpp/src/{SnnAcrobotTask,SnnDiscMCTask,
// SnnCarTask}.cpp; tests/test_wann_envs.py fija trayectorias de referencia generadas con esos archivos.
//  Acrobot: torque ±1 N·m, escala de Gymnasium (RLTools usa ±5 por defecto). Mismo límite de 500 pasos.
struct GymAcrobotParams : rlt::rl::environments::acrobot::DefaultParameters<T> {
    static constexpr T MIN_TORQUE = -1.0;
    static constexpr T MAX_TORQUE = +1.0;
};
using GymAcrobotSpec = rlt::rl::environments::acrobot::Specification<T, TI, GymAcrobotParams>;
//  MountainCarDiscrete: MountainCar-v0 de Gymnasium. Fuerza {-1, 0, 1} * power, meta en 0.5, sin coste de
//  acción ni bonus de meta; la recompensa es -1 por paso (REWARD_SHIFT) y el episodio dura 200 pasos.
struct DiscMCParams {
    static constexpr T min_action = -1.0;
    static constexpr T max_action = 1.0;
    static constexpr T min_position = -1.2;
    static constexpr T max_position = 0.6;
    static constexpr T max_speed = 0.07;
    static constexpr T power = 0.001;
    static constexpr T goal_position = 0.5;
    static constexpr T goal_velocity = 0.0;
    static constexpr T action_cost = 0.0;
    static constexpr T goal_reward = 0.0;
    static constexpr T initial_position_min = -0.6;
    static constexpr T initial_position_max = -0.4;
};
using DiscMCSpec = rlt::rl::environments::mountain_car::Specification<T, TI, DiscMCParams>;
static constexpr TI MC_DISCRETE_STEPS = 200;
//  CarTrack: pista de 100x100 celdas de 50 mm (arena de 5 x 5 m), 1000 pasos.
using CarTrackSpec = rlt::rl::environments::car::SpecificationTrack<T, TI, 100, 100, 50>;

// Bottleneck con los parámetros del zoo de RLTools (src/rl/zoo/bottleneck-v0/environment.h).
struct BottleneckParameters : rlt::rl::environments::multi_agent::bottleneck::DefaultParameters<T, TI> {
    static constexpr TI N_AGENTS = 5;
    static constexpr TI LIDAR_RESOLUTION = 5;
    static constexpr T LIDAR_FOV = rlt::math::PI<T> * 90 / 180;
    static constexpr T BOTTLENECK_WIDTH = 5;
    static constexpr TI EPISODE_STEP_LIMIT = 200;
    static constexpr bool SPAWN_BOTH_SIDES = false;
    static constexpr T AGENT_MAX_SPEED = 4;
    static constexpr T AGENT_MAX_ANGULAR_VELOCITY = 4;
    static constexpr T AGENT_MAX_ACCELERATION = 20;
    static constexpr T AGENT_MAX_ANGULAR_ACCELERATION = 20;
    static constexpr T DT = 0.05;
};
using BottleneckSpec = rlt::rl::environments::multi_agent::bottleneck::Specification<rlt::numeric_types::Policy<T>, TI, BottleneckParameters>;

std::unique_ptr<VecEnvBase> make(const std::string& name, TI n, unsigned seed, int threads, TI max_steps) {
    namespace E = rlt::rl::environments;
    if (name == "Pendulum") return std::make_unique<VecEnv<E::Pendulum<PendulumSpec>>>(n, seed, threads, max_steps);
    if (name == "Acrobot") return std::make_unique<VecEnv<E::Acrobot<GymAcrobotSpec>>>(n, seed, threads, max_steps);
    if (name == "AcrobotSwingup") return std::make_unique<VecEnv<E::AcrobotSwingup<AcrobotSpec>>>(n, seed, threads, max_steps);
    if (name == "MountainCarContinuous")
        return std::make_unique<VecEnv<E::MountainCarContinuous<MountainCarSpec>>>(n, seed, threads, max_steps);
    if (name == "MountainCarDiscrete")
        return std::make_unique<VecEnv<E::MountainCarContinuous<DiscMCSpec>, MC_DISCRETE_STEPS, -1>>(n, seed, threads, max_steps);
    if (name == "CarTrack") return std::make_unique<VecEnv<E::CarTrack<CarTrackSpec>>>(n, seed, threads, max_steps);
    if (name == "Bottleneck")
        return std::make_unique<VecEnv<E::multi_agent::Bottleneck<BottleneckSpec>>>(n, seed, threads, max_steps);
#ifdef SGRL_WITH_BIPEDAL
    using BWSpec = E::bipedal_walker::Specification<T, TI>;
    if (name == "BipedalWalker") return std::make_unique<VecEnv<E::BipedalWalker<BWSpec>>>(n, seed, threads, max_steps);
    if (name == "BipedalWalkerHardcore")
        return std::make_unique<VecEnv<E::BipedalWalkerHardcore<BWSpec>>>(n, seed, threads, max_steps);
#endif
    throw std::invalid_argument("entorno desconocido: " + name);
}

PYBIND11_MODULE(_rlt_envs, m) {
    m.doc() = "Entornos vectorizados de RLTools (autoreset, OpenMP opcional)";
    py::class_<VecEnvBase>(m, "VecEnv")
        .def("reset", &VecEnvBase::reset, py::arg("seed") = py::none())
        .def("step", &VecEnvBase::step, py::arg("actions"))
        .def_property_readonly("obs_dim", &VecEnvBase::obs_dim)
        .def_property_readonly("act_dim", &VecEnvBase::act_dim)
        .def_property_readonly("episode_step_limit", &VecEnvBase::episode_step_limit)
        .def_property_readonly("num_envs", &VecEnvBase::num_envs)
        .def_property_readonly("n_agents", &VecEnvBase::n_agents);
    m.def("make", &make, py::arg("name"), py::arg("num_envs") = 1, py::arg("seed") = 0,
          py::arg("num_threads") = 1, py::arg("max_steps") = 0);
}
