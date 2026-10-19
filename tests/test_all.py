import numpy as np
import pytest
import torch

from sgrl.envs import ENV_NAMES, VecEnv
from sgrl.policy import ActorCritic
from sgrl.ppo import PPO, PPOConfig, evaluate
from sgrl.noise import Chip, NoiseConfig, NoisyLinear, noise_disabled
from sgrl.snn import SNNActor, SpikingPopulation


@pytest.mark.parametrize("name", ENV_NAMES)
def test_env_shapes_and_autoreset(name):
    env = VecEnv(name, 3, seed=1, max_steps=20)
    obs = env.reset()
    assert obs.shape == (3, env.obs_dim) and env.act_dim >= 1
    seen_done = False
    for _ in range(25):
        obs, rew, term, trunc, final_obs, finished = env.step(np.random.uniform(-1, 1, (3, env.act_dim)))
        assert obs.shape == final_obs.shape == (3, env.obs_dim)
        assert np.isfinite(obs).all() and np.isfinite(rew).all()
        seen_done |= bool((term | trunc).any())
    assert seen_done  # max_steps=20 fuerza al menos un fin de episodio


def test_pendulum_truncates_not_terminates():
    env = VecEnv("Pendulum", 2, seed=0)
    env.reset()
    for t in range(env.episode_step_limit):
        _, _, term, trunc, _, fin = env.step(np.zeros((2, 1)))
    assert not term.any() and trunc.all() and len(fin) == 2 and fin[0][1] == env.episode_step_limit


def test_env_is_deterministic_given_seed():
    def run():
        env = VecEnv("Pendulum", 2, seed=7)
        o = env.reset(7)
        for _ in range(10):
            o, *_ = env.step(np.full((2, 1), 0.3))
        return o

    assert np.allclose(run(), run())


def test_env_rejects_bad_action_shape():
    env = VecEnv("Pendulum", 2)
    env.reset()
    with pytest.raises(Exception):
        env.step(np.zeros((2, 3)))


def test_population_spikes_are_binary_and_batched():
    pop = SpikingPopulation(8)
    cur = torch.full((5, 8), 0.3)
    mem, total = pop.initial_state(cur), 0
    for _ in range(20):
        z, mem = pop(cur, mem)
        assert z.shape == (5, 8) and set(z.unique().tolist()) <= {0.0, 1.0}
        total += z.sum().item()
    assert total > 0  # con corriente fuerte tiene que haber disparo


@pytest.mark.parametrize("surrogate", ["fast_sigmoid", "atan", "triangular", "straight_through"])
def test_surrogate_gradient_reaches_first_layer(surrogate):
    torch.manual_seed(0)
    net = SNNActor(3, 1, (32, 32), "lif", steps=16, surrogate=surrogate)
    mean, rates = net(torch.randn(64, 3))
    assert (rates > 0).all(), "capa silenciosa a la inicialización"
    mean.pow(2).sum().backward()
    for p in net.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert net.fcs[0].weight.grad.abs().sum() > 0


def test_snn_state_does_not_leak_between_calls():
    net = SNNActor(3, 1, (16, 16), "lif", steps=16)
    x = torch.randn(4, 3)
    assert torch.allclose(net(x)[0], net(x)[0])  # el estado siempre arranca en reposo


def test_unknown_surrogate_and_neuron_messages():
    with pytest.raises(ValueError, match="fast_sigmoid"):
        SNNActor(3, 1, (8,), "lif", surrogate="nope")
    with pytest.raises(ValueError, match="lif"):
        SNNActor(3, 1, (8,), "izhikevich")


@pytest.mark.parametrize("neuron", ["ann", "lif"])
def test_ppo_smoke(neuron):
    torch.manual_seed(0)
    env = VecEnv("Pendulum", 4, seed=0)
    agent = ActorCritic(env.obs_dim, env.act_dim, neuron, (16, 16))
    ppo = PPO(agent, env, PPOConfig(num_envs=4, rollout_steps=64, minibatch_size=64, epochs=2))
    logs = []
    ppo.learn(512, logs.append)
    assert len(logs) == 2 and all(np.isfinite(l["pg"]) and np.isfinite(l["vf"]) for l in logs)
    ev = evaluate(agent, "Pendulum", ppo.norm, 2, 0, ppo.device)
    assert np.isfinite(ev["return_mean"])


@pytest.mark.parametrize("task", ["MountainCarDiscrete", "CarTrack", "Bottleneck", "Acrobot"])
@pytest.mark.parametrize("neuron", ["lif", "ann"])
def test_ppo_smoke_per_task(task, neuron):
    torch.manual_seed(0)
    env = VecEnv(task, 2, seed=0)
    agent = ActorCritic(env.obs_dim, env.act_dim, neuron, (16, 16), n_actions=env.n_actions,
                        n_agents=env.n_agents, steps=8)
    ppo = PPO(agent, env, PPOConfig(num_envs=2, rollout_steps=32, minibatch_size=32, epochs=1))
    logs = []
    ppo.learn(128, logs.append)
    assert all(np.isfinite(l["pg"]) and np.isfinite(l["vf"]) for l in logs)
    assert evaluate(agent, task, ppo.norm, 2, 0, ppo.device)["length_mean"] > 0


def test_discrete_policy_distribution_and_actions():
    ac = ActorCritic(2, 1, "lif", (8, 8), n_actions=3, steps=4)
    a, logp, v, _ = ac.act(torch.randn(5, 2))
    assert a.shape == (5, 1) and set(a.unique().tolist()) <= {0.0, 1.0, 2.0}
    lp2, ent, _, _ = ac.evaluate(torch.randn(5, 2), a)
    assert lp2.shape == ent.shape == (5,) and (ent <= np.log(3) + 1e-6).all()


def test_multiagent_actor_is_shared_across_agents():
    ac = ActorCritic(24, 10, "lif", (8, 8), n_agents=2, steps=4)  # 2 agentes: 12 obs / 5 acc cada uno
    one = torch.randn(1, 12)
    d, _ = ac.dist(torch.cat([one, one], 1))
    assert d.mean.shape == (1, 10) and torch.allclose(d.mean[:, :5], d.mean[:, 5:])  # mismos pesos
    assert ac.log_std.shape == (5,)


def test_bottleneck_task_dimensions():
    env = VecEnv("Bottleneck", 1)
    assert (env.n_agents, env.obs_dim, env.act_dim) == (5, 60, 10)
    assert VecEnv("MountainCarDiscrete", 1).n_actions == 3


def test_sb3_preset_matches_sb3_defaults():
    c = PPOConfig.sb3()
    assert (c.minibatch_size, c.epochs, c.actor_lr, c.critic_lr) == (64, 10, 3e-4, 3e-4)
    assert c.rollout_steps * c.num_envs == 2048 and c.target_kl is None and not c.anneal_lr and c.global_grad_clip


def test_ppo_smoke_sb3_preset_ortho_init():
    torch.manual_seed(0)
    env = VecEnv("Pendulum", 4, seed=0)
    agent = ActorCritic(env.obs_dim, env.act_dim, "ann", (16, 16), init_std=1.0, ortho_init=True)
    assert agent.actor.net[-1].weight.norm() < 0.5 and (agent.critic[0].bias == 0).all()
    ppo = PPO(agent, env, PPOConfig.sb3(num_envs=4, rollout_steps=32, minibatch_size=32, epochs=2))
    logs = []
    ppo.learn(256, logs.append)
    assert all(np.isfinite(l["pg"]) and np.isfinite(l["grad_norm"]) for l in logs)


def test_sb3_flag_sets_defaults_and_explicit_flags_win():
    from sgrl.train import SB3_DEFAULTS, build_parser
    ap = build_parser()
    ap.set_defaults(**SB3_DEFAULTS)
    a = ap.parse_args(["--sb3", "--minibatch", "128"])
    assert a.sb3 and a.epochs == 10 and a.minibatch == 128 and a.no_anneal_lr and a.init_std == 1.0


# ----------------------------------------------------------------------------------------------
# Ruido de memristores (sgrl/noise.py)
# ----------------------------------------------------------------------------------------------
def _read_stats(model, t, w=0.5, w_max=2.0, n=40000, seed=1):
    from sgrl.noise import Crossbar
    gen = torch.Generator().manual_seed(seed)
    xb = Crossbar(torch.full((n, 1), w), w_max, model)
    xb.program(gen)
    o = xb.read(t, gen)
    return o.mean().item(), o.std().item()


@pytest.mark.parametrize("name,t,mean,std", [
    # (media, desviación) del peso 0.5 con w_max=2 sobre 40 000 chips, medidas con el C++ de
    # ../wann-cpp/memristor_noise.hpp (los modelos de aihwkit, puerto literal)
    ("simple", 0.0, 0.4765, 0.0704),
    ("pcm", 0.0, 0.4914, 0.0730),
    ("pcm", 3600.0, 0.3830, 0.0634),
    ("pcm", 86400.0, 0.3298, 0.0620),
    ("reram", 1.0, 0.4865, 0.1592),
    ("reram", 86400.0, 0.4808, 0.2162),
    ("reramcmo", 1.0, 0.4194, 0.0343),
    ("reramcmo", 3600.0, 0.3613, 0.0614),
])
def test_noise_models_match_cpp_port(name, t, mean, std):
    from sgrl import noise
    model = {"simple": noise.SimpleModel, "pcm": noise.PCMModel, "reram": noise.ReRamWanModel,
             "reramcmo": noise.ReRamCMOModel}[name]()
    m, s = _read_stats(model, t)
    assert abs(m - mean) < 0.004 and abs(s - std) < 0.004 + 0.03 * std


def _actor(kind):
    torch.manual_seed(0)
    return ActorCritic(3, 1, kind, (16, 16), steps=4).actor


@pytest.mark.parametrize("kind", ["ann", "lif"])
def test_chip_none_is_exact_noop_and_detach_restores(kind):
    actor, x = _actor(kind), torch.randn(5, 3)
    ref = actor(x)[0]
    with Chip(actor, NoiseConfig("none"), seed=0) as chip:
        chip.program()
        assert chip.read() == 0.0 and torch.equal(actor(x)[0], ref)
    with Chip(actor, NoiseConfig("pcm", noise_scale=3.0), seed=0) as chip:
        chip.program()
        assert chip.read() > 0 and not torch.allclose(actor(x)[0], ref)
        with noise_disabled(actor):
            assert torch.equal(actor(x)[0], ref)
    assert torch.equal(actor(x)[0], ref)  # al salir del `with` el actor vuelve a ser nominal


def test_chip_programming_fixed_reads_fresh_and_seeded():
    def weights(seed, n_reads, read_scale):
        actor = _actor("ann")
        chip = Chip(actor, NoiseConfig("simple", read_noise_scale=read_scale), seed=seed)
        chip.program()
        out = []
        for _ in range(n_reads):
            chip.read()
            out.append(actor.net[0].effective_weight().clone())
        return out

    a, b = weights(0, 2, 0.0)
    assert torch.equal(a, b)                       # sin ruido de lectura la programación fija el chip
    assert not torch.equal(weights(0, 1, 0.0)[0], weights(1, 1, 0.0)[0])   # otro chip, otra programación
    assert torch.equal(weights(5, 1, 1.0)[0], weights(5, 1, 1.0)[0])       # misma semilla, mismo chip
    c, d = weights(0, 2, 1.0)
    assert not torch.equal(c, d)                   # cada lectura trae ruido nuevo


def test_noise_quantization_levels():
    from sgrl.noise import build_model
    cfg = NoiseConfig("simple", noise_scale=0.0, n_levels=5, w_max=1.0)
    lin = NoisyLinear(8, 8)
    with torch.no_grad():
        lin.weight.copy_(torch.linspace(-1, 1, 64).reshape(8, 8))
    lin.program(cfg, build_model(cfg), torch.Generator().manual_seed(0))
    lin.read(0.0, torch.Generator().manual_seed(0))
    levels = lin.effective_weight().detach().unique()
    # 5 niveles por rama del par diferencial -> pesos en múltiplos de 1/(5-1) = 0.25 (2·5-1 = 9 valores)
    assert len(levels) <= 9 and torch.allclose(levels, torch.round(levels * 4) / 4)


def test_reram_requires_measured_time():
    actor = _actor("ann")
    chip = Chip(actor, NoiseConfig("reram"), seed=0)
    chip.program()
    chip.read(86400.0)
    with pytest.raises(ValueError, match="86400"):
        chip.read(1234.0)


def test_noise_is_straight_through_for_gradients():
    actor, x = _actor("lif"), torch.randn(8, 3)
    with Chip(actor, NoiseConfig("pcm", noise_scale=2.0), seed=0) as chip:
        chip.resample()
        actor(x)[0].pow(2).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
    assert actor.fcs[0].weight.grad.abs().sum() > 0


def test_ppo_trains_with_noise_chip_each_iteration():
    torch.manual_seed(0)
    env = VecEnv("Pendulum", 4, seed=0)
    agent = ActorCritic(env.obs_dim, env.act_dim, "lif", (16, 16), steps=4)
    ppo = PPO(agent, env, PPOConfig(num_envs=4, rollout_steps=32, minibatch_size=32, epochs=2))
    chip = Chip(agent.actor, NoiseConfig("pcm"), seed=0)
    seen = []
    logs = []
    ppo.learn(256, logs.append, iter_hook=lambda it: seen.append(chip.resample()))
    assert len(seen) == 2 and all(r > 0 for r in seen)
    assert all(np.isfinite(l["pg"]) and np.isfinite(l["kl"]) for l in logs)
    # el primer minilote usa el mismo chip que el rollout: la razón de PPO vale 1 (KL ~ 0)
    assert logs[0]["kl"] < 1e-3


def test_train_and_noise_eval_cli(tmp_path):
    import csv
    import importlib.util
    from pathlib import Path
    from sgrl.train import main as train_main

    common = ["--env", "Pendulum", "--steps", "256", "--num-envs", "4", "--rollout-steps", "32", "--minibatch", "32",
              "--epochs", "1", "--hidden", "8", "8", "--sim-steps", "4", "--device", "cpu", "--eval-every", "0",
              "--eval-episodes", "2", "--out", str(tmp_path)]
    train_main([*common, "--name", "a", "--train-noise-model", "pcm"])
    spec = importlib.util.spec_from_file_location("noise_eval", Path(__file__).parent.parent / "scripts" / "noise_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "n.csv"
    mod.main([str(tmp_path / "a" / "final.pt"), "--noise-model", "pcm", "--noise-scales", "0,1", "--t-inference", "0,3600",
              "--n-chips", "2", "--episodes", "2", "--out", str(out)])
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 2 * 2 * 2 and all(np.isfinite(float(r["return_mean"])) for r in rows)
    zero = [r for r in rows if float(r["noise_scale"]) == 0.0 and float(r["t_inference"]) == 0.0]
    assert all(float(r["weight_rmse"]) < 0.3 for r in zero) and len({r["return_mean"] for r in zero}) >= 1
