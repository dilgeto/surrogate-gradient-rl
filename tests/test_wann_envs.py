"""Los entornos Acrobot, MountainCarDiscrete y CarTrack deben ser EXACTAMENTE los de wann-cpp.

Las trayectorias de referencia (semilla 12345, acciones deterministas) se generaron con una réplica de la
construcción de entornos de wann-cpp (SnnAcrobotTask.cpp, SnnDiscMCTask.cpp, SnnCarTask.cpp): torque ±1 en
Acrobot, MountainCar-v0 de Gymnasium (fuerza 0.001, meta 0.5, -1 por paso, 200 pasos) y pista de 50 mm.
Se fijan los primeros pasos (obs antes de actuar, recompensa) y la duración del episodio.
"""
import math

import numpy as np
import pytest

from sgrl.envs import VecEnv

SEED = 12345
# tarea -> (pasos que dura el episodio con estas acciones, ¿termina?, [(recompensa, obs)] de los primeros pasos)
REF = {
    "Acrobot": (500, False, [
        (-1.0, [0.999935203217444, -0.011383732537662603, 0.9952179495167196, -0.0976792350488893, 0.05415944593159089, 0.02400892557143905]),
        (-1.0, [0.9999701130627832, -0.007731298804509971, 0.9977274231883505, -0.06737944061755126, -0.01736849781899555, 0.27521642392436985]),
        (-1.0, [0.9997760707189709, -0.021161484298963616, 0.9998374836304439, 0.018027931870898604, -0.11139194598847105, 0.5622537151079782]),
        (-1.0, [0.9986817422183973, -0.0513300863005964, 0.9881181263556934, 0.15369635118412528, -0.17962981841073042, 0.7715395095710467]),
        (-1.0, [0.996078317637208, -0.0884759014253563, 0.9507554433254629, 0.30994207037284, -0.17994266809373685, 0.8031646207847467]),
    ]),
    "MountainCarDiscrete": (200, False, [
        (-1.0, [-0.5113839784204496, 0.0]),
        (-1.0, [-0.5124755688979484, -0.0010915904774987941]),
        (-1.0, [-0.5136505679440099, -0.001174999046061478]),
        (-1.0, [-0.5139001674718672, -0.00024959952785731367]),
        (-1.0, [-0.5152224962979552, -0.0013223288260880272]),
    ]),
    "CarTrack": (354, True, [
        (0.0, [0.6002231392859763, -1.7208221645653714, -0.7010868021911727, 0.0, 0.0, 0.0, 0.22000000000000003, 0.3, 0.42000000000000004]),
        (0.0, [0.6002231392859763, -1.7208221645653714, -0.7010868021911727, 0.0, 0.0, 0.0, 0.22000000000000003, 0.3, 0.42000000000000004]),
        (0.025313080237547338, [0.6002231392859763, -1.7208221645653714, -0.7010868021911727, 0.025313080237547338, 0.0, 0.0, 0.22000000000000003, 0.3, 0.42000000000000004]),
        (0.06515072748078657, [0.6004165670619421, -1.7209854462195648, -0.7010868021911727, 0.06515072748078657, 0.006503642000941511, 0.2781593647165272, 0.22000000000000003, 0.3, 0.42000000000000004]),
        (0.12216310328515623, [0.6009563625200663, -1.7213560030012376, -0.6983052085440075, 0.12216310328515623, 0.014431519319529776, 0.4979408589174192, 0.22000000000000003, 0.3, 0.42000000000000004]),
    ]),
}


def _actions(task, k):
    if task == "Acrobot":
        return np.array([[math.sin(0.37 * k + 0.5)]])
    if task == "MountainCarDiscrete":
        return np.array([[(k * 7) % 3]])  # índice -> fuerza índice-1
    return np.array([[math.sin(0.37 * k), math.cos(0.23 * k + 0.1)]])


@pytest.mark.parametrize("task", list(REF))
def test_env_matches_wann_cpp(task):
    length, terminates, head = REF[task]
    env = VecEnv(task, 1, seed=0)
    obs = env.reset(SEED)
    for k in range(length + 1):
        _, rew, term, trunc, final_obs, finished = env.step(_actions(task, k))
        if k < len(head):
            r_ref, o_ref = head[k]
            np.testing.assert_allclose(obs[0], o_ref, rtol=1e-6, atol=1e-9, err_msg=f"{task} obs paso {k}")
            assert rew[0] == pytest.approx(r_ref, rel=1e-6, abs=1e-9), f"{task} recompensa paso {k}"
        if term[0] or trunc[0]:
            break
        obs = final_obs
    assert k + 1 == length, f"{task}: el episodio dura {k + 1} pasos y wann-cpp {length}"
    assert bool(term[0]) == terminates
    assert bool(trunc[0]) == (not terminates)


# --- reward shaping basado en potencial (escala 10, como wann-cpp/p/acrobot_snn.json y disc_mc_snn.json) ---
# F = 10·(φ(s') - φ(s)) por paso, de la fórmula de SnnAcrobotTask.cpp / SnnDiscMCTask.cpp (semilla 12345).
SHAPING_REF = {
    "Acrobot": [-0.03170075118769433, -0.026243726122059385, 0.06384638122540864, 0.2285891696011655],
    "MountainCarDiscrete": [-0.0011461647778132633, -0.0011139640541157902, -0.0002206406715132836, -0.001075410041930791],
}


@pytest.mark.parametrize("task", list(SHAPING_REF))
def test_shaping_term_matches_wann_cpp(task):
    env = VecEnv(task, 1, seed=0)
    obs = env.reset(SEED)
    got = []
    for k in range(len(SHAPING_REF[task])):
        _, _, _, _, final_obs, _ = env.step(_actions(task, k))
        got.append(10.0 * (env.potential(final_obs)[0] - env.potential(obs)[0]))
        obs = final_obs
    np.testing.assert_allclose(got, SHAPING_REF[task], rtol=1e-6, atol=1e-9)


def test_shaping_trains_on_shaped_reward_but_reports_original_return():
    import torch
    from sgrl.policy import ActorCritic
    from sgrl.ppo import PPO, PPOConfig

    torch.manual_seed(0)
    env = VecEnv("Acrobot", 2, seed=0)
    agent = ActorCritic(env.obs_dim, env.act_dim, "ann", (8, 8))
    ppo = PPO(agent, env, PPOConfig(num_envs=2, rollout_steps=600, minibatch_size=128, epochs=1,
                                    reward_shaping=10.0, normalize_reward=False))
    batch, _ = ppo.collect()
    assert len(ppo.recent) > 0
    assert all(ret == -length for ret, length in ppo.recent)  # Acrobot: -1 por paso, sin shaping en lo reportado
    assert not np.allclose(batch["ret"].cpu().numpy(), batch["val"].cpu().numpy())
    # el aprendizaje sí ve el shaping: con la misma semilla, la recompensa de entrenamiento difiere de -1
    torch.manual_seed(0)
    env0 = VecEnv("Acrobot", 2, seed=0)
    ppo0 = PPO(ActorCritic(env0.obs_dim, env0.act_dim, "ann", (8, 8)), env0,
               PPOConfig(num_envs=2, rollout_steps=600, minibatch_size=128, epochs=1, normalize_reward=False))
    batch0, _ = ppo0.collect()
    assert not np.allclose(batch["adv"].cpu().numpy(), batch0["adv"].cpu().numpy())


def test_shaping_not_available_for_tasks_without_potential():
    import torch
    from sgrl.policy import ActorCritic
    from sgrl.ppo import PPO, PPOConfig

    env = VecEnv("CarTrack", 1, seed=0)
    agent = ActorCritic(env.obs_dim, env.act_dim, "ann", (8, 8))
    with pytest.raises(ValueError, match="potencial"):
        PPO(agent, env, PPOConfig(num_envs=1, reward_shaping=10.0))
