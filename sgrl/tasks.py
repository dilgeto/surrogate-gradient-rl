"""Hiperparámetros de partida por tarea (se aplican como valores por defecto del CLI).

Son puntos de partida razonables, no óptimos: se comprobó que cada tarea corre y entrena (ver
README para qué se verificó); no se ajustaron para convergencia salvo donde el README lo diga.
Cualquier flag explícito en la línea de comandos tiene prioridad.
"""

PRESETS = {
    "Acrobot": dict(steps=1_000_000, gamma=0.99, epochs=10, minibatch=512, actor_lr=1e-3,
                    num_envs=16, rollout_steps=128),
    # Recompensa dispersa (+100 en la meta): algo de entropía ayuda a descubrir la meta.
    "MountainCarDiscrete": dict(steps=1_000_000, gamma=0.99, epochs=10, minibatch=512, actor_lr=1e-3,
                                ent_coef=0.02, num_envs=16, rollout_steps=256),
    "CarTrack": dict(steps=2_000_000, gamma=0.99, epochs=10, minibatch=1024, actor_lr=5e-4,
                     num_envs=16, rollout_steps=256),
    # 5 agentes con actor compartido: cada muestra son 5 pasadas del actor, por eso minilote menor.
    "Bottleneck": dict(steps=3_000_000, gamma=0.99, epochs=10, minibatch=512, actor_lr=5e-4,
                       num_envs=16, rollout_steps=200),
    "BipedalWalker": dict(steps=5_000_000, gamma=0.99, epochs=10, minibatch=2048, actor_lr=3e-4,
                          num_envs=32, rollout_steps=256, hidden=[128, 128], env_threads=8),
}
