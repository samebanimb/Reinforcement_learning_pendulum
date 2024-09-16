from gym.envs.registration import register

register(
    id="pendulum_environment/Pendulum-v0",
    entry_point="pendulum_environment.envs:Pendulum",
    max_episode_steps=500,
)

register(
    id="pendulum_environment_evaluation/Pendulum_eval-v0",
    entry_point="pendulum_environment.envs:Pendulum_Evaluation",
    max_episode_steps=500,
)
register(
    id="pendulum_environment_stabilization/Pendulum_eval-v0",
    entry_point="pendulum_environment.envs:Pendulum_Stabilization",
    max_episode_steps=500,
)
register(
    id="pendulum_environment_Test1/Pendulum_eval-v0",
    entry_point="pendulum_environment.envs:Pendulum_Test1",
    max_episode_steps=500,
)
register(
    id="pendulum_environment_Test2/Pendulum_eval-v0",
    entry_point="pendulum_environment.envs:Pendulum_Test2",
    max_episode_steps=500,
)
