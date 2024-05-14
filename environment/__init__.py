from gym.envs.registration import register

register(
    id="pendulum_environment/Pendulum-v0",
    entry_point="environment.envs:Pendulum",
    max_episode_steps=1500,
)
