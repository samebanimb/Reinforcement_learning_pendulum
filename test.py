if __name__ == "__main__":
    import pendulum_environment  # noqa
    import gym

    env = gym.make("pendulum_environment/Pendulum-v0", render_mode="human")
    env.reset()
    env.render()
