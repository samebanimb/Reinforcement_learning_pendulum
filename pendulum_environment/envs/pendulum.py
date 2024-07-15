"""
Summary
This code produces an environment for a pendulum on a cart
It can be used to train an agent in Reinforcement Learning
"""

from math import pi, cos
from typing import Optional

import numpy as np
import gym
from gym import spaces, logger
import pygame
from pygame import gfxdraw

from pendulum_environment.utils import integrate_RK4


class Pendulum(gym.Env):
    """
    This class create a gym environment which contains a pendulum on cart
     for Reinforcement Learning

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        render_mode="rgb_array",
        max_voltage=5,
        max_track_length=1,
        track_limitation=1,
        action_step=0.01,
    ):
        # TODO make sure that the entered voltage is not decimal
        # TODO make sure the discretization step is smaller than than the max_voltage
        self.dt = 0.01

        self.kinematics_integrator = "RK4"

        self.x_threshold = max_track_length / 2 * track_limitation
        threshold = self.x_threshold + 0.5 * (max_track_length / 2 - self.x_threshold)

        high = np.array(
            [
                threshold,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        # action_number = int(max_voltage * 2 / action_step + 1)

        # step = 0.5
        # assert isinstance(action_number, int), "The maximal"

        # self.action_space = spaces.Discrete(action_number)
        self.voltage = max_voltage
        self.action_space = spaces.Box(
            -max_voltage, max_voltage, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._action_to_voltage = []
        # for k in range(21):
        #   self._action_to_voltage.append(-max_voltage + k * step)

        self.render_mode = render_mode

        self.length = 0.1
        self._first_time_upright = False

        self.screen_width = 608
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.surf = None
        self.last_voltage = 0.0

        self.steps_beyond_terminated = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, theta, x_dot, theta_dot = self.state
        # voltage = self._action_to_voltage[action]
        # if action == 1:
        #    voltage = self.voltage
        # else:
        #    voltage = -self.voltage
        voltage = action[0]

        state = np.reshape(
            np.array([x, theta, x_dot, theta_dot], dtype=np.float32), (4, 1)
        )

        state = integrate_RK4(states=state, action=voltage, dt=self.dt)
        x = state[0, 0]
        theta = state[1, 0]
        x_dot = state[2, 0]
        theta_dot = state[3, 0]
        self.state = (x, theta, x_dot, theta_dot)

        x_out_of_bounds = x < -self.x_threshold or x > self.x_threshold
        # pendulum_upright = cos(theta) < -0.995

        # pendulum_near_center = x < 0.1 and x > -0.1
        # pendulum_over_track = theta % (2 * pi) > (pi / 2) and theta % (2 * pi) < (
        #    3 * pi / 2
        # )
        terminated = bool(x_out_of_bounds)
        reward = 0
        # if terminated:
        #    reward = -600
        # self._first_time_upright = pendulum_upright
        # if not terminated:
        #    if pendulum_upright and self._first_time_upright:
        #        reward += 100
        #    elif pendulum_upright:
        #        reward += 1 - 0.1 * x**2
        #    else:
        #        reward = (
        #            -0.01 * (theta % (2 * pi) - pi) ** 2
        #            - 0.2 * (x) ** 2
        #            # - 0.0001 * theta_dot**2
        #        )
        #        if cos(theta) < 0:
        #            reward -= 0.05 * cos(theta)
        if terminated:
            reward = -300
        if not terminated:
            reward += 0.1 * (0.5 * (1 - cos(theta) - (x / self.x_threshold) ** 2))
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0
        self.last_voltage = voltage
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # self.state = [
        #    self.np_random.uniform(
        #        low=(-self.x_threshold + 0.1), high=(self.x_threshold - 0.1)
        #    ),
        #    self.np_random.uniform(low=-(8 * pi) / 9, high=(8 * pi) / 9),
        #    0.0,
        #    0.0,
        # ]
        self.state = [
            self.np_random.uniform(
                low=(-self.x_threshold + 0.3), high=(self.x_threshold - 0.3)
            ),
            self.np_random.uniform(low=-pi / 2, high=pi / 2),
            0.0,
            self.np_random.uniform(low=-0.5, high=0.5),
        ]
        self.np_random.uniform(low=-self.x_threshold, high=self.x_threshold)
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32)

    def render(self, *args):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 200  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(x[1] + np.pi)
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
