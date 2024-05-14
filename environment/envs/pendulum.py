"""
Summary
This code produces an environment for a pendulum on a cart
It can be used to train an agent in Reinforcement Learning
"""

import math
from typing import Optional

import numpy as np

import gym
from gym import spaces, logger
import pygame
from pygame import gfxdraw

from environment.utils.RK4 import integrate_RK4


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
        render_mode=None,
        max_voltage=5,
        max_track_length=1,
        track_limitation=0.6,
        action_step=0.001,
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

        action_number = max_voltage * 2 / action_step + 1
        assert action_number.is_integer(), "The maximal"

        self.action_space = spaces.Discrete(action_number)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._action_to_voltage = []
        for k in range(action_number):
            self._action_to_voltage.append(-max_voltage + k * action_step)

        self.render_mode = render_mode

        self.length = 0.5

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.surf = None

        self.steps_beyond_terminated = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, theta, x_dot, theta_dot = self.state
        voltage = self._action_to_voltage[action]

        state = np.reshape(np.array([x, theta, x_dot, theta_dot]), (4, 1))

        state = integrate_RK4(states=state, action=voltage, dt=self.dt)
        x = state[1, 1]
        theta = state[2, 1]
        x_dot = state[3, 1]
        theta_dot = state[4, 1]
        self.state = (x, theta, x_dot, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta % math.pi == math.pi / 2
        )
        reward = 0.0
        if not terminated:
            reward += -0.1
        elif self.steps_beyond_terminated is None:

            if (x < -self.x_threshold) or (x > self.x_threshold):
                reward += -200
            else:
                reward += 200

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
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        high = np.array(
            [
                self.x_threshold,
                0,
                0,
                0,
            ],
            dtype=np.float32,
        )
        self.state = self.np_random.uniform(low=-high, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
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
        carty = 100  # TOP OF CART
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
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
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
