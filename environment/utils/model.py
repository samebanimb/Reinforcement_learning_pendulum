import numpy as np
from typing import Union
from math import sin, cos


def Model(states: np.array, action: Union[int, np.ndarray]) -> np.ndarray:
    x = states[1, 1]
    theta = states[2, 1]
    x_dot = states[3, 1]
    theta_dot = states[4, 1]
    gravity = 9.03788858603972
    mass_cart = 0.94
    mass_pole = 0.127
    length = 0.2066
    damping_factor = 8.28866833710732
    gain_factor = 1.16342493738728

    new_states = np.array(
        [
            x,
            theta,
            (
                mass_pole * length * sin(theta) * theta_dot**2
                + mass_pole * gravity * sin(theta) * cos(theta)
                + (-damping_factor * x_dot + gain_factor * action)
            )
            / (mass_cart + mass_pole * sin(theta) ** 2),
            -(
                mass_pole * length * sin(theta) * cos(theta) * theta_dot**2
                + (mass_pole + mass_cart) * gravity * sin(theta)
                + cos(theta) * (-damping_factor * x_dot + gain_factor * action)
            )
            / (length * (mass_cart + mass_pole * sin(theta) ^ 2)),
        ]
    )

    new_states = np.reshape(new_states, (4, 1))
    return new_states
