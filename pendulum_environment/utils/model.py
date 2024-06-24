"""
Class, which contains
 the model and subsequently
the parameters of the model

"""

from typing import Union
from math import sin, cos
import numpy as np


def Model(states: np.array, action: Union[int, np.ndarray]) -> np.ndarray:
    """_summary_

    Args:
        states (np.array): position, angle, velocity, angular velocity
        action (Union[int, np.ndarray]): voltage of the motor

    Returns:
        np.ndarray: _description_
    """
    theta = states[1, 0]
    x_dot = states[2, 0]
    theta_dot = states[3, 0]
    gravity = 9.03788858603972
    mass_cart = 0.94
    mass_pole = 0.127
    length = 0.2066
    damping_factor = 8.28866833710732
    gain_factor = 1.16342493738728

    new_states = np.array(
        [
            x_dot,
            theta_dot,
            (
                mass_pole * length * sin(theta) * theta_dot**2
                + mass_pole * gravity * sin(theta) * cos(theta)
                + (-damping_factor * x_dot + gain_factor * action)
            )
            / (mass_cart + mass_pole * (sin(theta) ** 2)),
            -(
                mass_pole * length * sin(theta) * cos(theta) * (theta_dot**2)
                + (mass_pole + mass_cart) * gravity * sin(theta)
                + cos(theta) * (-damping_factor * x_dot + gain_factor * action)
            )
            / (length * (mass_cart + mass_pole * sin(theta) ** 2)),
        ],
        dtype=np.float32,
    )

    new_states = np.reshape(new_states, (4, 1))
    return new_states
