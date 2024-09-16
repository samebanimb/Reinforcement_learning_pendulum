"""
Class, which contains
 the model and subsequently
the parameters of the model

"""

from typing import Union
from math import sin, cos, tanh, exp
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
    # m = 0.127790354461162
    # M = 1.001707436687852
    # l = 0.172872963286384
    # g = 9.81000000000000
    # Am = 11.551714191138592
    # Beq = 2.199519470593672
    # Fc_x = 4.162906348117589
    # Fs_x = 1.508990358342391
    # vs_x = -15.656676084215142
    # e_x = 1.419991679308769
    # e_o = 0.001582169336626
    #
    # new_states = np.array(
    #    [
    #        x_dot,
    #        theta_dot,
    #        (
    #            3 * e_o * theta_dot * cos(theta)
    #            - 4 * Fc_x * l * tanh(1000 * x_dot)
    #            + 4 * Beq * l * action
    #            - 4 * Am * l * x_dot
    #            - 4 * e_x * l * x_dot
    #            + 4 * l**2 * m * theta_dot**2 * sin(theta)
    #            + 4 * Fc_x * l * tanh(1000 * x_dot) * exp(-(x_dot**2) / vs_x**2)
    #            - 4 * Fs_x * l * tanh(1000 * x_dot) * exp(-(x_dot**2) / vs_x**2)
    #            + 3 * g * l * m * cos(theta) * sin(theta)
    #        )
    #        / (l * (4 * M + 4 * m - 3 * m * cos(theta) ** 2)),
    #        -(
    #            3
    #            * (
    #                M * e_o * theta_dot
    #                + e_o * m * theta_dot
    #                + g * l * m**2 * sin(theta)
    #                + l**2 * m**2 * theta_dot**2 * cos(theta) * sin(theta)
    #                - Fc_x * l * m * tanh(1000 * x_dot) * cos(theta)
    #                + Beq * l * m * action * cos(theta)
    #                - Am * l * m * x_dot * cos(theta)
    #                + M * g * l * m * sin(theta)
    #                - e_x * l * m * x_dot * cos(theta)
    #                + Fc_x
    #                * l
    #                * m
    #                * tanh(1000 * x_dot)
    #                * exp(-(x_dot**2) / vs_x**2)
    #                * cos(theta)
    #                - Fs_x
    #                * l
    #                * m
    #                * tanh(1000 * x_dot)
    #                * exp(-(x_dot**2) / vs_x**2)
    #                * cos(theta)
    #            )
    #        )
    #        / (l**2 * m * (4 * M + 4 * m - 3 * m * cos(theta) ** 2)),
    #    ],
    #    dtype=np.float32,
    # )

    new_states = np.reshape(new_states, (4, 1))
    return new_states
