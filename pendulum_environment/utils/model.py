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

    # Km = 0.007677634454753
    # Kt = 0.007682969729280
    # Kg = 3.710000000000000
    # Mp = 0.127000000000000
    # eta_g = 1
    # lp = 0.177800000000000
    # Jp = 0.001198730801458
    # r_mp = 0.006350000000000
    # Rm = 2.600000000000000
    # Bp = 0.002400000000000
    # Jeq = 1.073129910054978
    # Beq = 5.400000000000000
    # g = 9.810000000000000
    #
    # new_states = np.array(
    #    [
    #        x_dot,
    #        theta_dot,
    #        (
    #            -Km * Kt * theta_dot * Kg**2 * Mp * eta_g * lp**2
    #            - Jp * Km * Kt * theta_dot * Kg**2 * eta_g
    #            + Kt * action * Kg * Mp * eta_g**2 * lp**2 * r_mp
    #            + Jp * Kt * action * Kg * eta_g**2 * r_mp
    #            + Rm * sin(theta) * Mp**2 * x_dot**2 * lp**3 * r_mp**2
    #            + Rm * g * cos(theta) * sin(theta) * Mp**2 * lp**2 * r_mp**2
    #            + Jp * Rm * sin(theta) * Mp * x_dot**2 * lp * r_mp**2
    #            + Bp * Rm * cos(theta) * Mp * x_dot * lp * r_mp**2
    #            - Beq * Rm * theta_dot * Mp * lp**2 * r_mp**2
    #            - Beq * Jp * Rm * theta_dot * r_mp**2
    #        )
    #        / (
    #            Rm
    #            * r_mp**2
    #            * (
    #                -(Mp**2) * lp**2 * cos(theta) ** 2
    #                + Mp**2 * lp**2
    #                + Jeq * Mp * lp**2
    #                + Jp * Mp
    #                + Jeq * Jp
    #            )
    #        ),
    #        -(
    #            Bp * Jeq * Rm * x_dot * r_mp**2
    #            + Bp * Mp * Rm * x_dot * r_mp**2
    #            + Mp**2 * Rm * g * lp * r_mp**2 * sin(theta)
    #            - Beq * Mp * Rm * lp * r_mp**2 * theta_dot * cos(theta)
    #            + Jeq * Mp * Rm * g * lp * r_mp**2 * sin(theta)
    #            + Mp**2 * Rm * x_dot**2 * lp**2 * r_mp**2 * cos(theta) * sin(theta)
    #            - Kg**2 * Km * Kt * Mp * eta_g * lp * theta_dot * cos(theta)
    #            + Kg * Kt * Mp * eta_g**2 * lp * r_mp * action * cos(theta)
    #        )
    #        / (
    #            Rm
    #            * r_mp**2
    #            * (
    #                -(Mp**2) * lp**2 * cos(theta) ** 2
    #                + Mp**2 * lp**2
    #                + Jeq * Mp * lp**2
    #                + Jp * Mp
    #                + Jeq * Jp
    #            )
    #        ),
    #    ],
    #    dtype=np.float32,
    # )

    new_states = np.reshape(new_states, (4, 1))
    return new_states
