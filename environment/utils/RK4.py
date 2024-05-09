"""_summary_
Utility code to appply the runge kutta mehtod
Returns:
    _type_: _description_
"""

import numpy as np
from .model import Model


def integrate_RK4(states: np.ndarray, action, dt, N_steps=1):

    h = dt / N_steps

    states_end = states

    for _ in range(N_steps):
        k_1 = Model(states_end, action)
        k_2 = Model(states_end + 0.5 * h * k_1, action)
        k_3 = Model(states_end + 0.5 * h * k_2, action)
        k_4 = Model(states_end + k_3 * h, action)

        states_end = states_end + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h

    return states_end
