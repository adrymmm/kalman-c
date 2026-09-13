import numpy as np
import math
from dataclasses import dataclass

@dataclass
class KalmanStatePy():
    a: np.ndarray
    P: np.ndarray
    H: float
    Q: np.ndarray
    Z: np.ndarray
    T: np.ndarray
    n: int


def kalman_predict_py(state: KalmanStatePy) -> None:
    state.a =  state.T @ state.a
    state.P = state.T @ state.P @ state.T.T + state.Q

def kalman_update_py(state: KalmanStatePy, y) -> float:
    v = y - state.Z @ state.a
    F = state.Z @ state.P @ state.Z.T + state.H
    assert F != 0.0, "F is zero, singular"


    K = state.P @ state.Z.T / F
    state.a += K * v
    state.P -= np.outer(K,state.Z) @ state.P

    # log likelihood contribution  
    return -0.5 * (np.log(2*math.pi) + np.log(F) + (v**2 / F))

def kalman_filter_py(state: KalmanStatePy, y, n_obs) -> tuple:
    Lt_sum=0
    a_out = np.zeros((n_obs, state.n))

    for i in range(n_obs):
        kalman_predict_py(state)
        Lt = kalman_update_py(state, y[i])
        Lt_sum += Lt
        # Save whole row into a_out
        a_out[i, :] = state.a

    return Lt_sum, a_out