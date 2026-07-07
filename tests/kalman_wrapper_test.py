import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from kalman_wrapper import run_kalman_filter


def python_reference(y, a0, P0, H, Q):
    a, P = a0, P0
    ll = 0.0
    a_history = []

    for yt in y:
        P += Q
        v = yt - a
        F = P + H
        K = P / F
        a += K * v
        P = (1 - K) * P
        ll += -0.5 * (np.log(2 * np.pi) + np.log(F) + v**2 / F)
        a_history.append(a)

    return np.array(a_history), ll


def test_local_level_matches_reference():
    y = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
    a0, P0, H, Q = 0.0, 1.0, 1.0, 0.1

    a_out, loglik, final_state = run_kalman_filter(y, a0, P0, H, Q)
    expected_a, expected_ll = python_reference(y, a0, P0, H, Q)

    assert np.allclose(a_out, expected_a, atol=1e-9)
    assert np.isclose(loglik, expected_ll, atol=1e-9)


if __name__ == "__main__":
    test_local_level_matches_reference()
    print("Test passed.")