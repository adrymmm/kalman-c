import statsmodels.api as sm
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from kalman_wrapper import run_kalman_filter

def test_local_linear_trend_matches_statsmodels():
    T = 1000

    H = 1.0
    Q_level, Q_slope = 0.01, 0.01
    P0_level, P0_slope = 1.0, 1.0
    level0, slope0 = 0.0, 0.0

    # Slope as its own random walk
    h_slope = np.random.normal(0, np.sqrt(Q_slope), T)
    slope = np.cumsum(h_slope)

    # Level accumulates slope plus own noise
    h_level = np.random.normal(0, np.sqrt(Q_level), T)
    level = level0 + np.cumsum(slope + h_level)

    e = np.random.normal(0, np.sqrt(H), T)
    y = level + e

    mod = sm.tsa.UnobservedComponents(y, level='local linear trend')
    print(mod.param_names)

    T_mat = np.array([[1.0, 1.0], [0.0, 1.0]])
    P0 = np.array([[P0_level, 0.0], [0.0, P0_slope]])
    Q = np.array([[Q_level, 0.0], [0.0, Q_slope]])

    initial_state_cov = T_mat @ P0 @ T_mat.T + Q

    mod.initialize_known(
    initial_state=np.array([level0, slope0]),
    initial_state_cov=initial_state_cov
)
    mod.ssm.loglikelihood_burn = 0

    res = mod.filter(params=[H, Q_level, Q_slope])
    level_filtered_sm = res.filtered_state[0]
    slope_filtered_sm = res.filtered_state[1]
    loglik_sm = res.llf

    a_out, loglik, state = run_kalman_filter(
        y,
        a0=[level0, slope0],
        P0=[P0_level, 0.0, 0.0, P0_slope],
        H=H,
        Q=[Q_level, 0.0, 0.0, Q_slope],
        Z=[1.0, 0.0],
        Tmat=[1.0, 1.0, 0.0, 1.0],
        n=2
    )

    level_diff = np.abs(a_out[:, 0] - level_filtered_sm)
    slope_diff = np.abs(a_out[:, 1] - slope_filtered_sm)
    print("max abs diff (level):", level_diff.max())
    print("max abs diff (slope):", slope_diff.max())
    print("first 5 C level:  ", a_out[:5, 0])
    print("first 5 sm level: ", level_filtered_sm[:5])
    print("first 5 C slope:  ", a_out[:5, 1])
    print("first 5 sm slope: ", slope_filtered_sm[:5])
    print("loglik C:", loglik, " loglik sm:", loglik_sm)

    assert np.allclose(a_out[:, 0], level_filtered_sm, atol=1e-6)
    assert np.allclose(a_out[:, 1], slope_filtered_sm, atol=1e-6)
    assert np.isclose(loglik, loglik_sm, atol=1e-6)

if __name__ == "__main__":
    test_local_linear_trend_matches_statsmodels()
    print("Test passed.")