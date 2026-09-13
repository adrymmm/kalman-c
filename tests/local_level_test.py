import statsmodels.api as sm
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from kalman_wrapper import run_kalman_filter

def test_local_to_statslevels():
    T = 1000

    H, Q = 1.0, 0.1

    h = np.random.normal(0, np.sqrt(Q), T)
    e = np.random.normal(0, np.sqrt(H), T)
    a0, P0 = 0.0, 1.0
    a = a0 + np.cumsum(h)
    y = a + e

    mod = sm.tsa.UnobservedComponents(y, level='local level')
    mod.initialize_known(initial_state=np.array([a0]), initial_state_cov=np.array([[P0 + Q]]))
    mod.ssm.loglikelihood_burn = 0

    res = mod.filter(params=[H, Q])
    a_filtered_sm = res.filtered_state[0]
    loglik_sm = res.llf

    a_out, loglik, state = run_kalman_filter(
        y, a0=[a0], P0=[P0], H=H, Q=[Q], Z=[1.0], Tmat=[1.0], n=1
    )

    diff = np.abs(a_out[:, 0] - a_filtered_sm)
    print("max abs diff (states):", diff.max())
    print("first 5 C states:      ", a_out[:5, 0])
    print("first 5 statsmodels:   ", a_filtered_sm[:5])
    print("loglik C:", loglik, " loglik sm:", loglik_sm)

    assert np.allclose(a_out[:, 0], a_filtered_sm, atol=1e-9)
    assert np.isclose(loglik, loglik_sm, atol=1e-9)

if __name__ == "__main__":
    test_local_to_statslevels()
    print("Test passed.")