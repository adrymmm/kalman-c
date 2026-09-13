import statsmodels.api as sm
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from kalman_python import KalmanStatePy, kalman_filter_py

def test_python_matches_c_reference():
    Y = [1.0, 2.0, 3.0, 4.0, 5.0]
    state = KalmanStatePy(
        a=np.array([0.0]), P=np.array([[1.0]]), H=1.0,
        Q=np.array([[0.1]]), Z=np.array([1.0]), T=np.array([[1.0]]), n=1
    )
    loglik, a_out = kalman_filter_py(state, Y, n_obs=5)
    assert np.isclose(loglik, -12.142234, atol=1e-6)

if __name__ == "__main__":
    test_python_matches_c_reference()
    print("Test passed.")