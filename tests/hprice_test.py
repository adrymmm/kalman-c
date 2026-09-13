import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from kalman_wrapper import run_kalman_filter
from data_utils import load_house_price_series

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def test_stats_local_houseprice():
    df_eng = load_house_price_series(DATA_DIR)

    mod = sm.tsa.UnobservedComponents(df_eng.values, level='local level')
    res_fit = mod.fit()

    H, Q = res_fit.params[0],res_fit.params[1]

    # First value of series
    a0 = df_eng.values[0]
    # Sample variance as stand in
    P0 = np.var(df_eng.values)

    # Start recursion from C starting points
    mod.initialize_known(initial_state=np.array([a0]), initial_state_cov=np.array([[P0 + Q]]))
    # Include first observations - statsmodels drops it
    mod.ssm.loglikelihood_burn = 0

    res = mod.filter(params=[H, Q])
    a_filtered_sm = res.filtered_state[0]
    loglik_sm = res.llf

    a_out, loglik, state = run_kalman_filter(df_eng.values, a0, P0, H, Q)

    assert np.allclose(a_out, a_filtered_sm, atol=1e-6)
    assert np.isclose(loglik, loglik_sm, atol=1e-6)

if __name__ == "__main__":
    test_stats_local_houseprice()
    print("Test passed.")
