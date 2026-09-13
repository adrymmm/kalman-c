import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from python.kalman_wrapper import run_kalman_filter

IN = "data/"

def test_stats_local_houseprice():
    df = pd.read_csv(f'{IN}/Average-prices-2026-03.csv', index_col=0, parse_dates=[0])
    df.drop(columns=['Area_Code', 'Monthly_Change', 'Annual_Change', 'Average_Price_SA'], inplace=True)
    df_eng = df[df['Region_Name'] == 'England']
    df_eng = df_eng.loc['1975-01-01':]
    # Taking sums of months for quarters
    df_eng = df_eng.resample('QE').mean(numeric_only=True)
    df_eng['Region_Name'] = 'England'
    # Move to front
    df_eng = df_eng[['Region_Name'] + [col for col in df_eng.columns if col != 'Region_Name']]
    # Converting to periods
    df_eng.index = df_eng.index.to_period('Q')
    # Keeping only columns we need
    df_eng = df_eng["Average_Price"]

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
