import sys, os, time
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from kalman_wrapper import run_kalman_filter
from kalman_python import KalmanStatePy, kalman_filter_py

N_REPEATS = 7
SERIES_LENGTHS = [100, 1_000, 10_000, 100_000, 1_000_000]

H, Q = 1.0, 0.1
a0, P0 = 0.0, 1.0


def simulate_local_level(n_obs, seed=42):
    rng = np.random.default_rng(seed)
    h = rng.normal(0, np.sqrt(Q), n_obs)
    e = rng.normal(0, np.sqrt(H), n_obs)
    a = a0 + np.cumsum(h)
    return a + e


def time_c(y):
    start = time.perf_counter()
    run_kalman_filter(y, [a0], [P0], H, [Q], [1.0], [1.0], 1)
    return time.perf_counter() - start


def time_statsmodels(y):
    mod = sm.tsa.UnobservedComponents(y, level='local level')
    mod.initialize_known(initial_state=np.array([a0]), initial_state_cov=np.array([[P0 + Q]]))
    mod.ssm.loglikelihood_burn = 0
    start = time.perf_counter()
    mod.filter(params=[H, Q])
    return time.perf_counter() - start


def time_python(y, n_obs):
    state = KalmanStatePy(
        a=np.array([a0]), P=np.array([[P0]]), H=H,
        Q=np.array([[Q]]), Z=np.array([1.0]), T=np.array([[1.0]]), n=1
    )
    start = time.perf_counter()
    kalman_filter_py(state, y, n_obs)
    return time.perf_counter() - start


def median_time(fn, *args, repeats=N_REPEATS):
    return np.median([fn(*args) for _ in range(repeats)])


def main():
    rows = []
    for n_obs in SERIES_LENGTHS:
        y = simulate_local_level(n_obs)

        c_t = median_time(time_c, y)
        sm_t = median_time(time_statsmodels, y)
        py_t = median_time(time_python, y, n_obs)

        rows.append({
            "n_obs": n_obs,
            "c_time_s": c_t,
            "statsmodels_time_s": sm_t,
            "python_time_s": py_t,
            "speedup_vs_python": py_t / c_t,
            "speedup_vs_statsmodels": sm_t / c_t,
        })
        print(f"n_obs={n_obs:>7}  C={c_t:.6f}s  sm={sm_t:.6f}s  py={py_t:.6f}s")

    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(__file__), "results.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()