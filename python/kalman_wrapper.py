import ctypes
import numpy as np
import os

MAXN = 2

_lib_path = os.path.join(os.path.dirname(__file__), "..", "libkalman.so")
lib = ctypes.CDLL(_lib_path)


class KalmanState(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_double * MAXN),
        ("P", ctypes.c_double * (MAXN * MAXN)),
        ("H", ctypes.c_double),
        ("Q", ctypes.c_double * (MAXN * MAXN)),
        ("Z", ctypes.c_double * MAXN),
        ("T", ctypes.c_double * (MAXN * MAXN)),
        ("n", ctypes.c_int),
    ]


lib.kalman_filter.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # y[]
    ctypes.c_int,                     # n_obs
    ctypes.POINTER(KalmanState),      # s
    ctypes.POINTER(ctypes.c_double),  # a_out[]
]
lib.kalman_filter.restype = ctypes.c_double


def run_kalman_filter(y, a0, P0, H, Q, Z, Tmat, n):
    y = np.ascontiguousarray(y, dtype=np.float64)
    n_obs = len(y)

    a0 = np.ascontiguousarray(a0, dtype=np.float64).flatten()
    P0 = np.ascontiguousarray(P0, dtype=np.float64).flatten()
    Q = np.ascontiguousarray(Q, dtype=np.float64).flatten()
    Z = np.ascontiguousarray(Z, dtype=np.float64).flatten()
    Tmat = np.ascontiguousarray(Tmat, dtype=np.float64).flatten()

    state = KalmanState()
    state.n = n
    for i in range(n):
        state.a[i] = a0[i]
        state.Z[i] = Z[i]
    for i in range(n * n):
        state.P[i] = P0[i]
        state.Q[i] = Q[i]
        state.T[i] = Tmat[i]
    state.H = H

    a_out = np.zeros(n_obs * n, dtype=np.float64)

    y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    a_out_ptr = a_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    loglik = lib.kalman_filter(y_ptr, n_obs, ctypes.byref(state), a_out_ptr)

    a_out = a_out.reshape(n_obs, n)
    return a_out, loglik, state


if __name__ == "__main__":
    y = [1.0, 2.0, 1.5, 3.0, 2.5]
    a_out, loglik, final_state = run_kalman_filter(
        y, a0=[0.0], P0=[1.0], H=1.0, Q=[0.1], Z=[1.0], Tmat=[1.0], n=1
    )
    print("a_out:\n", a_out)
    print("loglik:", loglik)