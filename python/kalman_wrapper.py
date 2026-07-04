import ctypes
import numpy as np
import os

# Load the shared library
_lib_path = os.path.join(os.path.dirname(__file__), "..", "libkalman.so")
lib = ctypes.CDLL(_lib_path)
 
class KalmanState(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_double),
        ("P", ctypes.c_double),
        ("H", ctypes.c_double),
        ("Q", ctypes.c_double),
    ]

lib.kalman_filter.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # y[]
    ctypes.c_int,                     # T
    ctypes.POINTER(KalmanState),      # s
    ctypes.POINTER(ctypes.c_double),  # a_out[]
]
lib.kalman_filter.restype = ctypes.c_double


def run_kalman_filter(y, a0, P0, H, Q):
    # Force contiguous array for C compatibility
    y = np.ascontiguousarray(y, dtype=np.float64)
    T = len(y)

    state = KalmanState(a=a0, P=P0, H=H, Q=Q)
    # A out array only allocated in python
    a_out = np.zeros(T, dtype=np.float64)

    y_ptr = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    a_out_ptr = a_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    loglik = lib.kalman_filter(y_ptr, T, ctypes.byref(state), a_out_ptr)

    return a_out, loglik, state

if __name__ == "__main__":
    y = [1.0, 2.0, 1.5, 3.0, 2.5]
    a_out, loglik, final_state = run_kalman_filter(y, a0=0.0, P0=1.0, H=1.0, Q=0.1)
    print("a_out:", a_out)
    print("loglik:", loglik)
    print("final a, P:", final_state.a, final_state.P)