from numba import njit, complex128, int64, float64
import numpy as np
from scipy.interpolate import interp1d,CubicSpline

@njit(int64 (float64[:],float64),cache = True)
def find_interval(x, xi):
    n = len(x) - 1
    if xi <= x[0]:
        return 0
    if xi >= x[-1]:
        return n - 1
    low = 0
    high = n - 1
    while low <= high:
        mid = (low + high) // 2
        if x[mid] <= xi < x[mid + 1]:
            return mid
        elif xi < x[mid]:
            high = mid - 1
        else:
            low = mid + 1
    return n - 2

@njit(float64[:,:](float64[:],float64[:,:,:],float64[:]),cache = True)
def _evaluate_vector_spline(x, coefs, x_eval):
    """
    Evaluate vector-valued cubic spline.

    x: shape (n,)
    coefs: shape (n-1, N, 4), ordered as [c0, c1, c2, c3] for each dimension (SciPy order)
    x_eval: shape (M,)

    Returns: shape (M, N)
    """
    n_minus1, N, _ = coefs.shape
    M = len(x_eval)
    out = np.empty((M, N), dtype=np.float64)

    for i in range(M):
        xi = x_eval[i]
        idx = find_interval(x, xi)
        dx = xi - x[idx]

        for j in range(N):
            # Correct Horner evaluation: ((((c0 * dx) + c1) * dx) + c2) * dx + c3
            c0 = coefs[idx, j, 0]  # dx^3
            c1 = coefs[idx, j, 1]  # dx^2
            c2 = coefs[idx, j, 2]  # dx^1
            c3 = coefs[idx, j, 3]  # dx^0
            out[i, j] = ((c0 * dx + c1) * dx + c2) * dx + c3
    return out

def make_fast_vector_spline(x: np.ndarray, coefs: np.ndarray):
    x = np.ascontiguousarray(x, dtype=np.float64)
    coefs = np.ascontiguousarray(coefs, dtype=np.float64)

    def spline(x_eval):
        x_eval = np.ascontiguousarray(x_eval, dtype=np.float64)
        return _evaluate_vector_spline(x, coefs, x_eval)

    return spline

def numba_interpolate(x,y,real_imag = False):
    if real_imag == True:
        cs = CubicSpline(x, np.real(y))#,     kind=interpol_kind, axis=0, bounds_error=False, fill_value='extrapolate')
        coefs = np.transpose(cs.c, (1, 2, 0))
        y_interp_real = make_fast_vector_spline(x, coefs)

        cs = CubicSpline(x, np.imag(y))#,     kind=interpol_kind, axis=0, bounds_error=False, fill_value='extrapolate')
        coefs = np.transpose(cs.c, (1, 2, 0))
        y_interp_imag = make_fast_vector_spline(x, coefs)
        y_interp = (y_interp_real,y_interp_imag)
    else:
        cs = CubicSpline(x, y)
        coefs = np.transpose(cs.c, (1, 2, 0))
        y_interp = make_fast_vector_spline(x, coefs)#,     kind=interpol_kind, axis=0, bounds_error=False, fill_value='extrapolate')
    return y_interp