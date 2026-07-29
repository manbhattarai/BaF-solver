# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

cdef int find_interval(double[:] x, double xi):
    cdef int n = x.shape[0] - 1
    if xi <= x[0]:
        return 0
    if xi >= x[n]:
        return n - 1

    cdef int low = 0
    cdef int high = n - 1
    cdef int mid

    while low <= high:
        mid = (low + high) // 2
        if x[mid] <= xi < x[mid + 1]:
            return mid
        elif xi < x[mid]:
            high = mid - 1
        else:
            low = mid + 1
    return n - 2

def make_spline(np.ndarray[np.float64_t, ndim=1] x,
                  np.ndarray[np.float64_t, ndim=3] coefs):
    """
    Create a fast cubic spline evaluator.
    
    Parameters:
        x     : (n,)
        coefs : (n-1, N, 4) in SciPy format: [dx^3, dx^2, dx^1, dx^0]
    
    Returns:
        spline(x_eval) -> (len(x_eval), N)
    """
    cdef double[:, :] coefs2d
    x = np.ascontiguousarray(x, dtype=np.float64)
    coefs = np.ascontiguousarray(coefs, dtype=np.float64)

    cdef double[:, :] dummy

    def spline(np.ndarray[np.float64_t, ndim=1] x_eval):
        cdef Py_ssize_t i, j
        cdef int idx
        cdef double dx, xi
        cdef Py_ssize_t M = x_eval.shape[0]
        cdef Py_ssize_t n_minus1 = coefs.shape[0]
        cdef Py_ssize_t N = coefs.shape[1]

        x_eval = np.ascontiguousarray(x_eval, dtype=np.float64)
        cdef double[:] x_eval_view = x_eval
        cdef double[:, :, :] coefs_view = coefs
        cdef double[:] x_view = x

        out = np.empty((M, N), dtype=np.float64)
        cdef double[:, :] out_view = out

        for i in range(M):
            xi = x_eval_view[i]
            idx = find_interval(x_view, xi)
            dx = xi - x_view[idx]
            for j in range(N):
                # Horner: (((c0 * dx) + c1) * dx + c2) * dx + c3
                out_view[i, j] = ((coefs_view[idx, j, 0] * dx +
                                   coefs_view[idx, j, 1]) * dx +
                                   coefs_view[idx, j, 2]) * dx + coefs_view[idx, j, 3]
        return out

    return spline
