from .states import SigmaLevel,PiLevelOmega

from .spin_params import S,I1,I2
cdef double S_ = <double>S
cdef double I1_ = <double>I1
cdef double I2_ = <double>I2


from .fast_wigners import wigner_6j,wigner_3j,wigner_9j

#conditional import
#from molecular_parameters import *

"""
import ctypes
libm = ctypes.CDLL('libm.dylib')

libm.sqrt.argtypes = [ctypes.c_double]
libm.sqrt.restype = ctypes.c_double

libm.fabs.argtypes = [ctypes.c_double]
libm.fabs.restype = ctypes.c_double
"""

cdef extern from "math.h" nogil:
    double sqrt(double)
    double fabs(double)

cdef inline double kdel(double x,double y) nogil:
    return 1.0 if <int>(2*x) == <int>(2*y) else 0.0

cdef inline double reduced(double x) nogil:
    return sqrt(x*(x+1.0)*(2.0*x+1.0))

cdef inline double nreduced(double x,double y) nogil:
    return sqrt((2.0*x+1.0)*(2.0*y+1.0))

cdef inline double minus_1_pow(double x) nogil:
    cdef int temp = <int>x
    return 1.0 if temp%2 == 0 else -1.0


#### Dipole matrix element between Sigma and Pi states #######################


cpdef double H_int_omega_optimized(state1:SigmaLevel, state2:PiLevelOmega, double pol=0.0):    #pol convention changed. pol defined from ground (state1) to excited (state2).
                                                                    # pol +1 -> mF_state2 - mF_state1 = +1
    cdef double G,N,F1,F,mF,Lambda,Sigma,Omega,Jex,F1p,Fp,mFp

    G,N,F1,F,mF=state1.G,state1.N,state1.F1,state1.F,state1.mF
    Lambda,Sigma,Omega,Jex,F1p,Fp,mFp = state2.Lambda, \
                                        state2.Sigma, \
                                        state2.Omega, \
                                        state2.parity_state.J, \
                                        state2.parity_state.F1, \
                                        state2.parity_state.F, \
                                        state2.parity_state.mF
    
    cdef int i
    cdef double J,sigma,omega
    cdef int q,iter_idx
    cdef double val = 0.0
    cdef double mult_J,mult_sigma,pre_factor

    pre_factor = (minus_1_pow(G+S_+I1_+F-mF+Fp+I2_+F1+1)*
                    sqrt(2*N+1)*
                    wigner_3j(F,1,Fp,-mF,-pol,mFp)*
                    nreduced(F,Fp)*
                    wigner_6j(F1p,Fp,I2_,F,F1,1)*
                    nreduced(F1,F1p)
                    )

    for iter_idx in range(<int>(2*fabs(N-S_)),<int>(2*(N+S_+1)),2):
        #J = 0.5*float(iter_idx)
        J = 0.5*<double>iter_idx
        mult_J = (nreduced(J,G)*
                    wigner_6j(F1,G,N,S_,J,I1_)*
                    minus_1_pow(F1p+I1_+J+1)*
                    wigner_6j(Jex,F1p,I1_,F1,J,1)*
                    nreduced(J,Jex)
                )

        for sigma in [-1.0/2,1.0/2]:
            omega = sigma
            mult_sigma = (mult_J*
                            minus_1_pow(N-S_+omega+J-omega)*wigner_3j(J,S_,N,omega,-sigma,0)*
                            kdel(sigma,Sigma)
                        )
            for q in range(-1,2,2):#[-1,1]: # removed q= 0 value because Lambda (from Pi state) cannot be 0.
                val += mult_sigma*wigner_3j(J,1,Jex,-omega,q,Omega)*kdel(Lambda,-q)
        #iter_idx += 2
    return val*pre_factor

####################################################################################################