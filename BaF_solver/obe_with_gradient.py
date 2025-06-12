from functools import lru_cache
import numpy as np
from numba import njit, complex128, int64
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import warnings
import time

#Diffeqpy import
try:
    from diffeqpy import de,ode
except Exception as e:
    # Fallback to default and warn
    warnings.warn(f"Could not detect diffeqpy. Only Python package for OBE solver available.")

#Sympy import
import sympy as sp
from sympy import I
t_sp=sp.symbols('t',real = sp.true)

#Symengine import
import symengine as se
t_se = se.Symbol("t", real=True)

#Local import
from .molecular_parameters import Gamma;
Gamma *= 2*np.pi


def pulse_se(t_symbol,center,width):
    k = 200
    return 0.5 * (se.tanh(k * (t_symbol-center + width / 2)) - se.tanh(k * (t_symbol-center - width / 2)))

def pulse_sp(t_symbol,center,width):
    k = 200
    return 0.5 * (sp.tanh(k * (t_symbol-center + width / 2)) - sp.tanh(k * (t_symbol-center - width / 2)))

def pulse_np(t_num,center,width):
    k = 200
    return 0.5 * (np.tanh(k * (t_num-center + width / 2)) - np.tanh(k * (t_num-center - width / 2)))


class Excitation():
    """ Excitation class defines the properties of the optical field.
    Parameters:
    rabi : float
            Rabi frequency of the field. The actual Rabi frequency is rabi*dipole_matrix_element for the particular transition
    pol : int
            Polarization of the field. The dipole matrix element that is created represents transition from Pi to Sigma level. Thus the sense of circular
            polarization is reversed. A value of +1 represnts matrix element due to sigma minus light representing transition from sigma ground state to
            pi excited state. -1 represents sigma plus transiton. 0 represents z- polarized light-transition.
    
    ground_state: SigmaLevel
                    Ground sigma state for the transition
    
    excited_state: PiLevelParity
                    Excited Pi state for the transition
                    Specifying ground_state and excited state defines the frequency of the light only. It need not represnt the set of
                    physically realizable transition.
    detuning : float
                Detuning from the transition frequency specified by the specified ground and excited states
                
    position: float
                Position in time of the center of the beam.
    diameter : float
                The 1/e^2 diameter (in case of Gaussian beam) and the size of the beam (in case of Uniform beam) specified in units of time.
    shape : String
            "Gaussian" to represent a Gaussian beam
            "Uniform" to represent a uniform intensity beam.
    """
    
    def __init__(self, rabi:float, pol:int, ground_state,excited_state, detuning = 0, position = None, diameter = None, shape = None):
        self.rabi = rabi
        self.pol = pol
        self.ground_state = ground_state
        self.excited_state = excited_state
        self.detuning = detuning
        self.position = position
        self.diameter = diameter
        self.shape    = shape
    
    def __repr__(self):
        return f"rabi = {self.rabi}, pol = {self.pol}, Ground = {self.ground_state}, Excited = {self.excited_state}, detuning  = {self.detuning}, position = {self.position}, diameter = {self.diameter}, shape = "+ self.shape
    def __str__(self):
        return f"rabi = {self.rabi}, pol = {self.pol}, Ground = {self.ground_state}, Excited = {self.excited_state}, detuning  = {self.detuning}, position = {self.position}, diameter = {self.diameter}, shape = "+ self.shape
     


class obe:
    """obe class takes in light atom interaction fields, creates interaction Hamiltonian and solves optical bloch equations
    Parameters:
    E_field : Excitation class or a list of Excitation class
            It describes contain all the light fields interating with the molecule.
    states : list
            Contais the list of the ground (G) and the excited (E) states passed as a list [G,E].
    H0 : numpy.ndarray
        Bare Hamiltonian for the levels considered
    Hint : numpy.ndarray
        Matrix of dipole matrix elements. This matrix is later used to construct the actual interaction hamiltonian with the 
        the light detuning and time dependence, for solving the optical bloch equations.
    br: numpy.ndarray
        Matrix of branching ratio. The matrix has a dimension m x n, where m is the number of ground states and n is the number of 
        excited states.
        The element br[m,n] represents the probability of the excited state n decaying to ground state m by spontaneous emission. The sum
        of elements along the columns is 1.
    Hint_func: None or lambda function
        If None, the solver is called to create an interaction Hamiltonian, considering all the fields passed to the obe class.
        If passed as a lambda function of time, is used as interacting Hamiltonian by the solver.

    """
    def __init__(self,E_field,
                    states, #these are the states corresponding to initial B value
                    H0, # interpolation function
                    Hint, #interpolation function
                    br, #interpolation function
                    B_field, # a tuple (B0,grad)
                    test_factor,
                    mode = 'symengine',
                    **kwargs): 
                
        if isinstance(E_field,Excitation):
            self.E_field = [E_field]
        else:
            self.E_field = E_field

        self.B0 = B_field[0]  
        self.grad = B_field[1]  
        
        self.ground_states = states[0]
        self.excited_states = states[1]
        self._n_ground = len(self.ground_states)
        self._n_exec = len(self.excited_states)
        self._n_total = self._n_ground + self._n_exec
        
        if isinstance(Hint,tuple):
            self.Hint_list = [Hint]
        else:
            self.Hint_list = Hint

        
        self.test_factor=test_factor
        self.mode = mode
        self.__dict__.update(kwargs)


        self.H0 = H0 # interpolating function. Array of size _n_total x _n_total

        self._H0_base = 2*np.pi*self.get_interp_array(H0,(self._n_total,self._n_total),self.B0,real_imag = False)
        self._H0_base_diag = np.diag(self._H0_base)

        self.Hinit_scipy = 0
        self.Hinit_symengine = 0
        self.commutator_time_numba = 0
        self.decay_time = 0
        self.repop_time = 0
        
        A = np.zeros((self._n_total,self._n_total),dtype = np.complex128)
        np.fill_diagonal(A[self._n_ground:self._n_total, self._n_ground:self._n_total], 1.0)
        self.decay_matrix = Gamma * A 
        self.decay_matrix_diag = np.ascontiguousarray(np.diag(self.decay_matrix))
        
        self.br = br # interpolating function
        
        #the mF states are good quantum numbers for every magnetic field.
        #we rearrange the states so that the ordering of the mF states stay the same
        self._ground_mF_lists = [[s.mF for s in gs.states] for gs in self.ground_states]
        self._excited_mF_lists = [[s.mF for s in es.states] for es in self.excited_states]
        
        
        if mode == 'symengine':
            self.Hint = self.interaction_picture_symengine()
            #print(f"Type : {type(self.Hint)}, len : {len(self.Hint)}.")
        elif mode == 'sympy':
            print('Sympy not available with gradient calculating solve. Switching to symengine mode.')
            self.Hint = self.interaction_picture_symengine()
        else:
            print("Mode not recognized")
            raise ValueError(f"Unsupported mode: {mode}")
            return 0

        
            
    #Reconstruct matrix on demand
    @staticmethod
    def get_interp_array(A_interp,shape,t,real_imag = False):
        if real_imag:
            real, imag = A_interp
            return (real(t).reshape(shape),imag(t).reshape(shape))
        else:
            return A_interp(t).reshape(shape)
          
        
    def solve(self,npoints,r_init:np.ndarray, max_step_size = 1.0/Gamma, package = 'Python',method = 'RK45'):

        @njit(complex128[:,:](int64,complex128[:],complex128[:,:]),cache = True)
        def decay_product(n,G_diag, R):
            S = np.empty((n, n), dtype=np.complex128)
            for i in range(n):
                for j in range(i,n):
                    val = 0.5 * R[i, j] * (G_diag[j] + G_diag[i])
                    S[i, j] = val
                    if i != j:
                        S[j, i] = np.conj(val)
            return S

        @njit(complex128[:,:](int64,complex128[:,:]),cache = True)
        def numba_commutator(N,HR):
            comm = np.empty((N, N), dtype=np.complex128)
            for i in range(N):
                for j in range(i,N):
                    val = -1j * (HR[i, j] - np.conj(HR[j, i]))
                    comm[i, j] = val
                    if i != j:
                        comm[j, i] = np.conj(val)
            return comm
           
        Rm = np.zeros((self._n_total,self._n_total), dtype=np.complex128)        
        def Rdot_python(T,u):

            R = u.reshape((self._n_total, self._n_total))
            
            B = self.B0+self.grad*T
            
            if self.mode == 'symengine':
                #(H_temp_real,H_temp_imag) = self.Hint
                start = time.perf_counter()
                H = 2*np.pi*self.get_interp_array(self.H0,(self._n_total,self._n_total),B) - \
                    self._H0_base
                #if not (self.Hinit_time%1000):
                #    print(f"h max : {np.amax(H)}, H min : {np.amin(H)}")
                H = H.astype(np.complex128)
                stop = time.perf_counter()

                self.Hinit_scipy += stop-start
                for count,Hint_single in enumerate(self.Hint_list):
                    
                    start = time.perf_counter()
                    Hint_single_real,Hint_single_imag = self.get_interp_array(Hint_single,(self._n_total,self._n_total),B,real_imag = True)
                    #H_interpol = self.get_interp_array(Hint_single,(self._n_total,self._n_total),B)
                    H_interpol = Hint_single_real + 1.0j* Hint_single_imag
                    stop = time.perf_counter()
                    self.Hinit_scipy += stop-start
                    Hint_lambda_real,Hint_lambda_imag = self.Hint[count]
                    H_lambda = Hint_lambda_real(T) + 1.0j*Hint_lambda_imag(T)
                    self.Hinit_symengine += stop-start

                    #check if any is zero
                    
                    
                    H += H_interpol * H_lambda
            
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
                print("Mode not recognized.")
                return 0

            #commuter term
            start = time.perf_counter()
            HR = H @ R
            commuter_term = numba_commutator(self._n_total,HR)
            stop = time.perf_counter()
            self.commutator_time_numba += stop-start
            

            #compute the decay term
            start = time.perf_counter()
            decay_term = decay_product(self._n_total,self.decay_matrix_diag,R) #number of entries, diagonal form of relaxation matrix, and density matrix
            stop =time.perf_counter()
            self.decay_time += stop-start
        
            #compute the repopulation term
            start = time.perf_counter()
            R_exec = R.diagonal()[self._n_ground : self._n_total]

            br = self.get_interp_array(self.br,(self._n_ground,self._n_exec),B)            
            Rm_diag = br@R_exec

            indices = np.arange(self._n_ground)
            diag_values = Gamma * Rm_diag[:self._n_ground]
            Rm[indices, indices] = diag_values
            stop =time.perf_counter()
            self.repop_time += stop-start
            
            return (commuter_term-decay_term+Rm).ravel()

        
        #extract the max and the min of the interaction time
        tmax = -1e3
        tmin =  1e3
        for E_field in self.E_field:
            if E_field.shape == 'Gaussian':
                t_start = E_field.position - 1.5*E_field.diameter
                t_end   = E_field.position + 1.5*E_field.diameter
            else:
                t_start = E_field.position - 0.6*E_field.diameter
                t_end   = E_field.position + 0.6*E_field.diameter
            if t_start < tmin:
                tmin = t_start
            if t_end > tmax:
                tmax = t_end
        #print(tmin,tmax)
        tinterval = np.linspace(tmin,tmax,npoints)
        
        if package == 'Python':
            print("Solving started.")
            start = time.perf_counter()
            result = solve_ivp(Rdot_python,[tinterval[0],tinterval[-1]],r_init.flatten(),
                            t_eval = tinterval,
                            method = method,
                            max_step = max_step_size,
                            dense_output = False,
                            atol = 1e-6,rtol = 1e-3
                            )
            print("nfev:", result.nfev)
            print(f"ODE solver took : {time.perf_counter() - start} s")
            result = np.array(result.y).T
        
        elif package == 'Julia':

            start = time.perf_counter()
            prob = de.ODEProblem(Rdot_julia,r_init.flatten(),(tinterval[0],tinterval[-1]))

            temp_result = de.solve(prob,de.Tsit5(),reltol=1e-3,abstol=1e-6) #DP5()
            result = temp_result.u
            print(f"Julia solving took {time.perf_counter() - start} s.")
        
        #print(f"Magnetic field final : {B} G.")
        print(f'Time spent on lambdified Hinit symengine= {self.Hinit_symengine :.3f}s')
        print(f'Time spent on interpolated Hinit scipy= {self.Hinit_scipy :.3f}s')
        print(f'Time spent on commutator numba = {self.commutator_time_numba :.3f}s')
        print(f'Time spent on decay = {self.decay_time :.3f}s')
        print(f'Time spent on repopulation = {self.repop_time :.3f}s')
        return result
    

    def interaction_picture_symengine(self):
        """making the interaction Hamiltonian have the time dependence"""

        myHint = []
        #myHint will be constructed as a list of tuple of real and imag part of each of the Hints.
        # Each of the components are lambdify (not interpolant) functions
        

        for count_hint,Hint in enumerate(self.Hint_list):

            if self.max_Hints:
                max_Hint = self.max_Hints[count_hint]
            else:
                max_Hint = None
                max_Hint_ij = max([np.amax(np.abs(Hint[0](self.B0))), np.amax(np.abs(Hint[1](self.B0)))])


            
            #Hint_real , Hint_imag = np.real(Hint),np.imag(Hint) #no need for this now
            
            H_real = se.zeros(self._n_total,self._n_total)
            H_imag = se.zeros(self._n_total,self._n_total)
            for E_field in self.E_field:    
                #Field properties
                t0 = E_field.position
                tsigma =  E_field.diameter/4
                if E_field.shape == 'Gaussian':
                    beam_shape_factor = sp.exp(-(t_se-t0)**2/4/tsigma**2)
                elif E_field.shape == 'Uniform':
                    beam_shape_factor = pulse_se(t_se,t0,tsigma*4)
                else:
                    beam_shape_factor = pulse_se(t_se,t0,tsigma*4)
                
                #Initialize the Hamiltonian
                H_temp_real = se.zeros(self._n_total,self._n_total)
                H_temp_imag = se.zeros(self._n_total,self._n_total)
                
                rabi = E_field.rabi*2*np.pi  #Rabi expressed in angular unit
                idx_ground = self.ground_states.index(E_field.ground_state)
                idx_exec   = self.excited_states.index(E_field.excited_state)
                

                E_res = self._H0_base_diag[self._n_ground+idx_exec]- \
                        self._H0_base_diag[idx_ground]+ \
                        E_field.detuning*2*np.pi #detuning converted to angular unit
                
                
                coeff = 1.0/2*rabi*beam_shape_factor #just calculate it once.
                for i in range(self._n_ground): #index for ground states
                    #list of mF values of the ground states
                    mF_init_list = self._ground_mF_lists[i]
                    for j in range(self._n_ground,self._n_total): #index for excited states. Looking at the upper triangular region only
                        
                        # dont need this line too, or could make some use of it for speeding up the code
                        if max_Hint is None:
                            abs_Hintij = max_Hint_ij
                        else:
                            abs_Hintij = np.abs(max_Hint[i,j])
                        if abs_Hintij == 0 or abs_Hintij < 1e-8:
                            continue
                        
                        
                        E = self._H0_base_diag[j] - self._H0_base_diag[i] #angular
                        
                        eqv_detuning = np.real(E_res - E)
                        
                        if ( eqv_detuning**2 >=  self.test_factor**2 * ( 2 * (rabi *abs_Hintij )**2 + Gamma**2 ) ): #is far from resonance        
                            continue

                        mF_final_list = self._excited_mF_lists[j-self._n_ground]                        
                        dmF_list = [(mF_final - mF_init) for mF_final in mF_final_list for mF_init in mF_init_list]

                        if E_field.pol in dmF_list:
                            pol_multiplier = 1
                        else:
                            pol_multiplier = 0
                            continue
                        
                        #we will be adding the B dependent detunign into the energy level of each of the states
                        phase = eqv_detuning*t_se
                        
                        H_temp_real_ij = coeff * se.cos(phase) 
                        H_temp_imag_ij = coeff * se.sin(phase) 

                        H_temp_real[i,j] =  H_temp_real_ij
                        H_temp_imag[i,j] =  H_temp_imag_ij
                        H_temp_real[j,i] =  H_temp_real_ij
                        H_temp_imag[j,i] = -H_temp_imag_ij
                    
                H_real += H_temp_real
                H_imag += H_temp_imag
            
            #LAMBDIFY THE TEMP HAMILTONIANS HERE
            Hint_real_func = se.Lambdify([t_se], H_real, backend = 'llvm', cse = True)
            Hint_imag_func = se.Lambdify([t_se], H_imag, backend = 'llvm', cse = True) 
            
            #construct myHint such that it is a list with saame size as self.Hint_list
            #each element is a tuple of the real and imag part of the lambdified function containing the time varying part
            myHint.append((Hint_real_func,Hint_imag_func))
        
        return myHint
    