import sympy as sp
from sympy import I
from functools import lru_cache

t_sp=sp.symbols('t',real = sp.true)

import symengine as se
t_se = se.Symbol("t", real=True)

import numpy as np
from numba import njit, complex128, int64

from scipy.integrate import solve_ivp
from scipy.integrate import odeint

from .molecular_parameters import Gamma;
Gamma *= 2*np.pi
import warnings
try:
    from diffeqpy import de,ode
except Exception as e:
    # Fallback to default and warn
    warnings.warn(f"Could not detect diffeqpy. Only Python package for OBE solver available.")

import time


def pulse_se(t_symbol,center,width):
    k = 200
    return 0.5 * (se.tanh(k * (t_symbol-center + width / 2)) - se.tanh(k * (t_symbol-center - width / 2)))

def pulse_sp(t_symbol,center,width):
    k = 200
    return 0.5 * (sp.tanh(k * (t_symbol-center + width / 2)) - sp.tanh(k * (t_symbol-center - width / 2)))

def pulse_np(t_num,center,width):
    k = 200
    return 0.5 * (np.tanh(k * (t_num-center + width / 2)) - np.tanh(k * (t_num-center - width / 2)))
"""
def pulse(t_symbol, center, width):
    k = 100
    sigmoid = lambda x: sp.exp(k * x) / (1 + sp.exp(k * x))
    val = sigmoid(t_symbol - (center - width / 2)) - sigmoid(t_symbol - (center + width / 2))
    return val

def pulse(t_symbol, center, width):
    k = 100

    def fast_sigmoid(x):
        return x / (1 + sp.Abs(x))

    return 0.5 * (fast_sigmoid(k * (t_symbol - center + width / 2)) -
                  fast_sigmoid(k * (t_symbol - center - width / 2)))
"""



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
    def __init__(self,E_field,states,H0,Hint,br,test_factor,mode = 'symengine',Hint_func = None): #transitions added
                
        if isinstance(E_field,Excitation):
            self.E_field = [E_field]
        else:
            self.E_field = E_field
        
        self.ground_states = states[0]
        self.excited_states = states[1]
        self._n_ground = len(self.ground_states)
        self._n_exec = len(self.excited_states)
        self._n_total = self._n_ground + self._n_exec
        
        if type(Hint) is np.ndarray:
                self.Hint_list = [Hint]
        else:
            self.Hint_list = Hint
        
        self.test_factor=test_factor
        self.mode = mode


        self.H0 = 2*np.pi*H0#np.round(2*np.pi*H0,2)
        self._H0_diag = np.diag(self.H0)
        
        #self.U = None
        #self.Ud = None
        #self.generate_unitary_matrices()

        self.Hinit_time = 0
        self.commutator_time_mult = 0
        self.commutator_time_conj_add = 0
        self.commutator_time_numba = 0
        self.commutator_time_numpy = 0
        self.decay_time = 0
        self.repop_time = 0
        self.reshaping = 0
        
        A = np.zeros((self._n_total,self._n_total),dtype = np.complex128)
        np.fill_diagonal(A[self._n_ground:self._n_total, self._n_ground:self._n_total], 1.0)
        self.decay_matrix = Gamma * A 
        self.decay_matrix_diag = np.ascontiguousarray(np.diag(self.decay_matrix))
        
        self.br = br.astype(np.complex128)
        
        self._ground_mF_lists = [[s.mF for s in gs.states] for gs in self.ground_states]
        self._excited_mF_lists = [[s.mF for s in es.states] for es in self.excited_states]
        
        if Hint_func == None:
            #start = time.time()
            if mode == 'symengine':
                self.Hint = self.interaction_picture_symengine()
            elif mode == 'sympy':
                self.Hint = self.interaction_picture_sympy()
            else:
                print("Mode not recognized")
                return 0
        else:
            self.Hint = Hint_func
            
                
        
    def solve(self,npoints,r_init:np.ndarray, max_step_size = 1.0/Gamma, package = 'Python',method = 'RK45'):
        """
            r_init : is defined as .astype(np.complex128)
         """


        @njit(complex128[:, :](int64,complex128[:],complex128[:, :]),cache = True)
        def decay_product(n,G_diag, R):
            S = np.empty((n, n), dtype=np.complex128)
            for i in range(n):
                for j in range(i,n):
                    val = 0.5 * R[i, j] * (G_diag[j] + G_diag[i])
                    S[i, j] = val
                    if i != j:
                        S[j, i] = np.conj(val)
            return S

        @njit(complex128[:, :](int64,complex128[:,:]),cache = True)
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
        
        #S = np.zeros ((self._n_total,self._n_total), dtype=np.complex128)
        #comm = np.zeros ((self._n_total,self._n_total), dtype=np.complex128)
        
        def Rdot_python(T,u):
            

            

            R = u.reshape((self._n_total, self._n_total))
            
            start = time.perf_counter()
            if self.mode == 'symengine':
                #(H_temp_real,H_temp_imag) = self.Hint
                H = self.Hint[0](T)+1.0j*self.Hint[1](T)
                # add the interpolating function of 
            elif self.mode == 'sympy':
                H = self.Hint(T)
            else:
                print("Mode not recognized.")
                return 0
            stop = time.perf_counter()
            self.Hinit_time += (stop-start)

            """
            start = time.perf_counter()
            #compute the commutator term
            H_R = H@R
            #stop = time.perf_counter()
            #self.commutator_time_mult += (stop-start)
            #start = time.perf_counter()
            #R_H = H_R.T.conj()#H_R.conj().T
            commuter_term = -1.0j*(H_R-H_R.T.conj())
            stop = time.perf_counter()
            self.commutator_time_numpy += (stop-start)
            """

            
            start = time.perf_counter()
            HR = H @ R
            commuter_term = numba_commutator(self._n_total,HR)
            stop = time.perf_counter()
            self.commutator_time_numba += (stop-start)  
            

            #compute the decay term
            start = time.perf_counter()
            #numba version is nuch faster than numpy version
            decay_term = decay_product(self._n_total,self.decay_matrix_diag,R) #number of entries, diagonal form of relaxation matrix, and density matrix
            #G_R = self.decay_matrix_diag[:,np.newaxis]*R
            #R_G = G_R.conj().T
            #decay_term = 0.5*(G_R+R_G)
            
            stop = time.perf_counter()
            self.decay_time += (stop-start)

            #compute the repopulation term
            start = time.perf_counter()
            
            #Rm = np.zeros((self._n_total,self._n_total), dtype=np.complex128)
            R_exec = R.diagonal()[self._n_ground : self._n_total]            
            Rm_diag = self.br@R_exec
            indices = np.arange(self._n_ground)
            diag_values = Gamma * Rm_diag[:self._n_ground]
            Rm[indices, indices] = diag_values
            
            stop = time.perf_counter()
            self.repop_time += (stop-start)

            return (commuter_term-decay_term+Rm).ravel()

        def Rdot_julia(u,p,T):
            u = np.array(u)
            R = u.reshape((self._n_total, self._n_total))
            
            start = time.perf_counter()
            if self.mode == 'symengine':
                #(H_temp_real,H_temp_imag) = self.Hint
                H = self.Hint[0](T)+1.0j*self.Hint[1](T)
            elif self.mode == 'sympy':
                H = self.Hint(T)
            else:
                print("Mode not recognized.")
                return 0
            stop = time.perf_counter()
            self.Hinit_time += (stop-start)
            
            start = time.perf_counter()
            HR = H @ R
            commuter_term = numba_commutator(self._n_total,HR)
            #commuter_term = commutator_blas(HR,self._n_total)
            stop = time.perf_counter()
            self.commutator_time_numba += (stop-start)  
            

            #compute the decay term
            start = time.perf_counter()
            decay_term = decay_product(self._n_total,self.decay_matrix_diag,R) #number of entries, diagonal form of relaxation matrix, and density matrix
            stop = time.perf_counter()
            self.decay_time += (stop-start)

            #compute the repopulation term
            start = time.perf_counter()
            R_exec = R.diagonal()[self._n_ground : self._n_total]            
            Rm_diag = self.br@R_exec
            indices = np.arange(self._n_ground)
            diag_values = Gamma * Rm_diag[:self._n_ground]
            Rm[indices, indices] = diag_values
            
            stop = time.perf_counter()
            self.repop_time += (stop-start)

            
            return_val = (commuter_term-decay_term+Rm).flatten()
           
            return list(return_val)
            
                

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
        tinterval = np.linspace(tmin,tmax,npoints)
        
        if package == 'Python':
            print("Solving started.")
            start = time.perf_counter()
            result = solve_ivp(Rdot_python,[tinterval[0],tinterval[-1]],r_init.flatten(),
                            t_eval = tinterval,
                            method = method,
                            max_step = max_step_size,
                            dense_output = False,
                            atol = 1e-7,rtol = 1e-4
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
        
        print(f'Time spent on lambdified Hinit = {self.Hinit_time :.3f}s')
        print(f'Time spent on commutator numba = {self.commutator_time_numba :.3f}s')
        print(f'Time spent on decay = {self.decay_time :.3f}s')
        print(f'Time spent on repopulation = {self.repop_time :.3f}s')
        return result
    

    def solve_numba(self,npoints,r_init:np.ndarray, max_step_size = 1.0/Gamma, package = 'Python',method = 'RK45'):
        
        @njit(complex128[:, :](int64,complex128[:],complex128[:, :]),cache = True)
        def decay_product(n,G_diag, R):
            S = np.empty((n, n), dtype=np.complex128)
            #S = np.empty_like(R)
            for i in range(n):
                for j in range(i,n):
                    val = 0.5 * R[i, j] * (G_diag[j] + G_diag[i])
                    S[i, j] = val
                    if i != j:
                        S[j, i] = np.conj(val)
            return S

        @njit(complex128[:, :](int64,complex128[:,:]),cache = True)
        def numba_commutator(N,HR):
            comm = np.empty((N, N), dtype=np.complex128)
            for i in range(N):
                for j in range( i,N):
                    val = -1j * (HR[i, j] - np.conj(HR[j, i]))
                    comm[i, j] = val
                    
                    if i != j:
                        comm[j, i] = np.conj(val)
            return comm
        

           
        Rm = np.zeros((self._n_total,self._n_total), dtype=np.complex128)
        #S = np.zeros ((self._n_total,self._n_total), dtype=np.complex128)
        #comm = np.zeros ((self._n_total,self._n_total), dtype=np.complex128)
        
        n_total = self._n_total
        n_ground = self._n_ground
        n_exec = self._n_exec
        Hint_real = self.Hint[0]
        Hint_imag = self.Hint[0]
        decay_matrix_diag = self.decay_matrix_diag
        br = self.br
        def H_time(T):
            return Hint_real(T)+1.0j*Hint_imag(T)

        @njit()
        def Rdot_python(T,u):

            R = u.reshape((n_total,n_total))
            H = H_time(T)
            

            HR =H@R
            commuter_term = numba_commutator(n_total,HR)

            decay_term = decay_product(n_total,decay_matrix_diag,R) 

            R_exec = R.diagonal()[n_ground : n_total]            
            Rm_diag = br@R_exec
            indices = np.arange(n_ground)
            diag_values = Gamma * Rm_diag[:n_ground]
            Rm[indices, indices] = diag_values
            

            return (commuter_term-decay_term+Rm).ravel()

        #extract the max and the min of the interaction time
        start = time.perf_counter()
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
            
        tinterval = np.linspace(tmin,tmax,npoints)
        stop = time.perf_counter()
        print(f"Extracting time interval took : {stop - start :.3f} s.")

        #start = time.time()
        if package == 'Python':
            print("Solving started.")
            start = time.perf_counter()
            result = solve_ivp(Rdot_python,[tinterval[0],tinterval[-1]],r_init.flatten(),
                            t_eval = tinterval,
                            method = method,
                            max_step = max_step_size,
                            #atol = 1e-7,rtol = 1e-4
                            )
            print(f"ODE solver took : {time.perf_counter() - start} s")
            result = np.array(result.y).T
        
            
        return result


    def interaction_picture_symengine(self):
        """making the interaction Hamiltonian have the time dependence"""

        myHint_real = se.zeros(self._n_total,self._n_total)
        myHint_imag = se.zeros(self._n_total,self._n_total)


        cum_Htemp_construction = 0
        start_Hint_construction = time.perf_counter()
        

        for Hint in self.Hint_list:
            
            Hint_real , Hint_imag = np.real(Hint),np.imag(Hint)
            
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
                
                rabi = E_field.rabi*2*np.pi #Rabi expressed in angular unit
                idx_ground = self.ground_states.index(E_field.ground_state)
                idx_exec = self.excited_states.index(E_field.excited_state)
                
                E_res = self.H0[self._n_ground+idx_exec,self._n_ground+idx_exec]- \
                        self.H0[idx_ground,idx_ground]+ \
                        E_field.detuning*2*np.pi #angular unit
                
                
                coeff = 1.0/2*rabi*beam_shape_factor #just calculate it once.
                for i in range(self._n_ground): #index for ground states
                    #list of mF values of the ground states
                    mF_init_list = self._ground_mF_lists[i]
                    for j in range(self._n_ground,self._n_total): #index for excited states. Looking at the upper triangular region only
                        
                        abs_Hintij = np.abs(Hint[i,j])
                        if abs_Hintij == 0 or abs_Hintij < 1e-8:
                            continue
                        
                        E = self._H0_diag[j] - self._H0_diag[i] #angular
                        
                        eqv_detuning = np.real(E_res - E)
                        if ( eqv_detuning**2 >=  self.test_factor**2 * ( 2 * (rabi * abs_Hintij )**2 + Gamma**2 ) ): #is far from resonance                            
                            continue
                        

                        #Introduce pol multiplier that multiplies by rabi frequency of the correct polarization only
                        #In case there is transvere fields, states of different mF values are mixed. This necesssiates checkecking if any state in the superpostion have the same mF values
                        #List of mF values
                        
                        #mF_init_list = [self.ground_states[i].states[kk].mF for kk in range(len(self.ground_states[i].states))]
                        #mF_final_list =[self.excited_states[j-self._n_ground].states[kk].mF for kk in range(len(self.excited_states[j-self._n_ground].states))]
                        mF_final_list = self._excited_mF_lists[j-self._n_ground]                        
                        dmF_list = [(mF_final - mF_init) for mF_final in mF_final_list for mF_init in mF_init_list]

                        if E_field.pol in dmF_list:
                            pol_multiplier = 1#True#1
                        else:
                            pol_multiplier = 0#False#0
                            continue

                        
                        Hint_ij_real = Hint_real[i,j]
                        Hint_ij_imag = Hint_imag[i,j]
                        
                        phase = eqv_detuning*t_se
                        
                        cos_phase = se.cos(phase)
                        sin_phase = se.sin(phase)
                        H_temp_real_ij = coeff * ( Hint_ij_real * cos_phase - Hint_ij_imag * sin_phase )
                        H_temp_imag_ij = coeff * ( Hint_ij_imag * cos_phase + Hint_ij_real * sin_phase )

                        H_temp_real[i,j] =  H_temp_real_ij
                        H_temp_imag[i,j] =  H_temp_imag_ij
                        H_temp_real[j,i] =  H_temp_real_ij
                        H_temp_imag[j,i] = -H_temp_imag_ij

                #end_loop = time.time()
                #cum_loop += end_loop-start_loop
                
                myHint_real += H_temp_real
                myHint_imag += H_temp_imag
        #myHint_real = Matrix(myHint_real)
        #myHint_imag = Matrix(myHint_imag)
        stop_Hint_construction = time.perf_counter()
        cum_Htemp_construction += stop_Hint_construction-start_Hint_construction
        #print(f"Cum sp : {cum_spMatrix}, cum Eres = {cum_Eres}, cum_loop: {cum_loop}, cum_det : {cum_det_check}, cum mF : {cum_mF_check}, cum Hiint : {cum_Htemp_construction}")
        print(f"Hamiltonian construction : {cum_Htemp_construction}")
        
        
        start = time.perf_counter()
        
        Hint_real_func = se.Lambdify([t_se], myHint_real,backend = 'llvm',cse = True)
        lap = time.perf_counter()
        print(f"First Lambdify took {lap - start:.4f} s")
        Hint_imag_func = se.Lambdify([t_se], myHint_imag,backend = 'llvm',cse = True) 
        
        print(f"Second Lambdify took {time.perf_counter() - lap:.4f} s")
        return (Hint_real_func, Hint_imag_func)
        

    def interaction_picture_sympy(self):
        """making the interaction Hamiltonian have the time dependence"""
        
        N_pols = len(self.Hint_list)
        #print(N_pols)
        myHint = sp.Matrix(np.zeros(np.shape(self.Hint_list[0])))

        #cum_spMatrix = 0
        #cum_Eres = 0
        #cum_loop = 0
        #cum_det_check = 0
        #cum_Htemp_construction = 0
        #cum_mF_check = 0
        for Hint in self.Hint_list:
            for idx_field_temp,E_field in enumerate(self.E_field):
                #start_field = time.time()
                #Field properties
                t0 = E_field.position
                tsigma =  E_field.diameter/4
                if E_field.shape == 'Gaussian':
                    beam_shape_factor = sp.exp(-(t_sp-t0)**2/4/tsigma**2)
                elif E_field.shape == 'Uniform':
                    beam_shape_factor = pulse_sp(t_sp,t0,tsigma*4)
                else:
                    beam_shape_factor = pulse_sp(t_sp,t0,tsigma*4)
                
                #Initialize the Hamiltonian
                H_temp = sp.zeros(self._n_total,self._n_total)#np.copy(Hint)#sp.Matrix(Hint)
                

                #stop_sp_matrix = time.time()
                #cum_spMatrix +=stop_sp_matrix-start_field

                #start_Eres = time.time()
                rabi = np.round(E_field.rabi*2*np.pi,1) #Rabi expressed in angular unit
                idx_ground = self.ground_states.index(E_field.ground_state)
                idx_exec = self.excited_states.index(E_field.excited_state)
                
                E_res = self.H0[self._n_ground+idx_exec,self._n_ground+idx_exec]- \
                        self.H0[idx_ground,idx_ground]+ \
                        E_field.detuning*2*np.pi #angular unit
                #stop_Eres = time.time()
                #cum_Eres += stop_Eres - start_Eres
                #start_loop = time.time()
                
                for i in range(self._n_ground): #index for ground states
                    for j in range(self._n_ground,self._n_total): #index for excited states. Looking at the upper triangular region only
                        
                        if Hint[i,j] == 0:
                            continue
                        
                        #start_det_check = time.time()
                        E = self.H0[j,j] - self.H0[i,i] #angular
                        eqv_detuning = E_res - E
                        if (np.abs(eqv_detuning) >=  self.test_factor*(2*(rabi*np.abs(Hint[i,j]))**2+Gamma**2)**0.5):
                            isNearResonant = 0#False
                            #print(self.test_factor*(2*(rabi*np.abs(Hint[i,j]))**2+Gamma**2)**0.5/2/np.pi)
                            continue
                        else:
                            isNearResonant = 1#True
                        #stop_det_check = time.time()
                        #cum_det_check += stop_det_check-start_det_check

                        #Introduce pol multiplier that multiplies by rabi frequency of the correct polarization only
                        #In case there is transvere fields, states of different mF values are mixed. This necesssiates checkecking if any state in the superpostion have the same mF values
                        #List of mF values
                        
                        #start_mF_check = time.time()
                        mF_init_list = [self.ground_states[i].states[kk].mF for kk in range(len(self.ground_states[i].states))]
                        mF_final_list =[self.excited_states[j-self._n_ground].states[kk].mF for kk in range(len(self.excited_states[j-self._n_ground].states))]
                        dmF_list = [(mF_final - mF_init) for mF_final in mF_final_list for mF_init in mF_init_list]
                        if E_field.pol in dmF_list:
                            pol_multiplier = 1#True#1
                        else:
                            pol_multiplier = 0#False#0
                            continue
                        #stop_mF_check = time.time()
                        #cum_mF_check +=stop_mF_check-start_mF_check
                        
                        
                        #start_Hint_construction = time.time()


                        coeff = Hint[i,j]*1.0/2*rabi*beam_shape_factor
                        coeff_conj = Hint[j,i]*1.0/2*rabi*beam_shape_factor
                        phase = eqv_detuning*t_sp

                        H_temp[i,j] = coeff*sp.exp(I*phase) #Emission

                        H_temp[j,i] = coeff_conj*sp.exp(-I*phase) #Absorption

                        
                        """
                        if (isNearResonant and pol_multiplier):
                                H_temp[i,j] = Hint[i,j]*1.0/2*rabi*sp.exp(I*eqv_detuning*t)* \
                                                beam_shape_factor       #Emission

                                H_temp[j,i] = Hint[j,i]*1.0/2*rabi*sp.exp(-I*eqv_detuning*t)* \
                                                beam_shape_factor      #Absorption
                        """
                        #stop_Hint_construction = time.time()
                        #cum_Htemp_construction +=stop_Hint_construction-start_Hint_construction
                #end_loop = time.time()
                #cum_loop += end_loop-start_loop
                       
                myHint += H_temp
        
        #print(f"Cum sp : {cum_spMatrix}, cum Eres = {cum_Eres}, cum_loop: {cum_loop}, cum_det : {cum_det_check}, cum mF : {cum_mF_check}, cum Hiint : {cum_Htemp_construction}")
        
        print('Symbolic Hamiltonian created.')
        start = time.time()
        myHint = sp.lambdify(t_sp,myHint,['numpy'],cse = True)
        print(f"Lambdify took {time.time() - start} s.")
        print("Lambdified Hamiltonian returned.")
        return myHint

   


    def interaction_picture_symengine_optimized(self):
        """Optimized version: making the interaction Hamiltonian have the time dependence"""

        myHint_real = sp.Matrix.zeros(self._n_total, self._n_total)
        myHint_imag = sp.Matrix.zeros(self._n_total, self._n_total)

        # Pre-calculate mF lists for all states ONCE if they don't change
        # Assuming state objects have a structure like: state.states = [substate1, substate2]
        # and substate has .mF attribute
        ground_mF_lists = [
            [s.mF for s in gs.states] for gs in self.ground_states
        ]
        excited_mF_lists = [
            [s.mF for s in es.states] for es in self.excited_states
        ]
        # Pre-calculate diagonal elements of H0
        H0_diag = [self.H0[k,k] for k in range(self._n_total)]


        cum_Htemp_construction = 0
        start_Hint_construction = time.perf_counter()

        

        for Hint in self.Hint_list:
            for E_field in self.E_field:
                # Field properties - Calculated once per E_field
                t0 = E_field.position
                tsigma = E_field.diameter / 4
                if E_field.shape == 'Gaussian':
                    beam_shape_factor = se.exp(-(t_se - t0)**2 / 4 / tsigma**2)
                elif E_field.shape == 'Uniform':
                     # Ensure pulse_se returns a symbolic expression if needed pre-lambdify
                    beam_shape_factor = pulse_se(t_se, t0, tsigma * 4)
                else:
                    # Default or handle other shapes
                    beam_shape_factor = pulse_se(t_se, t0, tsigma * 4) # Assuming default

                rabi = E_field.rabi * 2 * np.pi # Angular unit
                # Calculate E_res once per E_field
                # Need indices corresponding to the specific states involved in *this* E_field transition
                # This assumes E_field object *knows* its ground/excited states directly
                # If E_field only stores labels, you need to find indices like before.
                # Assuming E_field has `ground_state_label` and `excited_state_label`
                # and you have a way to map these labels to indices i and j (or specific sub-levels)
                # For simplicity, let's reuse the original indexing method if mapping isn't direct.
                try:
                    # Find the *specific* indices involved in this E_field transition
                    # This might need adjustment based on how E_field stores state info
                    idx_ground_field = self.ground_states.index(E_field.ground_state) # Assumes E_field.ground_state is one of the objects in self.ground_states
                    idx_exec_field = self.excited_states.index(E_field.excited_state)   # Assumes E_field.excited_state is one of the objects in self.excited_states
                    E_res = H0_diag[self._n_ground + idx_exec_field] - \
                            H0_diag[idx_ground_field] + \
                            E_field.detuning * 2 * np.pi # Angular unit
                except ValueError:
                    print(f"Warning: Could not find E_field states {E_field.ground_state} or {E_field.excited_state} in state lists.")
                    continue # Skip this E_field if states aren't found


                # Initialize the temporary Hamiltonian inside E_field loop
                H_temp_real = sp.Matrix.zeros(self._n_total, self._n_total)
                H_temp_imag = sp.Matrix.zeros(self._n_total, self._n_total)

                coeff_base = 0.5 * rabi * beam_shape_factor # Calculate symbolic part once

                for i in range(self._n_ground): # Index for ground states
                    # Get pre-calculated mF list
                    mF_init_list = ground_mF_lists[i]
                    H0_ii = H0_diag[i] # Get pre-calculated diagonal element

                    for j in range(self._n_ground, self._n_total): # Index for excited states
                        Hint_ij = Hint[i, j] # Access matrix element once

                        # Early exit if element is zero
                        if Hint_ij == 0:
                            continue

                        # Get pre-calculated mF list and H0 element
                        # Index for excited_states list is j - n_ground
                        mF_final_list = excited_mF_lists[j - self._n_ground]
                        H0_jj = H0_diag[j] # Get pre-calculated diagonal element

                        E = H0_jj - H0_ii # Angular frequency difference
                        eqv_detuning = np.real(E_res - E) # Use np.real for numerical part

                        # Resonance Check (using pre-calculated Hint_ij magnitude if possible)
                        # Note: np.abs() on a symbolic Hint_ij might not work as expected if complex symbolic.
                        # Assuming Hint elements are numerical or symengine handles abs appropriately.
                        # If Hint elements are complex numbers (not symbolic), calculate abs_Hint_ij numerically.
                        abs_Hint_ij = abs(Hint_ij) # Use Python's abs() which works on complex numbers
                        resonance_threshold_sq = 2 * (rabi * abs_Hint_ij)**2 + Gamma**2
                        if (eqv_detuning**2 >= (self.test_factor**2) * resonance_threshold_sq): # Compare squares to avoid sqrt
                            continue

                        # Optimized mF / Polarization Check
                        pol_match_found = False
                        target_dmF = E_field.pol
                        for mF_final in mF_final_list:
                            for mF_init in mF_init_list:
                                if (mF_final - mF_init) == target_dmF:
                                    pol_match_found = True
                                    break # Exit inner mF loop
                            if pol_match_found:
                                break # Exit outer mF loop

                        if not pol_match_found:
                            continue # Skip if no matching polarization transition found

                        # Calculate symbolic Hamiltonian contribution
                        # Assuming Hint[i,j] elements are complex numbers (not symbolic expressions)
                        # If Hint[i,j] CAN be symbolic, you need se.re, se.im
                        Hint_ij_real = np.real(Hint_ij) # Use np.real/imag if Hint elements are numerical
                        Hint_ij_imag = np.imag(Hint_ij)

                        phase = eqv_detuning * t_se # Symbolic phase
                        cos_phase = se.cos(phase)
                        sin_phase = se.sin(phase)

                        # Combine coefficient calculation
                        # coeff_base is symbolic (due to beam_shape_factor)
                        # Hint_ij parts are numerical constants
                        H_temp_real_ij = coeff_base * ( Hint_ij_real * cos_phase - Hint_ij_imag * sin_phase )
                        H_temp_imag_ij = coeff_base * ( Hint_ij_imag * cos_phase + Hint_ij_real * sin_phase )

                        # Assign to temporary matrices
                        H_temp_real[i, j] = H_temp_real_ij
                        H_temp_imag[i, j] = H_temp_imag_ij

                        # Assign conjugate transpose part
                        H_temp_real[j, i] =  H_temp_real_ij # Real part is the same
                        H_temp_imag[j, i] = -H_temp_imag_ij # Imaginary part flips sign

                # Accumulate results for this E_field
                # This matrix addition might be a bottleneck if matrices are large
                myHint_real += H_temp_real
                myHint_imag += H_temp_imag

        stop_Hint_construction = time.perf_counter()
        cum_Htemp_construction = stop_Hint_construction - start_Hint_construction
        print(f"Optimized Hamiltonian construction : {cum_Htemp_construction:.4f} s")

        # Lambdify remains the same, but should be faster as the expressions might be simpler
        # due to numerical constants being folded in earlier where possible.
        start_lambdify = time.perf_counter()
        Hint_real_func = se.Lambdify([t_se], myHint_real, cse=True)
        lap_lambdify = time.perf_counter()
        print(f"First Lambdify took {lap_lambdify - start_lambdify:.4f} s")
        Hint_imag_func = se.Lambdify([t_se], myHint_imag, cse=True)
        end_lambdify = time.perf_counter()
        print(f"Second Lambdify took {end_lambdify - lap_lambdify:.4f} s")

        return (Hint_real_func, Hint_imag_func)

if __name__ == '__main__':
    print("Obe package test run.")