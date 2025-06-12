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


        self.H0 = 2*np.pi*H0
        self._H0_diag = np.diag(self.H0)


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

            
            start = time.perf_counter()
            HR = H @ R
            commuter_term = numba_commutator(self._n_total,HR)
            stop = time.perf_counter()
            self.commutator_time_numba += (stop-start)  
            

            #compute the decay term
            start = time.perf_counter()
            decay_term = decay_product(self._n_total,self.decay_matrix_diag,R) #number of entries, diagonal form of relaxation matrix, and density matrix
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
                        if ( eqv_detuning**2 >=  self.test_factor**2 * ( 2 * (rabi * abs_Hintij/abs_Hintij )**2 + Gamma**2 ) ): #is far from resonance                            
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

                
                
                myHint_real += H_temp_real
                myHint_imag += H_temp_imag
        
        stop_Hint_construction = time.perf_counter()
        cum_Htemp_construction += stop_Hint_construction-start_Hint_construction
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


if __name__ == '__main__':
    print("Obe package test run.")