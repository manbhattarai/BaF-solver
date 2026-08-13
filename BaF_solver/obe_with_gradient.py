from functools import lru_cache

import sympy as sp
from sympy import I
t_sp=sp.symbols('t',real = sp.true)

import symengine as se
t_se = se.Symbol("t", real=True)

import numpy as np
from numba import njit, complex128, int64, prange

import scipy
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

from .molecular_parameters_137 import Gamma;
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

def approx_form(t_num,sigma):
    return (np.exp(-(t_num-2*sigma)/2/sigma**2) + np.exp(-(t_num-6*sigma)/2/sigma**2) + np.exp(-(t_num-8*sigma)/2/sigma**2))*np.sqrt(8/np.pi)

def unipolar(t_symbol,center,sigma):
    return 1/se.cosh((t_symbol-center)/sigma)

def static_gaussian(t_symbol,center,sigma):
    return se.exp(-(t_symbol-center)**2/sigma**2)

def bipolar(t_symbol,center,sigma,x):
    return 1/2.0*x/sigma*1/se.cosh((t_symbol-center)/sigma)*se.tanh((t_symbol-center)/sigma)

def single_sinusoidal(t_symbol,center,width):
    freq = 1/width
    return se.sin(2*se.pi*freq*t_symbol)*pulse_sp(t_symbol,center,width)





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
     
class Static_Excitation():
    """
    E_amplitude : specified as magnitude of electric field in units of V/cm
    pol : direction of the electric field
    position : position of the center. Depends on the shape of the field
    diameter : spatial extent of the electric field. Depends on the shape of the field
    shape : Shape as defined by strings as 
            "Unipolar",
            "Bipolar",
            "Sinusoidal",
            "Pulse"    
    """
    def __init__(self, E_amplitude:float, pol:int, position = None, diameter = None, shape = None):
        self.E_amplitude = E_amplitude #expressed in V/cm
        self.pol = pol
        self.position = position
        self.diameter = diameter
        self.shape    = shape
    
    def __repr__(self):
        return f"E_amplitude = {self.E_amplitude}, pol = {self.pol}, position = {self.position}, diameter = {self.diameter}, shape = "+ self.shape
    def __str__(self):
        return f"E_amplitude = {self.E_amplitude} V/cm, pol = {self.pol}, position = {self.position}, diameter = {self.diameter}, shape = "+ self.shape
     

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

    """
    def __init__(self,E_field,
                    states,  #these are the states corresponding to initial B value
                    H0,      # interpolation function
                    Hint,    #interpolation function or list of function
                    
                    br,      #interpolation function
                    
                    test_factor,
                    mode = 'symengine',
                    #B_field, # a tuple (B0,grad)
                    #E_stat_field,
                    #Hstatic_int, #interpolation function; encapsulates static electric field interaction
                    **kwargs): 

        # Optical field checks        
        if isinstance(E_field,Excitation):
            self.E_field = [E_field]
        else:
            self.E_field = E_field

        #Assessing ground and excited states
        self.ground_states = states[0]
        self.excited_states = states[1]
        self._n_ground = len(self.ground_states)
        self._n_exec = len(self.excited_states)
        self._n_total = self._n_ground + self._n_exec
                
        self._ground_mF_lists = [[s.mF for s in gs.states] for gs in self.ground_states]
        self._excited_mF_lists = [[s.mF for s in es.states] for es in self.excited_states]

        self.br = br # interpolating function


        #Interaction Hamiltonian check. 
        #Checks for tuple to separate out the real and imaginary terms.
        #These are broken into real and imag part in a tuple for faster interpolation computation
        if isinstance(Hint,tuple):
            self.Hint_list = [Hint]
        else:
            self.Hint_list = Hint

        self.test_factor=test_factor
        self.mode = mode

        


        #Read the kwargs
        #The allowed kwargs are
        #a. B_field : sent as a tuple or as a list. The first element is the initial amg field, second element is the gradient of the field in terms of microsecond.
        #b.Hstatic_int : the interpolation Hamiltonian function for a static electric field. Unlike the magnetic field, this is not used to diagonalize the system hamiltonian but acts as a interaction term
        #c. E_stat_field : Static_Excitation class, containing static electric field information.

        self.__dict__.update(kwargs)
        self.overall_envelope = kwargs.get('overall_envelope', None)


        B_field = kwargs.get('B_field', (0,0))
        self.B0 = B_field[0]  
        self.grad = B_field[1] 
        
        E_stat_field = kwargs.get('E_stat_field', None)
        Hstatic_int = kwargs.get('Hstatic_int', None)

        self.H0 = H0 # interpolating function. Array of size _n_total x _n_total
        self._H0_base = 2*np.pi*self.get_interp_array(H0,(self._n_total,self._n_total),self.B0,real_imag = False)
        self._H0_base_diag = np.diag(self._H0_base)


        if isinstance(E_stat_field,Static_Excitation):
            self.E_stat_field = [E_stat_field]
        else:
            self.E_stat_field = E_stat_field

        if E_stat_field:
            self.H_static_multiplier = self.electric_static_multiplier()
        else:
            self.H_static_multiplier = [lambda var: 0,lambda var: 0]

         

        

        self.Hinit_scipy = 0
        self.Hinit_symengine = 0

        self.scipy_symengine_multiplication = 0
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
        
        
        
        
        if mode == 'symengine':
            self.Hint = self.interaction_picture_symengine()
            #print(f"Type : {type(self.Hint)}, len : {len(self.Hint)}.")
        elif mode == 'sympy':
            print('Sympy not available with gradient calculating solve. Switching to symengine mode.')
            self.Hint = self.interaction_picture_symengine()
            #self.Hint = self.interaction_picture_sympy()
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
          
        
    def solve(self,npoints,r_init:np.ndarray, max_step_size = 1.0/Gamma, package = 'Python',method = 'RK45',pass_number = 1):

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
            
            B = self.B0+self.grad*T
            #print(T,",",(B - self.B0)*2*1.399)
            
            if self.mode == 'symengine':
                #(H_temp_real,H_temp_imag) = self.Hint
                start = time.perf_counter()
                #2*np.pi*self.get_interp_array(H0,(self._n_total,self._n_total),self.B0,real_imag = False)
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
                    Hint_lambda_real, Hint_lambda_imag = self.Hint[count]
                    H_lambda = Hint_lambda_real(T) + 1.0j * Hint_lambda_imag(T)
                    # change here to incorporate multiple passes
                    if self.overall_envelope:
                        single_sigma = self.overall_envelope.params['single_sigma']
                        overall_sigma = self.overall_envelope.params['sigma']
                        #H_lambda *= approx_form(T + sigma_single * pass_number,self.overall_sigma)

                        H_lambda *= self.overall_envelope.func(T + 4*single_sigma * pass_number, overall_sigma,pass_number)
                        #print(T + 4*single_sigma * pass_number,end = ',')
                    self.Hinit_symengine += stop-start


                    #check if any is zero
                    
                    start = time.perf_counter()
                    H += H_interpol * H_lambda
                    stop = time.perf_counter()
                    self.scipy_symengine_multiplication += stop - start
            
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
                print("Mode not recognized.")
                return 0


            #Extract the static electric field interaction
            #H_static_electric is a real Hamiltonian 
            if self.Hstatic_int:
                H_static_electric_interpol = 2*np.pi*self.get_interp_array(self.Hstatic_int,(self._n_total,self._n_total),B)
            else:
                H_static_electric_interpol = 0

            #print(f"[{np.round(T,3)},  {np.round(H_static_electric[3,4],5)}]")
            H_static_electric_lambda = self.H_static_multiplier[0](T) +1.0j * self.H_static_multiplier[1](T)
            H_static_electric = H_static_electric_interpol*H_static_electric_lambda
            #print(H_static_electric[3:4+1,3:4+1])
            H += H_static_electric    
            #print(H[3:4+1,3:4+1])
            #print("-----")

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

            BR = self.get_interp_array(self.br,(self._n_ground,self._n_exec),B)            
            Rm_diag = BR@R_exec

            indices = np.arange(self._n_ground)
            diag_values = Gamma * Rm_diag[:self._n_ground]
            Rm[indices, indices] = diag_values
            stop =time.perf_counter()
            self.repop_time += stop-start
            
            return (commuter_term-decay_term+Rm).ravel()

        
        #extract the max and the min of the interaction time
        tmax = -1e3
        tmin =  1e3
        #Also need to consider the static electric fields as they are applied
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
            #print("Solving started.")
            start = time.perf_counter()
            result = solve_ivp(Rdot_python,[tinterval[0],tinterval[-1]],r_init.flatten(),
                            t_eval = tinterval,
                            method = method,
                            max_step = max_step_size,
                            dense_output = False,
                            atol = 1e-6,rtol = 1e-4
                            )
            #print("nfev:", result.nfev)
            #print(f"ODE solver took : {time.perf_counter() - start} s")
            result = np.array(result.y).T
            #print("")
        
        
        #print(f"Magnetic field final : {B} G.")
        if 1 == 0:
            print(f'Time spent on lambdified Hinit symengine= {self.Hinit_symengine :.3f}s')
            print(f'Time spent on interpolated Hinit scipy= {self.Hinit_scipy :.3f}s')
            print(f'Time spent on interpolated Hinit scipy symengine multiplication = {self.scipy_symengine_multiplication :.3f}s')
            
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
                        
                        if (i == 3) and (j == 25):
                            #print("Printing")
                            print(f"Eqv det = {eqv_detuning/2/np.pi}")
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

    
    def electric_static_multiplier(self):
        start_static = time.time()
        coeff_field_shape =[]
        coeff_field_real = []
        for static_field in self.E_stat_field:
            if static_field.shape == 'Unipolar':
                center = static_field.position
                sigma = static_field.diameter/4
                if np.isreal(static_field.E_amplitude):
                    ampl = static_field.E_amplitude
                else:
                    ampl = static_field.E_amplitude.imag

                
                beam_shape_factor = ampl*unipolar(t_se,center,sigma)
            elif static_field.shape == 'Bipolar':
                center = static_field.position
                sigma = static_field.diameter/4
                v_beam = 616e-4
                x = 1/v_beam
                if np.isreal(static_field.E_amplitude):
                    ampl = static_field.E_amplitude
                else:
                    ampl = static_field.E_amplitude.imag
                beam_shape_factor = ampl*bipolar(t_se,center,sigma,x)
            elif static_field.shape == 'Static_Gaussian':
                center = static_field.position
                sigma = static_field.diameter/4
                if np.isreal(static_field.E_amplitude):
                    ampl = static_field.E_amplitude
                else:
                    ampl = static_field.E_amplitude.imag
                beam_shape_factor = ampl*static_gaussian(t_se,center,sigma)
            elif static_field.shape == 'Sinusoidal':
                center = static_field.position
                width = static_field.diameter
                if np.isreal(static_field.E_amplitude):
                    ampl = static_field.E_amplitude
                else:
                    ampl = static_field.E_amplitude.imag
                beam_shape_factor = ampl*single_sinusoidal(t_se,center,width)
            elif static_field.shape == 'Pulse':
                center = static_field.position
                dia = static_field.diameter
                if np.isreal(static_field.E_amplitude):
                    ampl = static_field.E_amplitude
                else:
                    ampl = static_field.E_amplitude.imag
                beam_shape_factor = ampl*pulse_se(t_se,center,dia)
            
            coeff_field_shape.append(beam_shape_factor)
            if np.isreal(static_field.E_amplitude):
                coeff_field_real.append(1)
            else:
                coeff_field_real.append(0)

        H_temp_real = se.zeros(self._n_total,self._n_total)
        H_temp_imag = se.zeros(self._n_total,self._n_total)

        for kk,item in enumerate(coeff_field_shape):
            for i in range(self._n_ground):
                for j in range(i+1,self._n_ground):
                    if self.max_Hstatic[i,j] != 0:
                        delta = self._H0_base_diag[i] - self._H0_base_diag[j] #GH[i,i]- GH[j,j]
                        phase = delta*t_se
                        if np.abs(delta) > 2*np.pi*200.0:
                            continue
                        # H[i,j] = se.exp(I*t*delta)
                        
                        if coeff_field_real[kk] == 1:
                            H_temp_real[i,j] +=  coeff_field_shape[kk] * se.cos(phase)
                            H_temp_real[j,i] +=  coeff_field_shape[kk] * se.cos(phase)
                            H_temp_imag[i,j] +=  coeff_field_shape[kk] * se.sin(phase)
                            H_temp_imag[j,i] += -coeff_field_shape[kk] * se.sin(phase)
                        else:
                            # exp(1j*phase)
                            H_temp_real[i,j] +=  -coeff_field_shape[kk] * se.sin(phase)
                            H_temp_real[j,i] +=  coeff_field_shape[kk] *  se.sin(phase)
                            H_temp_imag[i,j] +=  coeff_field_shape[kk] * se.cos(phase)
                            H_temp_imag[j,i] +=  coeff_field_shape[kk] * se.cos(phase)
        

        H_real_func = se.Lambdify([t_se], H_temp_real, backend = 'llvm', cse = True)
        H_imag_func = se.Lambdify([t_se], H_temp_imag, backend = 'llvm', cse = True)
        stop_static = time.time()
        #print(f"Static electric hamiltonian took {(stop_static - start_static) :.3f} s")
        return (H_real_func,H_imag_func)
        