#Standart imports
import numpy as np
import scipy
import warnings
from joblib import Parallel, delayed
import time


#Local import
from .spin_params import *
from .hamiltonian import H_int
from .states import SigmaLevel, PiLevelParity
from .SigmaHamiltonian import *
from .PiHamiltonian import *


class System():
    """ What does the system class contain!"""
    def __init__(self,N_sigma,J_pi,B_field = [0,0,0],ignore_mF = False):
        """
        J_pi is a string or a list of string where each element is of the form 'J-' or 'J+' where 
        J is the total angular momentum and the sign following is the parity
        """ 
        if type(N_sigma) == list:
            self.N_list = N_sigma
        else:
            self.N_list = [N_sigma]
        
        if type(J_pi) == list:
            self.J_list = J_pi
        else:
            self.J_list = [J_pi]
        
        self.B_field = B_field
        self.sigma_states = []
        self.pi_states = []
        self.F_plus_sigma_all = []
        self.F_plus_pi_all = []
        self.generate_sigma_states(ignore_mF = ignore_mF)
        self.generate_pi_states(ignore_mF = ignore_mF)
        

        self.sigma_Hamiltonian = SigmaHamiltonian(self.sigma_states,self.F_plus_sigma_all,self.B_field)
        self.pi_Hamiltonian = PiHamiltonian(self.pi_states,self.F_plus_pi_all,self.B_field)
        
        self.interaction_Hamiltonian = None
        self.branching_ratios = None
        #self.amu = amu
    
    def generate_sigma_states(self,ignore_mF = False):
        """Use ignore_mF = True to generatte states without mFs"""
        for N in self.N_list:
            for G in np.arange(np.abs(I1-S),I1+S+1):
                for F1 in np.arange(np.abs(N-G),N+G+1):
                    for F in np.arange(np.abs(F1-I2),F1+I2+1):
                        if ignore_mF:
                            self.sigma_states.append(SigmaLevel(G,N,F1,F))
                        else:
                            #construct the rotation matrix here for 
                            self.F_plus_sigma_all.append(self.create_F_plus(F))
                            for mF in np.arange(-F,F+1):
                                self.sigma_states.append(SigmaLevel(G,N,F1,F,mF))
    
    @staticmethod
    def create_F_plus(F):
        MF = np.arange(-F,F+1)
        F_plus = np.zeros((len(MF),len(MF)))
        for i,mF in enumerate(MF):
            try:
                F_plus[i+1,i] = np.sqrt(F*(F+1)-mF*(mF+1))
            except:
                continue
        return F_plus
    
    
    def generate_pi_states(self,ignore_mF = False):
        """Use ignore_mF = True to generatte states without mFs"""
        for J_str in self.J_list:
            if J_str[-1] == '+':
                parity = 1
            elif J_str[-1] == '-':
                parity = -1
            J = eval(J_str[:-1])
            for F1 in np.arange(np.abs(J-I1),J+I1+1):
                for F in np.arange(np.abs(F1-I2),F1+I2+1):
                    if ignore_mF:
                        self.pi_states.append(PiLevelParity(parity,J,F1,F))
                    else:
                        #construct the rotation matrix here for 
                        self.F_plus_pi_all.append(self.create_F_plus(F))
                        for mF in np.arange(-F,F+1):
                            self.pi_states.append(PiLevelParity(parity,J,F1,F,mF))
                        

    
    @staticmethod
    def generate_interaction_matrix(state1:list,state2:list,pol):
        num_1 = len(state1)
        num_2 = len(state2)
        Htemp = Parallel(n_jobs = -1)(delayed(H_int)(state1[m],state2[n],pol) for m in range(num_1) for n in range(num_2))        
        Hmat = np.array(Htemp).reshape(num_1,num_2)
        return Hmat
    
    
    def generate_interaction_Hamiltonian(self,state1:list,state2:list,pol = 0):
        num_1 = len(state1)
        num_2 = len(state2)
        Hint = np.zeros((num_1+num_2,num_1+num_2),dtype=np.complex128)
        
        Htemp = Parallel(n_jobs = -1)(delayed(H_int)(state1[m],state2[n],pol) for m in range(num_1) for n in range(num_2))
        for m in range(num_1):
            for n in range(num_2):
                Hint[m,num_1+n] = Htemp[0]
                Hint[num_1+n,m] = np.conj(Hint[m,num_1+n])
                Htemp.pop(0)
        #Hint = Hint + np.conj(Hint.T)
        self.interaction_Hamiltonian = Hint
        
    def generate_branching_ratios(self,ground_state:list,excited_state:list):
        start = time.perf_counter()
        Trans_z = np.abs(self.generate_interaction_matrix(ground_state,excited_state,pol=0))**2
        stop_pi = time.perf_counter()
        print(f"Pi branching took : {stop_pi-start} sec")

        Trans_sigma_plus = np.abs(self.generate_interaction_matrix(ground_state,excited_state,pol=1))**2 
        stop_sigmaplus = time.perf_counter()
        print(f"Sigma+ branching took : {stop_sigmaplus-stop_pi} sec")

        Trans_sigma_minus = np.abs(self.generate_interaction_matrix(ground_state,excited_state,pol=-1))**2
        stop_sigmaminus = time.perf_counter()
        print(f"Sigma- branching took : {stop_sigmaminus-stop_sigmaplus} sec")

        Trans_tot = Trans_z+Trans_sigma_plus+Trans_sigma_minus
        sum_over_ground = np.sum(Trans_tot,axis=0) #sum over the rows
        rows,cols = np.shape(Trans_tot)
        for i in range(cols):
            Trans_tot[:,i] /= sum_over_ground[i]
        self.branching_ratios = Trans_tot