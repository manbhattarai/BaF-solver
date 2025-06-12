import numpy as np
from .spin_params import S,LAMBDA,SIGMA,OMEGA,I1,I2
from .fast_wigners import wigner_6j
class SigmaLevel:
    def __init__(self,G,N,F1,F,mF=None):
        self.G = G
        self.N = N
        self.F1 = F1
        self.F = F
        self.mF = mF    
        
    def __repr__(self):
        return f"|G = {self.G}, N = {self.N}, F1 = {self.F1}, F = {self.F}, mF = {self.mF}>"
    def __str__(self):
        return f"|G = {self.G}, N = {self.N}, F1 = {self.F1}, F = {self.F}, mF = {self.mF}>"
    
    def __eq__(self,other):
        if isinstance(other,SigmaLevel):
            return ((self.N == other.N) and 
                    (self.G == other.G) and
                    (self.F1 == other.F1) and
                    (self.F == other.F) and
                    (self.mF == other.mF)
                   )
        return False

    def __hash__(self):
        return hash((self.G, self.N,self.F1,self.F,self.mF))    
    
    def GtoJ(self) -> 'Superposition':
        #Define the J variable
        J_list = np.arange(np.abs(self.N-S),self.N+S+1)
        ampl_list = []
        state_list = []
        for J in J_list:
            ampl = (2*J+1)**0.5*\
                    (2*self.G+1)**0.5*\
                    wigner_6j(self.F1,self.G,self.N,S,J,I1)*\
                    (-1)**(self.G+S+I1)
            state = SigmaLevel_J(self.N,J,self.F1,self.F,self.mF)
            
            ampl_list.append(ampl)
            state_list.append(state)
        return Superposition(ampl_list,state_list)   

    """
    def to_uncoupled_basis(self) -> 'Superposition':
        #Define the J variable
        J_list = np.arange(np.abs(self.N-S),self.N+S+1)
        ampl_list = []
        state_list = []
        for J in J_list:
            ampl = (2*J+1)**0.5*\
                    (2*self.G+1)**0.5*\
                    wigner_6j(self.F1,self.G,self.N,S,J,I1)*\
                    (-1)**(self.G+S+I1)
            state = SigmaLevel_J(self.N,J,self.F1,self.F,self.mF)
            
            ampl_list.append(ampl)
            state_list.append(state)
        return Superposition(ampl_list,state_list)
    """
        

class SigmaLevel_J:
    def __init__(self,N,J,F1,F,mF=None):
        self.N = N
        self.J = J
        self.F1 = F1
        self.F = F
        self.mF = mF

    def __eq__(self,other):
        if isinstance(other,SigmaLevel_J):
            return ((self.N == other.N) and 
                    (self.J == other.J) and
                    (self.F1 == other.F1) and
                    (self.F == other.F) and
                    (self.mF == other.mF)
                   )
        return False

    def __hash__(self):
        return hash((self.N,self.J,self.F1,self.F,self.mF))
        
    def __repr__(self):
        return f"|N = {self.N}, J = {self.J}, F1 = {self.F1}, F = {self.F}, mF = {self.mF}>"
    def __str__(self):
        return f"|N = {self.N}, J = {self.J}, F1 = {self.F1}, F = {self.F}, mF = {self.mF}>"


"""
class SigmaLevel_uncoupled:
    def __init__(self,N,mS,mN,mI1,mI2):
        self.N = N
        self.mS = mS
        self.mN = mN
        self.mI1 = mI1
        self.mI2 = mI2

    def __eq__(self,other):
        if isinstance(other,SigmaLevel_uncoupled):
            return ((self.N == other.N) and 
                    (self.mS == other.mS) and
                    (self.mN == other.mN) and
                    (self.mI1 == other.mI1) and
                    (self.mI2 == other.mI2)
                   )
        return False
        
    def __repr__(self):
        return f"|S = {S}, mS = {self.mS},I1 = {I1}, mI1 = {self.mI1}, N = {self.N}, mN = {self.mN}, I2 = {I2}, mI2 = {self.mI2}>"
    def __str__(self):
        return f"|S = {S}, mS = {self.mS},I1 = {I1}, mI1 = {self.mI1}, N = {self.N}, mN = {self.mN}, I2 = {I2}, mI2 = {self.mI2}>"

    
"""


class PiLevelParity:
    def __init__(self,parity,J,F1,F,mF=None):
        self.J = J
        self.F1 = F1
        self.F = F
        self.mF = mF
        self.parity = parity
          
        
    def __str__(self):
        if self.parity == 1:
            return f"|J = {self.J}+, F1 = {self.F1}, F = {self.F}, mF = {self.mF}>"
        elif self.parity == -1:
            return f"|J = {self.J}-, F1 = {self.F1}, F = {self.F}, mF = {self.mF}>"
        
    def __repr__(self):
        if self.parity == 1:
            return f"|J = {self.J}+, F1 = {self.F1}, F = {self.F}, mF = {self.mF}>"
        elif self.parity == -1:
            return f"|J = {self.J}-, F1 = {self.F1}, F = {self.F}, mF = {self.mF}>"


    def __eq__(self,other):
        if isinstance(other,PiLevelParity):
            return ((self.parity == other.parity) and 
                    (self.J == other.J) and
                    (self.F1 == other.F1) and
                    (self.F == other.F) and
                    (self.mF == other.mF)
                   )
        return False

    def __hash__(self):
        return hash((self.J,self.F1,self.F,self.mF,self.parity))
        
    def parity_to_omega(self):
        state1 = PiLevelOmega(LAMBDA,SIGMA,OMEGA,self)
        state2 = PiLevelOmega(-LAMBDA,-SIGMA,-OMEGA,self)
        return Superposition([1/np.sqrt(2),1/np.sqrt(2)*self.parity*(-1)**(self.J-S)],[state1,state2]) 
        
        

class PiLevelOmega:
    def __init__(self,Lambda,Sigma,Omega,parity_state:PiLevelParity):
        self.Lambda = Lambda
        self.Sigma = Sigma
        self.Omega = Omega
        self.parity_state = parity_state
    
    def __hash__(self):
        return hash((self.Lambda,self.Sigma,self.Omega,self.parity_state))

    def __repr__(self):
        return f"|LAMBDA = {self.Lambda}, SIGMA = {self.Sigma}, OMEGA = {self.Omega} "+ \
                f"; J = {self.parity_state.J}, F1 = {self.parity_state.F1},F = {self.parity_state.F}, mF = {self.parity_state.mF}>"
    def __str__(self):
        return f"|LAMBDA = {self.Lambda}, SIGMA = {self.Sigma}, OMEGA = {self.Omega} "+ \
                f"; J = {self.parity_state.J}, F1 = {self.parity_state.F1},F = {self.parity_state.F}, mF = {self.parity_state.mF}>"
        
        
    
        
class Superposition:
    """Coefficients is a list of coeffieicients and states is a list of the SigmaLevel or PiLevel objects"""
    def __init__(self,amplitude:list,states:list):
        self.amplitude = amplitude
        self.states = states
    
    def __str__(self):
        val=""
        for i,amp_val in enumerate(self.amplitude):
            if i == len(self.amplitude)-1 and np.abs(amp_val)>=1e-2:
                val += str(np.round(amp_val,5)) + ' ' + str(self.states[i])
            else:
                val += str(np.round(amp_val,5)) + ' ' + str(self.states[i]) + ' + \n'
        return val
    
    def __repr__(self):
        val=""
        for i,amp_val in enumerate(self.amplitude):
            if i == len(self.amplitude)-1 and np.abs(amp_val)>=1e-2:
                val += str(np.round(amp_val,5)) + ' ' + str(self.states[i])
            else:
                val += str(np.round(amp_val,5)) + ' ' + str(self.states[i]) + ' + \n'
        return val


    def __eq__(self,other):
        if isinstance(other,Superposition):
            return ((self.amplitude == other.amplitude) and 
                    (self.states == other.states)
                   )
        return False

    def __hash__(self):
        return hash((tuple(self.amplitude),tuple(self.states)))

    
    
    def GtoJ(self):
        #Check if the object contains the states in SigmaLevel basis 
        if not isinstance(self.states[0],SigmaLevel):
            print(f"Cannot convert from {type(self.states[0])} to G basis.")
            return -1

        #Distill out the states. Each of the states are SigmaLevel.
        states = self.states
        amplitudes = self.amplitude

        superposition_list = []
        for ampl,st in zip(amplitudes,states):
            sup_J = st.GtoJ() #This is a superposition of J states 
            amp_J = sup_J.amplitude
            for idx in range(len(amp_J)):
                amp_J[idx] *= ampl
            superposition_list.append(sup_J)


        #make it into a single superposition
        amps_new_superposition_list = []
        states_new_superposition_list = []
        for item in superposition_list:
            amps_ = item.amplitude
            st_ = item.states
            for i in range(len(amps_)):
                #check for duplicate entry
                #is item[1][i] already in the list of states_new_superposition_list
                try:                    
                    index = states_new_superposition_list.index(st_[i])
                    #state found states_new_superposition_list at index = index 
                    #add the amplitude to the present amplitude
                    amps_new_superposition_list[index] += amps_[i]
                except ValueError:
                    if np.abs(amps_[i]) >= 1e-5:
                        amps_new_superposition_list.append(amps_[i])
                        states_new_superposition_list.append(st_[i])

        return Superposition(amps_new_superposition_list,states_new_superposition_list)

        
            
            
            
            
        
        
