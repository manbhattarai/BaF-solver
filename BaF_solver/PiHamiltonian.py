import numpy as np
import scipy
import time

from .hamiltonian import H0_pi_parity_basis,HZeeman_pi_parity_basis
from .states import PiLevelParity, Superposition
from .molecular_parameters import T00



class PiHamiltonian():
    def __init__(self,states,F_plus,B_field):
        self.bare = []
        #print("F_plus", F_plus)
        self.Zeeman = self.Zeeman(states,F_plus,B_field)
        self.states = states
        self.diagonalized_states = []
        self.diagonalized_Hamiltonian = None
        self.B_field = B_field
        self.diagonalized_states_as_vectors = []
    
    class Zeeman():
        def __init__(self,states,F_plus,B_field):
            self.Nlevels = len(states)
            self.states = states
            self.B_field = B_field
            #print(self.states)
            
            self.F_plus,self.F_minus = self.create_ladder_operators(F_plus)
            
            self.F_x = 1/2*(self.F_plus+self.F_minus)
            self.F_y = -1j/2*(self.F_plus-self.F_minus)
            self.X,self.Y,self.Z =[],[],[]
            #self.generate_Zeeman()
            
            
        
        def create_ladder_operators(self,F_plus):
            A_plus = np.zeros((self.Nlevels,self.Nlevels))
            current_index = 0
            for i,sub_F_plus in enumerate(F_plus):
                m = np.shape(sub_F_plus)[0] #sub_F_plus is a numpy array of dim 2F+1 x 2F+1
                A_plus[current_index:current_index+m,current_index:current_index+m] = sub_F_plus
                current_index += m
            A_minus = np.transpose(A_plus)
            return A_plus,A_minus
        
        
        def generate_Zeeman(self):
            #check if the states have been generated with the mF levels
            if self.states[0].mF == None:
                raise ValueError("Error. Cannot generate Zeeman Hamiltonian. States generated without magnetic sublevels")
            
            #generate the Z hamiltonian first
            HZ = np.empty((self.Nlevels,self.Nlevels),dtype = np.complex128)
            
            for row in range(self.Nlevels):
                for col in range(row+1):
                    temp_val = HZeeman_pi_parity_basis(self.states[row],self.states[col])
                    HZ[row,col] = temp_val
                    if row != col:
                        HZ[col,row] = temp_val.conj() ##############################################conjugated!

            self.Z = HZ

            if self.B_field[0] != 0:
                #generate the X Hamiltonian
                # X is obtained by rotating the Z Hamiltonian by pi/2 about the y axis
                print("Pi Zeeman X was built.")
                Ux = scipy.linalg.expm(-1j*self.F_y*np.pi/2)
                self.X = Ux@self.Z@(Ux.conj().T)#np.transpose(np.conjugate(Ux))
            else:
                self.X = np.zeros_like(HZ)
            
            if self.B_field[1] != 0:
                #generate the Y Hamiltonian
                #Y is obtained by rotating the Z Hamiltonian by -pi/2 about the x axis
                print("Pi Zeeman Y was built.")
                Uy = scipy.linalg.expm(1j*self.F_x*np.pi/2)
                self.Y = Uy@self.Z@(Uy.conj().T)#np.transpose(np.conjugate(Uy))
            else:
                self.Y = np.zeros_like(HZ)
            
    
    def generate_bare(self):
        num = len(self.states)
        #check if the states have been generated with the mF levels
        H0 = np.zeros((num,num),dtype=np.complex128)
        for row in range(num):
            for col in range(row+1):
                H0[row,col] = H0_pi_parity_basis(self.states[row],self.states[col])
                if row != col:
                    H0[col,row] = H0[row,col]
        self.bare = H0  + T00*np.identity(np.shape(H0)[0])
    
        
    def diagonalize(self):
        if len(self.bare) == 0 and len(self.Zeeman.Z) == 0:
            raise ValueError("Error. Hamiltonian not generated.")
        if len(self.Zeeman.Z) == 0 and len(self.bare) != 0:
            H_temp = self.bare
        if len(self.Zeeman.Z) != 0 and len(self.bare) != 0:
            H_temp = self.bare + self.B_field[0]*self.Zeeman.X+ \
                                    self.B_field[1]*self.Zeeman.Y+ \
                                    self.B_field[2]*self.Zeeman.Z
        
        w,v = np.linalg.eigh(H_temp) #columns of v are the eigenvectors
        sort_idx = np.argsort(w)
        w,v = w[sort_idx],v[:,sort_idx]
        self.diagonalized_Hamiltonian = np.conjugate(v.transpose())@H_temp@v
        w=np.round(w,6)
        #v=np.round(v,6)
        self.diagonalized_states_as_vectors = v #[v[:,i] for i in range(v.shape[0])]
        
        
        #to represent the diagonalized states
        for i in range(len(w)):
            v_temp = np.round(v[:,i],5) #############################################################
            #v_temp = v[:,i]
            non_zero_idx = np.nonzero(v_temp)[0]
            amp = []
            st = []
            for idx in non_zero_idx:
                amp.append(v_temp[idx])
                st.append(self.states[idx])
            self.diagonalized_states.append(Superposition(amp,st))