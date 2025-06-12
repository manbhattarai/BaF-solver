import numpy as np
import scipy
import time

from .hamiltonian import H0_sigma, HZeeman_sigma
from .states import SigmaLevel, Superposition


class SigmaHamiltonian():
    def __init__(self,states,F_plus,B_field):
        self.bare = []
        self.Zeeman = self.Zeeman(states,F_plus,B_field) 
        self.states = states
        self.diagonalized_states = []
        self.diagonalized_Hamiltonian = None
        self.B_field = B_field
        self.diagonalized_states_as_vectors = []
        #self.species = species
    
    class Zeeman():
        def __init__(self,states,F_plus,B_field):
            self.Nlevels = len(states)
            self.states = states
            self.B_field = B_field
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
            HZ = np.zeros((self.Nlevels,self.Nlevels))
            for row in range(self.Nlevels):
                for col in range(row+1):
                    temp_val = HZeeman_sigma(self.states[row],self.states[col])
                    HZ[row,col] = temp_val
                    if row != col:
                        HZ[col,row] = np.conjugate(temp_val) #########################conjugated
            self.Z = HZ
            
            if self.B_field[0] != 0:
                #generate the X Hamiltonian
                # X is obtained by rotating the Z Hamiltonian by pi/2 about the y axis
                Ux = scipy.linalg.expm(-1j*self.F_y*np.pi/2)
                Uxd = scipy.linalg.expm(1j*self.F_y*np.pi/2)
                self.X = Ux@self.Z@Uxd
            else:
                self.X = np.zeros_like(HZ)
            
            if self.B_field[1] != 0:
                #Y is obtained by rotating the Z Hamiltonian by -pi/2 about the x axis
                Uy = scipy.linalg.expm(1j*self.F_x*np.pi/2)
                Uyd = scipy.linalg.expm(-1j*self.F_x*np.pi/2)
                self.Y = Uy@self.Z@Uyd
            else:
                self.Y = np.zeros_like(HZ)
            
            
    
    def generate_bare(self):
        num = len(self.states)
        H0 = np.zeros((num,num),dtype=np.complex128)
        for row in range(num):
            for col in range(row+1):
                temp_val = H0_sigma(self.states[row],self.states[col])
                H0[row,col] = temp_val
                if row != col:
                    H0[col,row] = np.conjugate(temp_val) # conjugate necessary here. Possibly not because the static hamiltonian does not have any complex terms.
        self.bare = H0
        
        
    
        
    def diagonalize(self):
        if len(self.bare) == 0 and len(self.Zeeman.Z) == 0:
            raise ValueError("Error. Hamiltonian not generated.")
        if len(self.Zeeman.Z) == 0 and len(self.bare) != 0:
            H_temp = self.bare
        if len(self.Zeeman.Z) != 0 and len(self.bare) != 0:
            H_temp = self.bare + self.B_field[0]*self.Zeeman.X+ \
                                    self.B_field[1]*self.Zeeman.Y+ \
                                    self.B_field[2]*self.Zeeman.Z
        #print(H_temp)
        w,v = np.linalg.eigh(H_temp) #columns of v are the eigenvectors
        #print(w)
        sort_idx = np.argsort(w)
        #print(v)
        w,v = w[sort_idx],v[:,sort_idx]

        #Verify that v is a unitary matrix
        """
        _temp = np.conjugate(v.transpose())@v

        identity_matrix = np.eye(H_temp.shape[0])

        print(f"Unitarity check : {np.allclose(_temp, identity_matrix)}")
        """
        self.diagonalized_Hamiltonian = np.conjugate(v.transpose())@H_temp@v
        w=np.round(w,6)
        
        self.diagonalized_states_as_vectors = v#[v[:,i] for i in range(v.shape[0])]
        
        #print(w)
        #to represent the diagonalized states
        for i in range(len(w)):
            v_temp = np.round(v[:,i],5)###############################################################
            #v_temp = v[:,i]
            non_zero_idx = np.nonzero(v_temp)[0]
            amp = []
            st = []
            for idx in non_zero_idx:
                amp.append(v_temp[idx])
                st.append(self.states[idx])
            self.diagonalized_states.append(Superposition(amp,st))