#Importing the standard libraries
import numpy as np
import copy

#Importing the solver modules
import .system
from .obe import obe 
from .states import SigmaLevel,PiLevelParity
from .obe import Excitation


def field_optimizer(Bz,rabi,detuning,level_to_optimize):
    b=system.System([0,2],['1/2-','3/2-','5/2-'],B_field = [0.0,0.0,Bz],ignore_mF = False)

    b.sigma_Hamiltonian.generate_bare()
    b.sigma_Hamiltonian.Zeeman.generate_Zeeman()
    b.pi_Hamiltonian.generate_bare()
    b.pi_Hamiltonian.Zeeman.generate_Zeeman()
    b.sigma_Hamiltonian.diagonalize()
    b.pi_Hamiltonian.diagonalize()
    G_global = b.sigma_Hamiltonian.diagonalized_states
    GH_global = np.round(b.sigma_Hamiltonian.diagonalized_Hamiltonian,3)
    E_global =b.pi_Hamiltonian.diagonalized_states
    EH_global = np.round(b.pi_Hamiltonian.diagonalized_Hamiltonian,3)
    
    G = G_global
    E = E_global[0:16]
    GH = GH_global
    EH = EH_global[0:16,0:16]
    
    b.generate_branching_ratios(G,E)
    BR = b.branching_ratios
    

    GH -= np.amin(np.diag(GH))*np.identity(np.shape(GH)[0])
    EH -= np.amin(np.diag(EH))*np.identity(np.shape(EH)[0])

    H0 = np.zeros((len(G)+len(E),len(G)+len(E)),dtype=np.complex128)
    H0[0:len(G),0:len(G)] = GH
    H0[len(G):,len(G):] = EH
    b.generate_interaction_Hamiltonian(G,E)
    Hint_1=np.round(b.interaction_Hamiltonian,3)
    n0 = [1/len(G)]*len(G)+[0]*len(E)

    b.generate_interaction_Hamiltonian(G,E,pol=+1)
    Hint_2 = np.round(b.interaction_Hamiltonian,3)

    b.generate_interaction_Hamiltonian(G,E,pol=-1)
    Hint_3 = np.round(b.interaction_Hamiltonian,3)

    print(f"mF = {G[level_to_optimize].states[0].mF}")

    myList_1 = []
    myList_2 = []
    for i in range(0,len(G)):
        if i != level_to_optimize:
            tempList_1 = []
            tempList_2 = []
            for j in range(len(E)):
                #look at the Hint element for z pol light
                if np.abs(Hint_1[i,len(G)+j])> 1e-3:
                    tempList_1.append((i,j))
                if np.abs(Hint_2[i,len(G)+j])> 1e-3 or np.abs(Hint_3[i,len(G)+j])> 1e-3:
                    tempList_2.append((i,j))
                
            myList_1.append(tempList_1)
            myList_2.append(tempList_2)



    Gamma = 2*np.pi*2.7
    tsigma = 8.192/4
    transitions_Z = []
    transitions_X = []
    test_factor = 20
    vib_branch = 0.96
    obe_mode = 'symengine'

    def r22_single_Z(field):
        steps=50
        n=len(E)+len(G)
        r_init = np.zeros(np.shape(H0)).astype(np.complex128)
        for i in range(len(G)):
            r_init[i,i] = 1.0/len(G)
        
        Hint_Z = None
        my_obe_1 = obe(field,[G,E],H0,Hint_1,vib_branch*BR,test_factor,mode = obe_mode,Hint_func = Hint_Z)
        Hint_Z = my_obe_1.Hint
        ans = my_obe_1.solve(steps,r_init,max_step_size = 1/Gamma,package='Python')
        rho = np.array(ans[-1]) #gives the solution at the end of the time
        r_init = rho.reshape(n,n)
        r_level2optimize=np.real(r_init[level_to_optimize,level_to_optimize])
        return r_level2optimize #the new version of ax by default maximizes

    def r22_single_X(field):
        X_field = []
        X_field.append(field)
        field_pol_reversed = copy.copy(field)
        field_pol_reversed.pol *= -1
        X_field.append(field_pol_reversed)
        steps=50
        n=len(E)+len(G)
        r_init = np.zeros(np.shape(H0)).astype(np.complex128)
        for i in range(len(G)):
            r_init[i,i] = 1.0/len(G)
        
        Hint_X = None
        my_obe_2 = obe(X_field,[G,E],H0,[Hint_2,Hint_3],vib_branch*BR,test_factor,mode = obe_mode,Hint_func =Hint_X)
        Hint_X = my_obe_2.Hint
        ans = my_obe_2.solve(steps,r_init,max_step_size = 1/Gamma,package='Python')
        rho = np.array(ans[-1]) #gives the solution at the end of the time
        r_init = rho.reshape(n,n)
        r_level2optimize=np.real(r_init[level_to_optimize,level_to_optimize])
        return r_level2optimize
        
    
    sum_pop_z = 0
    for item in myList_1:
        if not item:
            continue
        max_val = -1
        
        for (i,j) in item:
            pol = 0
            #print((i,j),end = ",")
            groundState = G[i]
            excitedState = E[j]
            pos = 0
            dia = 4*tsigma
            temp_field = Excitation(rabi,pol,groundState,excitedState,detuning = detuning,position = pos,diameter = dia ,shape = "Uniform")
            
            pop = r22_single_Z(temp_field)
            if pop>max_val:
                max_val = pop
                choice = (i,j)
            #print(choice)
            #print((i,j),":",pop)
        
        sum_pop_z = sum_pop_z + (max_val- 1/len(G)) # the quantity in the bracket is the gain in population in level to optimize
        transitions_Z.append(choice)
        #print(f"Z added {choice}.")
    #print(np.round(sum_pop_z,3))
    #print("Transition_Z")
    #print(transitions_Z)

    sum_pop_x = 0
    for item in myList_2:
        if not item:
            continue
        max_val = 0
        
        for (i,j) in item:
            groundState = G[i]
            excitedState = E[j]
            pol = groundState.states[0].mF - excitedState.states[0].mF
            pos = 0
            dia = 4*tsigma
            temp_field = Excitation(1/np.sqrt(2)*rabi,pol,groundState,excitedState,detuning = detuning,position = pos,diameter = dia ,shape = "Uniform")
            
            pop = r22_single_X(temp_field)
            if pop>max_val:
                max_val = pop
                choice = (i,j)
            #print((i,j),":",pop)
        
        sum_pop_x = sum_pop_x + max_val - 1/len(G)
        transitions_X.append(choice)
        #print(f"X added {choice}.")
    #print(np.round(sum_pop_x,3))
    #print("transition_x")
    #print(transitions_X)
    #print(f"Sum : {np.round(sum_pop_z+sum_pop_x,2)}")  
    return sum_pop_z,sum_pop_x, transitions_Z,transitions_X   
            
