#Constants
uB = 1.399624604;
uN = uB/1836
cmIn2MHz = 0.0299792458e6

#137
#Ground state
BN = 6479.67249; DN = 5.53483e-3;
#BN=0.21614*cmIn2MHz#632.28165*cmIn2MHz
gamma = 80.9605;
delta_gamma=0
bBa = 2303.4; cBa = 75.1965; bFBa = bBa + cBa/3;
eq0Q = -143.6812;
bF = 63.41446; cF = 7.30504; bFF = bF + cF/3;
cI = 0
#
gS = 2.002; gI2 = 5.258; gI1 = 0.937365/1.5; grot = -0.048; gl = -0.028 ;


#Excited state
Gamma = 2.7 #In MHz. The program should convert it to angualr frequency as necessary
A = 632.28165*cmIn2MHz#632.28165*cmIn2MHz
AD = 0.0310*1e-3*cmIn2MHz
p2q=-0.2578*cmIn2MHz#-0.2578*cmIn2MHz

a_ex = 26.55 #this could be made larger to simulate larger separation betwee th
b_ex = -0.2303
c_ex = -5.3094
h_F_12 = (a_ex-1/2*(b_ex+c_ex))
d_F = 3.58*1

h_Ba_12  = 206.7
d_Ba  = 254.3
eq0Q1 = -89.1

Bex=0.21189575*cmIn2MHz#0.21189575*cmIn2MHz
T00=11946.316291675*cmIn2MHz#11946.316325*cmIn2MHz

glp = -0.536 ; gLp = 0.98;

