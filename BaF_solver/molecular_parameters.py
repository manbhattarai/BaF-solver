#Constants
uB = 1.399624604;
uN = uB/1836
uE = 1.3996246 # MHz/ (V/cm)
cmIn2MHz = 0.0299792e6


#138

#Ground state parameters

Y01=6491.3962#6491.3946
Y11=-34.8831#-34.8784
Y21=15.93e-3#13.0288e-3
Y02=-5.5250e-3#-5.5248e-3
Y12=-9.43e-6#-9.7632e-6

gamma_00=80.984
gamma_10=-58.4e-3
delta_gamma = 112e-6
BN=Y01+Y11*1/2+Y21*1/4
DN=-Y02-Y12*1/2
gamma=gamma_00+gamma_10*1/2

bBa = 2303.4; cBa = 75.1965; bFBa = bBa + cBa/3;
eq0Q = -143.6812;
bF = 63.509#63.41446; 
cF = 8.224#7.30504; 
bFF = bF + cF/3;
cI = 0

de_sigma = 3.179*uE/2.78
de_pi = 1.50*uE/2.78


#Ground state Zeeman parameters
gS = 2.00197
gI2 = 5.258
gI1 = 0.94/1.5
grot = -0.048
gl = -0.00594


#Excited state parameters
Gamma = 2.7 #In MHz. The program should convert it to angualr frequency as necessary


h_Ba_12 = 206.0
d_Ba = 254
eq0Q1 = -89.0
a_ex = 26.55 #this could be made larger to simulate larger separation betwee th
b_ex = -5.09/2
c_ex = -5.54-b_ex
h_F_12 =a_ex -1/2*(b_ex+c_ex)
d_F = 3.58
A=632.28175*cmIn2MHz
AD = 0.0
p2q=-0.25755*cmIn2MHz
Bex=0.2117414*cmIn2MHz
T00=11946.3168*cmIn2MHz

#Excited state Zeeman parameters
glp = -0.55
gLp = 0.98
