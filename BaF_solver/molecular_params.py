from dataclasses import dataclass
from .constants import *

@dataclass(frozen=True)
class Params:
    BN: float
    DN: float
    gamma: float
    delta_gamma: float
    bBa : float
    cBa : float
    eq0Q : float
    bF : float
    cF : float
    cI : float
    gS : float
    gI1 : float
    gI2 : float
    grot : float
    gl : float
    A : float
    AD : float
    p2q : float
    Bex : float
    T00 : float
    a_ex : float
    b_ex : float
    c_ex : float
    d_F : float
    h_Ba_12 : float
    d_Ba : float
    eq0Q1 : float
    h_F_12 : float
    glp : float
    gLp : float
    de_sigma : float
    de_pi : float

    @property
    def bFBa(self):
        return self.bBa + self.cBa/3
    @property
    def h_F_12(self):
        return (self.a_ex-1/2*(self.b_ex+self.c_ex))
    @property
    def A(self):
        return self.A_cmIN*cmIn2MHz
    @property
    def AD(self):
        return self.AD_cmIN*cmIn2MHz
    @property
    def p2q(self):
        return self.p2q_cmIN*cmIn2MHz
    @property
    def Bex(self):
        return self.Bex_cmIN*cmIn2MHz  
    @property
    def T00(self):
        return self.T00_cmIN*cmIn2MHz  
    

params_138 = Params(
    BN=6479.67249,
    DN=5.53483e-3,
    gamma=80.9605,
    delta_gamma=0,
    bBa=2303.4,
    cBa=75.1965,
    eq0Q=-143.6812,
    bF=63.41446,
    cF=7.30504,
    cI=0,
    gS=2.002,
    gI1=0.937365 / 1.5,
    gI2=5.258,
    grot=-0.048,
    gl=-0.028,
    A_cmIn=632.28165,
    AD_cmIn=0.0310 * 1e-3,
    p2q_cmIn=-0.2578,
    Bex_cmIn=0.21189575,
    T00_cmIn=11946.316291675,
    a_ex=26.55,
    b_ex=-0.2303,
    c_ex=-5.3094,
    d_F=3.58,
    h_Ba_12=206.7,
    d_Ba=254.3,
    eq0Q1=-89.1,
    glp=-0.536,
    gLp=0.98,
    de_sigma=3.179 * uE / 2.78,
    de_pi=1.50 * uE / 2.78,
    
)


params_137 = Params(
    BN=6479.67249,
    DN=5.53483e-3,
    gamma=80.9605,
    delta_gamma=0,
    bBa=2303.4,
    cBa=75.1965,
    eq0Q=-143.6812,
    bF=63.41446,
    cF=7.30504,
    cI=0,
    gS=2.002,
    gI1=0.937365 / 1.5,
    gI2=5.258,
    grot=-0.048,
    gl=-0.028,
    A_cmIn=632.28165,
    AD_cmIn=0.0310 * 1e-3,
    p2q_cmIn=-0.2578,
    Bex_cmIn=0.21189575,
    T00_cmIn=11946.316291675,
    a_ex=26.55,
    b_ex=-0.2303,
    c_ex=-5.3094,
    d_F=3.58,
    h_Ba_12=206.7,
    d_Ba=254.3,
    eq0Q1=-89.1,
    glp=-0.536,
    gLp=0.98,
    de_sigma=3.179 * uE / 2.78,
    de_pi=1.50 * uE / 2.78,
    
)

params_136 = Params(
    BN=6479.67249,
    DN=5.53483e-3,
    gamma=80.9605,
    delta_gamma=0,
    bBa=2303.4,
    cBa=75.1965,
    eq0Q=-143.6812,
    bF=63.41446,
    cF=7.30504,
    cI=0,
    gS=2.002,
    gI1=0.937365 / 1.5,
    gI2=5.258,
    grot=-0.048,
    gl=-0.028,
    A_cmIn=632.28165,
    AD_cmIn=0.0310 * 1e-3,
    p2q_cmIn=-0.2578,
    Bex_cmIn=0.21189575,
    T00_cmIn=11946.316291675,
    a_ex=26.55,
    b_ex=-0.2303,
    c_ex=-5.3094,
    d_F=3.58,
    h_Ba_12=206.7,
    d_Ba=254.3,
    eq0Q1=-89.1,
    glp=-0.536,
    gLp=0.98,
    de_sigma=3.179 * uE / 2.78,
    de_pi=1.50 * uE / 2.78,
    
)

params_135 = Params(
    BN=6479.67249,
    DN=5.53483e-3,
    gamma=80.9605,
    delta_gamma=0,
    bBa=2303.4,
    cBa=75.1965,
    eq0Q=-143.6812,
    bF=63.41446,
    cF=7.30504,
    cI=0,
    gS=2.002,
    gI1=0.937365 / 1.5,
    gI2=5.258,
    grot=-0.048,
    gl=-0.028,
    A_cmIn=632.28165,
    AD_cmIn=0.0310 * 1e-3,
    p2q_cmIn=-0.2578,
    Bex_cmIn=0.21189575,
    T00_cmIn=11946.316291675,
    a_ex=26.55,
    b_ex=-0.2303,
    c_ex=-5.3094,
    d_F=3.58,
    h_Ba_12=206.7,
    d_Ba=254.3,
    eq0Q1=-89.1,
    glp=-0.536,
    gLp=0.98,
    de_sigma=3.179 * uE / 2.78,
    de_pi=1.50 * uE / 2.78,
    
)
