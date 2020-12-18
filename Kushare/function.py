
#import functions and modules:
    
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt
import math
from pcec_init import pars, ptr

#Constants:
F = 96485 #C/mol
R = 8.313 #J/mol*k

def residual(t, SV):
    dSV_dt = np.empty_like(SV) 
    
    #----- Anode-----
    dphi_PNO = SV[ptr.dphi_int_PNO_0]
   
    #Mass Action Equations For Charge Transfer:
    
    k_fwd_PNO_H = pars.k_fwd_star_PNO_H*math.exp(-pars.beta_an*F*pars.n_PNO_H*dphi_PNO/R/pars.T)
    k_rev_PNO_H = pars.k_rev_star_PNO_O*math.exp(-pars.beta_an*F*pars.n_PNO_H*dphi_PNO/R/pars.T)
    k_fwd_PNO_O = pars.k_fwd_star_PNO_O*math.exp(-pars.beta_an*F*pars.n_PNO_O*dphi_PNO/R/pars.T)
    k_rev_PNO_O = pars.k_rev_star_PNO_O*math.exp((1-pars.beta_an)*F*pars.n_PNO_O*dphi_PNO/R/pars.T)
    i_Far_PNO_O = pars.n_PNO_O*F*(k_fwd_PNO_O*pars.conc_fwd_PNO_O-k_rev_PNO_O*pars.conc_rev_PNO_O)
    i_Far_PNO_H = pars.n_PNO_H*F*(k_fwd_PNO_H*pars.conc_fwd_PNO_H-k_rev_PNO_H*pars.conc_rev_PNO_H)
    i_Far_PNO = i_Far_PNO_O + i_Far_PNO_H
    
    #Final calculations for the change in potential difference on the negatrode
    i_dl_an = pars.i_ext - i_Far_PNO
    ddphi_int_PNO = -i_dl_an/pars.C_dl_an
    dSV_dt[ptr.dphi_int_PNO_0] = ddphi_int_PNO

    #----- cathode -----
    dphi_NiBZY = SV[1]
    
    #Mass-Action equations
    
    k_fwd_NiBZY_H = pars.k_fwd_star_NiBZY_H*math.exp(-pars.beta_ca*F*pars.n_NiBZY_H*dphi_NiBZY/R/pars.T)
    k_rev_NiBZY_H = pars.k_rev_star_NiBZY_H*math.exp(((1-pars.beta_ca)*pars.n_NiBZY_H*F*dphi_NiBZY)/(R*pars.T))
    k_fwd_NiBZY_O = pars.k_fwd_star_NiBZY_O*math.exp(-pars.beta_ca*F*pars.n_NiBZY_O*dphi_NiBZY/R/pars.T)
    k_rev_NiBZY_O = pars.k_rev_star_NiBZY_O*math.exp((1-pars.beta_ca)*F*pars.n_NiBZY_H*dphi_NiBZY/R/pars.T)
    i_Far_NiBZY_O = pars.n_NiBZY_O*F*(k_fwd_NiBZY_O*pars.conc_fwd_NiBZY_O-k_rev_NiBZY_O*pars.conc_rev_NiBZY_O)
    i_Far_NiBZY_H = pars.n_NiBZY_O*F*(k_fwd_NiBZY_H*pars.conc_fwd_NiBZY_O-k_rev_NiBZY_H*pars.conc_rev_NiBZY_O)
    i_Far_NiBZY = i_Far_NiBZY_H + i_Far_NiBZY_O
    
    i_dl_ca = pars.i_ext - i_Far_NiBZY
    ddphi_int_NiBZY = -i_dl_ca/pars.C_dl_ca
    dSV_dt[1] = ddphi_int_NiBZY
    return dSV_dt

t_final = 1000 #seconds

SV_0 = np.array([pars.dphi_int_PNO_0, pars.dphi_int_ca_0])

time_span = np.array([0,t_final])

solution = solve_ivp(residual,time_span,SV_0,rtol=1e-8, atol=1e-8,method = 'BDF')

for var in solution.y:
    plt.plot(solution.t,var)
    
plt.legend(['Anode double layer','Cathode double layer'])

# volatge plot
V_elyte = solution.y[0,:]
V_ca = V_elyte + solution.y[1,:]
plt.plot(solution.t,V_elyte)
plt.plot(solution.t,V_ca)

plt.xlabel('Time (s)',fontsize=14)
plt.ylabel('Electric Potential (V)',fontsize=14)
plt.legend(['Electrolyte','Cathode'])