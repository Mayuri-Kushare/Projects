
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
    dSV_dt = np.zeros_like(SV) 
    
    #----- Anode-----
    dphi_PNO = SV[ptr.dphi_PNO_0]
    # eta_an = SV[ptr.phi_dl_an] - pars.delta_Phi_eq_an 
    #Mass Action Equations For Charge Transfer:
    
    k_fwd_PNO_H = pars.k_fwd_star_PNO_H*math.exp(-pars.beta_an*F*pars.n_PNO_H*dphi_PNO/R/pars.T)
    k_rev_PNO_H = pars.k_rev_star_PNO_O*math.exp(-pars.beta_an*F*pars.n_PNO_H*dphi_PNO/R/pars.T)
    k_fwd_PNO_O = pars.k_fwd_star_PNO_O*math.exp(-pars.beta_an*F*pars.n_PNO_O*dphi_PNO/R/pars.T)
    k_rev_PNO_O = pars.k_rev_star_PNO_O*math.exp((1-pars.beta_an)*F*pars.n_PNO_O*dphi_PNO/R/pars.T)
    i_Far_PNO_O = pars.n_PNO_O*F*(k_fwd_PNO_O*pars.conc_fwd_PNO_O-k_rev_PNO_O*pars.conc_rev_PNO_O)
    i_Far_PNO_H = pars.n_PNO_H*F*(k_fwd_PNO_H*pars.conc_fwd_PNO_H-k_rev_PNO_H*pars.conc_rev_PNO_H)
    i_Far_PNO = i_Far_PNO_O + i_Far_PNO_H
    
    # potential difference
    i_dl_an = pars.i_ext - i_Far_PNO
    dphi_PNO = -i_dl_an/pars.C_dl_an
    dSV_dt[ptr.dphi_PNO_0] = dphi_PNO
    

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
    dphi_NiBZY = -i_dl_ca/pars.C_dl_ca
    dSV_dt[1] = dphi_NiBZY
    return dSV_dt 
      
     #Getting parameters from the SV
    C_k_gas_ca = SV[ptr.C_k_ca_gas]
    
    s1 = {'C_k': C_k_gas_ca, 'dy_ca':pars.dy_NiBZY, 'eps_g':pars.eps_g_ca, 
        'n_Brugg':pars.n_brugg, 'd_solid':pars.d_part_avg}
    s2 = {'C_k': C_k_gas_ca, 'dy_elyte':pars.dy_elyte, 'eps_g':pars.eps_elyte,
        'n_Brugg':pars.n_brugg, 'd_solid':pars.d_BZY_elyte}
    gas_props = {'T':pars.T, 'D_k':pars.D_k_an_g, 'mu':pars.mu_an_g}
   
    N_k_i = pcec_gas_flux(s1,s2,gas_props) #
    
    "Stoichiometric values for the gas transfer reactions at the cathode"
    #Hydrogen
    nu_H_NiBZY_s = -2
    nu_vac_NiBZY = 2
    nu_H2_gas = 1
       
    "cathode gas phase reactions- rate of progress are calculated for hydrogen"
    
    conc_fwd_H2 = (pars.C_H_NiBZY**nu_H_NiBZY_s) 
    conc_rev_H2 = (C_k_gas_ca**nu_H2_gas) * (pars.C_vac_NiBZY**nu_vac_NiBZY) 
    qdot_H2 = pars.k_fwd_H2* conc_fwd_H2 - pars.k_rev_H2 * conc_rev_H2
    sdot_H2 = nu_H2_gas * qdot_H2  
    dCk_dt = (N_k_i + sdot_H2*pars.A_fac_Ni)*pars.eps_g_dy_NiBZY
    dSV_dt[ptr.C_k_ca_gas] = dCk_dt  
  
def pcec_gas_flux(node1, node2, gas_props):
    N_k  = np.zeros_like(node1['C_k'])

    f1 = node1['dy_ca']/(node1['dy_ca'] + node2['dy_elyte'])
    f2 = 1-f1

    C_int = f1*node1['C_k'] + f2*node2['C_k']

    X_k_1 = node1['C_k']/np.sum(node1['C_k'])
    X_k_2 = node2['C_k']/np.sum(node2['C_k'])
    X_k_int = f1*X_k_1 + f2*X_k_2

    P_1 = np.sum(node1['C_k'])*R*gas_props['T']
    P_2 = np.sum(node2['C_k'])*R*gas_props['T']
    # print(P_1, P_2)

    eps_g = f1*node1['eps_g'] + f2*node2['eps_g']
    tau_fac = (f1*node1['eps_g']**node1['n_Brugg'] 
        + f2*node2['eps_g']**node2['n_Brugg'])
    D_k_eff = eps_g*gas_props['D_k']/tau_fac
   
    
    d_part = f1*node1['d_solid'] + f2*node2['d_solid']
    K_g = eps_g**3*d_part**2*tau_fac**(-2)*(1-eps_g)**(-2)/72

    dY = 0.5*(node1['dy'] + node2['dy'])

    V_conv = -K_g*(P_2 - P_1)/dY/gas_props['mu']
    V_k_diff = -D_k_eff*(X_k_2 - X_k_1)/dY/X_k_int

    V_k  = V_conv + V_k_diff

    N_k = C_int*X_k_int*V_k
  
    return N_k
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
t_final = 1000 #seconds

SV_0 = np.array([pars.dphi_PNO_0, pars.dphi_ca_0])

#C_k_gas_ca = SV[ptr.C_k_ca_gas]
#SV_0 = np.hstack((np.array([pars.phi_PNO_0 - pars.phi_elyte_0]),  C_k_gas_ca,
    #np.array([pars.phi_ca_0 - pars.phi_elyte_0])))

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