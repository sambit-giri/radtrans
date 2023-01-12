import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import quad, simpson
from scipy.interpolate import splev, splrep

import radtrans
from radtrans import constants as const

## Source model
Ndot  = 1e54    #s^1
nHI   = 1.87e-4 #cm^-3
C     = 5
t_evol= 500     #Myr
L_box = 5e24    #cm
T     = 1e4     #K

T1 = 1e5      # K
L1 = 1.9e31   # 1/eV^2
N1 = 5e48     # 1/s

S = radtrans.SourceSpectrum()
S.black_body(T, Ndot=Ndot)
L_nu = S.L_nu

nus = 10**np.linspace(10,20,200)  # Hz
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.loglog(nus, L_nu(nus), ls='-', c='k', label='Spectrum')
ax.axvline(S.Emin/S.h_P, label='$\\nu_\mathrm{min}$', ls='--')
ax.axvline(S.Emax/S.h_P, label='$\\nu_\mathrm{max}$', ls='-.')
ax.set_ylim(bottom=1)
# ax.axis([1e10,1e15,1e5,1e27])
ax.set_ylabel(r'$L_\nu$ [eV/s/Hz]')
ax.set_xlabel(r'$\nu$   [Hz]')
plt.legend()
plt.show()

Mpc_to_cm = 3.086e24 #cm
Myr_to_s  = 3.156e13 #s 

## Code setup
n_cells = 256
dt_coarse = t_evol/10
dt_fine   = t_evol/100

param = radtrans.par()
param.RTsolver.LB = L_box/Mpc_to_cm
param.RTsolver.dL = param.RTsolver.LB/n_cells
param.RTsolver.dt = dt_fine
param.RTsolver.t_evol = t_evol


## Analytical solution
alphaB = lambda T: 2.59e-13 * (T/1e4)**-0.7 #cm^3 s^-1
R_St   = lambda nH, T, C, Ndot: np.cbrt(3*Ndot/4/np.pi/alphaB(T)/C/nH**2) #cm
t_rec  = lambda nH, T, C: (C*alphaB(T)*nH)**-1 #s
R_St_Mpc  = lambda nH, T, C, Ndot: R_St(nHI,T,C,Ndot)/Mpc_to_cm #Mpc
t_rec_Myr = lambda nH, T, C: t_rec(nHI, 1e4, C)/Myr_to_s        #Myr

r_I = lambda nH, T, C, Ndot, t: R_St(nH,T,C,Ndot) \
                            * (1-np.exp(-t/t_rec(nHI,T,C)))**(1/3)
v_I = lambda nH, T, C, Ndot, t: R_St(nH,T,C,Ndot)/3/t_rec(nHI,T,C) \
          * np.exp(-t/t_rec(nHI,T,C))/(1-np.exp(-t/t_rec(nHI,T,C)))**(2/3)


print('t_rec  = {:.2e} s = {:.2f} Myr'.format(t_rec(nHI, T, C),t_rec_Myr(nHI, T, C)))
print('r_S  = {:.2e} cm = {:.3f} Mpc'.format(R_St(nHI,T,C,Ndot),R_St_Mpc(nHI,T,C,Ndot)))
print('t_evol/t_rec = {:.2f}'.format(t_evol/t_rec_Myr(nHI, T, C)))

## one-dimension
xHI_grid = np.ones_like(nHI_grid)
nHI_grid = nHI*np.ones(n_cells)
nH_grid  = nHI_grid*xHI_grid
xi_grid  = 1-xHI_grid
dL_box   = L_box/n_cells # cm.
rS_grid  = np.arange(n_cells)*dL_box # cm. In this grid, source is at r=0.

t_coarse_grid = np.arange(0,t_evol,dt_coarse)*Myr_to_s   #s
t_fine_grid   = np.arange(0,t_evol,dt_fine)*Myr_to_s     #s


#### Plot
t_plot_grid   = np.arange(0,t_evol,dt_fine/100)*Myr_to_s #s

fig, axs = plt.subplots(3,2,figsize=(7,8))
axs[1,0].plot(t_plot_grid/Myr_to_s, r_I(nHI, T, C, Ndot, t_plot_grid)/Mpc_to_cm*1000, 
                        ls='-', c='k', label='analytical')
axs[1,0].set_xlabel('$t$ [Myr]')
axs[1,0].set_ylabel('$r_I$ [kpc]')
axs[1,1].plot(t_plot_grid/Myr_to_s, r_I(nHI, T, C, Ndot, t_plot_grid)/Mpc_to_cm*1000, 
                        ls='-', c='k', label='analytical')
axs[1,1].set_xlabel('$t$ [Myr]')
axs[1,1].set_ylabel('$r_I$ [kpc]')
plt.tight_layout()
plt.show()