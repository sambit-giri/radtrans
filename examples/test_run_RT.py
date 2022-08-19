"""
A test run of the RT solver. Here we include Helium
"""

import pickle
import numpy as np
import radtrans as rad
from radtrans.solver_Helium import Source_MAR_Helium

param = rad.par()


Emin = 500
Nion = 4000

# solver param
param.solver.time_step = 0.05
param.solver.dn = 500  # rad bins
param.solver.z_end = 5
param.solver.z = 25
param.solver.r_end = 10

# cosmo
param.cosmo.Om = 0.31
param.cosmo.Ob = 0.045
param.cosmo.Ol = 0.69
param.cosmo.h = 0.68
param.cosmo.profile = 0

## Source sed
param.source.N_al = 9690
param.source.Nion = Nion
param.source.xray_in_ion = 1 #0
param.source.ion_in_xray = 1 #0
param.source.E_min_xray = Emin
param.source.E_max_xray = 10000
param.source.E_min_sed_xray = 500
param.source.E_max_sed_xray = 8000
param.source.E_min_sed_ion = 13.6
param.source.E_max_sed_ion = 273  # should be equal to E_min_xray to be sure that we get Nion photons in total !
param.source.alS_ion = 0
param.source.cX = 0.2 * 3.4e40  # * 1e-90

# fesc
param.source.f0_esc = 0.15

# fstar
param.source.f_st = 0.2
param.source.g1 = 0.49
param.source.g2 = -0.61
param.source.g3 = 4
param.source.g4 = -1
param.source.Mp = 1e11
param.source.Mt = 7e7

param.source.M_halo = 1e8

grid_model = Source_MAR_Helium(param)
grid_model.solve(param)
# pickle.dump(file=open('./TProfiles_test_AsHM_Nion'+str(Nion)+'_Emin'+str(Emin)+'.pkl','wb'),obj=grid_model )


## Plot profiles
import matplotlib.pyplot as plt 
import matplotlib as mpl

zs   = np.sort(grid_model.z_history[1::4])
norm = mpl.colors.Normalize(vmin=zs.min(), vmax=zs.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
cmap.set_array([])

for i, z in enumerate(zs):
    plt.plot(grid_model.r_grid_cell, grid_model.xHII_history['{:}'.format(z)], 
        c=cmap.to_rgba(i + 1), label='z={:.2f}'.format(z))
# plt.legend()
plt.colorbar(cmap)
plt.xlabel('R [Mpc/h]')
plt.ylabel('$x_\mathrm{HII}$')
plt.show()


