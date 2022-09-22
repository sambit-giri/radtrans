
"""
We use same parametrization as in  : 1809.08995
fst : 0.001-1 (log)
fesc : 0.001-1 (log)
alpha_esc : -1-0.5 (line)
Mt : 1e8-1e10 (log)
cX  : 1e38 - 1e42 (log)
Emin_xray / Emin_sed_xray : 100eV - 1.5keV
"""

import radtrans as rad
import time
import numpy as np
from radtrans.global_qty import global_signal
import pickle; import matplotlib.pyplot as plt

Sampling = rad.gen_Sampling( bounds = [(-3.0,0.0),(-3.0,0.0),(-1.0,0.5),(8.0,10.0),(38.0,42.0),(100.0,1500.0)] , Nsamples = 50, Niteration = 5000)

plt.figure(figsize=(5,5))
for i in range(len(Sampling)):
    plt.plot(Sampling[i][0],Sampling[i][1],'*')

plt.grid()
plt.show()


print('generating the training set')


param = rad.par()
#sim
param.sim.M_i_min = 1e2*np.exp(0.79*(25-40))
param.sim.M_i_max =4e7*np.exp(0.79*(25-40))
param.sim.model_name = '_matching_FAST_simple_1'   # 'sed_lowxray' #'
param.sim.mpi4py ='no'
param.sim.cores = 1
param.sim.binn = 20
param.sim.Nh_part_min = 10


#solver
param.solver.z = 40
param.solver.z_end = 6
param.solver.dn = 5000
param.solver.time_step = 0.05

#cosmo
param.cosmo.Om = 0.31
param.cosmo.Ob = 0.045
param.cosmo.Ol = 0.69
param.cosmo.h = 0.68
param.cosmo.profile = 0
param.cosmo.Temp_IC =  1e-50  #CAREFULLLLL

## Source sed
param.source.N_al = 4000
param.source.Nion = 10264.8
param.source.xray_in_ion = 0
param.source.ion_in_xray = 0
param.source.E_min_xray = 500
param.source.E_max_xray = 2000
param.source.E_min_sed_xray = 500
param.source.E_max_sed_xray = 2000
param.source.E_min_sed_ion = 13.6
param.source.E_max_sed_ion = 273 # should be equal to E_min_xray to be sure that we get Nion photons in total !
param.source.alS_ion = 0
param.source.alS_xray =  2.5
param.source.cX = 2.4 * 3.4e40   #* 1e-90

#fesc
param.source.f0_esc = 0.1

#fstar
param.source.f_st = 0.05
param.source.g1 = 0
param.source.g2 = 0
param.source.g3 = 40
param.source.g4= -40
param.source.Mp=1e10 * param.cosmo.h
param.source.Mt=1e9 * param.cosmo.h

param.sim.thresh_xHII = 0.01

Sampling = pickle.load(file=open('./sampling_arr.pkl', 'rb'))
rad.gen_training_set(param,Sampling,  Helium=False, simple_model = True)