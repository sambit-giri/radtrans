"""
Creating the training set of ionisation profiles using the RT solver.
"""

import pickle, sys, os 
import numpy as np
import radtrans as rad
from radtrans.solver_Helium import Source_MAR_Helium

import matplotlib.pyplot as plt 
import matplotlib as mpl
from joblib import Parallel, delayed
from tqdm import tqdm

def check_sysargv(knob, fall_back, d_type):
    smooth_info = np.array([knob in a for a in sys.argv])
    if np.any(smooth_info): 
        smooth_info = np.array(sys.argv)[smooth_info]
        smooth_info = smooth_info[0].split(knob)[-1]
        smooth = d_type(smooth_info)
    else:
        smooth = d_type(fall_back)
    return smooth 

param = rad.par()

## Emulator parameters
M_halo = 1e10   # [1e6,1e15]
Nion   = 4000   # [100,100000]
g1 = 0.49       # [0,2.]
g2 = -0.61      # [-2,0]
g3 = 4          # [-5,5]
g4 = -1         # [-5,5]
Mp = 1e11       # [1e9,1e13]
Mt = 7e7        # [1e6,1e10]

# solver param
param.solver.time_step = 0.05
param.solver.dn = 500  # rad bins
param.solver.z_end = 5
param.solver.z = 25
param.solver.r_end = 50

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
param.source.E_min_xray = 500 #Emin
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
param.source.g1 = g1
param.source.g2 = g2
param.source.g3 = g3
param.source.g4 = g4
param.source.Mp = Mp
param.source.Mt = Mt 
param.source.M_halo = M_halo

M_halo_list = 10**np.linspace(6,15)
Nion_list   = np.linspace(100,100000)
g1_list = np.linspace(0,2.)
g2_list = np.linspace(-2,0)
g3_list = np.linspace(-5,5)
g4_list = np.linspace(-5,5)
Mp_list = 10**np.linspace(9,13)
Mt_list = 10**np.linspace(6,10)
fstar = lambda Ms,par: 2*par.cosmo.Ob/par.cosmo.Om*par.source.f_st \
                        / ((Ms/par.source.Mp)**par.source.g1+(Ms/par.source.Mp)**par.source.g2) \
                        * (1+(par.source.Mt/Ms)**par.source.g3)**par.source.g4
Ms = 10**np.linspace(6,12,100)

# param.source.g1 = g1
# param.source.g2 = g2
# param.source.g3 = g3
# param.source.g4 = g4
# param.source.Mp = Mp
# param.source.Mt = Mt 
# plt.loglog(Ms, fstar(Ms,param), c='k')
# param.source.g1 = g1_list[0]
# param.source.g2 = g2
# param.source.g3 = g3
# param.source.g4 = g4
# param.source.Mp = Mp
# param.source.Mt = Mt 
# plt.loglog(Ms, fstar(Ms,param), label='g1={:.2f}'.format(param.source.g1))
# param.source.g1 = g1_list[-1]
# param.source.g2 = g2
# param.source.g3 = g3
# param.source.g4 = g4
# param.source.Mp = Mp
# param.source.Mt = Mt 
# plt.loglog(Ms, fstar(Ms,param), label='g1={:.2f}'.format(param.source.g1))
# param.source.g1 = g1
# param.source.g2 = g2_list[0]
# param.source.g3 = g3
# param.source.g4 = g4
# param.source.Mp = Mp
# param.source.Mt = Mt 
# plt.loglog(Ms, fstar(Ms,param), label='g2={:.2f}'.format(param.source.g2))
# param.source.g1 = g1
# param.source.g2 = g2_list[-1]
# param.source.g3 = g3
# param.source.g4 = g4
# param.source.Mp = Mp
# param.source.Mt = Mt 
# plt.loglog(Ms, fstar(Ms,param), label='g2={:.2f}'.format(param.source.g2))
# plt.axis([1e6,1e13,1e-4,1e-1])
# plt.legend()
# plt.show()


from skopt.sampler import Lhs
from skopt.space import Space

n_samples = 10000 
space = Space([(6., 15.),        #log10Mhalo
                (100., 100000.), #Nion
                (0,2),           #g1
                (-2,0),          #g2
                (-5,5),          #g3
                (-5,5),          #g4
                (9,13),          #log10Mp
                (6,10)           #log10Mt
                ])
file_train = 'xHII_profiles_log10Mhalo_Nion_g1_g2_g3_g4_log10Mp_log10Mt_nSamples{}.pkl'.format(n_samples)
try:
    data_train = pickle.load(open(file_train, 'rb'))
    grid_model = data_train['RT_model']
except:
    lhs = Lhs(criterion="maximin", iterations=100)
    x = lhs.generate(space.dimensions, n_samples)
    param.table.filename_table='qwerty'
    grid_model = Source_MAR_Helium(param)
    grid_model.solve(param)
    data_train = {'X': np.vstack(([[np.log10(M_halo),Nion,g1,g2,g3,g4,np.log10(Mp),np.log10(Mt)]],np.array(x))),
                  'y_dict': {0: np.array([it for ke,it in grid_model.xHII_history.items()])},
                  'z': np.array([ke for ke,it in grid_model.xHII_history.items()]).astype(float),
                  'RT_model': grid_model}
    pickle.dump(data_train, open(file_train, 'wb'))

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

def run_rt(i):
    data_train = pickle.load(open(file_train, 'rb'))
    pp = data_train['X'][i]
    try:
        prof = data_train['y_dict'][i]
    except:
        print('{} index...'.format(i))
        param.source.M_halo = 10**pp[0]   # [1e6,1e15]
        param.source.Nion   = pp[1]       # [100,100000]
        param.source.g1 = pp[2]           # [0,2.]
        param.source.g2 = pp[3]           # [-2,0]
        param.source.g3 = pp[4]           # [-5,5]
        param.source.g4 = pp[5]           # [-5,5]
        param.source.Mp = 10**pp[6]       # [1e9,1e13]
        param.source.Mt = 10**pp[7]       # [1e6,1e10]
        param.table.filename_table='qwerty'
        grid_model = Source_MAR_Helium(param)
        grid_model.solve(param)
        prof = np.array([it for ke,it in grid_model.xHII_history.items()])
        data_train['y_dict'][i] = prof
        pickle.dump(data_train, open(file_train, 'wb'))
    return prof 

n_jobs = check_sysargv('--n_jobs=', 64, int)
n_samples = check_sysargv('--n_samples=', data_train['X'].shape[0], int)
print(n_jobs,n_samples)

for i in tqdm(np.arange(n_samples)):
    pp = data_train['X'][i]
    print('{}/{}...'.format(i+1,data_train['X'].shape[0]))
    prof = run_rt(i)
    print('...done.')

import multiprocessing
print('{} CPUs'.format(multiprocessing.cpu_count()))

y = np.array(Parallel(n_jobs=n_jobs)(delayed(run_rt)(i) for i in tqdm(np.arange(n_samples)) ))
print('Output shape:', y.shape)




