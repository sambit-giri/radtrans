from scipy.signal import fftconvolve
import radtrans as rad
from scipy.interpolate import splrep,splev
import numpy as np
import pickle
import datetime
from radtrans.Only_Hydrogen import cm_per_Mpc, BB_Planck, profile_1D
from radtrans.bias import bias,profile
from radtrans.Astro import NGamDot
from radtrans.constants import *

start_time = datetime.datetime.now()


# Let's start with a simple model with only galaxies, with T = 50000K, a constant fstar and fesc (via f_c2ray = 4000).
# We bin Halo masses, and redshifts, and compute the profiles for all these.

M_min = 1e7
M_max = 1e12
binn = 11 # let's start with 10 bins
M_Bin = np.logspace(np.log10(M_min),np.log10(M_max),binn,base = 10)
z_start = 25
z_end  = 5
binn = 11
z_Bin   = np.linspace(z_start,z_end,binn)



def S_fct(Mh, Mt, g3, g4):
    return (1 + (Mt / Mh) ** g3) ** g4


def f_star_Halo(Mh, f_st=1, Mp=2e11, g1 = 0.49, g2=-0.61, Mt=1e8, g3=0, g4=0):
    return 2 * f_st / ((Mh / Mp) ** g1 + (Mh / Mp) ** g2) * S_fct(Mh, Mt, g3, g4)


# Fix the parameters, and loop over M_HALO and Z_BIN
parameters = rad.par()
parameters.solver.dn = 10
parameters.solver.dn_table = 1000
parameters.solver.precision = 1e10  # No adaptive mesh
parameters.solver.C = 1

parameters.source.E_0 = 13.6
parameters.source.E_upp = 1e3
parameters.source.type = 'Galaxies'
parameters.source.lifetime = 10
parameters.solver.evol = 10
parameters.solver.Nt = 10

parameters.cosmo.h = 0.7
parameters.cosmo.corr_fct = './../files/cosmofct.dat'
parameters.cosmo.ps = './../files/CDM_PLANCK_tk.dat'

parameters.table.import_table = False

# 2d Arrays with c1 and c2 values for each Mh, z.
c1_2d_array = np.zeros((len(z_Bin),len(M_Bin))) ## sharpness
c2_2d_array = np.zeros((len(z_Bin),len(M_Bin))) ##ion front
Ng_dot_array = np.zeros(len(M_Bin))

for ih,Mhalo in enumerate(M_Bin):
    parameters.source.M_halo = Mhalo
    parameters.source.fc2ray = 4000 * f_star_Halo(Mh=Mhalo)
    N_gam_dot = NGamDot(parameters)

    for iz,zz_ in enumerate(z_Bin):
        ### Let's deal with r_end :
        cosmofile = parameters.cosmo.corr_fct
        vc_r, vc_m, vc_bias, vc_corr = np.loadtxt(cosmofile, usecols=(0, 1, 2, 3), unpack=True)
        corr_tck = splrep(vc_r, vc_corr, s=0)
        r_MaxiMal =  max(vc_r)/(1+zz_)   ## Maximum k-value available in cosmofct.dat
        cosmo_corr = splev(r_MaxiMal * (1 + zz_), corr_tck)
        halo_bias = bias(zz_, parameters)
        # baryonic density profile in [cm**-3]
        nHI0_profile = profile(halo_bias, cosmo_corr, parameters, zz_) * parameters.cosmo.Ob / parameters.cosmo.Om * \
                       M_sun * parameters.cosmo.h ** 2 / (cm_per_Mpc) ** 3 / m_H
        r_End = (3 * N_gam_dot * 10 * sec_per_year * 1e6 / 4 / np.pi / parameters.cosmo.Ob / np.mean(nHI0_profile)) ** (
                    1.0 / 3) / cm_per_Mpc

        parameters.solver.r_end =  min(r_End/10, r_MaxiMal)
        parameters.solver.z = zz_
        parameters.table.filename_table = './Gamma_Tables/Gamma_Gal_HIOnly_10Myr_Mh_1e{}_z{}'.format(round(np.log10(Mhalo), 2),
                                                                                      round(zz_, 2))

        grid_model = rad.Source_Only_H(parameters)
        grid_model.solve(parameters)

        try:
            grid_model.fit()
            c1_2d_array[iz, ih] = grid_model.c1
            c2_2d_array[iz, ih] = grid_model.c2
        except Exception:
            c1_2d_array[iz, ih] = None
            c2_2d_array[iz, ih] = None
        #pickle.dump(
        #    file=open('./Solver_Model3_DPL_Gal_HIOnly_10Myr_Mh_1e{}_z{}_dn100_Nt100'.format(
       #         round(np.log10(Mhalo), 2), round(zz_, 2)), 'wb'), obj=grid_model)
    Ng_dot_array[ih] = N_gam_dot

pickle.dump(obj =(np.flip(z_Bin),M_Bin,c1_2d_array,c2_2d_array,Ng_dot_array), file = open('./TEST_Model2_z_Mh_c1_c2_NgDot','wb'))
if np.any(c1_2d_array==None):
    print('WARNING : Contains None')
print(np.flip(z_Bin),M_Bin,c1_2d_array,c2_2d_array,Ng_dot_array)
end_time = datetime.datetime.now()
print('took:',end_time-start_time)

