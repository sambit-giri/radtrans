
"""
In this script we define functions that can be called to :
1. run the RT solver and compute the evolution of the T, x_HI profiles, and store them
2. paint the profiles on a grid.
"""
import radtrans as rad
from scipy.interpolate import splrep,splev, interp1d
import numpy as np
import pickle
import datetime
from radtrans.constants import cm_per_Mpc, sec_per_year, M_sun, m_H, rhoc0, Tcmb0
from radtrans.astro import Read_Rockstar
from radtrans.cosmo import T_adiab
import os
import copy
from skopt import sampler


def run_RT_for_emul(Mhalo,fstar,parameters,Helium):
    """
    This is to parallelize with job lib. Make a copy of parameters, set the halo mass to Mhalo, and run the solver (and store the profiles.)
    Regarding r_end : since we vectorized the radial direction, we can set r_end to the value that we want and increase dn if needed !
    So no need to do as we previously did (estimate r_end from the Stromgren sphere radius...)
    """
    param = copy.deepcopy(parameters)
    param.source.M_halo = Mhalo
    param.source.f_st = fstar
    z_start = param.solver.z
    param.table.filename_table = './gamma_tables/gamma_fst_' + str(round(fstar,2)) + '_Mh_{:.1e}_z{}.pkl'.format(Mhalo,round(z_start, 2))


    print('Solving the RT equations ..',)
    if Helium == True:
        grid_model = rad.Source_MAR_Helium(param)
    else :
        grid_model = rad.Source_MAR(param)

    grid_model.solve(param)

    pickle.dump(file=open('./profiles_output/dict_profiles_fst_' + str(round(fstar,2)) + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, Mhalo), 'wb'),
                obj={'z':grid_model.z_history,'r':profile.r_grid_cell,'Temp':grid_model.T_history,'xHII':grid_model.xHII_history} )
    print('... RT equations solved. Profiles stored.')
    print(' ')


def gen_Sampling(bounds, Nsamples,Niteration = 2000):
    print('generating a training set made of',Nsamples,'samples, for Mhalo in range [',bounds[0][0],bounds[0][1], '], and fstar in the range[',bounds[1][0],bounds[1][1],'].')
    LHS = sampler.Lhs(lhs_type='centered', criterion='maximin', iterations=Niteration)
    Sampling = LHS.generate(dimensions=bounds, n_samples=Nsamples, random_state=None) ## shape is (Nsamples,size(bounds) )
    pickle.dump(file=open('./sampling_arr.pkl', 'wb'), obj=Sampling)
    print('storing it in ./sampling_arr.pkl')
    return Sampling


def gen_training_set(param,Sampling, Helium = True):
    """
    This function loops over astro parameters, starting with the halo mass, and generate profiles, scanning the parameter space according to a Latin Hypercube distribution.
    ----------
    bounds : list of tuples [(),(),()]. For now (Mhalo,fstar)
    Nsamples : total number of points in param space for which we solve RT eq.
    Sampling : output of gen_Sampling
    """
    LBox = param.sim.Lbox  # Mpc/h
    z_start = param.solver.z
    ### Let's deal with r_end :
    cosmofile = param.cosmo.corr_fct
    vc_r, vc_corr = np.loadtxt(cosmofile, usecols=(0, 1), unpack=True)
    r_MaxiMal = max(vc_r) / (1 + z_start)  ## Minimum k-value available in cosmofct.dat
    param.solver.r_end = max(LBox /10, r_MaxiMal)  # in case r_End is too small, we set it to LBox/10.

    if param.sim.mpi4py == 'yes':
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
    elif param.sim.mpi4py == 'no':
        rank = 0
        size = 1
    else:
        print('param.sim.mpi4py should be yes or no')

    for i in range(len(Sampling)):
        if rank == i % size:
            Mhalo = 10 ** float(Sampling[i][0])
            fstar = Sampling[i][1]
            run_RT_for_emul(Mhalo,fstar,param,Helium)

