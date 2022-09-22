
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
import copy
from skopt import sampler
from radtrans.profiles_on_grid import profile_to_3Dkernel, Spreading_Excess_Fast, put_profiles_group


def run_RT_for_emul(param,Helium,simple_model):
    """
    This is to parallelize with job lib. Make a copy of parameters, set the halo mass to Mhalo, and run the solver (and store the profiles.)
    Regarding r_end : since we vectorized the radial direction, we can set r_end to the value that we want and increase dn if needed !
    So no need to do as we previously did (estimate r_end from the Stromgren sphere radius...)
    """
    z_start = param.solver.z

    fstar = param.source.f_st
    f0_esc = param.source.f0_esc
    pl_esc = param.source.pl_esc
    Mt = param.source.Mt
    cX = param.source.cX
    Emin = param.source.E_min_xray
    Mhalo = param.source.M_halo
    param.table.filename_table = './gamma_tables/gamma_fst_' + str(round(fstar,2)) + '_Mh_{:.1e}_z{}.pkl'.format(Mhalo,round(z_start, 2))
    print('Solving the RT equations ..', )


    if simple_model:
        print('--SIMPLE MODEL--')
        grid_model = rad.simple_solver(param)
    elif Helium == True:
        print('--HELIUM--')
        grid_model = rad.Source_MAR_Helium(param)
    else:
        print('--ONLY HYDROGEN--')
        grid_model = rad.Source_MAR(param)

    grid_model.solve(param)

    if simple_model:
        pickle.dump(file=open( './profiles_output/profiles_' + param.sim.model_name + '_fst_' + str(round(fstar, 2)) + '_fesc_'+ str(round(f0_esc, 2)) +'_plesc_' + str(round(pl_esc, 2)) +
                '_Mt_'+ str(round(Mt, 2)) + '_cX_' + str(round(cX, 2)) + '_Emin_' + str(round(Emin, 2)) + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, Mhalo), 'wb'),
                obj={'z': grid_model.z_history, 'r_cMpc/h': grid_model.r_grid_cell, 'Temp': grid_model.T_history,'Rbubb_cMpc/h': grid_model.R_bubble})
    else:
        pickle.dump(file=open('./profiles_output/profiles_'+ param.sim.model_name +'_fst_' + str(round(fstar, 2)) + '_fesc_'+ str(round(f0_esc, 2)) +'_plesc_' + str(round(pl_esc, 2)) +
                               '_Mt_'+ str(round(Mt, 2)) + '_cX_' + str(round(cX, 2)) + '_Emin_' + str(round(Emin, 2)) + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, Mhalo), 'wb'),
                obj={'z':grid_model.z_history,'r_pMpc/h':grid_model.r_grid_cell,'Temp':grid_model.T_history,'xHII':grid_model.xHII_history} )


    print('... RT equations solved. Profiles stored.')
    print(' ')


def gen_Sampling(bounds, Nsamples,Niteration = 2000):
    print('generating a training set made of',Nsamples,'samples, for bounds :',bounds)#Mhalo in range [',bounds[0][0],bounds[0][1], '], and fstar in the range[',bounds[1][0],bounds[1][1],'].')
    LHS = sampler.Lhs(lhs_type='centered', criterion='maximin', iterations=Niteration)
    Sampling = LHS.generate(dimensions=bounds, n_samples=Nsamples, random_state=None) ## shape is (Nsamples,size(bounds) )
    pickle.dump(file=open('./sampling_arr.pkl', 'wb'), obj=Sampling)
    print('storing it in ./sampling_arr.pkl')
    return Sampling


def gen_training_set(param,Sampling, Helium = True,simple_model = False):
    """
    This function loops over astro parameters, starting with the halo mass, and generate profiles, scanning the parameter space according to a Latin Hypercube distribution.
    ----------
    bounds : list of tuples [(),(),()]. For now (Mhalo,fstar)
    Nsamples : total number of points in param space for which we solve RT eq.
    Sampling : output of gen_Sampling
    """

    print('generating the training set')

    LBox = param.sim.Lbox  # Mpc/h
    param.solver.r_end = LBox /10 #max(LBox /10, r_MaxiMal)  # in case r_End is too small, we set it to LBox/10.

    if param.sim.mpi4py == 'yes':
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
    elif param.sim.mpi4py == 'no':
        rank = 0
        size = 1
    else:
        print('param.sim.mpi4py should be yes or no')
    M_i_min = param.sim.M_i_min
    M_i_max = param.sim.M_i_max
    binn = param.sim.binn  # let's start with 10 bins
    M_Bin = np.logspace(np.log10(M_i_min), np.log10(M_i_max), binn, base=10)

    for m in range(len(M_Bin)):
        for i in range(len(Sampling)):
            nbr = m * len(Sampling) + i
            if rank == nbr % size:
                param_ = copy.deepcopy(param)
                param_.source.M_halo = M_Bin[m]
                param_.source.f_st   = 10 ** Sampling[i][0]
                param_.source.f0_esc = 10 ** Sampling[i][1]
                param_.source.pl_esc = Sampling[i][2]
                param_.source.Mt     = 10 ** Sampling[i][3]
                param_.source.cX     = 10 ** Sampling[i][4]

                param_.source.E_min_sed_xray = Sampling[i][5]
                param_.source.E_min_xray     = Sampling[i][5]

                run_RT_for_emul(param_,Helium,simple_model)






def paint_ion_profile_emulator(filename,param,fstar):
    """
    Paint the Tk, xHII and Lyman alpha profiles on a grid for a single snapshot named filename.

    Parameters
    ----------
    param : dictionnary containing all the input parameters
    filename : the name of the snapshot, contained in param.sim.halo_catalogs.

    Returns
    -------
    Does not return anything. Paints and stores the grids on the directory grid_outputs.
    """
    catalog_dir = param.sim.halo_catalogs
    starttimeprofile = datetime.datetime.now()
    z_start = param.solver.z
    M_Bin = np.logspace(np.log10(param.sim.M_i_min), np.log10(param.sim.M_i_max), param.sim.binn, base=10)

    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    catalog = catalog_dir + filename
    halo_catalog = Read_Rockstar(catalog, Nmin=param.sim.Nh_part_min)
    H_Masses, H_X, H_Y, H_Z, H_Radii = halo_catalog['M'], halo_catalog['X'], halo_catalog['Y'], halo_catalog['Z'], halo_catalog['R']
    z = halo_catalog['z']

    # Data
    Sampling = pickle.load(file=open('./sampling_arr.pkl', 'rb'))
    ## read in the first profile to get the redshift list
    Mhalo = 10 ** float(Sampling[0][0])
    profile = pickle.load(file=open('./profiles_output/dict_profiles_fst_0.1_zi25_Mh_1.0e+05.pkl','rb'))#'dict_profiles_fst_' + str(round(Sampling[0][1], 2)) + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, Mhalo),'rb'))
    radial_grid = profile['r']

    Indexing = np.argmin( np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))), axis=1)
    print('There are', H_Masses.size, 'halos at z=', z, )


    ###load the emulator
    reg_xi_load = pickle.load(open('./emul_xion_profile_z9.pkl', 'rb'))

    if H_Masses.size == 0:
        print('There is no sources')
        Grid_xHII = np.array([0])

    else:
        Pos_Bubles = np.vstack((H_X, H_Y, H_Z)).T  # Halo positions.
        Pos_Bubbles_Grid = np.array([Pos_Bubles / LBox * nGrid]).astype(int)[0]
        Pos_Bubbles_Grid[np.where(Pos_Bubbles_Grid == nGrid)] = nGrid - 1  # you don't want Pos_Bubbles_Grid==nGrid
        Grid_xHII = np.zeros((nGrid, nGrid, nGrid))
        for i in range(len(M_Bin)):
            indices = np.where(Indexing == i)  ## indices in H_Masses of halos that have an initial mass at z=z_start between M_Bin[i-1] and M_Bin[i]
            if len(indices[0]) > 0:

                x_HII_profile = reg_xi_load.predict([[np.log10(M_Bin[i]),fstar]])[0]
                profile_xHII = interp1d(radial_grid * (1 + z), x_HII_profile, bounds_error=False, fill_value=(1, 0))
                kernel_xHII = profile_to_3Dkernel(profile_xHII, nGrid, LBox)

                #plt.semilogx(radial_grid * (1 + z), x_HII_profile,label = 'Mbin = {:.2e}'.format(M_Bin[i]))
                #plt.legend()
                if np.any(kernel_xHII > 0) :
                    Grid_xHII += put_profiles_group(Pos_Bubbles_Grid[indices], kernel_xHII * 1e-7 / np.sum(kernel_xHII)) * np.sum(kernel_xHII) / 1e-7

                endtimeprofile = datetime.datetime.now()
                print(len(indices[0]), 'halos in mass bin ', i, 'took : ', endtimeprofile - starttimeprofile,'to paint profiles')

        Grid_Storage = np.copy(Grid_xHII)

        if np.sum(Grid_Storage) < nGrid ** 3:
            Grid_xHII = Spreading_Excess_Fast(Grid_Storage)
        else:
            Grid_xHII = np.array([1])

        endtimespread = datetime.datetime.now()
        print('It took:', endtimespread - endtimeprofile, 'to spread the excess photons')

        if np.all(Grid_xHII == 0):
            Grid_xHII = np.array([0])

        if np.all(Grid_xHII == 1):
            print('universe is fully inoinzed. return 1 for Grid_xHII.')
            Grid_xHII = np.array([1])


    pickle.dump(file=open('./grid_output/xHII_Grid' + str(nGrid) + 'Emulator_' + '_snap' + filename[4:-5], 'wb'),obj=Grid_xHII)








