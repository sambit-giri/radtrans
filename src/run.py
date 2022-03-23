
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
from radtrans.bias import bias,profile
from radtrans.astro import Read_Rockstar
import os
import copy
from radtrans.profiles_on_grid import profile_to_3Dkernel, Spreading_Excess_Fast,put_profiles_group
from radtrans.couplings import x_coll,rho_alpha, S_alpha
from joblib import Parallel, delayed
from radtrans.global_qty import xHII_approx

import matplotlib.pyplot as plt




def run_RT_single_source(Mhalo,parameters,Helium):
    """
    This is to parallelize with job lib. Make a copy of parameters, set the halo mass to Mhalo, and run the solver (and store the profiles.)
    Regarding r_end : since we vectorized the radial direction, we can set r_end to the value that we want and increase dn if needed !
    So no need to do as we previously did (estimate r_end from the Stromgren sphere radius...)
    """
    param = copy.deepcopy(parameters)
    param.source.M_halo = Mhalo
    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    z_start = param.solver.z
    model_name = param.sim.model_name
    ### Let's deal with r_end :
    cosmofile = param.cosmo.corr_fct
    vc_r, vc_corr = np.loadtxt(cosmofile, usecols=(0, 1), unpack=True)
    r_MaxiMal = max(vc_r) / (1 + z_start)  ## Minimum k-value available in cosmofct.dat

    param.solver.r_end = max(LBox /10, r_MaxiMal)  # in case r_End is too small, we set it to LBox/10.
    param.table.filename_table = './gamma_tables/gamma_' + model_name + '_Mh_{:.1e}_z{}.pkl'.format(Mhalo,round(z_start, 2))

    print('Solving the RT equations ..')
    if Helium == True:
        grid_model = rad.Source_MAR_Helium(param)
    else :
        grid_model = rad.Source_MAR(param)
    grid_model.solve(param)
    pickle.dump(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, Mhalo), 'wb'),obj=grid_model)
    print('... RT equations solved. Profiles stored.')
    print(' ')


def run_solver(parameters,Helium=False):
    """
    This function loops over Mbin, initial halo masses and compute the RT equation from zstart to zend for each halo mass. It uses joblib to parallelize.
    Parameters
    ----------
    param : dictionnary containing all the input parameters

    Returns
    -------
    Does not return anything. However, it solve the RT equation for a range of halo masses, following their evolution from CDawn to the end of reionization. It stores the profile in a directory called profiles_output.
    """
    # check if folder exist to store gamma tables and profiles
    if not os.path.isdir('./gamma_tables'):
        os.mkdir('./gamma_tables')
    if not os.path.isdir('./profiles_output'):
        os.mkdir('./profiles_output')

    start_time = datetime.datetime.now()

    ## Initial mass binning
    M_i_min = parameters.sim.M_i_min
    M_i_max = parameters.sim.M_i_max
    binn  = parameters.sim.binn  # let's start with 10 bins
    M_Bin = np.logspace(np.log10(M_i_min), np.log10(M_i_max), binn, base=10)

    if parameters.sim.mpi4py == 'yes':
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
    elif parameters.sim.mpi4py == 'no':
        rank = 0
        size = 1
    else :
        print('param.sim.mpi4py should be yes or no')

    for ih, Mhalo in enumerate(M_Bin):
        if rank == ih % size:
            run_RT_single_source(M_Bin[ih], parameters,Helium = Helium)

    #def run_RT_single(M):
    #    return run_RT_single_source(M,parameters)

    #Parallel(n_jobs=parameters.sim.cores, prefer="threads")(delayed(run_RT_single)(M_Bin[i]) for i in range(len(M_Bin)))
    end_time = datetime.datetime.now()
    print('It took in total:', end_time - start_time)




def paint_profile_single_snap(filename,param,epsilon_factor=10,temp =True,lyal=True,ion=True,check_approx = True):
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
    model_name = param.sim.model_name
    M_Bin = np.logspace(np.log10(param.sim.M_i_min), np.log10(param.sim.M_i_max), param.sim.binn, base=10)

    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    catalog = catalog_dir+filename
    halo_catalog = Read_Rockstar(catalog, Nmin=param.sim.Nh_part_min)
    H_Masses, H_X, H_Y, H_Z, H_Radii = halo_catalog['M'], halo_catalog['X'], halo_catalog['Y'], halo_catalog['Z'], halo_catalog['R']
    z = halo_catalog['z']
    T_adiab_z = Tcmb0 * (1 + z) ** 2 / (1 + param.cosmo.z_decoupl)  # to consistently put to T_adiab the large scale IGM regions (pb with overlaps)

    # quick load to find matching redshift between solver output and simulation snapshot.
    grid_model = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]), 'rb'))
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]

    ##screening for xal
    epsilon = LBox / nGrid / epsilon_factor

    Indexing = np.argmin(np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))), axis=1)
    print('There are', H_Masses.size, 'halos at z=', z, )


    if H_Masses.size == 0:
        print('There aint no sources')
        Grid_xHII = np.array([0])
        Grid_Temp = np.array([T_adiab_z])
        Grid_xal = np.array([0])

    else:

        ## Here we compute quickly the cumulative fraction of ionized volume and check if it largely exceeds the box volume
        if check_approx :
            Ionized_vol = xHII_approx(param,halo_catalog)[1]
        else:
            Ionized_vol = 0

        if Ionized_vol > 2:
            print('universe is fully inoinzed. Return [1] for the XHII, T and xtot grid.')
            Grid_xHII = np.array([1])
            Grid_Temp = np.array([1])
            Grid_xal = np.array([1])
        else:
            Pos_Bubles = np.vstack((H_X, H_Y, H_Z)).T # Halo positions.
            Pos_Bubbles_Grid = np.array([Pos_Bubles / LBox * nGrid]).astype(int)[0]
            Pos_Bubbles_Grid[np.where(Pos_Bubbles_Grid == nGrid)] = nGrid - 1  # you don't want Pos_Bubbles_Grid==nGrid
            Grid_xHII_i = np.zeros((nGrid, nGrid, nGrid))
            Grid_Temp = np.zeros((nGrid, nGrid, nGrid))
            Grid_xal = np.zeros((nGrid, nGrid, nGrid))

            for i in range(len(M_Bin)):
                indices = np.where( Indexing == i)  ## indices in H_Masses of halos that have an initial mass at z=z_start between M_Bin[i-1] and M_Bin[i]
                if len(indices[0]) > 0:

                    grid_model   = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]),'rb'))

                    Temp_profile = grid_model.T_history[str(round(zgrid, 2))]
                    radial_grid  = grid_model.r_grid_cell
                    x_HII_profile = grid_model.xHII_history[str(round(zgrid, 2))]
                    #x_al_profile  = grid_model.x_al_history[str(round(zgrid, 2))]

                    r_lyal = np.logspace(-5, 2, 1000, base=10)  ## physical distance for lyal profile
                    rho_alpha_ = rho_alpha(r_lyal, grid_model.Mh_history[ind_z][0], zgrid, param)[0]
                    T_extrap = np.interp(r_lyal, radial_grid, grid_model.T_history[str(round(zgrid, 2))])
                    xHII_extrap = np.interp(r_lyal, radial_grid, grid_model.xHII_history[str(round(zgrid, 2))])
                    x_alpha_prof = 1.81e11 * (rho_alpha_) * S_alpha(zgrid, T_extrap, 1 - xHII_extrap) / (1 + zgrid)

                    #### CAREFUL ! this step has to be done AFTER using Tk_profile to compute x_alpha (via Salpha)
                    Temp_profile[np.where(Temp_profile <= T_adiab_z + 0.2)] = 0 # set to zero to avoid spurious addition - we put the +0.2 to be sure....

                    ## here we assume they all have M_bin[i]
                    profile_xHII = interp1d(radial_grid * (1 + z), x_HII_profile, bounds_error=False, fill_value=(1, 0))
                    kernel_xHII = profile_to_3Dkernel(profile_xHII, nGrid, LBox)

                    profile_T = interp1d(radial_grid * (1 + z), Temp_profile, bounds_error=False,fill_value=0)  # rgrid*(1+z) is in comoving coordinate, box too.
                    kernel_T = profile_to_3Dkernel(profile_T, nGrid, LBox)

                    profile_xal = interp1d(r_lyal * (1 + z), x_alpha_prof * (r_lyal * (1 + z) / (r_lyal * (1 + z) + epsilon)) ** 2, bounds_error=False, fill_value=0)                    ##screening
                    kernel_xal = profile_to_3Dkernel(profile_xal, nGrid, LBox)
                    renorm = np.trapz(x_alpha_prof * 4 * np.pi * r_lyal ** 2, r_lyal) / (LBox / (1 + z)) ** 3 / np.mean(kernel_xal)

                    if temp ==True:
                        Grid_Temp += put_profiles_group(Pos_Bubbles_Grid[indices], kernel_T * 1e-7 / np.sum(kernel_T)) * np.sum(kernel_T)/ 1e-7
                    if lyal == True:
                        Grid_xal += put_profiles_group(Pos_Bubbles_Grid[indices], kernel_xal * 1e-7 / np.sum(kernel_xal)) * renorm * np.sum(kernel_xal) / 1e-7  # we do this trick to avoid error from the fft when nu.sum(kernel) is too close to zero.

                    #if np.any(kernel_xHII > 0) and np.max( kernel_xHII) > 1e-8 and ion==True:  ## To avoid error from convole_fft (renomalization)
                    if np.any(kernel_xHII > 0) and ion == True:
                        Grid_xHII_i += put_profiles_group(Pos_Bubbles_Grid[indices], kernel_xHII * 1e-7 / np.sum(kernel_xHII)) * np.sum(kernel_xHII) / 1e-7

                endtimeprofile = datetime.datetime.now()
                print(len(indices[0]), 'halos in mass bin ', i, 'took : ', endtimeprofile - starttimeprofile,'to paint profiles')

            Grid_Storage = np.copy(Grid_xHII_i)
            Grid_Temp[np.where(Grid_Temp < T_adiab_z + 0.2)] = T_adiab_z

            Grid_xHII = Spreading_Excess_Fast(Grid_Storage)
            endtimespread = datetime.datetime.now()
            print('It took:', endtimespread - endtimeprofile, 'to spread the excess photons')

            if np.all(Grid_xHII == 0):
                Grid_xHII = np.array([0])

            if np.all(Grid_xHII == 1):
                print('universe is fully inoinzed. return 1 for Grid_xHII but the other still computed.')#. Return [1] for the XHII, T and xtot grid.')
                #Grid_xal = np.array([1])
                # Grid_xtot_ov = np.array([1])
                #Grid_Temp = np.array([1])
                Grid_xHII = np.array([1])

    # Grid_dTb_over_rho_b = factor * np.sqrt(1+z) * Grid_xtot/(1+Grid_xtot)* (1-T_cmb_z/Grid_Temp) * (1-Grid_xHII) #careful, this is dTb/(1+deltab)

    # Store Tk, xHII, and xal on a grid.
    if param.sim.store_grids == True:
        if not os.path.isdir('./grid_output'):
            os.mkdir('./grid_output')

        pickle.dump(file=open('./grid_output/T_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'wb'),obj=Grid_Temp)
        pickle.dump(file=open('./grid_output/xHII_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'wb'),obj=Grid_xHII)
        pickle.dump(file=open('./grid_output/xal_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'wb'),obj=Grid_xal/4/np.pi) #### WARNING : WE DIVIDE BY 4PI TO MATCH HM








def paint_profiles(param,temp =True,lyal=True,ion=True):
    """
    Parameters
    ----------
    param : dictionnary containing all the input parameters

    Returns
    -------
    Does not return anything. Loop over all snapshots in param.sim.halo_catalogs and calls paint_profile_single_snap. Uses joblib for parallelisation.
    """

    starttimeprofile = datetime.datetime.now()
    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    catalog_dir = param.sim.halo_catalogs

    if catalog_dir is None :
        print('You should specify param.sim.halo_catalogs. Should be a file containing the rockstar halo catalogs.')

    print('Painting T and ion profiles on a grid with', nGrid,'pixels per dim. Box size is',LBox ,'cMpc/h.')


    if param.sim.mpi4py == 'yes':
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
    elif param.sim.mpi4py == 'no':
        rank = 0
        size = 1
    else :
        print('param.sim.mpi4py should be yes or no')

    for ii, filename in enumerate(os.listdir(catalog_dir)):
        if rank == ii % size:
            paint_profile_single_snap(filename,param,temp=temp, lyal=lyal, ion=ion)

    #def paint_single(filename):
    #    return paint_profile_single_snap(filename,param)

    #Parallel(n_jobs=param.sim.cores, prefer="threads")(delayed(paint_single)(filename) for filename in os.listdir(catalog_dir))

    endtime = datetime.datetime.now()
    print('END. Stored Tgrid, xal grid and xHII grid. It took in total: ',endtime-starttimeprofile,'to paint the grids.')
    print('  ')


def grid_dTb(param):
    """
    Creates a grid of xcoll and dTb. Needs to read in Tk grid, xHII grid and density field on grid.
    """
    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name
    nGrid = param.sim.Ncell
    Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob, param.cosmo.h
    factor = 27 * Ob * h0 ** 2 / 0.023 * np.sqrt(0.15 / Om / h0 ** 2 / 10)  # factor used in dTb calculation

    for ii, filename in enumerate(os.listdir(catalog_dir)):
        with open(catalog_dir+filename, "r") as file:
            file.readline()
            a = float(file.readline()[4:])
            zz_ = 1 / a - 1
        Grid_Temp           = pickle.load(file=open('./grid_output/T_Grid'    + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_xHII           = pickle.load(file=open('./grid_output/xHII_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        #Grid_xtot_ov        = pickle.load(file=open('./grid_output/xtot_ov_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_xal             = pickle.load(file=open('./grid_output/xal_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))

        dens_field = param.sim.dens_field
        if dens_field is not None and param.sim.Ncell == 256:
            dens = np.fromfile(dens_field + filename[4:-5]+ '.0',dtype=np.float32)
            pkd=dens.reshape(256,256,256)
            pkd = pkd.T  ### take the transpose to match X_ion map coordinates
            Lbox = param.sim.Lbox
            N_cell = param.sim.Ncell
            V_total = Lbox**3
            V_cell = (Lbox/N_cell)**3
            mass = pkd * rhoc0 * V_total
            rho_m = mass / V_cell
            delta_b = (rho_m)/np.mean(rho_m)
        else :
            delta_b = 1 #rho/rhomean and not 1+rho/rhomean

        T_cmb_z = Tcmb0 * (1 + zz_)
        Grid_xHI = 1-Grid_xHII  ### neutral fraction



        Grid_xcoll = x_coll(z=zz_, Tk=Grid_Temp, xHI= Grid_xHI, rho_b= delta_b * rhoc0 * Ob * (1 + zz_) ** 3 * M_sun / cm_per_Mpc ** 3 / m_H)
        Grid_Tspin = ((1 / T_cmb_z + (Grid_xcoll+Grid_xal) / Grid_Temp) / (1 + Grid_xcoll+Grid_xal)) ** -1
        Grid_dTb = factor * np.sqrt(1+zz_) * (1-T_cmb_z/Grid_Tspin) * Grid_xHI * delta_b    # this is dTb*(1+deltab)

        pickle.dump(file=open('./grid_output/Tspin_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'wb'),obj=Grid_Tspin)
        pickle.dump(file=open('./grid_output/dTb_Grid'+str(nGrid)+'MAR_'+model_name+'_snap'+filename[4:-5],'wb'),obj = Grid_dTb)
        pickle.dump(file=open('./grid_output/xcoll_Grid'+str(nGrid)+'MAR_'+model_name+'_snap'+filename[4:-5],'wb'),obj = Grid_xcoll)


def compute_GS(param):
    """
    Reads in the grids and compute the global quantities averaged.
    """
    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name
    nGrid = param.sim.Ncell
    Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob,param.cosmo.h
    factor = 27 * (1 / 10) ** 0.5 * (Ob * h0 ** 2 / 0.023) * (Om * h0 ** 2 / 0.15) ** (-0.5)
    Tadiab = []
    z_ = []
    Tk = []
    Tk_neutral = []
    dTb  =[]
    x_HII = []
    x_al = []
    x_coll=[]
    T_spin= []

    for ii, filename in enumerate(os.listdir(catalog_dir)):
        with open(catalog_dir+filename, "r") as file:
            file.readline()
            a = float(file.readline()[4:])
            zz_ = 1 / a - 1
        Grid_Tspin          = pickle.load(file=open('./grid_output/Tspin_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_Temp           = pickle.load(file=open('./grid_output/T_Grid'    + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_xHII           = pickle.load(file=open('./grid_output/xHII_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        #Grid_xtot_ov           = pickle.load(file=open('./grid_output/xtot_ov_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_dTb            = pickle.load(file=open('./grid_output/dTb_Grid'  + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_xal            = pickle.load(file=open('./grid_output/xal_Grid'  + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_xcoll            = pickle.load(file=open('./grid_output/xcoll_Grid'  + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))

        z_.append(zz_)
        Tk.append(np.mean(Grid_Temp))
        Tk_neutral.append(np.mean(Grid_Temp[np.where(Grid_xHII<0.5)]))
        T_spin.append(np.mean(Grid_Tspin))
        x_HII.append(np.mean(Grid_xHII))
        x_al.append(np.mean(Grid_xal))
        x_coll.append(np.mean(Grid_xcoll))
        #x_tot_ov_1plusxtot.append(np.mean(Grid_xtot_ov))
        dTb.append(np.mean(Grid_dTb))

        Tadiab.append( Tcmb0 * (1+zz_)**2/(1+param.cosmo.z_decoupl) )


    if not os.path.isdir('./physics'):
        os.mkdir('./physics')

    z_, Tk, Tk_neutral, x_HII, x_al, x_coll, Tadiab, T_spin ,dTb= np.array(z_),np.array(Tk),np.array(Tk_neutral),np.array(x_HII),np.array(x_al),np.array(x_coll),np.array(Tadiab),np.array(T_spin),np.array(dTb)
    matrice = np.array([z_,Tk,Tk_neutral,x_HII,x_al,x_coll,Tadiab,T_spin,dTb])
    z_,Tk,Tk_neutral,x_HII,x_al,x_coll,Tadiab,T_spin ,dTb= matrice[:, matrice[0].argsort()]  ## sort according to z_

    dTb_GS = factor * np.sqrt(1 + z_) * (1 - Tcmb0*(1+z_) / T_spin)  * (1-x_HII)

    Dict = {'Tk':Tk,'Tk_neutral':Tk_neutral,'x_HII':x_HII,'x_al':x_al,'x_coll':x_coll,'dTb':dTb,'Tadiab':Tadiab,'z':z_,'T_spin':T_spin,'dTb_GS':dTb_GS}
    pickle.dump(file=open('./physics/GS_' + str(nGrid) + 'MAR_' + model_name+'.pkl', 'wb'),obj=Dict)


def compute_PS(param):
    """
    Parameters
    ----------
    param : dictionnary containing all the input parameters

    Returns
    -------
    Computes the power spectra of the desired quantities
    """

    import tools21cm as t2c
    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name
    nGrid = param.sim.Ncell
    #Om, Ob, h0 = param.cosmo.Om, param.cosmo.Ob, param.cosmo.h
    Lbox = param.sim.Lbox  #Mpc/h
    kbins = np.logspace(np.log10(param.sim.kmin), np.log10(param.sim.kmax), param.sim.kbin, base=10) #h/Mpc

    z_arr = []
    for filename in os.listdir(catalog_dir): #count the number of snapshots
        with open(catalog_dir+filename, "r") as file:
            file.readline()
            a = float(file.readline()[4:])
            zz_ = 1 / a - 1
        z_arr.append(zz_)

    z_arr = np.sort(z_arr)
    nbr_snap = len(z_arr)

    PS_xHI = np.zeros((nbr_snap,len(kbins)-1))
    PS_T   = np.zeros((nbr_snap,len(kbins)-1))
    PS_xal = np.zeros((nbr_snap,len(kbins)-1))
    PS_rho = np.zeros((nbr_snap,len(kbins)-1))
    PS_dTb = np.zeros((nbr_snap,len(kbins)-1))
    PS_T_lyal = np.zeros((nbr_snap,len(kbins)-1))
    PS_T_xHI  = np.zeros((nbr_snap,len(kbins)-1))
    PS_rho_xHI = np.zeros((nbr_snap,len(kbins)-1))
    PS_rho_xal = np.zeros((nbr_snap,len(kbins)-1))
    PS_rho_T   = np.zeros((nbr_snap,len(kbins)-1))
    PS_lyal_xHI = np.zeros((nbr_snap,len(kbins)-1))

    for filename in os.listdir(catalog_dir):
        with open(catalog_dir+filename, "r") as file:
            file.readline()
            a = float(file.readline()[4:])
            zz_ = 1 / a - 1
        Grid_Temp           = pickle.load(file=open('./grid_output/T_Grid'    + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_xHII           = pickle.load(file=open('./grid_output/xHII_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_dTb            = pickle.load(file=open('./grid_output/dTb_Grid'  + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_xal            = pickle.load(file=open('./grid_output/xal_Grid'  + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))

        if Grid_Temp.size==1: ## to avoid error when measuring power spectrum
            Grid_Temp = np.full((nGrid, nGrid, nGrid),1)
        if Grid_xHII.size==1:
            Grid_xHII = np.full((nGrid, nGrid, nGrid),1)
        if Grid_dTb.size==1:
            Grid_dTb = np.full((nGrid, nGrid, nGrid),1)
        if Grid_xal.size==1:
            Grid_xal = np.full((nGrid, nGrid, nGrid),1)



        delta_XHI = (1-Grid_xHII) / np.mean((1-Grid_xHII)) - 1
        delta_T = Grid_Temp / np.mean(Grid_Temp) - 1
        delta_dTb = Grid_dTb / np.mean(Grid_dTb) - 1
        delta_x_al = Grid_xal / np.mean(Grid_xal) - 1

        ii = np.where(z_arr == zz_)

        dens_field = param.sim.dens_field
        if dens_field is not None and param.sim.Ncell == 256:
            dens = np.fromfile(dens_field + filename[4:-5] + '.0', dtype=np.float32)
            pkd = dens.reshape(256, 256, 256)
            pkd = pkd.T  ### take the transpose to match X_ion map coordinates
            Lbox = param.sim.Lbox
            N_cell = param.sim.Ncell
            V_total = Lbox ** 3
            V_cell = (Lbox / N_cell) ** 3
            mass = pkd * rhoc0 * V_total
            rho_m = mass / V_cell
            delta_rho  = (rho_m - np.mean(rho_m)) / np.mean(rho_m)
            PS_rho[ii]      = t2c.power_spectrum.power_spectrum_1d(delta_rho, box_dims=Lbox , kbins=kbins)[0]
            PS_rho_xHI[ii] = t2c.power_spectrum.cross_power_spectrum_1d(delta_XHI, delta_rho,box_dims=Lbox, kbins=kbins)[0]
            PS_rho_xal[ii] = t2c.power_spectrum.cross_power_spectrum_1d(delta_x_al, delta_rho, box_dims=Lbox, kbins=kbins)[0]
            PS_rho_T[ii]   = t2c.power_spectrum.cross_power_spectrum_1d(delta_T, delta_rho, box_dims=Lbox, kbins=kbins)[0]

        else:
            delta_rho = 0,0  #  rho/rhomean-1
            print('no density field provided.')

        z_arr[ii]  = zz_
        PS_xHI[ii], k_bins = t2c.power_spectrum.power_spectrum_1d(delta_XHI, box_dims=Lbox, kbins=kbins)
        PS_T[ii]   = t2c.power_spectrum.power_spectrum_1d(delta_T, box_dims=Lbox, kbins=kbins)[0]
        PS_xal[ii] = t2c.power_spectrum.power_spectrum_1d(delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_dTb[ii] = t2c.power_spectrum.power_spectrum_1d(delta_dTb , box_dims=Lbox, kbins=kbins)[0]

        PS_T_lyal[ii] = t2c.power_spectrum.cross_power_spectrum_1d(delta_T, delta_x_al, box_dims=Lbox, kbins=kbins)[0]
        PS_T_xHI[ii] = t2c.power_spectrum.cross_power_spectrum_1d(delta_T, delta_XHI, box_dims=Lbox, kbins=kbins)[0]
        PS_lyal_xHI[ii]  = t2c.power_spectrum.cross_power_spectrum_1d(delta_x_al, delta_XHI, box_dims=Lbox, kbins=kbins)[0]


    Dict = {'z':z_arr,'k':k_bins,'PS_xHI': PS_xHI, 'PS_T': PS_T, 'PS_xal': PS_xal, 'PS_dTb': PS_dTb, 'PS_T_lyal': PS_T_lyal, 'PS_T_xHI': PS_T_xHI,
                'PS_rho': PS_rho, 'PS_rho_xHI': PS_rho_xHI, 'PS_rho_xal': PS_rho_xal, 'PS_rho_T': PS_rho_T, 'PS_lyal_xHI':PS_lyal_xHI}

    pickle.dump(file=open('./physics/PS_' + str(nGrid) + 'MAR_' + model_name + '.pkl', 'wb'), obj=Dict)


def paint_ly_alpha_single_snap(filename, param, epsilon_factor=10):
    """
    Paint the  Lyman alpha profiles on a grid for a single snapshot named filename.

    Parameters
    ----------
    param : dictionnary containing all the input parameters
    filename : the name of the snapshot, contained in param.sim.halo_catalogs.

    Returns
    -------
    Does not return anything. Paints and stores the grids on the directory grid_outputs.
    """
    catalog_dir = param.sim.halo_catalogs
    z_start = param.solver.z
    model_name = param.sim.model_name
    M_Bin = np.logspace(np.log10(param.sim.M_i_min), np.log10(param.sim.M_i_max), param.sim.binn, base=10)

    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    catalog = catalog_dir + filename
    halo_catalog = Read_Rockstar(catalog, Nmin=param.sim.Nh_part_min)
    H_Masses, H_X, H_Y, H_Z, H_Radii = halo_catalog['M'], halo_catalog['X'], halo_catalog['Y'], halo_catalog['Z'], halo_catalog['R']
    z = halo_catalog['z']

    # quick load to find matching redshift between solver output and simulation snapshot.
    grid_model = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]), 'rb'))
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]

    ##screening for xal
    epsilon = LBox / nGrid / epsilon_factor

    Indexing = np.argmin( np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))), axis=1)
    print('There are', H_Masses.size, 'halos at z=', z, )

    if H_Masses.size == 0:
        print('There aint no sources')
        Grid_xal = np.array([0])

    else:
        Pos_Bubles = np.vstack((H_X, H_Y, H_Z)).T  # Halo positions.
        Pos_Bubbles_Grid = np.array([Pos_Bubles / LBox * nGrid]).astype(int)[0]
        Pos_Bubbles_Grid[np.where(Pos_Bubbles_Grid == nGrid)] = nGrid - 1  # you don't want Pos_Bubbles_Grid==nGrid
        Grid_xal = np.zeros((nGrid, nGrid, nGrid))

        for i in range(len(M_Bin)):
            indices = np.where(Indexing == i)  ## indices in H_Masses of halos that have an initial mass at z=z_start between M_Bin[i-1] and M_Bin[i]
            if len(indices[0]) > 0:
                r_lyal = np.logspace(-6, 2, 4000, base=10)  ## physical distance for lyal profile
                Mhalo_z = M_Bin[i] * np.exp(-param.source.alpha_MAR * (z - z_start))
                rho_alpha_ = rho_alpha(r_lyal, Mhalo_z, z, param)[0]
                x_alpha_prof = 1.81e11 * (rho_alpha_) / (1 + zgrid)  # S_alpha(zgrid, T_extrap, 1 - xHII_extrap)
                profile_xal = interp1d(r_lyal * (1 + z), x_alpha_prof * (r_lyal * (1 + z) / (r_lyal * (1 + z) + epsilon)) ** 2, bounds_error=False, fill_value=0)  ##screening
                kernel_xal = profile_to_3Dkernel(profile_xal, nGrid, LBox)
                renorm = np.trapz(x_alpha_prof * 4 * np.pi * r_lyal ** 2, r_lyal) / (LBox / (1 + z)) ** 3 / np.mean( kernel_xal)
                if not np.sum(kernel_xal) < 1e-8 :
                    Grid_xal += put_profiles_group(Pos_Bubbles_Grid[indices], kernel_xal) * renorm

    pickle.dump(file=open('./grid_output/xal_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'wb'),obj=Grid_xal / 4 / np.pi)  #### WARNING : WE DIVIDE BY 4PI TO MATCH HM










def grid_dTb_from_profile(filename, param):
    """
    Paint the  Lyman alpha profiles on a grid for a single snapshot named filename.

    Parameters
    ----------
    param : dictionnary containing all the input parameters
    filename : the name of the snapshot, contained in param.sim.halo_catalogs.

    Returns
    -------
    Does not return anything. Paints and stores the grids on the directory grid_outputs.
    """
    catalog_dir = param.sim.halo_catalogs
    z_start = param.solver.z
    model_name = param.sim.model_name
    M_Bin = np.logspace(np.log10(param.sim.M_i_min), np.log10(param.sim.M_i_max), param.sim.binn, base=10)

    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    catalog = catalog_dir + filename
    halo_catalog = Read_Rockstar(catalog, Nmin=param.sim.Nh_part_min)
    H_Masses, H_X, H_Y, H_Z, H_Radii = halo_catalog['M'], halo_catalog['X'], halo_catalog['Y'], halo_catalog['Z'], halo_catalog['R']
    z = halo_catalog['z']

    # quick load to find matching redshift between solver output and simulation snapshot.
    grid_model = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]), 'rb'))
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]

    Indexing = np.argmin( np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))), axis=1)
    print('There are', H_Masses.size, 'halos at z=', z, )

    if H_Masses.size == 0:
        print('There aint no sources')
        Grid_dTb = np.array([0])

    else:
        Pos_Bubles = np.vstack((H_X, H_Y, H_Z)).T  # Halo positions.
        Pos_Bubbles_Grid = np.array([Pos_Bubles / LBox * nGrid]).astype(int)[0]
        Pos_Bubbles_Grid[np.where(Pos_Bubbles_Grid == nGrid)] = nGrid - 1  # you don't want Pos_Bubbles_Grid==nGrid
        Grid_dTb = np.zeros((nGrid, nGrid, nGrid))
        radial_grid = grid_model.r_grid_cell
        for i in range(len(M_Bin)):
            indices = np.where(Indexing == i)  ## indices in H_Masses of halos that have an initial mass at z=z_start between M_Bin[i-1] and M_Bin[i]
            if len(indices[0]) > 0:
                grid_model = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]),'rb'))
                dTb_profile = grid_model.dTb_history[str(round(zgrid, 2))]
                profile_dTb = interp1d(radial_grid * (1 + z), dTb_profile, bounds_error=False, fill_value=0)
                kernel_dTb = profile_to_3Dkernel(profile_dTb, nGrid, LBox)

                if not np.sum(kernel_dTb) < 1e-8 :
                    Grid_dTb += put_profiles_group(Pos_Bubbles_Grid[indices], kernel_dTb)

    pickle.dump(file=open('./grid_output/dTb_Grid_from_prof' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'wb'),obj=Grid_dTb)

