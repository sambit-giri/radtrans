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
from radtrans.astro import Read_Rockstar, NGamDot
import os
import copy
from radtrans.profiles_on_grid import profile_to_3Dkernel, Spreading_Excess_Fast,put_profiles_group
from radtrans.couplings import x_coll,rho_alpha, S_alpha


def run_solver(param):
    """
    Parameters
    ----------
    param : dictionnary containing all the input parameters

    Returns
    -------
    Does not return anything. However, it solve the RT equation for a range of halo masses, following their evolution from CDawn to the end of reionization. It stores the profile in a directory that it creates.
    """

    if param.sim.mpi4py == 'yes':
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
    elif param.sim.mpi4py == 'no':
        rank = 0
        size = 1
    else :
        print('param.sim.mpi4py should be yes or no')



    # check if folder exist to store gamma tables and profiles
    if not os.path.isdir('./gamma_tables'):
        os.mkdir('./gamma_tables')
    if not os.path.isdir('./profiles_output'):
        os.mkdir('./profiles_output')


    start_time = datetime.datetime.now()

    ##Simulation grid parameters
    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells

    ## Initial mass binning
    M_i_min = param.sim.M_i_min
    M_i_max = param.sim.M_i_max
    binn  = param.sim.binn  # let's start with 10 bins
    M_Bin = np.logspace(np.log10(M_i_min), np.log10(M_i_max), binn, base=10)

    z_start = param.solver.z
    z_end   = param.solver.z_end

    # Give a name to your sim, will be used to name all the files created.
    model_name = param.sim.model_name
    M_dark = param.source.M_min

    for ih, Mhalo in enumerate(M_Bin):
        if rank == ih % size:
            parameters = copy.deepcopy(param)
            if Mhalo < M_dark:
                parameters.source.M_halo = M_dark  # to compute rmax via NGamDot
            else:
                parameters.source.M_halo = Mhalo

            N_gam_dot = NGamDot(parameters)[0]
            parameters.source.M_halo = Mhalo

            print("Mh = %s starting at z = %d, being done by processor %d / %d" % (Mhalo, z_start, rank, size))

            ### Let's deal with r_end :
            cosmofile = parameters.cosmo.corr_fct
            vc_r, vc_corr = np.loadtxt(cosmofile, usecols=(0, 1), unpack=True)
            corr_tck = splrep(vc_r, vc_corr, s=0)
            r_MaxiMal = max(vc_r) / (1 + z_start)  ## Minimum k-value available in cosmofct.dat
            cosmo_corr = splev(r_MaxiMal * (1 + z_start), corr_tck)
            halo_bias = bias(z_start, parameters, Mass=Mhalo)
            # baryonic density profile in [cm**-3]
            nHI0_profile = profile(halo_bias, cosmo_corr, parameters,
                                   z_start) * parameters.cosmo.Ob / parameters.cosmo.Om * \
                           M_sun * parameters.cosmo.h ** 2 / (cm_per_Mpc) ** 3 / m_H
            r_end = (3 * N_gam_dot * 10 * sec_per_year * 1e6 / 4 / np.pi / parameters.cosmo.Ob / np.mean(
                nHI0_profile)) ** (1.0 / 3) / cm_per_Mpc
            ### All this above is to set the r_end (depending on the source strenght)

            parameters.solver.r_end = max(10 * LBox / nGrid, min(50 * r_end,r_MaxiMal))  # in case r_End is too small, we set it to 10*LBox/nGrid.
            print('r_end is ', '{:.2e}'.format(min(50 * r_end, r_MaxiMal)), 'Mpc.')

            parameters.table.filename_table = './gamma_tables/gamma_'+model_name+'_Mh_{:.1e}_z{}.pkl'.format(Mhalo, round(z_start, 2))

            print('Solving the RT equations ..')
            grid_model = rad.Source_MAR(parameters)
            grid_model.solve(parameters)
            pickle.dump(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, Mhalo), 'wb'), obj=grid_model)
            print('... RT equations solved. Profiles stored.')
            print(' ')

    end_time = datetime.datetime.now()
    print('It took in total:', end_time - start_time)




def paint_profiles(param):
    """
    Parameters
    ----------
    param : dictionnary containing all the input parameters

    Returns
    -------
    Does not return anything.
    """
    if param.sim.mpi4py == 'yes':
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
    elif param.sim.mpi4py == 'no':
        rank = 0
        size = 1
    else :
        print('param.sim.mpi4py should be yes or no')
    starttimeprofile = datetime.datetime.now()
    ##Simulation grid parameters
    LBox = param.sim.Lbox  # Mpc/h
    nGrid = param.sim.Ncell  # number of grid cells
    catalog_dir = param.sim.halo_catalogs
    if catalog_dir is None :
        print('You should specify param.sim.halo_catalogs. Should be a file containing the rockstar halo catalogs.')

    print('Painting T and ion profiles on a grid with', nGrid,'pixels per dim. Box size is',LBox ,'cMpc/h.')

    M_Bin = np.logspace(np.log10(param.sim.M_i_min), np.log10(param.sim.M_i_max), param.sim.binn, base=10)
    z_start = param.solver.z

    model_name = param.sim.model_name
    Om, Ob = param.cosmo.Om, param.cosmo.Ob
    h0 = param.cosmo.h
    factor = 27 * Om * h0 ** 2 / 0.023 * np.sqrt(0.15 / Om / h0 ** 2 / 10) # factor used in dTb calculation


    for ii, filename in enumerate(os.listdir(catalog_dir)):
        if rank == ii%size :
            catalog = catalog_dir+filename
            print('Halo catalog is ',filename , 'rank is:',rank)

            halo_catalog = Read_Rockstar(catalog)
            H_Masses, H_X, H_Y, H_Z, H_Radii = halo_catalog['M'],halo_catalog['X'],halo_catalog['Y'],halo_catalog['Z'],halo_catalog['R']
            z = halo_catalog['z']
            T_adiab_i = Tcmb0 * (1 + z_start) ** 2 / (1 + 250) # to consistently put to T_adiab the large scale IGM regions (pb with overlaps)
            T_adiab_z = Tcmb0 * (1 + z) ** 2 / (1 + 250) # to consistently put to T_adiab the large scale IGM regions (pb with overlaps)
            T_cmb_z = Tcmb0 * (1+z)


            # quick load to find matching redshift between solver output and simulation snapshot.
            grid_model = pickle.load(file=open('./profiles_output/SolverMAR_'+model_name+'_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]),'rb'))
            ind_z = np.argmin(np.abs(grid_model.z_history-z))
            zgrid = grid_model.z_history[ind_z]

            Indexing = np.digitize(H_Masses * np.exp(param.source.alpha_MAR*(z-z_start)), M_Bin) ## values of Mh at z_start, binned via M_Bin.
            print('There are',H_Masses.size,'halos at z=',z,)

            if H_Masses.size == 0:
                print('There aint no sources')
                Grid_xHII = np.array([0])
                Grid_Temp = np.array([T_adiab_z])
                Grid_xal = np.array([0])
                    #np.array([x_coll(z=z, Tk=T_adiab_z, xHI=1, rho_b= (1+z)**3 * Ob * rhoc0 * M_sun / cm_per_Mpc ** 3 / m_H)]) #xcoll value

            else :
                ## Here we compute quickly the cumulative fraction of ionized volume and check if it largely exceeds the box volume
                Ionized_vol = 0
                for i in range(len(M_Bin)+1):
                    nbr_halos = np.where(Indexing==i)[0].size
                    if i == len(M_Bin):# # for masses that are larger than the max M_Bin mass
                        i = i-1
                        grid_model = pickle.load(file=open('./profiles_output/SolverMAR_'+model_name+'_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]), 'rb'))
                        volume = np.trapz(4 * np.pi * grid_model.r_grid ** 2 * grid_model.xHII_History[str(round(zgrid, 2))], grid_model.r_grid)
                        Ionized_vol += volume * nbr_halos


                if Ionized_vol > 10 * LBox**3: ### I put the number 10 by hand, since this does not exactly give the same result as on the grid.
                    print('universe is fully inoinzed. Return [1] for the XHII, T and xtot grid.')
                    Grid_xHII = np.array([1])
                    Grid_Temp = np.array([1])
                    Grid_xal = np.array([1])

                else :
                    H_positions = np.vstack((H_X,H_Y,H_Z)).T
                    Pos_Bubles = H_positions

                    Pos_Bubbles_Grid = np.array([Pos_Bubles / LBox  * nGrid]).astype(int)[0]
                    Pos_Bubbles_Grid[np.where(Pos_Bubbles_Grid==nGrid)] = nGrid-1  #you don't want Pos_Bubbles_Grid==nGrid
                    Grid_xHII_i = np.zeros((nGrid, nGrid, nGrid))
                    Grid_Temp = np.zeros((nGrid, nGrid, nGrid))
                    Grid_xal = np.zeros((nGrid, nGrid, nGrid))



                    for i in range(len(M_Bin)+1):
                        indices = np.where(Indexing==i) ## indices in H_Masses of halos that have an initial mass at z=z_start between M_Bin[i-1] and M_Bin[i]

                        if i==len(M_Bin):## for masses that are larger than the max M_Bin, stick them to M_Bin[-1]
                            i = i-1

                        grid_model   = pickle.load(file=open('./profiles_output/SolverMAR_'+model_name+'_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]), 'rb'))
                        Temp_profile = grid_model.T_history[str(round(zgrid,2))]
                        radial_grid   = grid_model.r_grid

                        x_HII_profile = grid_model.xHII_History[str(round(zgrid,2))]

                        rho_alpha_ = rho_alpha(radial_grid, np.array([M_Bin[i]]), np.array([z]), param)[0][0]
                        x_al_profile = 1.81e11 * rho_alpha_ * S_alpha(z, Temp_profile, 1 - x_HII_profile) / (1 + z)  # grid_model.x_al_history[str(round(zgrid,2))]

                        Temp_profile[np.where(Temp_profile<=T_adiab_i+0.2)] = 0 # set to zero to avoid spurious addition - we put the +0.2 to be sure....


                        if len(indices[0])>0 :
                            ## here we assume they all have M_bin[i]
                            profile_xHII = interp1d(radial_grid*(1+z), x_HII_profile,bounds_error = False,fill_value=(1,0))
                            kernel_xHII = profile_to_3Dkernel(profile_xHII, nGrid, LBox)

                            profile_T = interp1d(radial_grid*(1+z),Temp_profile,bounds_error = False,fill_value=0)  #rgrid*(1+z) is in comoving coordinate, box too.
                            kernel_T = profile_to_3Dkernel(profile_T, nGrid, LBox)

                            profile_xal = interp1d(radial_grid*(1+z),x_al_profile,bounds_error = False,fill_value=0)
                            kernel_xal = profile_to_3Dkernel(profile_xal, nGrid, LBox)

                            if not np.sum(kernel_T)<1e-8:
                                Grid_Temp += put_profiles_group(Pos_Bubbles_Grid[indices],kernel_T)
                            if not np.sum(kernel_xal)<1e-8:
                                Grid_xal += put_profiles_group(Pos_Bubbles_Grid[indices],kernel_xal)


                            if np.any(kernel_xHII>0) and np.max(kernel_xHII)>1e-8: ## To avoid error from convole_fft (renomalization)
                                Grid_xHII_i += put_profiles_group(Pos_Bubbles_Grid[indices],kernel_xHII)
                            else:
                                continue
                        endtimeprofile = datetime.datetime.now()
                        print('Putting profiles : done. took : ',endtimeprofile-starttimeprofile)

                    Grid_Storage = np.copy(Grid_xHII_i)
                    Grid_Temp[np.where(Grid_Temp<T_adiab_i)] = T_adiab_z

                    Grid_xHII    = Spreading_Excess_Fast(Grid_Storage)
                    endtimespread= datetime.datetime.now()
                    print('It took:', endtimespread-endtimeprofile,'to spread the excess photons')


                    if np.all(Grid_xHII==0):
                        Grid_xHII = np.array([0])
                    if np.all( np.round((Grid_xal/(1+Grid_xal)),4)==1.): ## check if we are in the saturated regime
                        Grid_xal = np.array([1e10])

                    if np.all(Grid_xHII==1):
                        print('universe is fully inoinzed. Return [1] for the XHII, T and xtot grid.')
                        Grid_xal = np.array([1])
                        Grid_Temp = np.array([1])
                        Grid_xHII = np.array([1])

            #Grid_dTb_over_rho_b = factor * np.sqrt(1+z) * Grid_xtot/(1+Grid_xtot)* (1-T_cmb_z/Grid_Temp) * (1-Grid_xHII) #careful, this is dTb/(1+deltab)

            # Store Tk, xHII, and xtot on a grid.
            if param.sim.store_grids == True:
                if not os.path.isdir('./grid_output'):
                    os.mkdir('./grid_output')
                pickle.dump(file=open('./grid_output/T_Grid'+str(nGrid)+'MAR_'+model_name+'_snap'+filename[4:-5],'wb'),obj = Grid_Temp)
                pickle.dump(file=open('./grid_output/xHII_Grid'+str(nGrid)+'MAR_'+model_name+'_snap'+filename[4:-5],'wb'),obj = Grid_xHII)
                pickle.dump(file=open('./grid_output/xal_Grid'+str(nGrid)+'MAR_'+model_name+'_snap'+filename[4:-5],'wb'),obj = Grid_xal)
                #pickle.dump(file=open('./grid_output/dTb_Grid'+ str(nGrid) +'MAR_' + model_name + '_snap' + filename[4:-5],'wb'), obj=Grid_dTb_over_rho_b)
        print(' ')
    endtime = datetime.datetime.now()
    print('END. it took in total: ',endtime-starttimeprofile,'to paint the grids.')
    print('  ')


def grid_xcoll(param):
    """
    Creates a grid of xcoll. Needs to read in Tk grid, xHII grid and density field on grid.
    """
    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name
    nGrid = param.sim.Ncell
    Om, Ob = param.cosmo.Om, param.cosmo.Ob

    for ii, filename in enumerate(os.listdir(catalog_dir)):
        with open(catalog_dir+filename, "r") as file:
            file.readline()
            a = float(file.readline()[4:])
            zz_ = 1 / a - 1
        Grid_Temp           = pickle.load(file=open('./grid_output/T_Grid'    + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_xHII           = pickle.load(file=open('./grid_output/xHII_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))

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
            rho_m_pkd = mass / V_cell
        else :
            rho_m_pkd = rhoc0 * Om

        rho_bar_grid = rho_m_pkd * Ob/Om * (1 + zz_) ** 3 * M_sun / (cm_per_Mpc) ** 3 / m_H
        xHI_grid = 1-Grid_xHII  ### neutral fraction
        xcoll_grid = x_coll(zz_, Grid_Temp, xHI_grid, rho_bar_grid)

        pickle.dump(file=open('./grid_output/xcoll_Grid'+str(nGrid)+'MAR_'+model_name+'_snap'+filename[4:-5],'wb'),obj = xcoll_grid)


def compute_GS(param):
    """
    Reads in the grids and compute the global quantities averaged.
    """
    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name
    nGrid = param.sim.Ncell
    Om, Ob = param.cosmo.Om, param.cosmo.Ob
    Tadiab = []
    z_ = []
    Tk = []
    dTb  =[]
    x_HII = []
    x_al = []

    for ii, filename in enumerate(os.listdir(catalog_dir)):
        with open(catalog_dir+filename, "r") as file:
            file.readline()
            a = float(file.readline()[4:])
            zz_ = 1 / a - 1
        Grid_Temp           = pickle.load(file=open('./grid_output/T_Grid'    + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_xHII           = pickle.load(file=open('./grid_output/xHII_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        Grid_xal            = pickle.load(file=open('./grid_output/xal_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))
        #Grid_dTb_over_rho_b = pickle.load(file=open('./grid_output/dTb_Grid'  + str(nGrid) + 'MAR_' + model_name + '_snap' + filename[4:-5], 'rb'))

        dens_field = param.sim.dens_field
        if dens_field is not None and param.sim.Ncell == 256:
            dens = np.fromfile(dens_field + filename[4:-5]+ '.0',dtype=np.float32)
            pkd=dens.reshape(256,256,256)
            pkd = pkd.T ### take the transpose to match X_ion map coordinates
            Lbox = param.sim.Lbox
            N_cell = param.sim.Ncell
            V_total = Lbox**3
            V_cell = (Lbox/N_cell)**3
            mass = pkd * rhoc0 * V_total
            rho_m = mass / V_cell
            delta_b = (rho_m)/np.mean(rho_m)

        z_.append(zz_)
        Tk.append(np.mean(Grid_Temp))
        x_HII.append(np.mean(Grid_xHII))
        x_al.append(np.mean(Grid_xal))
        dTb.append(np.mean(Grid_dTb_over_rho_b*delta_b))
        Tadiab.append( Tcmb0 * (1+zz_)**2/(1+250) )


    if not os.path.isdir('./physics'):
        os.mkdir('./physics')
    Dict = {'Tk':Tk,'x_HII':x_HII,'x_al':x_al,'dTb':dTb,'Tadiab':Tadiab,'z':z_}
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
    kbins = np.logspace(np.log10(3e-2), np.log10(4), 400, base=10)
    # (input_array_nd, kbins=100, box_dims=None, return_n_modes=False, binning='log', breakpoint=0.1

    PS_XHI = t2c.power_spectrum.power_spectrum_1d(delta_XHI, box_dims=100 / 0.7, kbins=20)
    PS_rho = t2c.power_spectrum.power_spectrum_1d(delta_rho, box_dims=100 / 0.7, kbins=20)
    PS_cross = t2c.power_spectrum.cross_power_spectrum_1d(delta_XHI, delta_rho, box_dims=100 / 0.7, kbins=20)




