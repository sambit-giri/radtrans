
import radtrans as rad
from scipy.interpolate import splrep,splev, interp1d
import numpy as np
import pickle
import datetime
from radtrans.constants import cm_per_Mpc, sec_per_year, M_sun, m_H, rhoc0, Tcmb0
from radtrans.cosmo import T_adiab
import os
import copy
from radtrans.profiles_on_grid import profile_to_3Dkernel, Spreading_Excess_Fast, put_profiles_group, stacked_lyal_kernel
from radtrans.couplings import x_coll,rho_alpha, S_alpha
from radtrans.global_qty import J_alpha_n
import pkg_resources
from .constants import *
from .cosmo import comoving_distance, Hubble, hubble
from scipy.integrate import cumtrapz
from .couplings import eps_lyal




def Read_Ross_halo(file,param,Npart_ = 4000):
    """
    Read in a Ross halo catalog and return a dictionnary with all the information stored.
    M= Mgrid*Particle_Mesh**3 (mass is expressed in grid units)
    Grid cells : 1,1,1 is 0,0,0 (Fortran)
    output :
    Positions in cMpc/h and masses in Msol/h
    """

    Lbox  = param.sim.Lbox   # Mpc/h, Ross 2019 box size
    if Lbox!= 244 :
        print('WARNING : in C2ray sim, Lbox should be 244 Mpc/h')
    Npart = Npart_  # Npart per length from the sim
    Ncell = 250     # number of cells used to give halo positions. Fixed number, corresponds to Ross simulation

    Om = param.cosmo.Om
    M_box = rhoc0 * Om * (Lbox) ** 3  # total mass in box Msol/h
    Conversion = M_box / Npart ** 3 / param.cosmo.h  # conversion factor. to get mass in Msol see tools21cm/conv.py

    Halo_File = []
    with open(file) as f:
        for line in f:
            Halo_File.append(line)
    LMACHs, HMACHs = [], []
    X, Y, Z = [], [], []  ### Grid Box coordinates
    for i in range(len(Halo_File)):
        line = Halo_File[i].split()
        if i%2 == 1: ## One line over two
            LMACHs.append(float(line[4])) ###There are many more non zero lines in column 4; so we guess this means column 4 is LMACH and 3 HMACH
            HMACHs.append(float(line[3]))
            X.append(int(line[0]))
            Y.append(int(line[1]))
            Z.append(int(line[2]))

    LMACHs, HMACHs = np.array(LMACHs) * Conversion * param.cosmo.h , np.array(HMACHs) * Conversion * param.cosmo.h  ### in Msol/h

    X_pos = (np.array(X) - 1) / Ncell * Lbox  # Grid_X goes from 1 to Ncell. Hence the "-1"
    Y_pos = (np.array(Y) - 1) / Ncell * Lbox
    Z_pos = (np.array(Z) - 1) / Ncell * Lbox  # This is the true comoving position in Mpc/h

    Dict = {'HMACHs': HMACHs, 'LMACHs': LMACHs, 'X': X_pos, 'Y': Y_pos, 'Z': Z_pos}

    return Dict




def paint_profile_single_snap(filename,param,temp =True,lyal=True,ion=True):
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
    catalog = catalog_dir + filename
    halo_catalog = Read_Ross_halo(catalog,param)
    HMACHs, H_X, H_Y, H_Z, LMACHs = halo_catalog['HMACHs'], halo_catalog['X'], halo_catalog['Y'], halo_catalog['Z'],halo_catalog['LMACHs'] ### Fortran so 1,1,1 is 0,0,0
    H_Masses = 7.1 * LMACHs + 1.7 * HMACHs

    z = float(filename[0:-30])
    T_adiab_z = T_adiab(z,param)  # to consistently put to T_adiab the large scale IGM regions (pb with overlaps)


    ### Add up the adiabatic temperature
    dens_field = param.sim.dens_field

    if dens_field is not None and nGrid == 250:
        dens = np.fromfile(dens_field + filename[4:-5] + '.0', dtype=np.float32)
        pkd = dens.reshape(250, 250, 250)
        pkd = pkd.T  ### take the transpose to match X_ion map coordinates
        V_total = LBox ** 3
        V_cell = (LBox / nGrid) ** 3
        mass = pkd * rhoc0 * V_total
        rho_m = mass / V_cell
        delta_b = (rho_m) / np.mean(rho_m)-1
    else:
        delta_b = np.array([0])  # rho/rhomean-1 (usual delta here..)

    # quick load to find matching redshift between solver output and simulation snapshot.
    grid_model = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]), 'rb'))
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]
    T_adiab_z_solver = grid_model.T_history[str(round(zgrid, 2))][-1]  ## solver does not give exactly the correct adiabatic temperature, and this causes troubles

    Indexing = np.argmin( np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))), axis=1)
    print('There are', H_Masses.size, 'halos at z=', z, )

    if H_Masses.size == 0:
        print('There aint no sources')
        Grid_xHII = np.array([0])
        Grid_Temp = T_adiab_z * (1+delta_b)**(2/3)
        Grid_xal = np.array([0])

    else:
        Pos_Bubles = np.vstack((H_X, H_Y, H_Z)).T  # comoving Halo positions
        Pos_Bubbles_Grid = np.array([Pos_Bubles / LBox * nGrid]).astype(int)[0]
        Pos_Bubbles_Grid[np.where(Pos_Bubbles_Grid == nGrid)] = nGrid - 1  # you don't want Pos_Bubbles_Grid==nGrid

        Grid_xHII_i = np.zeros((nGrid, nGrid, nGrid))
        Grid_Temp = np.zeros((nGrid, nGrid, nGrid))
        Grid_xal = np.zeros((nGrid, nGrid, nGrid))

        for i in range(len(M_Bin)):
            indices = np.where( Indexing == i)  ## indices in H_Masses of halos that have an initial mass at z=z_start between M_Bin[i-1] and M_Bin[i]
            if len(indices[0]) > 0:
                grid_model = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]),'rb'))
                if temp == 'neutral':
                    Temp_profile = grid_model.T_neutral_hist[str(round(zgrid, 2))]
                else:
                    Temp_profile = grid_model.T_history[str(round(zgrid, 2))]

                radial_grid = grid_model.r_grid_cell
                x_HII_profile = grid_model.xHII_history[str(round(zgrid, 2))]
                # x_al_profile  = grid_model.x_al_history[str(round(zgrid, 2))]

                r_lyal = np.logspace(-5, 2, 1000, base=10)  ## physical distance for lyal profile. Never goes further away than 100 pMpc/h (checked)
                rho_alpha_ = rho_alpha_Ross(r_lyal, grid_model.Mh_history[ind_z], zgrid, param)[0]
                # T_extrap = np.interp(r_lyal, radial_grid, grid_model.T_history[str(round(zgrid, 2))])
                # xHII_extrap = np.interp(r_lyal, radial_grid, grid_model.xHII_history[str(round(zgrid, 2))])
                x_alpha_prof = 1.81e11 * (rho_alpha_) / (1 + zgrid)  # * S_alpha(zgrid, T_extrap, 1 - xHII_extrap)

                #### CAREFUL ! this step has to be done AFTER using Tk_profile to compute x_alpha (via Salpha)
                # Temp_profile[np.where(Temp_profile <= T_adiab_z + 0.2)] = 0 # set to zero to avoid spurious addition - we put the +0.2 to be sure....
                Temp_profile = (Temp_profile - T_adiab_z_solver).clip(min=0)

                ## here we assume they all have M_bin[i]
                profile_xHII = interp1d(radial_grid * (1 + z), x_HII_profile, bounds_error=False, fill_value=(1, 0))
                kernel_xHII = profile_to_3Dkernel(profile_xHII, nGrid, LBox)


                profile_T = interp1d(radial_grid * (1 + z), Temp_profile, bounds_error=False,  fill_value=0)  # rgrid*(1+z) is in comoving coordinate, box too.
                kernel_T = profile_to_3Dkernel(profile_T, nGrid, LBox)



                if lyal == True:
                    kernel_xal = stacked_lyal_kernel(r_lyal * (1 + z), x_alpha_prof, LBox, nGrid, nGrid_min=32)
                    renorm = np.trapz(x_alpha_prof * 4 * np.pi * r_lyal ** 2, r_lyal) / (LBox / (1 + z)) ** 3 / np.mean(kernel_xal)

                if (temp == True or temp == 'neutral') and np.any(kernel_T > 0):
                    Grid_Temp += put_profiles_group(Pos_Bubbles_Grid[indices], kernel_T * 1e-7 / np.sum(kernel_T)) * np.sum(kernel_T) / 1e-7

                if lyal == True:
                    Grid_xal += put_profiles_group(Pos_Bubbles_Grid[indices], kernel_xal * 1e-7 / np.sum(kernel_xal)) * renorm * np.sum(kernel_xal) / 1e-7  # we do this trick to avoid error from the fft when np.sum(kernel) is too close to zero.


                # if np.any(kernel_xHII > 0) and np.max(kernel_xHII) > 1e-8 and ion==True:  ## To avoid error from convole_fft (renomalization)
                if np.any(kernel_xHII > 0) and ion == True:

                    #### To deal with the situation where the ion front is smaller than the grid cell size
                    ion_front = np.argmin(np.abs(x_HII_profile - 0.5))
                    cell_length = LBox / nGrid
                    cell_vol = cell_length ** 3
                    central_cell = (int(nGrid / 2), int(nGrid / 2), int(nGrid / 2))

                    if radial_grid[ion_front] * (1 + z) < cell_length:  # if ion fron tis smaller than grid cell
                        inner_ind = np.where(radial_grid * (1 + z) < cell_length)
                        kernel_xHII[central_cell] = 1 / cell_vol * np.trapz(4 * np.pi * radial_grid[inner_ind] ** 2 * x_HII_profile[inner_ind],radial_grid[inner_ind]) * (1 + z) ** 3

                    Grid_xHII_i += put_profiles_group(Pos_Bubbles_Grid[indices], kernel_xHII * 1e-7 / np.sum(kernel_xHII)) * np.sum(kernel_xHII) / 1e-7


                #ara = ara+2

            endtimeprofile = datetime.datetime.now()
            print(len(indices[0]), 'halos in mass bin ', i, 'took : ', endtimeprofile - starttimeprofile,'to paint profiles')

        Grid_Storage = np.copy(Grid_xHII_i)

        if np.sum(Grid_Storage) < nGrid ** 3 and ion == True:
            Grid_xHII = Spreading_Excess_Fast(Grid_Storage)
        else:
            Grid_xHII = np.array([1])

        endtimespread = datetime.datetime.now()
        print('It took:', endtimespread - endtimeprofile, 'to spread the excess photons')

        if np.all(Grid_xHII == 0):
            Grid_xHII = np.array([0])

        if np.all(Grid_xHII == 1):
            print( 'universe is fully inoinzed. return 1 for Grid_xHII.')
            Grid_xHII = np.array([1])

        # Grid_dTb_over_rho_b = factor * np.sqrt(1+z) * Grid_xtot/(1+Grid_xtot)* (1-T_cmb_z/Grid_Temp) * (1-Grid_xHII) #careful, this is dTb/(1+deltab)
        Grid_Temp += T_adiab_z * (1+delta_b)**(2/3)


    # Store Tk, xHII, and xal on a grid.
    if param.sim.store_grids == True:
        if not os.path.isdir('./grid_output'):
            os.mkdir('./grid_output')
        if temp == True or temp == 'neutral':
            pickle.dump(file=open('./grid_output/T_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + str(z), 'wb'),obj=Grid_Temp)
        if ion == True:
            pickle.dump(file=open('./grid_output/xHII_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + str(z), 'wb'),obj=Grid_xHII)
        if lyal == True:
            pickle.dump(file=open('./grid_output/xal_Grid' + str(nGrid) + 'MAR_' + model_name + '_snap' + str(z), 'wb'),obj=Grid_xal/4/np.pi) #### WARNING : WE DIVIDE BY 4PI TO MATCH HM





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

    print('Painting T (',temp,'), lyal (',lyal, '), ion (' , ion,'),  profiles on a grid with', nGrid,'pixels per dim. Box size is',LBox ,'cMpc/h.')


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

    endtime = datetime.datetime.now()
    print('END. Stored Tgrid, xal grid and xHII grid. It took in total: ',endtime-starttimeprofile,'to paint the grids.')
    print('  ')







def rho_alpha_Ross(r_grid, MM, zz, param):
    """
    Ly-al coupling profile
    of shape (r_grid)
    - r_grid : physical distance around halo center in [pMpc/h]
    - zz  : redshift
    - MM  : halo mass
    """
    zstar = 35
    h0 = param.cosmo.h
    rectrunc = 23

    # rec fraction
    names = 'n, f'
    path_to_file = pkg_resources.resource_filename('radtrans', 'input_data/recfrac.dat')
    rec = np.genfromtxt(path_to_file, usecols=(0, 1), comments='#', dtype=float, names=names)

    # line frequencies
    nu_n = nu_LL * (1 - 1 / rec['n'][2:] ** 2)
    nu_n = np.insert(nu_n, [0, 0], np.inf)

    #rho_alpha = np.zeros((len(z_Bin), len(M_Bin), len(r_grid)))
    rho_alpha = np.zeros_like(r_grid)

    if zz < zstar:
        flux = []
        for k in range(2, rectrunc):
            zmax = (1 - (rec['n'][k] + 1) ** (-2)) / (1 - (rec['n'][k]) ** (-2)) * (1 + zz) - 1
            zrange = np.minimum(zmax, zstar) - zz

            N_prime = int(zrange / 0.01)  # dz_prime_lyal

            if (N_prime < 4):
                N_prime = 4

            z_prime = np.logspace(np.log(zz), np.log(zmax), N_prime, base=np.e)
            rcom_prime = comoving_distance(z_prime, param) * h0  # comoving distance in [cMpc/h]

            # What follows is the emissivity of the source at z_prime (such that at z the photon is at rcom_prime)
            # We then interpolate to find the correct emissivity such that the photon is at r_grid*(1+z) (in comoving unit)

            ### cosmidawn stuff, to compare
            alpha = param.source.alpha_MAR
            dMdt_int = MM * np.exp(alpha * (zz - z_prime)) / h0 * param.cosmo.Ob / param.cosmo.Om / ( 10 * 1e6)   # SFR Msol/h/yr Adapted from Ross Ngam dot expression

           # dMdt_int = alpha * MM * np.exp(alpha * (zz - z_prime)) * (z_prime + 1) * Hubble(z_prime,param) * f_star_Halo(param, MM * np.exp(alpha * (zz - z_prime))) * param.cosmo.Ob / param.cosmo.Om  # SFR Msol/h/yr

            eps_al = eps_lyal(nu_n[k] * (1 + z_prime) / (1 + zz), param)[None,:] * dMdt_int  # [photons.yr-1.Hz-1]
            eps_int = interp1d(rcom_prime, eps_al, axis=1, fill_value=0.0, bounds_error=False)

            #print('zz',zz, 'rgrid',r_grid * (1 + zz))
            flux_m = eps_int(r_grid * (1 + zz)) * rec['f'][k]  # want to find the z' corresponding to comoving distance r_grid * (1 + z).
            flux += [np.array(flux_m)]

        flux = np.array(flux)
        flux_of_r = np.sum(flux, axis=0)  # shape is (Mbin,rgrid)

        rho_alpha = flux_of_r / (4 * np.pi * r_grid ** 2) ## physical flux in [(pMpc/h)-2.yr-1.Hz-1]

    rho_alpha = rho_alpha * (h0 / cm_per_Mpc) ** 2 /sec_per_year  # [pcm-2.s-1.Hz-1]

    return rho_alpha



def xHII_approx_Ross(filename,param):

    catalog_dir = param.sim.halo_catalogs
    model_name = param.sim.model_name
    halo_catalog = Read_Ross_halo(catalog_dir + filename, param)
    z_start = param.solver.z
    HMACHs, H_X, H_Y, H_Z, LMACHs = halo_catalog['HMACHs'], halo_catalog['X'], halo_catalog['Y'], halo_catalog['Z'],   halo_catalog['LMACHs']  ### Fortran so 1,1,1 is 0,0,0
    H_Masses = 7.1 * LMACHs + 1.7 * HMACHs
    M_Bin = np.logspace(np.log10(param.sim.M_i_min), np.log10(param.sim.M_i_max), param.sim.binn, base=10)
    z = float(filename[0:-30])
    Indexing = np.argmin(np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))), axis=1)
    LBox = param.sim.Lbox
    grid_model = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]),'rb'))
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]

    Ionized_vol = 0
    for i in range(len(M_Bin)):
        indices = np.where( Indexing == i)  ## indices in H_Masses of halos that have an initial mass at z=z_start between M_Bin[i-1] and M_Bin[i]

        if len(indices[0]) > 0:
            nbr_halos = len(indices[0])
            print(nbr_halos,'halos in mass bin', i)
            grid_model = pickle.load( file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]), 'rb'))
            radial_grid = grid_model.r_grid_cell

            x_HII_profile = grid_model.xHII_history[str(round(zgrid, 2))]

            bubble_volume = np.trapz(4 * np.pi * radial_grid ** 2 * x_HII_profile, radial_grid)
            Ionized_vol += bubble_volume * nbr_halos  ##physical volume !!
            print('bubble_volume is ',bubble_volume)
    x_HII = Ionized_vol / (LBox / (1 + z)) ** 3  # normalize by total physical volume
    return zgrid, x_HII


def GS_Ross(param):
    catalog_dir = param.sim.halo_catalogs
    xHII = []
    z = []
    for ii, filename in enumerate(os.listdir(catalog_dir)):
        zz,xHII__ = xHII_approx_Ross(filename,param)
        xHII.append(xHII__)
        z.append(zz)
    xHII, z_array = np.array(xHII), np.array(z)
    matrice = np.array([xHII, z_array])
    z, xHII = matrice[:, matrice[0].argsort()] ## sort according to zarray
    Dict : {'z':z,'xHII':xHII}
    pickle.dump(file = open('./GS'+param.sim.model_name+'.pkl','wb'),obj = Dict)
    return Dict