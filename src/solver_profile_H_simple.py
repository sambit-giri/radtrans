"""""""""
Here we solve the coupled heat/ion RT equation with hydrogen but we simplify these equations to its most basic form.
"""""""""



from scipy.interpolate import splrep,splev
import scipy.integrate as integrate
from numpy import *
import numpy as np
from astropy import units as u
from astropy.cosmology import WMAP7 as pl
from scipy.optimize import fsolve
from sys import exit
import datetime
from .bias import *
from .astro import *
from .cosmo import T_adiab, correlation_fct
from .cross_sections import *
from .couplings import x_coll, rho_alpha, S_alpha, J0_xray_lyal
import copy
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz


###Constants
facr = 1 * u.Mpc
cm_per_Mpc = (facr.to(u.cm)).value
eV_per_erg = 6.242 * 10 ** 11   #(1 * u.erg).to(u.eV).value





def generate_table(param, z, n_HI):
    '''
    Generate the interpolation tables for the integrals for the radiative transfer equations.

    This function generates 3-D tables for each integral in the radiative transfer equation. The elements of the tables
    are the values of the integrals for a given set of variables (n_HI, n_HeI, n_HeII). This is done by initializing a
    array for each variable within a specified range and then loop over the arrays and use each set of variables to
    evaluate the integrals. The integral range is set up to include UV ionizing photons.

    Parameters
    ----------
    param : dictionary with source parameters
    z : float
     Redshift of the source.
    n_HI n_HeI n_HeII: array_like
     Cumulative (column density) Neutral hydrogen, HeI, HeII density array in Mpc/h.cm**-3 along r_grid.
    E_upp_ is the maximum energy above which we cut the integral. It's in a sense the max energy of the ionizing photon in our model
    Returns
    ------
    dict of {str:dict}
        Dictionary containing two sub-dictionaries: The first one containing the function variables and the second one
        containing the 12 tables for the integrals
    '''
    h0 = param.cosmo.h
    if param.table.import_table:
        if param.table.filename_table == None:
            print('Asking to import a table but filename_table is None. Exit')
            exit()
        else:
            Gamma_input_info = pickle.load(open(param.table.filename_table, 'rb'))
            print('Reading in table ', param.table.filename_table)

    else:

        print('Calculating table...')


        if (param.source.type == 'SED'):
            Ngam_dot_ion, E_dot_xray = NGamDot(param,param.solver.z)
            sed_ion = param.source.alS_ion
            sed_xray = param.source.alS_xray

            norm_ion = (1 - sed_ion) / ((param.source.E_max_sed_ion / h_eV_sec) ** (1 - sed_ion) - (param.source.E_min_sed_ion / h_eV_sec) ** (1 - sed_ion))
            norm_xray = (1 - sed_xray) / ((param.source.E_max_sed_xray / h_eV_sec) ** (1 - sed_xray) - (param.source.E_min_sed_xray / h_eV_sec) ** (1 - sed_xray))
            # nu**-alpha*norm  is a count of [photons.Hz**-1]

            def Nion(E, n_HI0):
                nu_ = Hz_per_eV * E
                int = cm_per_Mpc / param.cosmo.h * (n_HI0 * sigma_HI(E*0+13.6))  ##
                return np.exp(-int) * Ngam_dot_ion * nu_ ** (-sed_ion) * norm_ion * nu_  # this is [Hz*Hz**-1 . s**-1] normalized to Ngdot in the range min-max

            def Nxray(E, n_HI0):
                """
                input : E in eV, n_HIO in Mpc/h.cm**-3 (column density, see in generate table how column density are initialized)
                output : Divide it by 4*pi*r**2 ,and you get a flux [eV.s-1.r-2.eV-1], r meaning the unit of r
                """
                nu_ = Hz_per_eV * E
                int = cm_per_Mpc / h0 * (n_HI0 * sigma_HI(E*0+13.6))
                #print('WARNING : in Nion and Nxray we dont include frequency dependent sigma')
                return np.exp(-int) * E_dot_xray * norm_xray * nu_ ** (-sed_xray) * Hz_per_eV  # [eV/eV/s]

            nu_range = np.logspace(np.log10(param.source.E_min_sed_ion / h_eV_sec),np.log10(param.source.E_max_sed_ion / h_eV_sec), 3000, base=10)
            XraySed = Nxray(nu_range / Hz_per_eV, 1e-9)
            Ion_Sed = Nion(nu_range / Hz_per_eV, 1e-9)

        else :
            print('NEED SED FOR THE MODEL TYPE SOURCE')
            exit()

        IHI_1 = zeros((n_HI.size))
        IHI_2 = zeros((n_HI.size))

        IT_HI_1 = zeros((n_HI.size))
        IT_2a = zeros((n_HI.size))
        IT_2b = zeros((n_HI.size))

        E_range_ion  = np.logspace(np.log10(E_HI), np.log10(param.source.E_max_sed_ion), 500, base=10)
        E_range_xray = np.logspace(np.log10(param.source.E_min_xray), np.log10(param.source.E_max_xray), 500, base=10) #xray photon range



        IHI_1[:] = np.trapz(1 / E_range_ion * Nion(E_range_ion, n_HI[:, None]), E_range_ion)                             + param.source.xray_in_ion * np.trapz(1 / E_range_xray * Nxray(E_range_xray, n_HI[:, None]), E_range_xray)
        IHI_2[:] = np.trapz((E_range_ion - E_HI) / (E_HI * E_range_ion) * Nion(E_range_ion, n_HI[:, None]), E_range_ion) + param.source.xray_in_ion * np.trapz((E_range_xray - E_HI) / (E_HI * E_range_xray) * Nxray(E_range_xray, n_HI[:, None]), E_range_xray)


        ##xray and ionizing photon are included in heating. Integrate from E_HI to Emax
        IT_HI_1[:] = np.trapz((E_range_xray - E_HI) / E_range_xray * Nxray(E_range_xray, n_HI[:, None]), E_range_xray)   + param.source.ion_in_xray * np.trapz((E_range_ion - E_HI) / E_range_ion * Nion(E_range_ion, n_HI[:, None]), E_range_ion) #[eV/s]then divide by r2
        IT_2a[:]   = np.trapz(Nxray(E_range_xray, n_HI[:, None]) * E_range_xray, E_range_xray)                           + param.source.ion_in_xray * np.trapz(Nion(E_range_ion, n_HI[:, None]) * E_range_ion, E_range_ion)
        IT_2b[:]   = np.trapz(Nxray(E_range_xray, n_HI[:, None]) * (-4 * kb_eV_per_K), E_range_xray)                     + param.source.ion_in_xray * np.trapz(Nion(E_range_ion, n_HI[:, None]) * (-4 * kb_eV_per_K), E_range_ion)


        print('...done')

        Gamma_info = {'HI_1': IHI_1, 'HI_2': IHI_2,  'T_HI_1': IT_HI_1,'T_2a': IT_2a, 'T_2b': IT_2b}

        input_info = {'M': param.source.M_halo, 'z': z, 'type': param.source.type, 'N_ion_ph_dot': Ngam_dot_ion, 'E_dot_xray': E_dot_xray,
                      'n_HI': n_HI,  'E_0': param.source.E_min_sed_ion,
                      'E_upp': param.source.E_max_sed_ion,'SED':{'nu':nu_range,'ion':Ion_Sed,'xray':XraySed} }

        Gamma_input_info = {'Gamma': Gamma_info, 'input': input_info}

        if param.table.filename_table is None:
            filename_table = 'qwerty'
            print('No filename_table given. We will call it qwerty.')
        else:
            filename_table = param.table.filename_table
        print('saving table in pickle file :', filename_table)
        pickle.dump(Gamma_input_info, open(filename_table, 'wb'))

    return Gamma_input_info





class Source_H_ion_simple:
    """
    Source which ionizes the surrounding H and He gas along a radial direction.
    """

    def __init__(self, param):

        self.z = param.solver.z  # starting redshift
        self.alpha = param.source.alpha
        h0 = param.cosmo.h
        self.M_halo = param.source.M_halo #Msol/h
        self.R_halo = R_halo(self.M_halo/h0, self.z, param)  # physical halo size in Mpc
        self.r_start = self.R_halo * param.cosmo.h # Mpc/h
        print('R_halo is :', '{:.2e}'.format(self.R_halo), 'Mpc/h')
        self.r_end = param.solver.r_end  # maximal distance from source
        self.dn = param.solver.dn
        self.dn_table = param.solver.dn_table

        self.r_grid = linspace(self.r_start, self.r_end, self.dn) #Mpc/h, phyisical distance from halo center

        correlation_fct(param)
        dt_init =  param.solver.time_step * 1e6 * sec_per_year * u.s  ### time step of 0.1 Myr
        self.dt_init = dt_init

    def create_table(self, param):
        """
        Call the function to create the interpolation tables.

        Parameters
        ----------
        par : dict of {str:float}, optional
         Variables to pass on to the table generator. If none is given the parameters of the Source initialization will
         be used.
        """

         # phyisical distance from halo center in cMpc/h
        self.M_initial = param.source.M_halo
        self.z_initial = param.solver.z

        # Column densities in physical Mpc/h.cm**-3.
        nH_column = np.trapz( self.profiles(param, self.z_initial, Mass = self.M_initial), self.r_grid) * (1 + self.z_initial) ** 3
        print('n_H_column max : ', '{:.2e}'.format(nH_column), 'Mpc/h.cm**-3.')
        n_HI  = logspace(log10(nH_column  * 1e-10),  log10(1.05 * nH_column), self.dn_table, base=10)
        n_HI  = np.concatenate((np.array([0]), n_HI))

        Gamma_grid_info = generate_table(param, self.z_initial, n_HI)
        self.Gamma_grid_info = Gamma_grid_info



    def profiles(self,param,z,Mass = None):
        """
        2h profiles in nbr of [H atoms /co-cm**3] (comoving cm)
        output is the mean density between the values of r_grid
        size is size(r_grid)-1
        """
        # Profiles
        cosmofile = param.cosmo.corr_fct
        vc_r, vc_corr = np.loadtxt(cosmofile, usecols=(0, 1), unpack=True)
        corr_tck = splrep(vc_r, vc_corr, s=0)
        cosmo_corr = splev(self.r_grid * (1 + z), corr_tck)  # r_grid * (1+self.z) in cMpc/h --> To reach the correct scales and units for the correlation fucntion
        halo_bias = bias(z, param, Mass, bias_type='ST')
        # baryonic density profile in [cm**-3]
        norm_profile = profile(halo_bias, cosmo_corr, param,z) * param.cosmo.Ob / param.cosmo.Om * M_sun / (cm_per_Mpc) ** 3 / m_H
        #norm_profile = np.concatenate(([0],norm_profile)) ## store 0 corresponding to inside the halo (since it's fully ionized)
        return norm_profile

    def solve(self, param):
        """
        Solves the radiative transfer equation for the given source and the parameters.

        The solver calls grid parameters and initializes the starting grid with the initial conditions for the densities
        and the temperature. Using the time step, the radiative transfer equations are used to update the k-th cell from
        the radial grid for a certain time step dt_init. Then the solver moves on to the (k+1)-th cell, and uses the
        values calculated from the k-th cell in order to calculate the optical depth, which requires information of the
        densities from all prior cells. For each cell, we sum up the three densities from the starting cell up to the
        current cell and use these values to evaluate the 12 integrals in the equations, which is done by interpolation
        of the tables we generated previously. After each cell is updated for some time dt_init, we start again with the
        first cell and use the calculation from the previous time step as the initial condition and repeat the same
        process until the radial cells are updated l times such that l*dt_init has reached the evolution time.
        After the solver is finished we compare the ionization fronts of two consecutive runs and require an accuracy of 5% in
        order to finish the calculations. If the accuracy is not reached we store the values from the run and start
        again with time step size dt_init/2 and a radial grid with half the step size from the previous run.
        This process is repeated until the desired accuracy is reached.
        """

        print('Solving the radiative equations...')
        t_start_solver = datetime.datetime.now()
        z = self.z  #starting redshift
        h0 = param.cosmo.h
        dt_init = self.dt_init
        r_grid = self.r_grid

        self.create_table(param)

        n_HI = self.Gamma_grid_info['input']['n_HI']

        dr_max = r_grid[-1] - r_grid[-2]
        if dr_max * sigma_HI(13.6) * n_H_0 * cm_per_Mpc > 0.1:
            print('needs more spatial stepping. taumax is ', dr_max * sigma_HI(13.6) * n_H_0 * cm_per_Mpc)



        while True:

            Mh_history,z_grid,Nh_history = [],[],[]
            xHII_history = {}
            dTb_history = {}
            I1_HI_history, nHI_history ={},{}
            n_HII_cell, T_grid = zeros_like(self.r_grid), zeros_like(self.r_grid)
            T_grid += T_adiab(z,param) ### assume gas decoupled from cmb at z=param.cosmo.z_decoupl and then adiabatically cooled

            l = 0
            zstep_l = self.z_initial

            Gamma_info = self.Gamma_grid_info['Gamma']

            JHI_1, JHI_2  = Gamma_info['HI_1'], Gamma_info['HI_2']
            JT_HI_1       = Gamma_info['T_HI_1']
            JT_2a, JT_2b  = Gamma_info['T_2a'], Gamma_info['T_2b']
            Ng_dot_initial_ion= self.Gamma_grid_info['input']['N_ion_ph_dot']
            E_dot_initial_xray= self.Gamma_grid_info['input']['E_dot_xray']

            print('Solver will solve from z=', self.z_initial, 'to z=',param.solver.z_end, ', without turning off the source, with a time step of',param.solver.time_step,'Myr.')

            copy_param = copy.deepcopy(param)
            count__ = 0

            Mh_step_l = self.M_initial

            while Mh_step_l < param.source.M_min and zstep_l > param.solver.z_end  : ## while Mh(z) <Mmin, nothing happens
                age  = pl.age(self.z_initial)
                age  = age.to(u.s)
                age += l * self.dt_init
                func = lambda z: pl.age(z).to(u.s).value - age.value
                zstep_l = fsolve(func, x0 = zstep_l) ### zstep_l for initial guess
                Mh_step_l = self.M_initial * np.exp(param.source.alpha_MAR * (self.z_initial-zstep_l))
                self.nB_profile = self.profiles(param, zstep_l, Mass=Mh_step_l)
                if l*param.solver.time_step % 10 == 0 and l != 0:
                    Mh_history.append(Mh_step_l)
                    z_grid.append(zstep_l[0])
                    Tadiab = 2.725 * (1 + zstep_l) ** 2 / (1 + param.cosmo.z_decoupl)
                    nB_profile_z = self.nB_profile* (1 + zstep_l) ** 3
                    xHII_history[str(round(zstep_l[0], 2))] = 0
                    Nh_history.append(0)
                    I1_HI_history[str(round(zstep_l[0], 2))] = 0
                    nHI_history[str(round(zstep_l[0], 2))]  = 0
                l += 1

            T_grid = zeros_like(self.r_grid)
            T_grid += T_adiab(zstep_l,param)

            while zstep_l > param.solver.z_end :
                z_previous = zstep_l
                # Calculate the redshift z(t)
                age  = pl.age(self.z_initial) # careful. Here you add to z_initial l*dt_init and find the corresponding z.
                age  = age.to(u.s)
                age += l * self.dt_init
                func = lambda z: pl.age(z).to(u.s).value - age.value
                zstep_l = fsolve(func, x0 = zstep_l) ### zstep_l for initial guess

                ### Update halo mass, exponential growth
                Mh_step_l = self.M_initial * np.exp(param.source.alpha_MAR * (self.z_initial-zstep_l))
                copy_param.source.M_halo = Mh_step_l

                if param.source.type == 'SED' :
                    Ngam_dot_step_l_ion, E_dot_step_l_xray = NGamDot(copy_param,zstep_l) #[s**-1,eV/s]

                #### Update the profile due to expansion and Halo Growth
                self.nB_profile = self.profiles(param, zstep_l, Mass = Mh_step_l)



                ## dilute the ionized density according to redshift
                n_HII_cell = n_HII_cell * (1+zstep_l)**3 / (1+z_previous)**3

                #update the baryon profile
                nB_profile_z = self.nB_profile * (1 + zstep_l) ** 3  ## physical total baryon density

                n_HII_cell[n_HII_cell > nB_profile_z] = nB_profile_z[n_HII_cell > nB_profile_z] #xHII can't be >1
                n_HI_cell = nB_profile_z - n_HII_cell                       #set neutral physical density in cell

                dr = r_grid[1]-r_grid[0]                #lin spacing so tha'ts ok.
                r2 = self.r_grid ** 2

                n_HI_edge = (n_HI_cell[:-1]+n_HI_cell[1:])/2
                n_HI_edge = np.concatenate((np.array([0]),n_HI_edge,np.array([n_HI_edge[-1]])) )## value of the density at the edge of the cells. use this to cumpte the cumumlative densities at boundaries of cells size is (rgrid)


                K_HI = cumtrapz(n_HI_edge, np.concatenate((self.r_grid,np.array([self.r_grid[-1]+dr]))), initial=0.0) ## cumulative densities : before first cell, before second cell... etc up to right after last one. size is (r_grid)


                ionized_indices = np.where(n_HI_cell == 0)
                n_HI_cell[ionized_indices] = 1e-50 # to avoid division by zero

                """""""""""
                I1_HI   = np.interp(K_HI[:-1], n_HI, JHI_1) / r2 / cm_per_Mpc ** 2 / 4 / pi * h0**3
                I2_HI   = np.interp(K_HI[:-1], n_HI, JHI_2)  / r2 / cm_per_Mpc ** 2 / 4 / pi * h0**3
                I1_T_HI = np.interp(K_HI[:-1], n_HI, JT_HI_1) / r2 / cm_per_Mpc ** 2 / 4 / pi * h0**3 ## eV/s

                I2_Ta = np.interp(K_HI[:-1], n_HI, JT_2a) / r2 / cm_per_Mpc ** 2 / 4 / pi * h0**2
                I2_Tb = np.interp(K_HI[:-1] ,n_HI, JT_2b) / r2 / cm_per_Mpc ** 2 / 4 / pi * h0**2
                """""""""""

                I1_HI   = (np.interp(K_HI[:-1], n_HI, JHI_1)   - np.interp(K_HI[1:], n_HI, JHI_1)) / r2 / cm_per_Mpc ** 3 / 4 / pi / n_HI_cell / dr * h0**3
                I2_HI   = (np.interp(K_HI[:-1], n_HI, JHI_2)   - np.interp(K_HI[1:], n_HI, JHI_2)) / r2 / cm_per_Mpc ** 3 / 4 / pi / n_HI_cell / dr * h0**3
                I1_T_HI = (np.interp(K_HI[:-1], n_HI, JT_HI_1) - np.interp(K_HI[1:], n_HI, JT_HI_1)) / r2 / cm_per_Mpc ** 3 / 4 / pi / n_HI_cell / dr * h0**3 ## eV/s

                I2_Ta = np.interp(K_HI[:-1], n_HI, JT_2a) / r2 / cm_per_Mpc ** 2 / 4 / pi * h0**2
                I2_Tb = np.interp(K_HI[:-1], n_HI, JT_2b) / r2 / cm_per_Mpc ** 2 / 4 / pi * h0**2



                I1_HI[ionized_indices] = 0
                I2_HI[ionized_indices] = 0
                I1_T_HI[ionized_indices] = 0
                n_HI_cell[ionized_indices] = 0



                if param.source.type == 'SED':
                    I1_HI, I2_HI = np.nan_to_num((I1_HI, I2_HI)) * Ngam_dot_step_l_ion / Ng_dot_initial_ion # to account for source growth (via MAR)
                    I1_T_HI, I2_Ta, I2_Tb = np.nan_to_num((I1_T_HI, I2_Ta, I2_Tb)) * E_dot_step_l_xray / E_dot_initial_xray
                elif param.source.type == 'Ross':
                    I1_HI, I2_HI = np.nan_to_num((I1_HI, I2_HI)) * Ngam_dot_step_l_ion /Ng_dot_initial_ion  # to account for source growth (via MAR in Ngamma dot formula Ross et al)
                    I1_T_HI, I2_Ta, I2_Tb = np.nan_to_num((I1_T_HI, I2_Ta, I2_Tb)) * Mh_step_l / self.M_initial # no change in g_gamma in HMXB. see ross et al 2019..
                else:
                    I1_HI, I2_HI, I1_T_HI, I2_Ta, I2_Tb = np.nan_to_num((I1_HI, I2_HI, I1_T_HI, I2_Ta,I2_Tb)) * Ngam_dot_step_l / Ng_dot_initial_ion  ### added correctrion for halo growth



                n_ee = n_HII_cell  #e- density
                mu = nB_profile_z/ (nB_profile_z + n_HII_cell)  #molecular weigth

                ##coeff for Temp eq
                A1_HI = zeta_HI(T_grid) * n_HI_cell * n_ee
                A2_HII = eta_HII(T_grid) * n_HII_cell * n_ee
                A4_HI = psi_HI(T_grid) * n_HI_cell * n_ee
                A5 = theta_ff(T_grid) * (n_HII_cell) * n_ee # eV/s/cm**3
                H = pl.H(zstep_l)
                H = H.to(u.s ** -1).value
                A6 = (15 / 2 * H * kb_eV_per_K * T_grid * nB_profile_z / mu)


                A = I1_HI * n_HI_cell - alpha_HII(1e4) * n_HII_cell * n_ee

                #D = (2 / 3) * mu / (kb_eV_per_K) * (f_Heat(n_HII_cell / nB_profile_z) * (n_HI_cell * I1_T_HI) - A6 )  # sigma_s * n_ee / m_e_eV * (I2_Ta + T_grid * I2_Tb) - (A1_HI + A2_HII + A4_HI + A5 + A6)) ##K/s/cm**3   ###SIMPLE HEATING VERSION
                D = (2 / 3) * mu / (kb_eV_per_K) * (f_Heat(n_HII_cell / nB_profile_z) * (n_HI_cell * I1_T_HI) + sigma_s * n_ee / m_e_eV * (I2_Ta + T_grid * I2_Tb) - (A1_HI + A2_HII + A4_HI + A5 + A6)) ##K/s/cm**3

                n_HII_cell = n_HII_cell + dt_init.value * A # discretize the diff equation.
                n_HII_cell[np.where(n_HII_cell<0)] = 0

                T_nB = T_grid * nB_profile_z + dt_init.value * D #product of Tk * baryon physical density
                T_grid = T_nB/( nB_profile_z *(1+zstep_l)**3/(1+z_previous)**3 ) # to get the correct T~(1+z)**2 adiabatic regime, you need to account for the shift in baryon density

                n_HII_cell, T_grid = np.nan_to_num(n_HII_cell), np.nan_to_num(T_grid)

                n_HII_cell[n_HII_cell>nB_profile_z] = nB_profile_z[n_HII_cell>nB_profile_z]   ## xHII can't be >1



                if np.exp(-cm_per_Mpc * (K_HI[-1] * sigma_HI(13.6))) > 0.1 and count__==0:
                    print('WARNING : np.exp(-tau(rmax)) > 0.1. Some photons are not absorbed. Maybe you need larger rmax. ')
                    count__ = 1 #  to print this only once

                if l*param.solver.time_step % 10  == 0 and l != 0:
                    ydata = (nB_profile_z - n_HII_cell) / nB_profile_z
                    ydata = ydata.clip(min=0)  # remove neg elements

                    Mh_history.append(Mh_step_l)
                    z_grid.append(zstep_l[0])
                    xHII_history[str(round(zstep_l[0], 2))] = 1-ydata
                    nHI_history[str(round(zstep_l[0], 2))] = n_HI_cell
                    I1_HI_history[str(round(zstep_l[0], 2))] = I1_HI
                    Nh_history.append(Ngam_dot_step_l_ion)
                l += 1


            time_end_solve = datetime.datetime.now()
            print('solver took :', time_end_solve - t_start_solver)
            break

        #self.A_hist = A_grid_hist
        self.Nh_history = Nh_history
        self.n_HI_grid = nB_profile_z - n_HII_cell
        self.n_HII_cell = n_HII_cell
        self.n_B = nB_profile_z ## total density in nbr of Hatoms per physical-cm**3
        self.T_grid = np.copy(T_grid)
        self.z_history = np.array(z_grid)
        self.Mh_history = np.array(Mh_history)
        self.I1_HI_history = I1_HI_history
        self.nHI_history = nHI_history
        self.xHII_history = xHII_history
        self.r_grid_cell = r_grid

    def fit(self):
        p0 = [30/self.Ion_front_grid[-1][0],  self.Ion_front_grid[-1][0]]  # intial guess for the fit. c1 has to be increased when the ion front goes to smaller scales (sharpness, log scale)
        xdata, ydata = self.r_grid, self.n_HI_grid / self.n_H
        Fit_ = curve_fit(profile_1D_HI, xdata, ydata, p0=p0)
        self.c1 = Fit_[0][0]
        self.c2 = Fit_[0][1]



def profile_1D_HI(r, c1, c2):
    '''
    Sigmoid Function that we fit to the neutral fraction profile. In order to just store 2 parameters per profiles

    Parameters
    ----------
    r :  distance from the source in Mpc.
    c1 : shaprness of the profile (sharp = high c1)
    c2 : ionization front, value where profile_1D = 0.5
    '''
    out = 1/(1+np.exp(-c1*(r-c2)))
    return out