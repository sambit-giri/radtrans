
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

##Cosmology
Om      = 0.31
Ob      = 0.048
Ol      = 1-Om-Ob
h0      = 0.68
ns      = 0.97
s8      = 0.81


def n_H(z ,C):
    return C * n_H_0 * (1 + z) ** 3



def gamma_HI(n_HIIx, n_HI ,Tx, I1_HI, I2_HI, gamma_2c):
    """
    Calculate gamma_HI given the densities and the temperature
    Parameters
    ----------
    n_HIIx : float
     Ionized hydrogen density in cm**-3.
    n_HeIx : float
     Neutral helium density in cm**-3.
    n_HeIIx : float
     Single ionized helium density in cm**-3.
    n_HeIIIx : float
     Double ionized helium density in cm**-3.
    Tx : float
     Temperature of the gas in K.
    n_HI, n_HeI : float
     Neutral H and He densities
    Returns
    -------
    float
     Gamma_HI for the radiative transfer equation. [s-1]
    """
    n_e = n_HIIx
    X_HII = n_HIIx / (n_HIIx + n_HI)
    return gamma_2c + beta_HI(Tx) * n_e + I1_HI + f_H(X_HII) * I2_HI



def find_Ifront(x, r, show=False):
    """
    Finds the ionization front, the position where the ionized fraction is 0.5.

    Parameters
    ----------
    x : array_like
     Ionized hydrogen density along the radial grid.
    r : array_like
     Radial grid in linear space.
    show : bool, optional
     Decide whether or not to print the position of the ionization front

     Returns
     -------
     float
      Returns the position of the ionization front, one of the elements in the array r.
    """
    #m = argmin(abs(0.5 - x))
    outside_ = r[np.where(x<0.5)] ## where ionized fraction is basically 0
    inside_  = r[np.where(x>0.5)] ## where ionized fraction is basically 1

    if outside_.size  == 0 :  # fully ionized
        front = r[-1]
    elif inside_.size == 0 :  # fully neutral
        front = r[0]
    else :
        front = (np.max(inside_)+np.min(outside_))/2
    if show == True:
        print('Pos. of Ifront:', front )
    return front  #r[m]



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
    E_0_ = param.source.E_min_sed_ion
    E_upp_ = param.source.E_max_sed_ion  # [eV]
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

        if (param.source.type == 'Miniqsos'):  ### Choose the source type
            alpha = param.source.alpha
            M = param.source.M_miniqso  # Msol
            L = 1.38 * 10 ** 37 * eV_per_erg * M  # eV.s-1 , assuming 10% Eddington lumi.
            E_range_E0 = np.logspace(np.log10(E_0_), np.log10(E_upp_), 100, base=10)
            Ag = L / (np.trapz(E_range_E0 ** -alpha, E_range_E0))
            print('Miniqsos model chosen. M_qso is ', M)

            # Spectral energy function [eV.s-1.eV-1], emitted from source
            def I(E):
                miniqsos = E ** -alpha
                return Ag * miniqsos

            # Spectral energy function received after passing through column density n_HI0
            def N(E, n_HI0):
                """
                input : E in eV, n_HIO in Mpc/h.cm**-3 (column density, see in generate table how column density are initialized)
                output : Divide it by 4*pi*r**2 ,and you get a flux [eV.s-1.r-2.eV-1], r meaning the unit of r
                """
                int = cm_per_Mpc / param.cosmo.h * (n_HI0 * sigma_HI(E))
                return exp(-int) * I(E)

            E_range_HI_ = np.logspace(np.log10(13.6), np.log10(E_upp_), 1000, base=10)
            Ngam_dot_ion = np.trapz(I(E_range_HI_) / E_range_HI_, E_range_HI_)
            print('source emits', '{:.2e}'.format(Ngam_dot_ion), 'ionizing photons per seconds, in the energy range [', param.source.E_min_sed_ion, ',', param.source.E_upp, '] eV')

        elif (param.source.type == 'Galaxies_MAR'):
            Ngam_dot_ion = NGamDot(param,param.solver.z)
            print('Galaxies_MAR model chosen. M_halo is ', '{:.2e}'.format(M_halo))

            T_Galaxy = param.source.T_gal
            nu_range = np.logspace(np.log10(param.source.E_min_sed_ion / h_eV_sec), np.log10(param.source.E_max_sed_ion / h_eV_sec), 3000,base=10)
            norm__ = np.trapz(BB_Planck(nu_range, T_Galaxy) / h__, np.log(nu_range))

            I__ = Ngam_dot_ion / norm__
            print('BB spectrum normalized to ', '{:.2e}'.format(Ngam_dot_ion), ' ionizing photons per s, in the energy range [',
                  param.E_min_sed_ion, ' ', param.source.E_max_sed_ion, '] eV')

            def N(E, n_HI0):
                nu_ = Hz_per_eV * E
                int = cm_per_Mpc/ param.cosmo.h * (n_HI0 * sigma_HI(E))
                return exp(-int) * I__ * BB_Planck(nu_, T_Galaxy) / h__

            print('source emits', '{:.2e}'.format(Ngam_dot_ion), 'ionizing photons per seconds.')



        elif (param.source.type == 'SED'):
            Ngam_dot_ion, E_dot_xray = NGamDot(param,param.solver.z)
            sed_ion = param.source.alS_ion
            sed_xray = param.source.alS_xray

            norm_ion = (1 - sed_ion) / ((param.source.E_max_sed_ion / h_eV_sec) ** (1 - sed_ion) - (param.source.E_min_sed_ion / h_eV_sec) ** (1 - sed_ion))
            norm_xray = (1 - sed_xray) / ((param.source.E_max_sed_xray / h_eV_sec) ** (1 - sed_xray) - (param.source.E_min_sed_xray / h_eV_sec) ** (1 - sed_xray))
            # nu**-alpha*norm  is a count of [photons.Hz**-1]

            def Nion(E, n_HI0):
                nu_ = Hz_per_eV * E
                int = cm_per_Mpc / param.cosmo.h * (n_HI0 * sigma_HI(E))  ##
                return np.exp(-int) * Ngam_dot_ion * nu_ ** (-sed_ion) * norm_ion * nu_  # this is [Hz*Hz**-1 . s**-1] normalized to Ngdot in the range min-max

            def Nxray(E, n_HI0):
                """
                input : E in eV, n_HIO in Mpc/h.cm**-3 (column density, see in generate table how column density are initialized)
                output : Divide it by 4*pi*r**2 ,and you get a flux [eV.s-1.r-2.eV-1], r meaning the unit of r
                """
                nu_ = Hz_per_eV * E
                int = cm_per_Mpc / h0 * (n_HI0 * sigma_HI(E))
                return np.exp(-int) * E_dot_xray * norm_xray * nu_ ** (-sed_xray) * Hz_per_eV  # [eV/eV/s]

            nu_range = np.logspace(np.log10(param.source.E_min_sed_ion / h_eV_sec),np.log10(param.source.E_max_sed_ion / h_eV_sec), 3000, base=10)
            #plt.loglog(nu_range, Nion(nu_range / Hz_per_eV, 1e-9))
            XraySed = Nxray(nu_range / Hz_per_eV, 1e-9)
            Ion_Sed = Nion(nu_range / Hz_per_eV, 1e-9)



        elif (param.source.type == 'Ross'):
            Ngam_dot_ion, Ngam_dot_xray = NGamDot(param, param.solver.z)
            sed_xray = param.source.alS_xray

            # norm_ion = (1 - sed_ion) / ((param.source.E_max_sed_ion / h_eV_sec) ** (1 - sed_ion) - (param.source.E_min_sed_ion / h_eV_sec) ** (1 - sed_ion))
            norm_xray = sed_xray / (-(param.source.E_max_sed_xray / h_eV_sec) ** -sed_xray + (param.source.E_min_sed_xray / h_eV_sec) ** - sed_xray)
            # nu**-alpha*norm  is a count of [photons.Hz**-1]
            T_Galaxy = param.source.T_gal
            nu_range = np.logspace(np.log10(param.source.E_min_sed_ion / h_eV_sec), np.log10(param.source.E_max_sed_ion / h_eV_sec), 3000, base=10)
            norm_ion = np.trapz(BB_Planck(nu_range, T_Galaxy) / h__, np.log(nu_range))

            I__ = Ngam_dot_ion / norm_ion
            print('BB spectrum normalized to ', '{:.2e}'.format(Ngam_dot_ion), ' ionizing photons per s, in the energy range [', param.source.E_min_sed_ion, ' ',param.source.E_max_sed_ion, '] eV')

            def Nion(E, n_HI0):
                nu_ = Hz_per_eV * E
                int = cm_per_Mpc / param.cosmo.h * (n_HI0 * sigma_HI(E) )  ##
                return np.exp(-int) * I__ * BB_Planck(nu_,  T_Galaxy) / h__  # this is [Hz*Hz**-1 . s**-1] normalized to Ngdot in the range min-max
                # np.exp(-int) * Ngam_dot_ion * nu_ ** (-sed_ion) * norm_ion * nu_

            def Nxray(E, n_HI0):
                """
                input : E in eV, n_HIO in Mpc/h.cm**-3 (column density, see in generate table how column density are initialized)
                output : Divide it by 4*pi*r**2 ,and you get a flux [eV.s-1.r-2.eV-1], r meaning the unit of r
                """
                nu_ = Hz_per_eV * E
                int = cm_per_Mpc / h0 * (n_HI0 * sigma_HI(E))
                return np.exp(-int) * Ngam_dot_xray * norm_xray * nu_ ** -sed_xray  # [eV/eV/s]

            nu_range_xray = np.logspace(np.log10(param.source.E_min_xray / h_eV_sec),   np.log10(param.source.E_max_xray / h_eV_sec), 500, base=10)
            E_dot_xray = Ngam_dot_xray  # np.trapz(Ngam_dot_xray * norm_xray * nu_range_xray ** -sed_xray,nu_range_xray)

            # plt.loglog(nu_range, Nion(nu_range / Hz_per_eV, 1e-9))
            XraySed = Nxray(nu_range / Hz_per_eV, 1e-9)
            Ion_Sed = Nion(nu_range / Hz_per_eV, 1e-9)


        else:
            print('Source Type not available. Should be Galaxies or Miniqsos or SED')
            exit()

        IHI_1 = zeros((n_HI.size))
        IHI_2 = zeros((n_HI.size))

        IT_HI_1 = zeros((n_HI.size))
        IT_2a = zeros((n_HI.size))
        IT_2b = zeros((n_HI.size))

        E_range_ion  = np.logspace(np.log10(E_HI), np.log10(param.source.E_max_sed_ion), 500, base=10)
        E_range_xray = np.logspace(np.log10(param.source.E_min_xray), np.log10(param.source.E_max_xray), 500, base=10) #xray photon range

        """""""""
        IHI_1[:] = np.trapz(sigma_HI(E_range_ion_HI) / E_range_ion_HI * Nion(E_range_ion_HI, n_HI[:, None]),E_range_ion_HI) + param.source.xray_in_ion * np.trapz(sigma_HI(E_range_xray) / E_range_xray * Nxray(E_range_xray, n_HI[:, None]),E_range_xray)
        IHI_2[:] = np.trapz(sigma_HI(E_range_ion_HI) * (E_range_ion_HI - E_HI) / (E_HI * E_range_ion_HI) * Nion(E_range_ion_HI,n_HI[:, None]),  E_range_ion_HI) + param.source.xray_in_ion * np.trapz( sigma_HI(E_range_xray) * (E_range_xray - E_HI) / (E_HI * E_range_xray) * Nxray(E_range_xray, n_HI[:, None]),   E_range_xray)

        IT_HI_1[:] = np.trapz(sigma_HI(E_range_xray) * (E_range_xray - E_HI) / E_range_xray * Nxray(E_range_xray, n_HI[:,None]), E_range_xray)     + param.source.ion_in_xray * np.trapz(sigma_HI(E_range_ion_HI)   * (E_range_ion_HI - E_HI) / E_range_ion_HI * Nion(E_range_ion_HI,  n_HI[:,None]), E_range_ion_HI) #[eV/s]then divide by r2


        IT_2a[:] = np.trapz(Nxray(E_range_xray, n_HI[:, None]) * E_range_xray,E_range_xray) + param.source.ion_in_xray * np.trapz(Nion(E_range_ion_HI, n_HI[:, None]) * E_range_ion_HI, E_range_ion_HI)
        IT_2b[:] = np.trapz(Nxray(E_range_xray, n_HI[:, None]) * (-4 * kb_eV_per_K), E_range_xray) + param.source.ion_in_xray * np.trapz( Nion(E_range_ion_HI, n_HI[:, None]) * (-4 * kb_eV_per_K), E_range_ion_HI)

        """""""""

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





class Source_MAR:
    """
    Source which ionizes the surrounding H and He gas along a radial direction.

    This class initiates a source with a given mass, redshift and a default power law source, which can be changed to
    any spectral energy distribution. Given a starting point and a ending point, the radiative transfer equations are
    solved along the radial direction and the H and He densities are evolved for some given time.

    Parameters
    ----------
    M : float
     Mass of the source in solar masses.
    z : float
     Redshift of the source.
    r_start : float, optional
     Starting point of the radial Mpc-grid in logspace, default value is log10(0.0001).
    r_end : float, optional
     Ending point of the radial Mpc-grid in logspace, default value is log10(3).
    dn : number of initial subdivisions of the radial grid. Can increase significantly computing time.
    LE : bool, optional
     Decide whether to include UV ionizing photons, default is True (include).
    alpha : int, optional
     Spectral index for the power law source, default is -1.
    sed : callable, optional
     Spectral energy distribution. Default is a power law source.
    filename_table : str, optional
     Name of the external table, will be imported if available to skip the table generation
    recalculate_table : bool, default False
     Decide whether to import or generate the interpolation table. If nothing is given it will be set to False and then
     be changed whether or not a external table is available.
    """

    def __init__(self, param):

        if param.source.type == 'Miniqsos':
            self.M = param.source.M_miniqso
        elif param.source.type == 'SED':
            self.M = param.source.M_halo
        elif param.source.type == 'Galaxies_MAR' or param.source.type == 'Ross' :
            self.M = param.source.M_halo

        else:
            print('source.type should be Galaxies or Miniqsos')
            exit()

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
        self.Gamma_grid_info = None

        self.E_0 = param.source.E_min_sed_ion
        self.E_upp = param.source.E_max_sed_ion  # In eV
        self.r_grid = linspace(self.r_start, self.r_end, self.dn) #Mpc/h, phyisical distance from halo center
        correlation_fct(param)
        dt_init =  param.solver.time_step * 1e6 * sec_per_year * u.s  ### time step of 0.1 Myr
        self.dt_init = dt_init

        if param.source.g4 <-4 :
            print('g4 too high, you should not have a too sharp cutoff, otherwise the calculations of the gamma tables give zero.' )


    def create_table(self, param):
        """
        Call the function to create the interpolation tables.

        Parameters
        ----------
        par : dict of {str:float}, optional
         Variables to pass on to the table generator. If none is given the parameters of the Source initialization will
         be used.
        """

        dn_table = self.dn_table
         # phyisical distance from halo center in cMpc/h
        self.r_grid_cell = (self.r_grid[1:] + self.r_grid[:-1]) / 2  ## middle of each cell, where the density is computed
        self.M_initial = param.source.M_halo
        self.z_initial = param.solver.z

        # Column densities in physical Mpc/h.cm**-3.
        nH_column = np.trapz( self.profiles(param, self.z_initial, Mass = self.M_initial), self.r_grid_cell) * (1 + self.z_initial) ** 3
        print('n_H_column max : ', '{:.2e}'.format(nH_column), 'Mpc/h.cm**-3.')
        n_HI  = logspace(log10(nH_column  * 1e-6),  log10(1.05 * nH_column), dn_table, base=10)
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
        cosmo_corr = splev(self.r_grid_cell * (1 + z), corr_tck)  # r_grid * (1+self.z) in cMpc/h --> To reach the correct scales and units for the correlation fucntion
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

        # r_grid0 = logspace(log10(self.grid_param['r_start']), log10(self.grid_param['r_end']), dn,base=10)
        # r_grid  = logspace(log10(self.grid_param['r_start']), log10(self.grid_param['r_end']), dn,base=10)
        r_grid = self.r_grid

        self.create_table(param)

        n_HI = self.Gamma_grid_info['input']['n_HI']

        if self.M != self.Gamma_grid_info['input']['M']:
            print('correcting for different Qso Mass than table.')

        dr_max = r_grid[-1] - r_grid[-2]
        if dr_max * sigma_HI(13.6) * n_H_0 * cm_per_Mpc > 0.1:
            print('needs more spatial stepping. taumax is ', dr_max * sigma_HI(13.6) * n_H_0 * cm_per_Mpc)



        while True:

            Ion_front_grid,Mh_history,z_grid,Nh_history = [],[],[], []
            Ngdot_ratio = []
            Cumul_heating = np.zeros((len(r_grid)-1))
            # create a dictionnary to store the profiles at the desired redshifts
            #A_grid_hist,D_grid_hist = {},{}
            T_history = {}
            T_spin_history = {}
            xHII_history = {}
            dTb_history = {}
            I1_HI_history, I2_HI_history ={},{}
            nHI_norm = {} # neutral HI density normlized to mean baryon density. To be used in formula of dTb ~(1+delta_b)*xHI
            #x_tot_history, x_al_history, x_coll_history = {},{},{}
            #rho_al_history, rho_xray_history = {}, {}
            heat_history = {} ## profiles of heat energy deposited per baryons [eV/s]
            n_HII_cell, T_grid = zeros_like(self.r_grid_cell), zeros_like(self.r_grid_cell)
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
                    #time_grid.append(l * self.dt_init.value)
                    Ion_front_grid.append(0)
                    Mh_history.append(Mh_step_l)
                    z_grid.append(zstep_l[0])
                    #Ng_dot_history.append(0)
                    Tadiab = 2.725 * (1 + zstep_l) ** 2 / (1 + param.cosmo.z_decoupl)
                    nB_profile_z = self.nB_profile* (1 + zstep_l) ** 3
                    nHI_norm[str(round(zstep_l[0], 2))] = (nB_profile_z - n_HII_cell) * m_p_in_Msun * cm_per_Mpc ** 3 / rhoc_of_z(param, z) / Ob / (1 + z) ** 3
                    T_history[str(round(zstep_l[0],2))] = Tadiab
                    #x_al_history[str(round(zstep_l[0], 2))] = 0
                    rho_bar_mean = rhoc0 * h0**2 * Ob * (1 + zstep_l[0]) ** 3 * M_sun / (cm_per_Mpc) ** 3 / m_H  #mean physical bar density in [baryons /co-cm**3]
                    xcoll_ = x_coll(zstep_l[0], Tadiab, 1, rho_bar_mean)
                    #x_coll_history[str(round(zstep_l[0], 2))] = xcoll_
                    #x_tot_history[str(round(zstep_l[0], 2))] = xcoll_

                    T_spin_history[str(round(zstep_l[0], 2))] = Tspin(Tcmb0 * (1+zstep_l[0]), Tadiab,xcoll_ )

                    xHII_history[str(round(zstep_l[0], 2))] = 0
                    heat_history[str(round(zstep_l[0], 2))] = 0
                    dTb_history[str(round(zstep_l[0], 2))] = dTb(zstep_l[0], T_spin_history[str(round(zstep_l[0], 2))], nHI_norm[str(round(zstep_l[0], 2))], param)
                    Nh_history.append(0)
                    Ngdot_ratio.append(1)
                    #A_grid_hist[str(round(zstep_l[0], 2))], D_grid_hist[str(round(zstep_l[0], 2))] = 0,0
                    I1_HI_history[str(round(zstep_l[0], 2))], I2_HI_history[str(round(zstep_l[0], 2))] = 0,0

                l += 1

            T_grid = zeros_like(self.r_grid_cell)
            T_grid += T_adiab(zstep_l,param)

            while zstep_l > param.solver.z_end :
                z_previous = zstep_l
                # Calculate the redshift z(t)
                age  = pl.age(self.z_initial) # careful. Here you add to z_initial l*dt_init and find the corresponding z.
                age  = age.to(u.s)
                age += l * self.dt_init
                func = lambda z: pl.age(z).to(u.s).value - age.value
                zstep_l = fsolve(func, x0 = zstep_l) ### zstep_l for initial guess

                ##### CMB temperature for the collisional coupling
                T_gamma = 2.725 * (1 + zstep_l)  # [K]
                gamma_2c = alpha_HII(T_gamma) * 1e-6 * (m_e * kb * T_gamma / (2 * pi)) ** (3 / 2) / h__ ** 3 * exp(-3.4 / (T_gamma * kb_eV_per_K))  ## the 1e-6 factor is to go from cm**3 to m**3 for alpha_HII

                ### Update halo mass, exponential growth
                Mh_step_l = self.M_initial * np.exp(param.source.alpha_MAR * (self.z_initial-zstep_l))
                copy_param.source.M_halo = Mh_step_l

                if param.source.type == 'SED' :
                    Ngam_dot_step_l_ion, E_dot_step_l_xray = NGamDot(copy_param,zstep_l) #[s**-1,eV/s]

                if param.source.type == 'Ross':
                    Ngam_dot_step_l_ion = NGamDot(copy_param,zstep_l)[0]

                else :
                    Ngam_dot_step_l = NGamDot(copy_param,zstep_l)


                #### Update the profile due to expansion and Halo Growth
                self.nB_profile = self.profiles(param, zstep_l, Mass = Mh_step_l)


                ## dilute the ionized density according to redshift
                n_HII_cell = n_HII_cell * (1+zstep_l)**3 / (1+z_previous)**3

                #update the baryon profile
                nB_profile_z = self.nB_profile * (1 + zstep_l) ** 3  ## physical total baryon density

                n_HII_cell[n_HII_cell > nB_profile_z] = nB_profile_z[n_HII_cell > nB_profile_z] #xHII can't be >1
                n_HI_cell = nB_profile_z - n_HII_cell                       #set neutral physical density in cell

                dr = r_grid[1]-r_grid[0]                #lin spacing so tha'ts ok.
                r2 = self.r_grid_cell ** 2

                n_HI_edge = (n_HI_cell[:-1]+n_HI_cell[1:])/2
                n_HI_edge = np.concatenate((np.array([0]),n_HI_edge,np.array([n_HI_cell[-1]])) )## value of the density at the edge of the cells. use this to cumpte the cumumlative densities at boundaries of cells size is (rgrid)


                K_HI = cumtrapz(n_HI_edge, self.r_grid, initial=0.0) ## cumulative densities : before first cell, before second cell... etc up to right after last one. size is (r_grid)


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

                A = gamma_HI(n_HII_cell, n_HI_cell, T_grid, I1_HI, I2_HI, gamma_2c) * n_HI_cell - alpha_HII(T_grid) * n_HII_cell * n_ee

                #D = (2 / 3) * mu / (kb_eV_per_K) * (f_Heat(n_HII_cell / nB_profile_z) * (n_HI_cell * I1_T_HI) - A6 )  # sigma_s * n_ee / m_e_eV * (I2_Ta + T_grid * I2_Tb) - (A1_HI + A2_HII + A4_HI + A5 + A6)) ##K/s/cm**3   ###SIMPLE HEATING VERSION
                D = (2 / 3) * mu / (kb_eV_per_K) * (f_Heat(n_HII_cell / nB_profile_z) * (n_HI_cell * I1_T_HI) + sigma_s * n_ee / m_e_eV * (I2_Ta + T_grid * I2_Tb) - (A1_HI + A2_HII + A4_HI + A5 + A6)) ##K/s/cm**3

                Cumul_heating+= D
                if l * param.solver.time_step % 10 == 0 and l != 0:
                    heat_history[str(round(zstep_l[0], 2))] = np.copy(D)
                        #np.copy((2 / 3) * mu * f_Heat(n_HII_cell / nB_profile_z) * (n_HI_cell * I1_T_HI)/(nB_profile_z *(1+zstep_l)**3/(1+z_previous)**3))
                    #np.copy([f_Heat(n_HII_cell / nB_profile_z) * (n_HI_cell * I1_T_HI)+heat_history sigma_s * n_ee / m_e_eV * (I2_Ta + T_grid * I2_Tb),(A1_HI + A2_HII + A4_HI + A5 + A6)])

                n_HII_cell = n_HII_cell + dt_init.value * A # discretize the diff equation.
                n_HII_cell[np.where(n_HII_cell<0)] = 0

                T_nB = T_grid * nB_profile_z + dt_init.value * D #product of Tk * baryon physical density
                T_grid = T_nB/( nB_profile_z *(1+zstep_l)**3/(1+z_previous)**3 ) # to get the correct T~(1+z)**2 adiabatic regime, you need to account for the shift in baryon density

                n_HII_cell, T_grid = np.nan_to_num(n_HII_cell), np.nan_to_num(T_grid)

                n_HII_cell[n_HII_cell>nB_profile_z] = nB_profile_z[n_HII_cell>nB_profile_z]   ## xHII can't be >1

                front_step = find_Ifront(n_HII_cell / nB_profile_z, self.r_grid_cell )


                if np.exp(-cm_per_Mpc * (K_HI[-1] * sigma_HI(13.6))) > 0.1 and count__==0:
                    print('WARNING : np.exp(-tau(rmax)) > 0.1. Some photons are not absorbed. Maybe you need larger rmax. ')
                    count__ = 1 #  to print this only once

                if l*param.solver.time_step % 10  == 0 and l != 0:
                    xdata, ydata = self.r_grid_cell, (nB_profile_z - n_HII_cell) / nB_profile_z
                    ydata = ydata.clip(min=0)  # remove neg elements

                    Ion_front_grid.append(front_step)
                    Mh_history.append(Mh_step_l)
                    z_grid.append(zstep_l[0])
                    T_history[str(round(zstep_l[0],2))] = np.copy(T_grid)
                    xHII_history[str(round(zstep_l[0], 2))] = 1-ydata

                    nHI_norm[str(round(zstep_l[0],2))] = (nB_profile_z - n_HII_cell) * m_p_in_Msun * cm_per_Mpc ** 3 / rhoc_of_z(param, z) / Ob / (1 + z) ** 3


                    # xray contrib to Lya coupling
                    #J0xray = J0_xray_lyal(self.r_grid_cell, (1-ydata), (nB_profile_z - n_HII_cell) , E_dot_step_l_xray, z, param)

                    ### x_alpha
                    rho_bar = bar_density_2h(self.r_grid_cell, param, zstep_l[0], Mh_step_l) * (1 + zstep_l[0]) ** 3 #bar/physical cm**3
                    xHI_    = ydata   ### neutral fraction
                    xcoll_  = x_coll(zstep_l[0], T_grid, xHI_, rho_bar)
                    rho_alpha_ = rho_alpha(self.r_grid_cell, Mh_step_l[0], zstep_l[0], param)[0]
                    x_alpha_ = 1.81e11 * rho_alpha_ * S_alpha(zstep_l[0], T_grid, xHI_) / (1 + zstep_l[0])  #(rho_alpha_ + J0xray)
                    x_tot_   = (x_alpha_ + xcoll_)
                    #x_tot_history[str(round(zstep_l[0],2))]   = np.copy(x_tot_)
                    #x_al_history[str(round(zstep_l[0], 2))]   = np.copy(x_alpha_)
                    #x_coll_history[str(round(zstep_l[0], 2))] = np.copy(xcoll_)
                    T_spin_history[str(round(zstep_l[0], 2))] = Tspin(Tcmb0 * (1 + zstep_l[0]), T_grid, x_tot_)

                        #np.copy(f_Heat(n_HII_cell / nB_profile_z) * (n_HI_cell * I1_T_HI)/( nB_profile_z *(1+zstep_l)**3/(1+z_previous)**3 ) ) # eV/s

                    #rho_al_history[str(round(zstep_l[0], 2))] = np.copy(rho_alpha_)
                    #rho_xray_history[str(round(zstep_l[0], 2))] = np.copy(J0xray)

                    dTb_history[str(round(zstep_l[0], 2))] = dTb(zstep_l[0], T_spin_history[str(round(zstep_l[0], 2))], nHI_norm[str(round(zstep_l[0], 2))], param)
                    Nh_history.append(Ngam_dot_step_l_ion)
                    Ngdot_ratio.append(Ngam_dot_step_l_ion / Ng_dot_initial_ion)
                   # A_grid_hist[str(round(zstep_l[0], 2))], D_grid_hist[str(round(zstep_l[0], 2))] =A, D
                    I1_HI_history[str(round(zstep_l[0], 2))], I2_HI_history[str(round(zstep_l[0], 2))] = I1_HI, I2_HI
                l += 1


            Ion_front_grid = array([Ion_front_grid])
            Ion_front_grid = Ion_front_grid.reshape(Ion_front_grid.size, 1)
            time_end_solve = datetime.datetime.now()
            print('solver took :', time_end_solve - t_start_solver)
            break

        #self.A_hist = A_grid_hist
        self.Ngdot_ratio = Ngdot_ratio
        self.Cumul_heating = Cumul_heating
        self.Nh_history = Nh_history
        self.I1_HI_history = I1_HI_history
        self.I2_HI_history = I2_HI_history

        self.n_HI_grid = nB_profile_z - n_HII_cell
        self.n_HII_cell = n_HII_cell
        self.n_B = nB_profile_z ## total density in nbr of Hatoms per physical-cm**3
        self.T_grid = np.copy(T_grid)
        self.Ion_front_grid = Ion_front_grid
        self.z_history = np.array(z_grid)
        self.Mh_history = np.array(Mh_history)

        self.T_history  = T_history
        self.heat_history = heat_history
       # self.rho_al_history = rho_al_history
        #self.rho_xray_history = , rho_xray_history
        #self.x_tot_history = x_tot_history
        #self.x_coll_history = x_coll_history
        #self.x_al_history = x_al_history
        self.dTb_history = dTb_history
        self.T_spin_history = T_spin_history
        self.xHII_history=xHII_history

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