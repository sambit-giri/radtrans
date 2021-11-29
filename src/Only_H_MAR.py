
from scipy.interpolate import splrep,splev
import scipy.integrate as integrate
from numpy import *
import numpy as np
import scipy.interpolate as interpolate
from tqdm import tqdm
from astropy import units as u
from astropy.cosmology import WMAP7 as pl
from scipy.optimize import fsolve
from sys import exit
import pickle
import datetime
import matplotlib.pyplot as plt
from .bias import *
from .constants import *
from .Astro import *
import copy
import os
from scipy.optimize import curve_fit


###Constants
facr = 1 * u.Mpc
cm_per_Mpc = (facr.to(u.cm)).value
eV_per_erg = 6.242 * 10 ** 11   #(1 * u.erg).to(u.eV).value

# Hydro density and mass
n_H_0  = 1.87 * 10 ** -12        # [cm**-3]
m_H    = 1.6 * 10 ** - 27       # [kg]
m_He   = 6.6464731 * 10 ** - 27 # [kg]

# Energy limits and ionization energies, in [eV]
E_0 = 10.4
E_HI = 13.6
E_HeI = 24.5
E_HeII = 54.42

# Constants
c = 2.99792 * 10 ** 10    # [cm/s]
kb = 1.380649 * 10 ** -23 # [J/K] or [kg.m2.s-2.K-1 ]
kb_eV_per_K = 8.61733e-5  # [eV/K]

m_e = 9.10938 * 10 ** -31 # [kg]
m_e_eV = 511e3            # [eV], 511keV

sigma_s = 6.6524 * 10 ** -25 # [cm**2]
M_sun = 1.988 * 10 ** 30     # [kg]
sec_per_year = 3600*24*365.25
Hz_per_eV = 241799050402293


##Cosmology
Om      = 0.31
Ob      = 0.048
Ol      = 1-Om-Ob
h0      = 0.68
ns      = 0.97
s8      = 0.81

h__ = 6.626e-34         ## [J.s] or [m2 kg / s]
c__ = 2.99792e+8        ## [m/s]
k__ = 1.380649e-23      ## [m 2 kg s-2 K-1]
h_eV_sec = 4.135667e-15 ## [eV.s]

def BB_Planck( nu , T):  #  BB Spectrum [J s-1 m−2 Hz−1 ]
    a_ = 2.0 * h__ * nu**3 / c__**2
    intensity = 4 * pi * a_ / ( exp(h__*nu/(k__*T)) - 1.0)
    return intensity

def n_H(z ,C):
    return C * n_H_0 * (1 + z) ** 3



def sigma_HI(E):
    """
    Input : E is in eV.
    Returns : bound free photo-ionization cross section ,  [cm ** 2]
    """
    sigma_0 = 5.475 * 10 ** 4 * 10 ** -18 ## cm**2 1Mbarn = 1e-18 cm**2
    E_01 = 4.298 * 10 ** -1
    y_a = 3.288 * 10 ** 1
    P = 2.963
    y_w = y_0 = y_1 = 0
    x = E / E_01 - y_0
    y = sqrt(x ** 2 + y_1 ** 2)
    F = ((x - 1) ** 2 + y_w ** 2) * y ** (0.5 * P - 5.5) * (1 + sqrt(y / y_a)) ** -P
    sigma = sigma_0 * F
    return sigma


# Ionization and Recombination coefficients. Expressions taken from Fukugita and Kawasaki 1994.
def alpha_HII(T):
    """
    Recombination coefficient for Hydrogen :  [cm3.s-1]
    Input : temperature in K
    """
    return 2.6 * 10 ** -13 * (T / 10 ** 4) ** -0.85


def beta_HI(T):
    """
    Collisional ionization coefficient for Hydrogen :  [cm3.s-1]
    Input : temperature in K
    """
    return 5.85 * 10 ** -11 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-1.578 * 10 ** 5 / T)


def xi_HI(T):
    """
    Collisional ionization cooling (see Fukugita & Kawazaki 1994) [eV.cm3.s-1]
    """
    return eV_per_erg * 1.27 * 10 ** -21 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-1.58 * 10 ** 5 / T)

def eta_HII(T):
    """
    Recombination cooling [eV.cm3.s-1]
    """
    return eV_per_erg * 6.5 * 10 ** -27 * T ** 0.5 * (T / 10 ** 3) ** -0.2 * (1 + (T / 10 ** 6) ** 0.7) ** -1


def psi_HI(T):
    """
    Collisional excitation cooling [eV.cm3.s-1]
    """
    return eV_per_erg * 7.5 * 10 ** -19 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-1.18 * 10 ** 5 / T)


def theta_ff(T):
    """
    Free-free cooling coefficient [eV.cm3.s-1]
    """
    return eV_per_erg * 1.3 * 1.42 * 10 ** -27 * (T) ** 0.5


def f_H(x_ion):
    """
    Factor for secondary ionization, due to kinetic energy carried by secondary e- (see Shull & van Steenberg 1985). [Dimensionless]
    Input : x is the ionized fraction of hydrogen
    """
    if isnan(x_ion):
        x_ion = 1
    return nan_to_num(0.3908 * (1 - max(min(x_ion,1),0) ** 0.4092) ** 1.7592)

def f_Heat(xion):
    """
    Amount of heat deposited by secondary electrons. (Shull & van Steenberg (1985) fig.3 - according to Thomas&Zaroubi Miniqso). [Dimensionless]
    """
    if isnan(xion):
        xion = 1
    if xion> 10 ** -4:
        output =  nan_to_num(0.9971 * (1 - (1 - min(xion,1) ** 0.2663) ** 1.3163))
    else :
        output = 0.15
    return output


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
     Cumulative (column density) Neutral hydrogen, HeI, HeII density array in cm**-2 along r_grid.
    E_upp_ is the maximum energy above which we cut the integral. It's in a sense the max energy of the ionizing photon in our model
    Returns
    ------
    dict of {str:dict}
        Dictionary containing two sub-dictionaries: The first one containing the function variables and the second one
        containing the 12 tables for the integrals
    '''
    E_0_ = param.source.E_0
    E_upp_ = param.source.E_upp  # [eV]

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
                input : E in eV, n_HIO in Mpc.cm**-3 (column density, see in generate table how column density are initialized)
                output : Divide it by 4*pi*r**2 ,and you get a flux [eV.s-1.r-2.eV-1], r meaning the unit of r
                """
                int = cm_per_Mpc * (n_HI0 * sigma_HI(E))
                return exp(-int) * I(E)

            E_range_HI_ = np.logspace(np.log10(13.6), np.log10(E_upp_), 1000, base=10)
            Ngam_dot = np.trapz(I(E_range_HI_) / E_range_HI_, E_range_HI_)
            print('source emits', Ngam_dot, 'ionizing photons per seconds, in the energy range [', param.source.E_0,
                  ',', param.source.E_upp, '] eV')

        elif (param.source.type == 'Galaxies_MAR'):
            z = param.solver.z
            M_halo = param.source.M_halo
            M = M_halo
            dMh_dt = param.source.alpha_MAR * M_halo * (z + 1) * Hubble(z, param)  ## [(Msol/h) / yr]
            Ngam_dot = dMh_dt * f_star_Halo(param, M_halo) * param.cosmo.Ob / param.cosmo.Om * f_esc(param,M_halo) * param.source.Nion / sec_per_year / m_H * M_sun
            print('Galaxies_MAR model chosen. M_halo is ', M_halo)

            T_Galaxy = param.source.T_gal
            nu_range = np.logspace(np.log10(param.source.E_0 / h_eV_sec), np.log10(param.source.E_upp / h_eV_sec), 3000,base=10)
            norm__ = np.trapz(BB_Planck(nu_range, T_Galaxy) / h__, np.log(nu_range))

            I__ = Ngam_dot / norm__
            print('BB spectrum normalized to ', Ngam_dot, ' ionizing photons per s, in the energy range [',
                  param.source.E_0, ' ', param.source.E_upp, '] eV')

            def N(E, n_HI0):
                nu_ = Hz_per_eV * E
                int = cm_per_Mpc * (n_HI0 * sigma_HI(E))
                return exp(-int) * I__ * BB_Planck(nu_, T_Galaxy) / h__

            print('source emits', Ngam_dot, 'ionizing photons per seconds.')



        else:
            print('Source Type not available. Should be Galaxies or Miniqsos')
            exit()

        IHI_1 = zeros((n_HI.size))
        IHI_2 = zeros((n_HI.size))

        IT_HI_1 = zeros((n_HI.size))
        IT_2a = zeros((n_HI.size))
        IT_2b = zeros((n_HI.size))

        E_range_HI = np.logspace(np.log10(E_HI), np.log10(E_upp_), 1000, base=10)
        E_range_0    = np.logspace(np.log10(E_0_), np.log10(E_upp_), 1000, base=10)

        IHI_1[:] = np.trapz(1 / E_range_HI * N(E_range_HI, n_HI[:, None]), E_range_HI)  # sigma_HI(E_range_HI)
        IHI_2[:] = np.trapz((E_range_HI - E_HI) / (E_HI * E_range_HI) * N(E_range_HI, n_HI[:, None]), E_range_HI)
        IT_HI_1[:] = np.trapz((E_range_HI - E_HI) / E_range_HI * N(E_range_HI, n_HI[:, None]), E_range_HI)
        IT_2a[:] = np.trapz(N(E_range_0, n_HI[:, None]) * E_range_0, E_range_0)
        IT_2b[:] = np.trapz(N(E_range_0, n_HI[:, None]) * (-4 * kb_eV_per_K), E_range_0)

        print('...done')

        Gamma_info = {'HI_1': IHI_1, 'HI_2': IHI_2,  'T_HI_1': IT_HI_1,'T_2a': IT_2a, 'T_2b': IT_2b}

        input_info = {'M': M, 'z': z, 'type': param.source.type, 'N_ion_ph_dot': Ngam_dot,
                      'n_HI': n_HI,  'E_0': param.source.E_0,
                      'E_upp': param.source.E_upp}

        Gamma_input_info = {'Gamma': Gamma_info, 'input': input_info}

        if param.table.filename_table is None:
            filename_table = 'qwerty'
            print('No filename_table given. We will call it qwerty.')
        else:
            filename_table = param.table.filename_table
        print('saving table in pickle file ', filename_table)
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
    evol : float
     Evolution time of the densities in Myr.
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
    lifetime : float, optional
     Lifetime of the source, after which it will be turned off and the radiative transfer will continue to be calculated
     without a source
    filename_table : str, optional
     Name of the external table, will be imported if available to skip the table generation
    recalculate_table : bool, default False
     Decide whether to import or generate the interpolation table. If nothing is given it will be set to False and then
     be changed whether or not a external table is available.
    """

    def __init__(self, param):

        if param.source.type == 'Miniqsos':
            self.M = param.source.M_miniqso
        elif param.source.type == 'Galaxies':
            self.M = param.source.M_halo
        elif param.source.type == 'Galaxies_MAR':
            self.M = param.source.M_halo
        else:
            print('source.type should be Galaxies or Miniqsos')
            exit()

        self.z = param.solver.z  # redshift
        self.evol = param.solver.evol * 1e6 * sec_per_year * u.s  # *10**6*365*24*60*60*u.s #s # evolution time, typically 3-10 Myr
        self.lifetime = param.source.lifetime * 1e6 * sec_per_year * u.s  # *10**6*365*24*60*60*u.s   #Myr, lifetime of the source
        self.alpha = param.source.alpha

        self.M_halo = param.source.M_halo
        self.R_halo = R_halo(self.M_halo, self.z, param)  # physical halo size
        self.r_start = self.R_halo
        print('R_halo is :', self.R_halo, 'Mpc')
        self.r_end = param.solver.r_end  # maximal distance from source
        self.dn = param.solver.dn
        self.Nt = param.solver.Nt
        self.dn_table = param.solver.dn_table
        self.Gamma_grid_info = None
        self.C = param.solver.C  # Clumping factor

        self.E_0 = param.source.E_0
        self.E_upp = param.source.E_upp  # In eV
        self.r_grid = linspace(self.r_start, self.r_end, self.dn)

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
        r_grid = self.r_grid

        self.M_initial = param.source.M_halo
        self.z_initial = param.solver.z

        # Column densities in physical cm**-2
        nH_column = np.trapz( self.profiles(param,self.z_initial,Mass = self.M_initial), r_grid) * (1 + self.z_initial) ** 3
        print('n_H_column max : ', nH_column, 'cm**-3.')
        n_HI   = logspace(log10(nH_column  * 1e-6),  log10(1.05 * nH_column), dn_table, base=10)
        n_HI  = np.concatenate((np.array([0]), n_HI))

        Gamma_grid_info = generate_table(param, self.z_initial, n_HI)
        self.Gamma_grid_info = Gamma_grid_info



    def profiles(self,param,z,Mass = None):
        # Profiles
        cosmofile = param.cosmo.corr_fct
        vc_r, vc_m, vc_bias, vc_corr = np.loadtxt(cosmofile, usecols=(0, 1, 2, 3), unpack=True)
        corr_tck = splrep(vc_r, vc_corr, s=0)
        cosmo_corr = splev(self.r_grid * (1 + z),corr_tck)  # r_grid * (1+self.z) is the comoving value of r_grid at z. To reach the correct scales for the correlation fucntion
        halo_bias = bias(z, param,Mass)
        # baryonic density profile in [cm**-3]
        norm_profile = profile(halo_bias, cosmo_corr, param,z) * param.cosmo.Ob / param.cosmo.Om * M_sun * param.cosmo.h ** 2 / (cm_per_Mpc) ** 3 / m_H
        return norm_profile




    def initialise_grid_param(self):
        """
        Initialize the grid parameters for the solver.

        The grid parameters and the initial conditions are saved in a dictionary. A initial time step size and a initial
         radial grid is chosen. T_gamma is the CMB temperature for the given redshift and gamma_2c is used to calculate
         the contribution from the background photons.
        """

        grid_param = {'M': self.M, 'z': self.z}

        C = self.C



        t_evol = self.evol
        dt_init =  0.1 * 1e6 * sec_per_year * u.s  ### time step of 0.1 Myr

        dn = self.dn
        r_start = self.r_start
        r_end = self.r_end
        dn_table = self.dn_table

        grid_param['dn'] = dn
        grid_param['dn_table'] = dn_table
        grid_param['r_start'] = r_start
        grid_param['r_end'] = r_end
        grid_param['t_evol'] = t_evol
        grid_param['dt_init'] = dt_init
        grid_param['C'] = C
        grid_param['Source_lifeTime'] = self.lifetime

        self.grid_param = grid_param

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
        self.initialise_grid_param()
        z = self.grid_param['z']


        dt_init = self.grid_param['dt_init']

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

            time_grid = []
            Ion_front_grid = []
            Mh_history = []
            Ng_dot_history = []
            z_grid = []
            c1_history, c2_history = [], []


            n_HII_grid = zeros_like(r_grid)

            T_grid = zeros_like(r_grid)
            T_grid += 2.725 * (1 + z) ** 2 / (1 + 250) ### assume gas decoupled from cmb at z=250 and then adiabatically cooled

            l = 0
            zstep_l = self.z_initial

            print('Number of time steps: ', self.Nt)

            Gamma_info = self.Gamma_grid_info['Gamma']

            JHI_1, JHI_2  = Gamma_info['HI_1'], Gamma_info['HI_2']
            JT_HI_1       = Gamma_info['T_HI_1']
            JT_2a, JT_2b  = Gamma_info['T_2a'], Gamma_info['T_2b']
            Ng_dot_initial= self.Gamma_grid_info['input']['N_ion_ph_dot']

            print('Solver will solve for', param.solver.evol, ' Myr, turning off the source after',param.source.lifetime, 'Myr')

            copy_param = copy.deepcopy(param)




            while zstep_l > param.solver.z_end :    # l * dt_init <= self.grid_param['t_evol']:
                if l % 5 == 0 and l != 0:
                    print('Current Time step: ', l, 'z is ', zstep_l)

                # Calculate the redshift z(t)
                age  = pl.age(self.z_initial) # careful. Here you add to z_initial l*dt_init and find the corresponding z.
                age  = age.to(u.s)
                age += l * self.grid_param['dt_init']
                func = lambda z: pl.age(z).to(u.s).value - age.value
                zstep_l = fsolve(func, z)


                ##### CMB temperature for the collisional coupling
                T_gamma = 2.725 * (1 + zstep_l)  # [K]
                gamma_2c = alpha_HII(T_gamma) * 1e-6 * (m_e * kb * T_gamma / (2 * pi)) ** (3 / 2) / h__ ** 3 * exp(-3.4 / (T_gamma * kb_eV_per_K))  ## the 1e-6 factor is to go from cm**3 to m**3 for alpha_HII


                ### Update halo mass, exponential growth
                Mh_step_l = self.M_initial * np.exp(param.source.alpha_MAR * (self.z_initial-zstep_l))
                copy_param.source.M_halo = Mh_step_l
                Ngam_dot_step_l = NGamDot(copy_param)

                #### Update the profile due to expansion and Halo Growth
                self.nHI0_profile = self.profiles(param, zstep_l, Mass = Mh_step_l)

                nHI0_profile_z = self.nHI0_profile * (1 + zstep_l) ** 3

                # Initialize the values to evaluate the integrals
                K_HI = 0

                for k in (arange(0, r_grid.size - 1, 1)):
                    n_H_z_r = (nHI0_profile_z[k] + nHI0_profile_z[k + 1]) / 2

                    dr_current = r_grid[k + 1] - r_grid[k]
                    n_HI00 = n_H_z_r - n_HII_grid[k]

                    if n_HI00 < 0:
                        n_HI00 = 0

                    if n_HI00 > n_H_z_r:
                        n_HI00 = n_H_z_r
                        print('wtf')

                    K_HI += dr_current * abs(nan_to_num(n_HI00))
                    K_HI_previous = K_HI - dr_current * abs(nan_to_num(n_HI00))



                    r2 = r_grid[k] ** 2
                    m_corr = 1

                    if K_HI < np.min(n_HI) or K_HI < n_HI[0] or K_HI > np.max(n_HI):
                        print('Too narrow HI cumulative density range in tables init.')

                    if n_HI00 == 0:
                        I1_HI, I2_HI = 0, 0
                    else :
                       I1_HI = (np.interp(K_HI_previous, n_HI, JHI_1) - np.interp(K_HI, n_HI, JHI_1)) * m_corr / r2 / cm_per_Mpc ** 3 / 4 / pi / n_HI00 / dr_current
                       I2_HI = (np.interp(K_HI_previous, n_HI, JHI_2) - np.interp(K_HI, n_HI, JHI_2)) * m_corr / r2 / cm_per_Mpc ** 3 / 4 / pi / n_HI00 / dr_current
                       I1_T_HI = (np.interp(K_HI_previous, n_HI, JT_HI_1) - np.interp(K_HI, n_HI, JT_HI_1)) * m_corr / r2 / cm_per_Mpc ** 3 / 4 / pi / n_HI00 / dr_current

                    I2_Ta = np.interp(K_HI_previous, n_HI, JT_2a) * m_corr / r2 / cm_per_Mpc ** 2 / 4 / pi
                    I2_Tb = np.interp(K_HI_previous, n_HI, JT_2b) * m_corr / r2 / cm_per_Mpc ** 2 / 4 / pi

                    I1_HI, I2_HI, I1_T_HI, I2_Ta, I2_Tb = np.nan_to_num((I1_HI, I2_HI, I1_T_HI, I2_Ta, I2_Tb )) * Ngam_dot_step_l / Ng_dot_initial ### added correctrion for halo growth



                    def rhs(t, n):
                        """
                         Calculate the RHS of the radiative transfer equations.

                         RHS of the coupled nHII, n_HeII, n_HeIII and T equations. The equations are labelled as A,B,C,
                         and D and the rest of the variables are the terms contained in the respective equations.

                         Parameters
                         ----------
                         t : float
                          Time of evaluation in s.
                         n : array-like
                          1-D array containing the variables nHII, nHeII, nHeIII, T for evaluating the RHS.

                         Returns
                         -------
                         array_like
                          The RHS of the radiative transfer equations.
                         """
                        if isnan(n[0]) or isnan(n[1]) :
                            # print('n is :',n)
                            print('Warning: calculations contain nan values, check the rhs')

                        n_HIIx = n[0]
                        n_HIx = n_H_z_r - n[0]

                        if isnan(n_HIIx):
                            n_HIIx = n_H_z_r
                            n_HIx = 0

                        if n_HIIx > n_H_z_r:
                            # if n_HIIx > 1.01 * n_H(zstep_l, C):
                            #    print('n_HII becomes larger than mean n_H. Be careful.')
                            n_HIIx = n_H_z_r
                            n_HIx = 0

                        if n_HIIx < 0:
                            # print('n_HII becomes negative. Be careful.')
                            n_HIIx = 0
                            n_HIx = n_H_z_r

                        Tx = n[1]

                        if isnan(Tx):
                            print('Tx is nan')
                        if (Tx < T_gamma * (1 + zstep_l) ** 1 / (1 + 250)):
                            Tx = T_gamma * (1 + zstep_l) ** 1 / (1 + 250)

                        n_ee = n_HIIx

                        mu = (n_H_z_r ) / (n_H_z_r + n_ee)
                        n_B = n_H_z_r  + n_ee

                        ##coeff for Temp eq
                        A1_HI = xi_HI(Tx) * n_HIx * n_ee
                        A2_HII = eta_HII(Tx) * n_HIIx * n_ee
                        A4_HI = psi_HI(Tx) * n_HIx * n_ee

                        A5 = theta_ff(Tx) * (n_HIIx) * n_ee

                        H = pl.H(zstep_l)
                        H = H.to(u.s ** -1).value

                        A6 = (2 * H * kb * Tx * n_B / mu)

                        # A,B,C for the ionization equation. D for the Temp eq.

                        A = gamma_HI(n_HIIx, n_HIx, Tx, I1_HI, I2_HI,gamma_2c) * n_HIx - alpha_HII(Tx) * n_HIIx * n_ee

                        D = (2 / 3) * mu / (kb_eV_per_K * n_B) * (f_Heat(n_HIIx / n_H_z_r) * (n_HIx * I1_T_HI ) + sigma_s * n_ee / m_e_eV * (I2_Ta + Tx * I2_Tb) - (A1_HI  + A2_HII  + A4_HI + A5 + A6))

                        return ravel(array([A, D], dtype="object"))





                    y0 = zeros(2)
                    y0[0] = n_HII_grid[k]
                    y0[1] = T_grid[k]

                    if param.solver.method == 'sol':
                        sol = integrate.solve_ivp(rhs, [l * dt_init.value, (l + 1) * dt_init.value], y0, method='RK45')
                        n_HII_grid[k] = sol.y[0, -1]
                        T_grid[k] = nan_to_num(sol.y[1, -1])

                    if param.solver.method == 'bruteforce' :
                        kick = rhs(l * dt_init.value, y0)
                        n_HII_grid[k] += dt_init.value * kick[0]
                        T_grid[k] += dt_init.value * kick[1]
                        # n_HII_grid[k] = n_HII_grid[k] + dt_init.value * (I1_HI * (n_H_z_r - n_HII_grid[k]) - alpha_HII(T_grid[k]) * n_HII_grid[k] ** 2)  # sol.y[0, -1]  + beta_HI(1e4) * n_HII_grid[k]



                    if isnan(n_HII_grid[k]):
                        n_HII_grid[k] = n_H_z_r

                    if n_HII_grid[k] > n_H_z_r:
                        n_HII_grid[k] = n_H_z_r

                time_grid.append(l * self.grid_param['dt_init'].value)

                front_step = find_Ifront(n_HII_grid / nHI0_profile_z, self.r_grid)
                Ion_front_grid.append(front_step)
                Mh_history.append(Mh_step_l)
                z_grid.append(zstep_l[0])
                Ng_dot_history.append(Ngam_dot_step_l)


                nHI0_profile_step = (self.nHI0_profile[1:] + self.nHI0_profile[:-1]) / 2 * ( 1 + zstep_l) ** 3  # n_H(znow,self.C)
                nHI0_profile_step = np.concatenate((nHI0_profile_step, [nHI0_profile_step[-1]]))

                p0 = [30 / front_step, front_step]  # intial guess for the fit. c1 has to be increased when the ion front goes to smaller scales (sharpness, log scale)
                xdata, ydata = self.r_grid, (nHI0_profile_step- n_HII_grid)/ nHI0_profile_step

                try :
                    Fit_ = curve_fit(profile_1D, xdata, ydata, p0=p0)
                    c1, c2 = Fit_[0][0], Fit_[0][1]
                except Exception:
                    c1, c2 = 0, 0

                c1_history.append(c1)
                c2_history.append(c2)



                l += 1

                if np.exp(-cm_per_Mpc * (K_HI * sigma_HI(13.6))) > 0.1:
                    print('np.exp(-tau(rmax)) > 0.1. Some photons are not absorbed. Maybe you need larger rmax. ')

            time_grid = array([time_grid])
            time_grid = time_grid.reshape(time_grid.size, 1)
            Ion_front_grid = array([Ion_front_grid])
            Ion_front_grid = Ion_front_grid.reshape(Ion_front_grid.size, 1)
            time_end_solve = datetime.datetime.now()
            print('solver took :', time_end_solve - t_start_solver)
            break

        age = pl.age(self.z)
        age = age.to(u.s)
        age += self.evol
        func = lambda z: pl.age(z).to(u.s).value - age.value
        znow = fsolve(func, self.z)

        nHI0_profile_now = (self.nHI0_profile[1:] + self.nHI0_profile[:-1])/2  * (1 + znow) ** 3   #n_H(znow,self.C)
        nHI0_profile_now  = np.concatenate((nHI0_profile_now,[nHI0_profile_now[-1]]))


        self.n_HI_grid = nHI0_profile_now - n_HII_grid
        self.n_HII_grid = n_HII_grid
        self.n_H = nHI0_profile_now
        self.T_grid = T_grid
        self.time_grid = time_grid
        self.Ion_front_grid = Ion_front_grid
        self.z_now = znow
        self.z_history = z_grid
        self.Ngdot_history = Ng_dot_history
        self.Mh_History = Mh_history
        self.c1_history = c1_history
        self.c2_history = c2_history



    def fit(self):
        p0 = [30/self.Ion_front_grid[-1][0],  self.Ion_front_grid[-1][0]]  # intial guess for the fit. c1 has to be increased when the ion front goes to smaller scales (sharpness, log scale)
        xdata, ydata = self.r_grid, self.n_HI_grid / self.n_H
        Fit_ = curve_fit(profile_1D, xdata, ydata, p0=p0)
        self.c1 = Fit_[0][0]
        self.c2 = Fit_[0][1]



def profile_1D(r, c1, c2):
    '''
    Sigmoid Function that we fit to the ionization profile. In order to just store 2 parameters per profiles

    Parameters
    ----------
    r :  distance from the source in Mpc.
    c1 : shaprness of the profile (sharp = high c1)
    c2 : ionization front, value where profile_1D = 0.5
    '''
    out = 1/(1+np.exp(-c1*(r-c2)))
    return out