import scipy.integrate as integrate
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as interpolate
from tqdm import tqdm
from astropy import units as u
from astropy.cosmology import WMAP7 as pl
from scipy.optimize import fsolve
from sys import exit
import dill
from glob import glob
import pickle
from scipy.signal import fftconvolve
import datetime 


# Initialize the units and conversion factors
facr = 1 * u.Mpc
facr = (facr.to(u.cm)).value
facE = (1 * u.eV).to(u.erg).value
cm_3 = u.cm ** -3
s1 = u.s ** -1
u1 = u.erg * u.cm ** 3 * u.s ** -1
u2 = u.cm ** 3 * u.s ** -1
u3 = u.erg * u.s ** -1
diff = 3.371399040053564 * 10 ** -17

# H, He density and mass
n_H_0 = 1.9 * 10 ** -7
n_He_0 = 1.5 * 10 ** -8
m_H = 1.6 * 10 ** - 24 * u.g
m_He = 6.6464731 * 10 ** - 24 * u.g

# Energy limits and ionization energies
E_0 = 10.4 * facE
E_HI = 13.6 * facE
E_HeI = 24.59 * facE
E_HeII = 54.42 * facE
E_upp = 10 ** 4 * facE
E_cut = 200. * facE

# Constants
h = 1.0546 * 10 ** -27 * u.erg * u.s
c = 2.99792 * 10 ** 10 * u.cm / u.s
kb = 1.380649 * 10 ** -23 * u.J / u.K
kb = kb.to(u.erg / u.K)
m_e = 9.10938 * 10 ** -28 * u.g
sigma_s = 6.6524 * 10 ** -25 * u.cm ** 2
M_sun = 1.988 * 10 ** 33 * u.g
sec_per_year = 3600*24*365.25
Hz_per_eV = 241799050402293

##Cosmology
Om      = 0.31
Ob      = 0.048
Ol      = 1-Om-Ob
h0      = 0.68
ns      = 0.97
s8      = 0.81

h__ = 6.626e-34     ### [J.s] or [m2 kg / s]
c__ = 2.99792e+8    ### [m/s]
k__ = 1.380649e-23  ### [m 2 kg s-2 K-1]
h_eV_sec = 4.135667e-15

def BB_Planck( nu , T):  #  BB Spectrum [J s-1 m−2 Hz−1 ]
    a_ = 2.0*h__*nu**3/c__**2
    intensity = 4*pi*a_ / ( exp(h__*nu/(k__*T)) - 1.0)
    return intensity


# Photoionization and Recombination coefficients
def alpha_HII(T):
    return(6.28 * 10 ** -11 * T ** -0.5 * (T/10**3) ** -0.2 * (1+(T/10**6)**0.7)**-1)
    #return 2.6 * 10 ** -13 * (T / 10 ** 4) ** -0.8


def alpha_HeII(T):
    return 1.5 * 10 ** -10 * T ** -0.6353


def alpha_HeIII(T):
    return 3.36 * 10 ** -10 * T ** -0.5 * (T / 10 ** 3) ** -0.2 * (1 + (T / (4 * 10 ** 6)) ** 0.7) ** -1


def beta_HI(T):
    # print(5.85*10**-11*T**0.5*(1+(T/10**5)**0.5)**-1*exp(-1.578*10**5/T))
    return 5.85 * 10 ** -11 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-1.578 * 10 ** 5 / T)


def beta_HeI(T):
    return 2.38 * 10 ** -11 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-2.853 * 10 ** 5 / T)


def beta_HeII(T):
    return 5.68 * 10 ** -12 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-6.315 * 10 ** 5 / T)


def xi_HI(T):
    return 1.27 * 10 ** -21 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-1.58 * 10 ** 5 / T)


def xi_HeI(T):
    return 9.38 * 10 ** -22 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-2.85 * 10 ** 5 / T)


def xi_HeII(T):
    return 4.95 * 10 ** -22 * T ** 0.5 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-6.31 * 10 ** 5 / T)


def eta_HII(T):
    return 6.5 * 10 ** -27 * T ** 0.5 * (T / 10 ** 3) ** -0.2 * (1 + (T / 10 ** 6) ** 0.7) ** -1


def eta_HeII(T):
    return 1.55 * 10 ** -26 * T ** 0.3647


def eta_HeIII(T):
    return 3.48 * 10 ** -26 * T ** 0.5 * (T / 10 ** 3) ** -0.2 * (1 + (T / (4 * 10 ** 6)) ** 0.7) ** -1


def psi_HI(T):
    return 7.5 * 10 ** -19 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-1.18 * 10 ** 5 / T)


def psi_HeI(T, neT, n_HeIIT):
    return 9.1 * 10 ** -27 * T ** -0.1687 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-1.31 * 10 ** 4 / T)


def psi_HeII(T):
    return 5.54 * 10 ** -17 * T ** -0.397 * (1 + (T / 10 ** 5) ** 0.5) ** -1 * exp(-4.73 * 10 ** 5 / T)


def omega_HeII(T):
    return 1.24 * 10 ** -13 * T ** -1.5 * exp(-4.7 * 10 ** 5 / T) * (1 + 0.3 * exp(-9.4 * 10 ** 4 / T))


def theta_ff(T):
    return 1.3 * 1.42 * 10 ** -27 * (T) ** 0.5


def zeta_HeII(T):
    return 1.9 * 10 ** -3 * T ** -1.5 * exp(-4.7 * 10 ** 5 / T) * (1 + 0.3 * exp(-9.4 * 10 ** 4 / T))


# HI, HeI, HeII Photoionization cross sections
def sigma_HI(E):
    sigma_0 = 5.475 * 10 ** 4 * 10 ** -18
    E_01 = 4.298 * 10 ** -1 * facE
    y_a = 3.288 * 10 ** 1
    P = 2.963
    y_w = y_0 = y_1 = 0
    x = E / E_01 - y_0
    y = sqrt(x ** 2 + y_1 ** 2)
    F = ((x - 1) ** 2 + y_w ** 2) * y ** (0.5 * P - 5.5) * (1 + sqrt(y / y_a)) ** -P
    sigma = sigma_0 * F
    return sigma


def sigma_HeI(E):
    sigma_0 = 9.492 * 10 ** 2 * 10 ** -18
    E_01 = 1.361 * 10 ** 1 * facE
    y_a = 1.469
    P = 3.188
    y_w = 2.039
    y_0 = 4.434 * 10 ** -1
    y_1 = 2.136
    x = E / E_01 - y_0
    y = sqrt(x ** 2 + y_1 ** 2)
    F = ((x - 1) ** 2 + y_w ** 2) * y ** (0.5 * P - 5.5) * (1 + sqrt(y / y_a)) ** -P
    sigma = sigma_0 * F
    return sigma


def sigma_HeII(E):
    sigma_0 = 1.369 * 10 ** 4 * 10 ** -18  # cm**2
    E_01 = 1.72 * facE
    y_a = 3.288 * 10 ** 1
    P = 2.963
    y_w = 0
    y_0 = 0
    y_1 = 0
    x = E / E_01 - y_0
    y = sqrt(x ** 2 + y_1 ** 2)
    F = ((x - 1) ** 2 + y_w ** 2) * y ** (0.5 * P - 5.5) * (1 + sqrt(y / y_a)) ** -P
    sigma = sigma_0 * F
    return sigma


# H and He IGM densities
def n_H(z,C):
    return C*n_H_0 * (1 + z) ** 3


def n_He(z,C):
    return C*n_He_0 * (1 + z) ** 3


# Factor for secondary ionization and heating factor
def f_H(x, z_reion):

    if isnan(x):
        x = 1
    if x < 0:
        return 0.3908 * (1 - 0 ** 0.4092) ** 1.7592
    if x > 1:
        return 0.3908 * (1 - 1 ** 0.4092) ** 1.7592

    return nan_to_num(0.3908 * (1 - x ** 0.4092) ** 1.7592)


def f_He(x, z_reion):

    if isnan(x):
        x = 1
    if x < 0:
        return 0.0554 * (1 - 0 ** 0.4614) ** 1.6660
    if x > 1:
        return 0.0554 * (1 - 1 ** 0.4614) ** 1.6660

    return nan_to_num(0.0554 * (1 - x ** 0.4614) ** 1.6660)


def f_Heat(xion, z_reion):

    if isnan(xion):
        xion = 1
    if xion < 0:
        return 0.9971 * (1 - (1 - 0 ** 0.2663) ** 1.3163)
    if xion > 1:
        return 0.9971 * (1 - (1 - 1 ** 0.2663) ** 1.3163)
    if xion> 10 ** -4:
        return nan_to_num(0.9971 * (1 - (1 - xion ** 0.2663) ** 1.3163))
    return 0.15


def find_Ifront(x, r, z_reion, show=None):
    """
    Finds the ionization front, the position where the ionized fraction is 0.5.

    Parameters
    ----------
    n : array_like
     Ionized hydrogen density along the radial grid.
    r : array_like
     Radial grid in linear space.
    z_reion :
     Redshift of the source.
    show : bool, optional
     Decide whether or not to print the position of the ionization front

     Returns
     -------
     float
      Returns the position of the ionization front, one of the elements in the array r.
    """
    m = 0
    m = argmin(abs(0.5 - x))
    check = show if show is not None else False

    if check == True:
        print('Pos. of Ifront:', r[m])
    return r[m]

def adaptive_mesh(r,x,lowtol,uptol):
    """
    Builds a new adapted grid based on the ionization front.

    The function increases the resolution around the radial space where the neutral fraction is changing the most. As an
    example, the function can increase the resolution of the grid where the neutral fraction is in the range 0.01 to
    0.99 and leave the rest of the grid as it is.

    Parameters
    ----------
    r : array_like
     Initial radial grid.
    x : array_like
     Neutral fraction profile along the radial grid.
    lowtol : flaot
     Lower value of the area of the neutral fraction profile we like to increase the resolution of. Between 0.0 to 1.0
    uptol : float
     Upper value of the area of the neutral fraction profile we like to increase the resolution of. Between 0.0 to 1.0

    Returns
    -------
    array_like
     Newly adapted grid with an increased resolution where the neutral fraction is between lowtol and uptol. Careful:
     The new grid is not uniform anymore, but the calculations in solve() are adjusted such that this is taken into
     account.
    """

    """
    # adaptive grid + add logspace points between r[0] and r[1]
    lower = argmin(abs(x - lowtol))
    upper = argmin(abs(x - uptol))
    dr_grid = zeros(r.size - 1)
    for k in range(dr_grid.size):
        dr_grid[k] = r[k + 1] - r[k]
    dr_prev = r[upper]-r[upper-1]
    dr_new = dr_prev /7
    crit = argmin(abs(r-0.001))
    r1a = 10**linspace(log10(r[0]),log10(r[1]),5,endpoint = True)
    r1b = r[2:lower]
    r2 = arange(r[lower], r[upper], dr_new)
    r3 = r[upper + 1:]
    r_new = concatenate((r1a,r1b, r2, r3))
    print('lower: ', r[lower],x[lower])
    print('upper: ', r[upper], x[upper])
    dr_grid_new = zeros(r_new.size)
    for k in range(dr_grid.size):
        dr_grid_new[k] = r_new[k + 1] - r_new[k]
        assert dr_grid_new[k] > 0
    """

    # regular logspace adaptive grid
    lower = argmin(abs(x - lowtol))
    upper = argmin(abs(x - uptol))
    dr_grid = zeros(r.size - 1)
    for k in range(dr_grid.size):
        dr_grid[k] = r[k + 1] - r[k]
    #dr_prev = min(dr_grid)
    #dr_new = dr_prev /3
    #r1 = r[0:lower]
    #r2 = arange(r[lower], r[upper], dr_new)
    N_prev = r[lower:upper].size
    N_new = N_prev * 3
    r1 = r[0:lower]
    r2 = logspace(log10(r[lower]), log10(r[upper]), N_new,base=10)
    r3 = r[upper + 1:]
    r_new = concatenate((r1, r2, r3))
    print('check:',N_prev,N_new,r.size,r1.size,r2.size,r3.size)
    print('lower: ', r[lower],x[lower])
    print('upper: ', r[upper], x[upper])
    dr_grid_new = zeros(r_new.size)
    for k in range(dr_grid.size):
        dr_grid_new[k] = r_new[k + 1] - r_new[k]
        assert dr_grid_new[k] > 0

    """
    # non-adaptive gridding - new grid with half the step size
    n = r.size
    r_new = linspace(r[0],r[-1],2*n)
    """
    return r_new

def generate_table(param, z, r_grid, n_HI, n_HeI, alpha, sed):
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
    r_grid : array_like
     Radial Mpc grid in logspace from the source to a given distance, the starting distance is typically 0.0001 Mpc from
     the source.
    n_HI : array_like
     Neutral hydrogen density array in cm**-3 along r_grid. Typically a linear space grid starting from 0.
    n_HeI : array_like
     Neutral helium density array in cm**-3 along r_grid. Typically a linear space grid starting from 0.
    n_HI : array_like
     HeII density array in cm**-3 along r_grid. Typically a linear space grid starting from 0.
    E_0 : in eV, minimum energy of photons (depends on whether or not ionising photons get absorbed), we then convert to erg.
    alpha : int, default -1
     Spectral index for a power-law source.
    sed : callable, optional, or string
     Spectral energy distribution to be used instead of the default power-law function. sed is a function of energy.
    filename_table : str, optional
     Used to import a table and skip the calculation
    recalculate_table : bool, default False
     Parameter to either calculate the table or to check if a table is already given

    Returns
    ------
    dict of {str:dict}
        Dictionary containing two sub-dictionaries: The first one containing the function variables and the second one
        containing the 12 tables for the integrals
    '''
    E_0 = facE*param.source.E_upp 
    E_upp = facE*param.source.E_0 ##erg

    if param.table.import_table :
        if param.table.filename_table == None :
            print('Asking to import a table but filename_table is None. Exit')
            exit()
        else :
            Gamma_input_info = pickle.load(open(param.table.filename_table, 'rb'))
            print('Reading in table ', param.table.filename_table)

    else :

        print('Calculating table...')
        r_min = r_grid[0]
        dr = r_grid[1] - r_grid[0]

        if (param.source.type == 'Miniqsos'):  ### Choose the source type
                M = param.source.M_miniqso
                L = 1.38 * 10 ** 37 * M
                Ag = L / (4 * pi * facr ** 2 * (integrate.quad(lambda x: x ** -alpha, E_0, E_upp)[0]))
                print('Miniqsos model chosen. M_qso is ', M)
                                
                # Spectral energy function, power law for a quasar source
                def I(E):
                        #if sed is not None: return sed(E)
                        miniqsos = E ** -alpha
                        return Ag * miniqsos

                #print(I(E_0),I(E_upp),I(E_0+ E_upp/4),L /(E_upp-E_0)*log(E_upp/E_0))
                # Radiation flux times unit distance
                def N(E, n_HI0, n_HeI0):
            	        int = dr * facr * (n_HI0 * sigma_HI(E) + n_HeI0 * sigma_HeI(E))
            	        return exp(-int) * I(E)

        elif (param.source.type == 'Galaxies'):
                f_c2ray = param.source.fc2ray
                M_halo = param.source.M_halo                
                Delta_T = 10 ** 7 * sec_per_year
                N_ion_dot = f_c2ray * M_halo * M_sun /m_H * Ob/Om /Delta_T #### M_sun is in gramms
                print('Galaxy model chosen. M_halo is ', M_halo)

                T_Galaxy = param.source.T_gal

                #nu_0 = E_0/h.value      ### This is in Hz (E_0 in erg and h in erg.s)
                #nu_upp = E_upp/h.value
                #nu_range=np.logspace(np.log10(nu_0),np.log10(nu_upp),500,base=10)

                nu_range = np.logspace(np.log10(E_0 / h_eV_sec), np.log10(E_upp / h_eV_sec), 300, base=10)
                norm__ = np.trapz( BB_Planck(nu_range,T_Galaxy)/ h__,np.log(nu_range) )
			
                I__ =  N_ion_dot / (4 * pi * facr ** 2 * norm__)
                print('BB spectrum normalized to ',N_ion_dot, ' ionizing photons per s, in the energy range [',param.source.E_0,' ',param.source.E_upp, '] in eV')

                def N(E, n_HI0, n_HeI0): ####[erg/sec/erg/cm^2]
            	        nu_ = Hz_per_eV * E
            	        int = dr * facr * (n_HI0 * sigma_HI(E) + n_HeI0 * sigma_HeI(E))
            	        return exp(-int) * I__ * BB_Planck( nu_ , T_Galaxy)/h__
            	        #return exp(-int) * I__ *(2.0 *  nu_ ** 3 / c__ ** 2) * 4 * pi / (exp(h__ * nu_ / (k__ * T_Galaxy)) - 1.0)

        else :
                print('Source Type not available. Should be Galaxies or Miniqsos' )
                exit()
        #E_values = logspace(log10(E_0),log10(E_upp),10,base=10)

        IHI_1 = zeros((n_HI.size, n_HeI.size))
        IHI_2 = zeros((n_HI.size, n_HeI.size))
        IHI_3 = zeros((n_HI.size, n_HeI.size))

        IHeI_1 = zeros((n_HI.size, n_HeI.size))
        IHeI_2 = zeros((n_HI.size, n_HeI.size))
        IHeI_3 = zeros((n_HI.size, n_HeI.size))

        IHeII = zeros((n_HI.size, n_HeI.size))

        IT_HI_1 = zeros((n_HI.size, n_HeI.size))
        IT_HeI_1 = zeros((n_HI.size, n_HeI.size))
        IT_HeII_1 = zeros((n_HI.size, n_HeI.size))

        IT_2a = zeros((n_HI.size, n_HeI.size))
        IT_2b = zeros((n_HI.size, n_HeI.size))
        for k2 in tqdm(range(0, n_HI.size, 1)):
            for k3 in range(0, n_HeI.size, 1):
                IHI_1[k2, k3] = \
                integrate.quad(lambda x: sigma_HI(x) / x * N(x, n_HI[k2], n_HeI[k3]), max(E_0, E_HI), E_upp)[0]
			
                IHI_2[k2, k3] = integrate.quad(
                    lambda x: sigma_HI(x) * (x - E_HI) / (E_HI * x) * N(x, n_HI[k2], n_HeI[k3]), max(E_0, E_HI),
                    E_upp)[0]
                IHI_3[k2, k3] = integrate.quad(
                    lambda x: sigma_HeI(x) * (x - E_HeI) / (x * E_HI) * N(x, n_HI[k2], n_HeI[k3]),
                    max(E_0, E_HeI), E_upp)[0]

                IHeI_1[k2, k3] = \
                integrate.quad(lambda x: sigma_HeI(x) / x * N(x, n_HI[k2], n_HeI[k3]), max(E_0, E_HeI), E_upp)[0]
                IHeI_2[k2, k3] = integrate.quad(
                    lambda x: sigma_HeI(x) * (x - E_HeI) / (x * E_HeI) * N(x, n_HI[k2], n_HeI[k3]),
                    max(E_0, E_HeI), E_upp)[0]
                IHeI_3[k2, k3] = integrate.quad(
                    lambda x: sigma_HI(x) * (x - E_HI) / (x * E_HeI) * N(x, n_HI[k2], n_HeI[k3]), max(E_0, E_HeI),
                    E_upp)[0]

                IHeII[k2, k3] = \
                integrate.quad(lambda x: sigma_HeII(x) / x * N(x, n_HI[k2], n_HeI[k3]), max(E_0, E_HeII), E_upp)[
                    0]

                IT_HI_1[k2, k3] = \
                integrate.quad(lambda x: sigma_HI(x) * (x - E_HI) / x * N(x, n_HI[k2], n_HeI[k3]), max(E_0, E_HI),
                                E_upp)[0]
                IT_HeI_1[k2, k3] = \
                integrate.quad(lambda x: sigma_HeI(x) * (x - E_HeI) / x * N(x, n_HI[k2], n_HeI[k3]),
                                max(E_0, E_HeI), E_upp)[0]
                IT_HeII_1[k2, k3] = \
                integrate.quad(lambda x: sigma_HeII(x) * (x - E_HeII) / x * N(x, n_HI[k2], n_HeI[k3]),
                                max(E_0, E_HeII), E_upp)[0]

                IT_2a[k2, k3] = integrate.quad(lambda x: N(x, n_HI[k2], n_HeI[k3]) * x, E_0, E_upp)[
                    0]
                IT_2b[k2, k3] = \
                integrate.quad(lambda x: N(x, n_HI[k2], n_HeI[k3]) * (-4 * kb.value), E_0, E_upp)[0]

        print('...done')

        Gamma_info = {'HI_1': IHI_1, 'HI_2': IHI_2, 'HI_3': IHI_3,
                      'HeI_1': IHeI_1, 'HeI_2': IHeI_2, 'HeI_3': IHeI_3, 'HeII': IHeII,
                      'T_HI_1': IT_HI_1, 'T_HeI_1': IT_HeI_1, 'T_HeII_1': IT_HeII_1,
                      'T_2a': IT_2a, 'T_2b': IT_2b}

        input_info = {'M': M, 'z': z,
                      'r_grid': r_grid,
                      'n_HI': n_HI, 'n_HeI': n_HeI, 'E_0': param.source.E_0 , 'E_upp': param.source.E_upp}

        Gamma_input_info = {'Gamma': Gamma_info, 'input': input_info}

        if param.table.filename_table is None:
            filename_table = 'qwerty'
            print('No filename_table given. We will call it qwerty.')
        else :
            filename_table = param.table.filename_table
        print('saving table in pickle file ',filename_table)
        pickle.dump(Gamma_input_info,open(filename_table,'wb'))


    return Gamma_input_info

def gamma_HI(n_HIIx, n_HeIx, n_HeIIx, n_HeIIIx, Tx, I1_HI, I2_HI, I3_HI, zstar, C, gamma_2c):
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
    Returns
    -------
    float
     Gamma_HI for the radiative transfer equation.
    """
    n_e = n_HIIx + n_HeIIx + 2 * n_HeIIIx
    if n_H(zstar,C) == n_HIIx or n_He(zstar, C) - n_HeIIx - n_HeIIIx == 0:
        if n_H(zstar,C) == n_HIIx and n_He(zstar, C) - n_HeIIx - n_HeIIIx == 0:
            factor = 1
        else:
            factor = 0
    else:
        factor = abs((n_He(zstar, C) - n_HeIIx - n_HeIIIx) / (n_H(zstar, C) - n_HIIx))

    return gamma_2c.value + beta_HI(Tx) * n_e + I1_HI + f_H(n_HIIx/n_H(zstar,C), zstar) * I2_HI + f_H(n_HIIx/n_H(zstar,C),zstar) * factor * I3_HI


def gamma_HeI(n_HIIx, n_HeIx, n_HeIIx, n_HeIIIx, I1_HeI, I2_HeI, I3_HeI, zstar, C):
    """
    Calculate gamma_HeI given the densities and the temperature
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
    Returns
    -------
    float
     Gamma_HeI for the radiative transfer equation.
    """
    if n_H(zstar, C) == n_HIIx or n_He(zstar, C) - n_HeIIx - n_HeIIIx == 0:
        if n_H(zstar, C) == n_HIIx and n_He(zstar, C) - n_HeIIx - n_HeIIIx == 0:
            factor = 1
        else:
            factor = 0
    else:
        factor = abs(
            nan_to_num(n_H(zstar, C) - n_HIIx) / (n_He(zstar, C) - n_HeIIx - n_HeIIIx))

    return I1_HeI + f_He(n_HIIx/n_H(zstar,C), zstar) * I2_HeI + f_He(n_HIIx/n_H(zstar,C), zstar) * factor * I3_HeI

def gamma_HeII(I1_HeII):
    """
    Calculate gamma_HeII given the densities and the temperature
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
    Returns
    -------
    float
     Gamma_HeII for the radiative transfer equation.
    """
    return I1_HeII




class Source:
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
        else :
            print('source.type should be Galaxies or Miniqsos')
            exit()

        self.z = param.solver.z  # redshift
        self.evol = (param.solver.evol*10**6*365*24*60*60)*u.s  # evolution time, typically 3-10 Myr
        self.LE = True #LE if LE is not None else True
        self.lifetime = None

        self.alpha = param.source.alpha
        self.sed = None

        self.r_start = param.solver.r_start * u.Mpc   # starting point from source
        self.r_end = param.solver.r_end * u.Mpc       # maximal distance from source
        self.dn = param.solver.dn
        self.dn_table = param.solver.dn_table
        self.Gamma_grid_info = None
        self.C = param.solver.C # Clumping factor

        self.E_0 = facE*param.source.E_0     
        self.E_upp = facE*param.source.E_upp # IN erg


    def create_table(self, param, par=None):
        """
        Call the function to create the interpolation tables.

        Parameters
        ----------
        par : dict of {str:float}, optional
         Variables to pass on to the table generator. If none is given the parameters of the Source initialization will
         be used.
        """

        if par is None:
            M, z_reion = self.M, self.z
            alpha, sed = self.alpha, self.sed
            dn_table = self.grid_param['dn_table']

            #r_grid =  logspace(log10(self.grid_param['r_start']), log10(self.grid_param['r_end']), dn,base=10)
            r_grid =  linspace(self.grid_param['r_start'], self.grid_param['r_end'], dn_table)

            N = r_grid.size
            n_HI = linspace(0, 2*(N) * n_H(z_reion,self.C),100)
            n_HeI = linspace(0,  (N) * n_He(z_reion,self.C), 100)


        else:
            M, z_reion = par['M'], par['z_reion']
            r_grid = par['r_grid']
            n_HI, n_HeI = par['n_HI'], par['n_HeI']
            alpha, sed = par['alpha'], par['sed']

        Gamma_grid_info = generate_table(param, z_reion, r_grid, n_HI, n_HeI, alpha, sed)
        self.Gamma_grid_info = Gamma_grid_info

    def initialise_grid_param(self):
        """
        Initialize the grid parameters for the solver.

        The grid parameters and the initial conditions are saved in a dictionary. A initial time step size and a initial
         radial grid is chosen.T_gamma is the CMB temperature for the given redshift and gamma_2c is used to calculate
         the contribution from the background photons.
        """
        grid_param = {'M': self.M, 'z_reion': self.z}

        T_gamma = 2.725 * (1 + grid_param['z_reion']) * u.K
        gamma_2c = (alpha_HII(T_gamma.value) * (m_e * kb * T_gamma / (2 * pi)) ** (3 / 2) / h ** 3 * exp(
            -(3.4 * u.eV).to(u.erg) / (T_gamma * kb))).decompose()
        C = self.C

        grid_param['T_gamma'] = T_gamma
        grid_param['gamma_2c'] = gamma_2c

        dt_init = self.evol/75  # time interval
        if self.M >= 10**8:
            dt_init = self.evol/150
            print('Due to the large mass of the source, the timestep is increased, which will increase the computation time.')
        t_life = self.evol.value

        dn = self.dn  ####100
        r_start = self.r_start
        r_end = self.r_end
        dn_table = self.dn_table

        grid_param['dn'] = dn
        grid_param['dn_table'] = dn_table
        grid_param['r_start'] = r_start.value
        grid_param['r_end'] = r_end.value
        grid_param['t_life'] = t_life
        grid_param['dt_init'] = dt_init
        grid_param['C'] = C

        self.grid_param = grid_param

    def solve(self,param):
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
        process until the radial cells are updated l times such that l*dt_init has reached the evolution time. After the
        solver is finished we compare the ionization fronts of two consecutive runs and require an accuracy of 5% in
        order to finish the calculations. If the accuracy is not reached we store the values from the run and start
        again with time step size dt_init/2 and a radial grid with half the step size from the previous run.
        This process is repeated until the desired accuracy is reached.
        """
        print('Solving the radiative equations...')
        t_start_solver = datetime.datetime.now()
        self.initialise_grid_param()
        z_reion = self.grid_param['z_reion']
        gamma_2c = self.grid_param['gamma_2c']
        T_gamma = self.grid_param['T_gamma']
        C = self.grid_param['C']

        dt_init = self.grid_param['dt_init']
        dn = self.grid_param['dn']

        #r_grid0 = linspace(self.grid_param['r_start'], self.grid_param['r_end'], dn)
        #r_grid  = linspace(self.grid_param['r_start'], self.grid_param['r_end'], dn)
        r_grid0 = logspace(log10(self.grid_param['r_start']), log10(self.grid_param['r_end']), dn,base=10)
        r_grid  = logspace(log10(self.grid_param['r_start']), log10(self.grid_param['r_end']), dn,base=10)




        n_HII_grid = zeros_like(r_grid0)
        n_HeII_grid = zeros_like(r_grid0)
        n_HeIII_grid = zeros_like(r_grid0)

        self.create_table(param = param)

        N = r_grid.size
        n_HI = self.Gamma_grid_info['input']['n_HI']
        n_HeI = self.Gamma_grid_info['input']['n_HeI']
        points = (n_HI, n_HeI)

        if self.M >= 10 ** 8:
            method = 'LSODA'
            
        else:
            method = 'LSODA'

        while True:

            time_grid = []
            Ion_front_grid = []

            dn = self.grid_param['dn']

            n_HII0 = copy(n_HII_grid[:])
            n_HeII0 = copy(n_HeII_grid[:])
            n_HeIII0 = copy(n_HeIII_grid[:])
            n_HII_grid = zeros_like(r_grid)
            n_HeII_grid = zeros_like(r_grid)
            n_HeIII_grid = zeros_like(r_grid)

            T_grid = zeros_like(r_grid)
            T_grid += T_gamma.value * (1 + z_reion) ** 1 / (1 + 250)

            l = 0

            print('Number of time steps: ', int(math.ceil(self.grid_param['t_life'] / dt_init.value)))

            Gamma_info = self.Gamma_grid_info['Gamma']

            JHI_1, JHI_2, JHI_3 = Gamma_info['HI_1'], Gamma_info['HI_2'], Gamma_info['HI_3']
            JHeI_1, JHeI_2, JHeI_3, JHeII = Gamma_info['HeI_1'], Gamma_info['HeI_2'], Gamma_info['HeI_3'], Gamma_info[
                'HeII']
            JT_HI_1, JT_HeI_1, JT_HeII_1 = Gamma_info['T_HI_1'], Gamma_info['T_HeI_1'], Gamma_info['T_HeII_1']
            JT_2a, JT_2b = Gamma_info['T_2a'], Gamma_info['T_2b']

            print('t_life is ',self.grid_param['t_life'])
            while l * self.grid_param['dt_init'].value <= self.grid_param['t_life']:
                if l % 5 == 0 and l!=0:
                    print('Current Time step: ', l)

                # Calculate the redshift z(t)
                age = pl.age(z_reion)
                age = age.to(u.s)
                age += l * self.grid_param['dt_init']
                func = lambda z: pl.age(z).to(u.s).value - age.value
                zstar = fsolve(func, z_reion)

                # Initialize the values to evaluate the integrals
                K_HI = 0
                K_HeI = 0
                K_HeII = 0
                #print('starting k loop. Gridsize is ',r_grid.size)
                for k in (arange(0, r_grid.size, 1)):

                    table_grid = self.Gamma_grid_info['input']['r_grid']
                    dr_initial = table_grid[1]-table_grid[0]
                    dr_current = r_grid[1] - r_grid[0]
                    correction = dr_current / dr_initial

                    if k > 0:
                        dr_current = r_grid[k] - r_grid[k-1]
                        correction = dr_current / dr_initial


                        n_HI00 = n_H(zstar,C) - n_HII_grid[k - 1]

                        if n_HI00 < 0:
                            n_HI00 = 0
                        n_HeI00 = n_He(zstar, C) - n_HeII_grid[k - 1] - n_HeIII_grid[k - 1]

                        if n_HeI00 < 0:
                            n_HeI00 = 0
                        if n_HeI00 > n_He(zstar, C):
                            n_HeI00 = n_He(zstar, C)

                        K_HI += abs(nan_to_num(n_HI00)+nan_to_num(n_HeII_grid[k - 1]))*correction
                        K_HeI += abs(nan_to_num(n_HeI00))*correction
                        K_HeII += abs(nan_to_num(n_HeII_grid[k - 1]))*correction


                    if self.lifetime is not None and l * self.grid_param['dt_init'].value > (
                    (self.lifetime * u.Myr).to(u.s)).value:

                        I1_HI = 0
                        I2_HI = 0
                        I3_HI = 0

                        I1_HeI = 0
                        I2_HeI = 0
                        I3_HeI = 0

                        I1_HeII = 0

                        I1_T_HI = 0
                        I1_T_HeI = 0
                        I1_T_HeII = 0

                        I2_Ta = 0
                        I2_Tb = 0

                    else:

                        r2 = r_grid[k] ** 2
                        n_corr = exp(-dr_current*diff*K_HeII)
                        corr_ag = (integrate.quad(lambda x: x ** -self.alpha, self.E_0, self.E_upp)[0]) / (
                        integrate.quad(lambda x: x ** -self.alpha, E_0, 0.1 * E_upp)[0]) # numerical correction 
                        if self.M == self.Gamma_grid_info['input']['M']:
                            m_corr = 1
                        else:
                            m_corr = self.M/self.Gamma_grid_info['input']['M']



                        I1_HI = interpolate.interpn(points, JHI_1, (K_HI, K_HeI), method='linear') * n_corr * m_corr / r2 *corr_ag
                        I2_HI = interpolate.interpn(points, JHI_2, (K_HI, K_HeI), method='linear') * n_corr * m_corr / r2 *corr_ag
                        I3_HI = interpolate.interpn(points, JHI_3, (K_HI, K_HeI), method='linear') * n_corr * m_corr / r2 *corr_ag

                        I1_HeI = interpolate.interpn(points, JHeI_1, (K_HI, K_HeI), method='linear') * n_corr * m_corr / r2 *corr_ag
                        I2_HeI = interpolate.interpn(points, JHeI_2, (K_HI, K_HeI), method='linear') * n_corr * m_corr / r2 *corr_ag
                        I3_HeI = interpolate.interpn(points, JHeI_3, (K_HI, K_HeI), method='linear') * n_corr * m_corr / r2 *corr_ag

                        I1_HeII = interpolate.interpn(points, JHeII, (K_HI, K_HeI), method='linear') * n_corr * m_corr / r2 *corr_ag

                        I1_T_HI = interpolate.interpn(points, JT_HI_1, (K_HI, K_HeI), method='linear') * n_corr * m_corr / r2 *corr_ag
                        I1_T_HeI = interpolate.interpn(points, JT_HeI_1, (K_HI, K_HeI), method='linear') * n_corr * m_corr / r2 *corr_ag
                        I1_T_HeII = interpolate.interpn(points, JT_HeII_1, (K_HI, K_HeI), method='linear') * n_corr * m_corr / r2 *corr_ag

                        I2_Ta = interpolate.interpn(points, JT_2a, (K_HI, K_HeI), method='linear') * n_corr * m_corr / r2 *corr_ag
                        I2_Tb = interpolate.interpn(points, JT_2b, (K_HI, K_HeI), method='linear') * n_corr * m_corr / r2 *corr_ag



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

                        if isnan(n[0]) or isnan(n[1]) or isnan(n[2]) or isnan(n[3]):
                            print('Warning: calculations contain nan values, check the rhs')

                        n_HIIx = n[0]
                        n_HIx = n_H(zstar, C) - n[0]
                        if isnan(n_HIIx):
                            n_HIIx = n_H(zstar, C)
                            n_HIx = 0
                        if n_HIIx > n_H(zstar, C):
                            n_HIIx = n_H(zstar, C)
                            n_HIx = 0
                        if n_HIIx < 0:
                            n_HIIx = 0
                            n_HIx = n_H(zstar, C)

                        n_HeIIx = n[1]
                        n_HeIIIx = n[2]
                        n_HeIx = n_He(zstar, C) - n[1] - n[2]
                        if isnan(n_HeIIIx):
                            n_HeIIIx = n_He(zstar, C)
                            n_HeIIx = 0
                            n_HeIx = 0

                        if n_HeIIIx > n_He(zstar, C):
                            n_HeIIIx = n_He(zstar, C)
                            n_HeIIx = 0
                            n_HeIx = 0

                        if n_HeIIIx < 0:
                            n_HeIIIx = 0
                            n_HeIIx = 0
                            n_HeIx = n_He(zstar, C)

                        Tx = n[3]

                        if isnan(Tx):
                            print('Tx is nan')
                        if (Tx < T_gamma.value * (1 + zstar) ** 1 / (1 + 250)):
                            Tx = T_gamma.value * (1 + zstar) ** 1 / (1 + 250)

                        n_ee = n_HIIx + n_HeIIx + 2 * n_HeIIIx

                        mu = (n_H(zstar, C) + 4 * n_He(zstar, C)) / (
                                    n_H(zstar, C) + n_He(zstar, C) + n_ee)
                        n_B = n_H(zstar, C) + n_He(zstar, C) + n_ee

                        A1_HI = xi_HI(Tx) * n_HIx * n_ee
                        A1_HeI = xi_HeI(Tx) * n_HeIx * n_ee
                        A1_HeII = xi_HeII(Tx) * n_HeIIx * n_ee
                        A2_HII = eta_HII(Tx) * n_HIIx * n_ee
                        A2_HeII = eta_HeII(Tx) * n_HeIIx * n_ee
                        A2_HeIII = eta_HeIII(Tx) * n_HeIIIx * n_ee

                        A3 = omega_HeII(Tx) * n_ee * n_HeIIIx

                        A4_HI = psi_HI(Tx) * n_HIx * n_ee
                        A4_HeI = psi_HeI(Tx, n_ee, n_HeIIx) * n_ee
                        A4_HeII = psi_HeII(Tx) * n_HeIIx * n_ee

                        A5 = theta_ff(Tx) * (n_HIIx + n_HeIIx + 4 * n_HeIIIx) * n_ee

                        H = pl.H(zstar)
                        H = H.to(u.s ** -1)

                        A6 = (2 * H * kb * Tx * n_B / mu).value

                        A = gamma_HI(n_HIIx, n_HeIx, n_HeIIx, n_HeIIIx, Tx, I1_HI, I2_HI, I3_HI, zstar,C,  gamma_2c) * n_HIx - alpha_HII(
                            Tx) * n_HIIx * n_ee
                        B = gamma_HeI(n_HIIx, n_HeIx, n_HeIIx, n_HeIIIx, I1_HeI, I2_HeI, I3_HeI, zstar, C) * n_HeIx + beta_HeI(
                            Tx) * n_ee * n_HeIx - beta_HeII(Tx) * n_ee * n_HeIIx - alpha_HeII(
                            Tx) * n_ee * n_HeIIx + alpha_HeIII(Tx) * n_ee * n_HeIIIx - zeta_HeII(
                            Tx) * n_ee * n_HeIIx
                        Cc = gamma_HeII(I1_HeII) * n_HeIIx + beta_HeII(
                            Tx) * n_ee * n_HeIIx - alpha_HeIII(Tx) * n_ee * n_HeIIIx
                        Dd = (Tx / mu) * (-mu / (n_H(zstar, C) + n_He(zstar, C) + n_ee)) * (A + B + 2 * Cc)
                        D = (2 / 3) * mu / (kb.value * n_B) * (
                                    f_Heat(n_HIIx/n_H(zstar,C), zstar) * n_HIx * I1_T_HI + f_Heat(n_HIIx/n_H(zstar,C),
                                                                                       zstar) * n_HeIx * I1_T_HeI + f_Heat(
                                n_HIIx/n_H(zstar,C), zstar) * n_HeIIx * I1_T_HeII + sigma_s.value * n_ee / (m_e * c ** 2).value * (
                                            I2_Ta + Tx * I2_Tb) - (
                                            A1_HI + A1_HeI + A1_HeII + A2_HII + A2_HeII + A2_HeIII + A3 + A4_HI + A4_HeI + A4_HeII + A5 + A6)) + Dd

                        return ravel(array([A, B, Cc, D]))


                    y0 = zeros(4)
                    y0[0] = n_HII_grid[k]
                    y0[1] = n_HeII_grid[k]
                    y0[2] = n_HeIII_grid[k]
                    y0[3] = T_grid[k]

                    t_start_solve_TIME = datetime.datetime.now()
                    sol = integrate.solve_ivp(rhs, [l * dt_init.value, (l + 1) * dt_init.value], y0, method=method)

                    n_HII_grid[k] = sol.y[0, -1]
                    n_HeII_grid[k] = sol.y[1, -1]
                    n_HeIII_grid[k] = sol.y[2, -1]
                    T_grid[k] = nan_to_num(sol.y[3, -1])

                    if isnan(n_HII_grid[k]):
                        n_HII_grid[k] = n_H(zstar, C)

                    if n_HII_grid[k] > n_H(zstar, C):
                        n_HII_grid[k] = n_H(zstar, C)


                    if n_HeII_grid[k] > n_He(zstar, C):
                        n_HeII_grid[k] = n_He(zstar, C)
                        n_HeIII_grid[k] = 0

                    if isnan(n_HeIII_grid[k]):
                        n_HeIII_grid[k] = n_He(zstar, C)


                    if n_HeIII_grid[k] > n_He(zstar, C):
                        n_HeIII_grid[k] = n_He(zstar, C)
                        n_HeII_grid[k] = 0

                    if isnan(n_HeII_grid[k]):
                        print('Warning: Calculations contains NaNs.')
                        n_HeII_grid[k] =  n_He(zstar, C)-n_HeIII_grid[k]


                time_grid.append(l * self.grid_param['dt_init'].value)
                Ion_front_grid.append(find_Ifront(n_HII_grid/n_H(zstar,C), r_grid, zstar))
                l += 1

            r1 = find_Ifront(n_HII0/n_H(zstar,C),  r_grid0, zstar, show=True)
            r2 = find_Ifront(n_HII_grid/n_H(zstar,C),  r_grid, zstar, show=True)
            time_step = datetime.datetime.now()
            print('The accuracy is: ', abs((r1 - r2) / min(abs(r1), abs(r2))), ' -> 0.05 needed. It took : ', time_step-t_start_solver)
            print('r1 is ',r1,'r2 is',r2)
            if abs((r1 - r2) / min(abs(r1), abs(r2))) > 0.05 or r2 == self.r_start.value:

                if r2 == self.r_start.value:
                    print('Ionization front is still at the starting point. Starting again with smaller steps... ')
                r_grid0 = copy(r_grid[:])
                print('old:', r_grid0.size)
                r_grid = adaptive_mesh(r_grid0,1-n_HII_grid/n_H(zstar, C), 0.01, 0.99)
                print('new:', r_grid.size)


            else:
                time_grid = array([time_grid])
                time_grid = time_grid.reshape(time_grid.size, 1)
                Ion_front_grid = array([Ion_front_grid])
                Ion_front_grid = Ion_front_grid.reshape(Ion_front_grid.size, 1)
                time_end_solve = datetime.datetime.now()	
                print('solver took :', time_end_solve-t_start_solver)		
                break
        age = pl.age(self.z)
        age = age.to(u.s)
        age += self.evol
        func = lambda z: pl.age(z).to(u.s).value - age.value
        znow = fsolve(func, z_reion)
        self.n_HI_grid = n_H(znow,self.C) - n_HII_grid
        self.n_HII_grid = n_HII_grid
        self.n_HeI_grid = n_He(znow,self.C) - n_HeII_grid - n_HeIII_grid
        self.n_HeII_grid = n_HeII_grid
        self.n_HeIII_grid = n_HeIII_grid
        self.n_H = n_H(znow,self.C)
        self.n_He = n_He(znow,self.C)
        self.T_grid = T_grid
        self.r_grid = r_grid
        self.time_grid = time_grid
        self.Ion_front_grid =  Ion_front_grid

    def r(self):
        return  self.r_grid

    def nHI(self):
        return self.n_HI_grid

    def nHII(self):
        return self.n_HII_grid

    def nHeI(self):
        return self.n_HeI_grid

    def nHeII(self):
        return self.n_HeII_grid

    def nHeIII(self):
        return self.n_HeIII_grid

    def T(self):
        return self.T_grid

    def nH(self):
        return self.n_H

    def nHe(self):
        return self.n_He

    def xHI(self):
        return self.n_HI_grid / self.n_H

    def xHII(self):
        return self.n_HII_grid / self.n_H

    def xHeI(self):
        return self.n_HeI_grid / self.n_He

    def xHeII(self):
        return self.n_HeII_grid / self.n_He

    def xHeIII(self):
        return self.n_HeIII_grid / self.n_He

    def xHI_profile(self, end):
        """
        Use interpolation to create a profile function from the grid values.

        Parameters
        ----------
        end : float, optional
         Endpoint of interpolation, default is the endpoint of the r_grid. If the given endpoint is larger than
         self.r_end then the grids have to be extended such that the interpolation covers the interval between
         self.r_start and end.

        Returns
        -------
        callable
         Interpolated function xHI(r) with r in Mpc.
        """

        endpoint = end if end is not None else self.r_end.value

        if endpoint > (self.r_end.value):
            assert self.xHI()[-1] > 0.99
            new_grid = linspace(self.r_end.value, endpoint, 100)
            new_grid2 = linspace(0, self.r_start.value, 100)
            new_xHI = ones(100) * self.xHI()[-1]
            new_xHI2 = ones(100) * self.xHI()[0]
            new_grid = concatenate((new_grid2, self.r_grid, new_grid))
            new_xHI = concatenate((new_xHI2, self.xHI(), new_xHI))
            return interpolate.interp1d(new_grid, new_xHI)

        return interpolate.interp1d((self.r_grid), self.xHI())

    def T_profile(self,end):

        if end > self.r_end.value:
            assert abs(self.T_grid[-1]- self.grid_param['T_gamma'].value * (1 + self.z) ** 1 / (1 + 250)) < 5
            new_grid2 = linspace(0,self.r_start.value,100)
            new_grid = linspace(self.r_end.value,end,100)
            new_T2 = ones(100) * self.T_grid[0]
            new_T = ones(100) * self.T_grid[-1]
            new_TT = concatenate((new_T2,self.T_grid,new_T))
            new_rr = concatenate((new_grid2,self.r_grid,new_grid))
            return interpolate.interp1d(new_rr,new_TT)


    def Ifront(self):  # return the ionization front of the source, the distance where n_HII = n_HI
        m = 0
        x = self.n_HII_grid / n_H(self.z,self.C)
        m = argmin(abs(0.5 - x))

        print('Pos. of Ifront:', self.r_grid[m])
        return self.r_grid[m]








