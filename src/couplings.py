"""
Here we compute the Lyman_alpha and collisional coupling coefficient, in order to produce full dTb maps
"""

import numpy as np
from .constants import *
import pkg_resources
from .cosmo import comoving_distance, Hubble, hubble
from .astro import f_star_Halo
from scipy.interpolate import splrep,splev,interp1d
from scipy.integrate import cumtrapz

def T_cmb(z):
    return Tcmb0 * (1+z)

def kappa_coll():
    """
    [cm^3/s]
    """

    names = 'T, kappa'
    path_to_file = pkg_resources.resource_filename('radtrans', "input_data/kappa_eH.dat")
    eH = np.genfromtxt(path_to_file, usecols=(0, 1), comments='#', dtype=float, names=names)

    names = 'T, kappa'
    path_to_file = pkg_resources.resource_filename('radtrans', 'input_data/kappa_HH.dat')
    HH = np.genfromtxt(path_to_file, usecols=(0, 1), comments='#', dtype=float, names=names)

    return HH, eH


def x_coll(z, Tk, xHI, rho_b):
    """
    Collisional coupling coefficient. 1d profile around a given halo.

    z     : redshift
    Tk    : 1d radial gas kinetic temperature profile [K]
    xHI   : 1d radial ionization fraction profile
    rho_b : baryon density profile in nbr of [H atoms /cm**3] (physical cm)
    Returns : x_coll 1d profile.

    """

    # nH and e- densities
    n_HI  = rho_b * xHI
    n_HII = rho_b * (1-xHI) # [1/cm^3]

    # prefac (Eq.10 in arXiv:1109.6012)
    Tcmb = T_cmb(z)
    prefac = Tstar / A10 / Tcmb  # [s]

    HH, eH = kappa_coll()
    kappa_eH_tck = splrep(eH['T'], eH['kappa'])
    kappa_eH = splev(Tk, kappa_eH_tck, ext=3)  # [cm^3/s]
    kappa_HH_tck = splrep(HH['T'], HH['kappa'])
    kappa_HH = splev(Tk, kappa_HH_tck, ext=3)

    x_HH = prefac * kappa_HH * n_HI
    x_eH = prefac * kappa_eH * n_HII
    return x_HH + x_eH


def S_alpha(zz, Tgas, xHI):
    """
    Suppression factor S_alpha, dimensionless.
    Following method in astro-ph/0608032
    """

    # Eq.43
    tau_GP = 3.0e5 * xHI * ((1 + zz) / 7.0) ** 1.5
   # gamma = 1 / tau_GP

    # Eq. 55
    S_al = np.exp(-0.803 * Tgas ** (-2 / 3) * (1e-6 * tau_GP) ** (1 / 3))

    return S_al


def eps_lyal(nu,param):
    """
    Lymam-alpha part of the spectrum.
    See cosmicdawn/sources.py
    Return : eps (multiply by SFR and you get some [photons.yr-1.Hz-1])
    """
    h0    = param.cosmo.h
    N_al  = param.source.N_al  #9690 number of lya photons per protons (baryons) in stars
    alS = param.source.alS_lyal

    nu_min_norm  = nu_al
    nu_max_norm  = nu_LL

    Anorm = (1-alS)/(nu_max_norm**(1-alS) - nu_min_norm**(1-alS))
    Inu   = lambda nu: Anorm * nu**(-alS)

    eps_alpha = Inu(nu)*N_al/(m_p_in_Msun * h0)

    return eps_alpha



def rho_alpha(r_grid, M_Bin, z_Bin, param):
    """
    Ly-al coupling profile
    of shape (z_Bin, M_Bin, r_grid)
    - r_grid : physical distance around halo center in [pMpc/h]
    - z_Bin  : redshift binns
    - M_Bin  : mass bins
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

    rho_alpha = np.zeros((len(z_Bin), len(M_Bin), len(r_grid)))

    for i in range(len(z_Bin)):
        if z_Bin[i]<zstar:
            flux = []


            for k in range(2, rectrunc):
                zmax = (1 - (rec['n'][k] + 1) ** (-2)) / (1 - (rec['n'][k]) ** (-2)) * (1 + z_Bin[i]) - 1
                zrange = np.minimum(zmax, zstar) - z_Bin[i]

                N_prime = int(zrange / 0.01)  # dz_prime_lyal

                if (N_prime < 4):
                    N_prime = 4

                z_prime = np.logspace(np.log(z_Bin[i]), np.log(zmax), N_prime, base=np.e)
                rcom_prime = comoving_distance(z_prime, param) * h0  # comoving distance in [cMpc/h]

                # What follows is the emissivity of the source at z_prime (such that at z the photon is at rcom_prime)
                # We then interpolate to find the correct emissivity such that the photon is at r_grid*(1+z) (in comoving unit)

                ### cosmidawn stuff, to compare
                alpha = param.source.alpha_MAR
                dMdt_int = alpha * M_Bin[:, None] * np.exp(alpha*(z_Bin[i]-z_prime)) * (z_prime + 1) * Hubble(z_prime, param) * f_star_Halo(param, M_Bin[:, None] ) * param.cosmo.Ob / param.cosmo.Om # SFR Msol/h/yr

                eps_al = eps_lyal(nu_n[k] * (1 + z_prime) / (1 + z_Bin[i]), param)[ None,:] * dMdt_int    # [photons.yr-1.Hz-1]
                eps_int = interp1d(rcom_prime, eps_al, axis=1, fill_value=0.0, bounds_error=False)

                flux_m = eps_int(r_grid * (1 + z_Bin[i])) * rec['f'][k]   # want to find the z' corresponding to comoving distance r_grid * (1 + z).
                flux += [np.array(flux_m)]

            flux = np.array(flux)
            flux_of_r = np.sum(flux, axis=0)  # shape is (Mbin,rgrid)

            rho_alpha[i, :, :] = flux_of_r / (4 * np.pi * r_grid ** 2)[None, :]  ## physical flux in [(pMpc/h)-2.yr-1.Hz-1]

    rho_alpha = rho_alpha * (h0 / cm_per_Mpc) ** 2 /sec_per_year  # [pcm-2.s-1.Hz-1]

    return rho_alpha



def phi_alpha(x,E):
    """
    Fraction of the absorbed photon energy that goes into excitation. [Dimensionless]
    From Dijkstra, Haiman, Loeb. Apj 2004.

    Parameters
    ----------
    x : ionized hydrogen fraction at location
    E : energy in eV

    Returns
    -------
    float
    """
    return 0.39*(1-x**(0.4092*a_alpha(x,E)))**1.7592

def a_alpha(x,E):
    """
    Used in phi_alpha.
    """
    return 2/np.pi * np.arctan(E/120 * (0.03/x**1.5 + 1)**0.25)


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
    y = np.sqrt(x ** 2 + y_1 ** 2)
    F = ((x - 1) ** 2 + y_w ** 2) * y ** (0.5 * P - 5.5) * (1 + np.sqrt(y / y_a)) ** -P
    sigma = sigma_0 * F
    return sigma

def J0_xray(r_grid,xHII, n_HI, Edot,z, param):
    """
    Xray flux that contributes to lyman alpha coupling. [pcm-2.s-1.Hz-1]. Will be added next to rho_alpha to cmopute x_alpha

    Parameters
    ----------
    r_grid : radial distance form source [pMpc/h-1]
    Edot : xray source energy in [eV.s-1] (float)
    xHII : ionized fraction. (array of size r_grid)
    n_HI : number density of hydrogen atoms in the cell [pcm-3] (array of size r_grid)
    z : redshift
    Returns
    -------
    float
    """
    xHII = xHII.clip(min=1e-50) #to avoid warnings

    sed_xray = param.source.alS_xray
    norm_xray = (1 - sed_xray) / ((param.source.E_max_sed_xray / h_eV_sec) ** (1 - sed_xray) - (param.source.E_min_sed_xray / h_eV_sec) ** (1 - sed_xray))   #Hz**(alpha-1)
    E_range = np.logspace(np.log10(50), np.log10(2000), 200, base=10)  # eq(3) Thomas.2011
    nu_range = Hz_per_eV * E_range

    cumul_nHI = cumtrapz(n_HI, r_grid, initial=0.0)  ## Mpc/h.cm-3
    Edotflux = Edot / (4 * np.pi * r_grid * cm_per_Mpc ** 2 / param.cosmo.h ** 2)  # eV.s-1.pcm-2

    tau = cm_per_Mpc / param.cosmo.h * (cumul_nHI[:, None] * sigma_HI(E_range))  # shape is (r_grid,E_range)

    Nxray_arr = np.exp(-tau) * Edotflux[:, None] * norm_xray * nu_range[None, :] ** (-sed_xray) * Hz_per_eV  # [eV/eV/s/pcm^2], (r_grid,E_range)  array to integrate

    to_int = Nxray_arr * sigma_HI(E_range)[None, :] * phi_alpha(xHII[:, None], E_range)

    integral = np.trapz(to_int, E_range, axis=1)  # shape is r_grid
    return c__ * 1e2 * sec_per_year / (4 * np.pi * Hubble(z, param) * nu_al) * n_HI * integral / (h_eV_sec * nu_al) # [s-1.pcm-2.Hz-1]

