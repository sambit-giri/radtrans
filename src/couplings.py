"""
Here we compute the Lyman_alpha and collisional coupling coefficient, in order to produce full dTb maps
"""

import numpy as np
from .constants import *
import pkg_resources
from .cosmo import comoving_distance, Hubble
from .astro import f_star_Halo
from scipy.interpolate import splrep,splev,interp1d

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
    gamma = 1 / tau_GP

    # Eq. 55
    S_al = np.exp(-0.803 * Tgas ** (-2 / 3) * (1e-6 / gamma) ** (1 / 3))

    return S_al


def eps_lyal(nu,param):
    """
    Lymam-alpha part of the spectrum.
    See cosmicdawn/sources.py
    """
    h0    = param.cosmo.h
    N_al  = param.source.N_al #9690
    alS = param.source.alS_lyal

    nu_min_norm  = nu_al
    nu_max_norm  = nu_LL

    Anorm = (1-alS)/(nu_max_norm**(1-alS) - nu_min_norm**(1-alS))
    Inu   = lambda nu: Anorm * nu**(-alS)        # [1/Hz]

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
                dMdt_int = alpha * M_Bin[:, None] * np.exp(alpha*(z_Bin[i]-z_prime)) * (z_prime + 1) * Hubble(z_prime, param) * f_star_Halo(param, M_Bin[:, None] ) * param.cosmo.Ob / param.cosmo.Om

                eps_al = eps_lyal(nu_n[k] * (1 + z_prime) / (1 + z_Bin[i]), param)[ None,:] * dMdt_int    # (z_prime)
                eps_int = interp1d(rcom_prime, eps_al, axis=1, fill_value=0.0, bounds_error=False)

                flux_m = eps_int(r_grid * (1 + z_Bin[i])) * rec['f'][k]   # want to find the z' corresponding to comoving distance r_grid * (1 + z).
                flux += [np.array(flux_m)]

            flux = np.array(flux)
            flux_of_r = np.sum(flux, axis=0)  # shape is (Mbin,rgrid)

            rho_alpha[i, :, :] = flux_of_r / (4 * np.pi * r_grid ** 2)[None, :]  ## physical flux in [(pMpc/h)-2.s-1.Hz-1]

    rho_alpha = rho_alpha * (h0 / cm_per_Mpc) ** 2 /sec_per_year # [pcm-2.s-1.Hz-1]

    return rho_alpha



