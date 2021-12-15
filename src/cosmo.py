"""

FUNCTIONS RELATED TO COSMOLOGY

"""
import numpy as np
from scipy.integrate import cumtrapz, trapz, quad
from scipy.interpolate import splrep,splev
from .constants import rhoc0,c_km_s, Tcmb0, sec_per_year, km_per_Mpc


def hubble(z,param):
    """
    Hubble parameter [km.s-1.Mpc-1]
    """
    Om = param.cosmo.Om
    Ol = 1.0-Om
    H0 = 100.0*param.cosmo.h
    return H0 * (Om*(1+z)**3 + (1.0 - Om - Ol)*(1+z)**2 + Ol)**0.5


def Hubble(z,param):
    """""
    Hubble factor [yr-1] 
    """""
    return param.cosmo.h * 100.0 * sec_per_year / km_per_Mpc * np.sqrt(param.cosmo.Om**(1+z)**3+(1-param.cosmo.Om*-param.cosmo.Ol)*(1+z)**2+param.cosmo.Ol)


def comoving_distance(z,param):
    """
    Comoving distance between z[0] and z[-1]
    """
    return cumtrapz(c_km_s/hubble(z,param),z,initial=0)  # [Mpc]


def T_cmb(z,param):
    """
    CMB temperature
    """
    return Tcmb0*(1+z)

def T_smooth_radio(z,param):
    """
    Smooth Background radiation temperature when a radio excess is present, i.e Ar is non zero
    """
    Tcmb0 = param.cosmo.Tcmb
    Ar = param.radio.Ar
    Ar = np.array(Ar) # this line is when you want a z-dependent Ar. (used it to reproduce fig 2 of 2008.04315)
    Beta_Rad = param.radio.Beta_Rad
    nu = 1420/(1+z) #### in MHz
    return Tcmb0*(1+z)*(Ar*(nu/78)**Beta_Rad)


def read_powerspectrum(param):
    """
    Linear power spectrum from file
    """
    names='k, P'
    PS = np.genfromtxt(param.file.ps,usecols=(0,1),comments='#',dtype=None, names=names)
    return PS

