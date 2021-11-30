"""
Contains functions to compute measurable quantities from an N-body snapshot. either the number of ionizing photons Ngamma_dot
"""

import numpy as np
from .constants import *


def Hubble(z,param):
    """""
    Hubble factor [yr-1] 
    """""
    return param.cosmo.h * 100.0 * sec_per_year / km_per_Mpc * np.sqrt(param.cosmo.Om**(1+z)**3+(1-param.cosmo.Om*-param.cosmo.Ol)*(1+z)**2+param.cosmo.Ol)



def NGamDot(param):
    """
    Number of ionising photons emitted per sec for a given source model and source parameter. [s**-1]
    """
    E_0_ = param.source.E_0
    E_upp_ = param.source.E_upp
    if (param.source.type == 'Miniqsos'):  ### Choose the source type
        alpha = param.source.alpha
        M = param.source.M_miniqso  # Msol
        L = 1.38 * 10 ** 37 * eV_per_erg * M  # eV.s-1 , assuming 10% Eddington lumi.
        E_range_E0 = np.logspace(np.log10(E_0_), np.log10(E_upp_), 100, base=10)
        Ag = L / (np.trapz(E_range_E0 ** -alpha, E_range_E0))
        E_range_HI_ = np.logspace(np.log10(13.6), np.log10(E_upp_), 1000, base=10)
        Ngam_dot = np.trapz(Ag * E_range_HI_ ** -alpha / E_range_HI_, E_range_HI_)
        return Ngam_dot

    elif (param.source.type == 'Galaxies'):
        f_c2ray = param.source.fc2ray
        M = param.source.M_halo
        Delta_T = 10 ** 7 * sec_per_year
        Ngam_dot = f_c2ray * M * M_sun / m_H * param.cosmo.Ob / param.cosmo.Om / Delta_T  #### M_sun is in gramms
        return Ngam_dot

    elif (param.source.type == 'Galaxies_MAR'):
        z = param.solver.z
        M_halo = param.source.M_halo
        dMh_dt = param.source.alpha_MAR * M_halo * (z + 1) * Hubble(z, param)  ## [(Msol/h) / yr]
        Ngam_dot = dMh_dt * f_star_Halo(param, M_halo) * param.cosmo.Ob / param.cosmo.Om * f_esc(param,M_halo) * param.source.Nion / sec_per_year / m_H * M_sun  # [s**-1]
        return Ngam_dot

    else:
        print('Source Type not available. Should be Galaxies or Miniqsos')
        exit()



def Read_Rockstar(file):
    """
    Read in a rockstar halo catalog and return a dictionnary with all the information stored.
    """

    Halo_File = []
    with open(file) as f:
        for line in f:
            Halo_File.append(line)
    a = float(Halo_File[1][4:])
    z = 1 / a - 1
    LBox = float(Halo_File[6][10:-7])
    Halo_File = Halo_File[16:]  ### Rockstar
    H_Masses, H_Radii = [], []
    H_X, H_Y, H_Z = [], [], []
    for i in range(len(Halo_File)):
        line = Halo_File[i].split(' ')
        H_Masses.append(float(line[2]))
        H_X.append(float(line[8]))
        H_Y.append(float(line[9]))
        H_Z.append(float(line[10]))
        H_Radii.append(float(line[6]))
    H_Masses, H_X, H_Y, H_Z, H_Radii = np.array(H_Masses), np.array(H_X), np.array(H_Y), np.array(H_Z), np.array(H_Radii)
    Dict = {'M':H_Masses,'X':H_X,'Y':H_Y,'Z':H_Z, 'R':H_Radii,'z':z,'Lbox':LBox}

    return Dict

def S_fct(Mh, Mt, g3, g4):
    return (1 + (Mt / Mh) ** g3) ** g4

def f_star_Halo(param,Mh):
    """
    Double power law. fstar * Mh_dot * Ob/Om = M*_dot. fstar is therefore the conversion from baryon accretion rate  to star formation rate.
    """
    f_st = param.source.f_st
    Mp = param.source.Mp
    g1 = param.source.g1
    g2 = param.source.g2
    Mt = param.source.Mt
    g3 = param.source.g3
    g4 = param.source.g4
    return 2 * f_st / ((Mh / Mp) ** g1 + (Mh / Mp) ** g2) * S_fct(Mh, Mt, g3, g4)


def f_esc(param,Mh):
    f0  = param.source.f0_esc
    Mp  = param.source.Mp_esc
    pl  = param.source.pl_esc
    fesc = f0 * (Mp / Mh) ** pl
    return np.minimum(fesc,1)



def Ng_dot_Snapshot(param,rock_catalog):
    """
    WORKS FOR EXP MAR
    Mean number of ionising photons emitted per sec for a given rockstar snapshot. [s**-1]
    rock_catalog : rockstar halo cataloa
    """
    Halos = Read_Rockstar(rock_catalog)
    H_Masses, z = Halos['M'], Halos['z']
    dMh_dt = param.source.alpha_MAR * H_Masses * (z+1) * Hubble(z, param) ## [(Msol/h) / yr]
    dNg_dt = dMh_dt * f_star_Halo(param, H_Masses) * param.cosmo.Ob/param.cosmo.Om * f_esc(param, H_Masses) * param.source.Nion /sec_per_year /m_H * M_sun  #[s**-1]
    return z, np.sum(dNg_d t) / Halos['Lbox'] ** 3




