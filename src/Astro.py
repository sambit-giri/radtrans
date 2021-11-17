

import numpy as np
from radtrans.constants import *


def NGamDot(param):
    """
    Number of ionising photons emitted per sec for a given source. [s**-1]
    """
    E_0_ = param.source.E_0
    E_upp_ = param.source.E_upp
    if (param.source.type == 'Miniqsos'):  ### Choose the source type
        alpha = param.source.alpha
        M = param.source.M_miniqso  # Msol
        L = 1.38 * 10 ** 37 * eV_per_erg * M  # eV.s-1 , assuming 10% Eddington lumi.
        E_range_E0 = np.logspace(np.log10(E_0_), np.log10(E_upp_), 100, base=10)
        Ag = L / (np.trapz(E_range_E0 ** -alpha, E_range_E0))
        # print('Miniqsos model chosen. M_qso is ', M)
        E_range_HI_ = np.logspace(np.log10(13.6), np.log10(E_upp_), 1000, base=10)
        Ngam_dot = np.trapz(Ag * E_range_HI_ ** -alpha / E_range_HI_, E_range_HI_)
        return Ngam_dot

    elif (param.source.type == 'Galaxies'):
        f_c2ray = param.source.fc2ray
        M = param.source.M_halo
        Delta_T = 10 ** 7 * sec_per_year
        Ngam_dot = f_c2ray * M * M_sun / m_H * param.cosmo.Ob / param.cosmo.Om / Delta_T  #### M_sun is in gramms
        # print('Galaxy model chosen. M_halo is ', M)
        return Ngam_dot

    else:
        print('Source Type not available. Should be Galaxies or Miniqsos')








