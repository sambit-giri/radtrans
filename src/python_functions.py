"""
Bunch of python functions useful to plot, load, read in...
"""
import numpy as np
import scipy
import sys
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from radtrans.constants import *


def load_f(file):
    import pickle
    prof = pickle.load(open(file,'rb'))
    return prof


def save_f(file,obj):
    import pickle
    pickle.dump(file=open(file,'wb'),obj=obj)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def Tgas_from_rho_heat(HM_PS):
    rmin = 0.1
    rmax = 6000
    Nr = 50
    rr = np.logspace(np.log10(rmin), np.log10(rmax), Nr)
    rho_heat_HM = HM_PS['rho_heat_prof']  #
    dn_dlnm = HM_PS['dndlnm']
    masses = HM_PS['M0']

    rho_int = np.trapz(rho_heat_HM * rr[:, None] ** 2 * 4 * np.pi, rr, axis=1)
    T_mean = np.trapz(dn_dlnm * rho_int, np.log(masses), axis=1)
    return T_mean


def Gheat_from_rho_x(rr,HM_PS):
    rho_xray_HM = HM_PS['rho_X_prof']  # [erg/s]
    dn_dlnm = HM_PS['dndlnm']
    masses = HM_PS['M0']

    rho_int = np.trapz(rho_xray_HM * rr[:, None] ** 2 * 4 * np.pi, rr, axis=1)
    Gh_mean = np.trapz(dn_dlnm * rho_int, np.log(masses), axis=1)
    return Gh_mean




















########### FUNCTIONS TO DO COMPARISON PLOTS WITH ------>HALO MODEL<--------

def Plotting_GS(physics, x__approx, GS, PS, save_loc=None):
    """""""""""
    physics : GS history dictionnary data from radtrans
    x__approx : approximation dictionnary (when we compute the global quantities from halo catalogs + profiles, without using a grid.)
    GS, PS : from halo model
    """""""""""
    ################################################ loading
    from matplotlib import pyplot as plt
    RT_dTb_GS = physics['dTb_GS']
    RT_dTb_GS_Tkneutr = physics['dTb_GS_Tkneutral']
    RT_dTb = physics['dTb']  # physics['dTb']
    RT_zz = physics['z']
    x_tot = physics['xal_coda_style'] + physics['x_coll']
    Tcmb = (1 + RT_zz) * Tcmb0

    HM_zz, HM_dTb = GS['z'], GS['dTb']

    fig = plt.figure(figsize=(10, 10))

    ################################################ dTb
    axis1 = fig.add_subplot(421)
    axis1.plot(HM_zz, HM_dTb, label='HM')

    axis1.plot(RT_zz, RT_dTb_GS, label='RT dTb=f(mean)', ls='--')
    axis1.plot(RT_zz, RT_dTb, label='RT dTb=mean(dTb))', ls='--')
    # axis1.plot(RT_zz , RT_dTb_GS_Tkneutr   ,label='RT dTb=f(mean) with Tkneutral)',ls='--')

    axis1.set_xlim(5, 21)
    axis1.set_ylabel('dTb')
    axis1.legend(loc='best')

    ################################################ Temperatures
    axis2 = fig.add_subplot(422)
    axis2.semilogy(HM_zz, GS['Tgas'], 'k', label='Tgas')
    axis2.semilogy(HM_zz, GS['Tcmb'], 'orange', label='Tcmb')
    axis2.semilogy(HM_zz, GS['Tspin'], 'r', label='Tspin')
    axis2.semilogy(RT_zz, physics['Tadiab'], 'cyan', ls='--', label='Tadiab')
    axis2.semilogy(RT_zz, physics['T_spin'], 'r', ls='--')
    axis2.semilogy(RT_zz, physics['Tk'], 'k', ls=':')
    # axis2.semilogy(RT_zz,physics['Tk_neutral_regions'],'k',ls='-.')

    if PS is not None:
        Tgas_rhoheat = Tgas_from_rho_heat(PS)
        axis2.semilogy(HM_zz, Tgas_rhoheat, 'g', label='Tgas from rhoheat')

    axis2.set_xlim(5, 22)
    axis2.set_ylabel('T')
    axis2.set_xlabel('z')
    axis2.legend(loc='best')

    axis2.set_ylim(1, 1e3)
    if save_loc is not None:
        plt.savefig(save_loc)

    ################################################ Gamma_heat and sfrd
    axis3 = fig.add_subplot(423)
    # axis3.semilogy(HM_zz,GS['Gamma_heat']*eV_per_erg,'r',label='Gamma_heat [eV]')
    # axis3.semilogy(x__approx['z'],x__approx['Gamma_heat'],'r',ls='--')

    axis3.semilogy(HM_zz, GS['sfrd_lyal'], 'b', ls='-', label='sfrd [Msol h^2/yr/Mpc]')
    axis3.semilogy(x__approx['z'], x__approx['sfrd'], 'b', ls='--')
    axis3.set_xlim(5, 24)
    axis3.set_ylim(1e-6, 1e2)
    axis3.set_ylabel('')
    axis3.set_xlabel('z')

    axis3.legend(loc='best')

    ################################################ xal

    axis4 = fig.add_subplot(424)
    axis4.semilogy(GS['z'], GS['x_al'], 'r', label='x_al')
    axis4.semilogy(x__approx['z'], physics['xal_coda_style'], 'r', ls='--')

    # axis4.semilogy(HM_zz,GS['sfrd_lyal'],'b',ls='-',label='sfrd [Msol h^2/yr/Mpc]')
    # axis4.semilogy(x__approx['z'],x__approx['sfrd'],'b',ls='--')
    axis4.set_xlim(5, 20)
    axis4.set_ylim(1e-3, 1e4)
    axis4.set_ylabel('')
    axis4.set_xlabel('z')

    axis4.legend(loc='best')

    ################################################ xHI

    axis5 = fig.add_subplot(425)
    axis5.plot(GS['z'], GS['x_HI'], 'r', label='xHI')
    axis5.plot(x__approx['z'], 1 - x__approx['xHII'], 'r', ls='--', label='approx')
    axis5.plot(physics['z'], 1 - physics['x_HII'], 'orange', ls=':', lw=5)
    axis5.set_xlim(5, 20)
    # axis5.set_ylim(1e-3,1e4)
    axis5.set_ylabel('')
    axis5.set_xlabel('z')

    if save_loc is not None:
        plt.savefig(save_loc)

    ################################################ XCOL

    # axis6 = fig.add_subplot(426)
    # axis6.semilogy(GS['z'],GS['x_cl'],'r',label='x_col')
    # axis6.semilogy(x__approx['z'],physics['x_coll'],'r',ls='--')
    # axis6.semilogy(GS['z'],GS['x_cl']+GS['x_al'],'r',label='x_col')
    # axis6.semilogy(x__approx['z'],physics['x_coll']+physics['xal_coda_style'],'r',ls='--')
    # axis6.set_xlim(5,20)
    ##axi65.set_ylim(1e-3,1e4)
    # axis6.set_ylabel('')
    # axis6.set_xlabel('z')

    # axis7 = fig.add_subplot(427)
    # axis7.semilogy(GS['z'],GS['Gamma_heat']*eV_per_erg,'k',label='Gamma_heat')
    # axis7.semilogy(x__approx['z'],x__approx['Gamma_heat'],'k',ls='--')
    # axis7.set_xlim(5,20)
    # axis7.set_ylim(1e-25,1e-15)
    # axis7.set_ylabel('')
    # axis7.set_xlabel('z')
    #
    # if save_loc is not None :
    #    plt.savefig(save_loc)


def Plotting_PS(k_array, z, physics, PS_Dict, GS, PS, save_loc=None):
    """""""""""
    k_array,z : values for the power spectra plot
    physics,PS_Dict : GS history and PS dictionnary data from radtrans
    GS, PS : from halo model
    """""""""""
    ################################################ loading
    from matplotlib import pyplot as plt
    RT_zz = physics['z']  #
    Tcmb = (1 + RT_zz) * Tcmb0
    RT_dTb_GS = physics['dTb_GS']  # *(1-Tcmb/physics['Tk'])/(1-Tcmb/physics['Tk_neutral'])
    x_tot = physics['xal_coda_style'] + physics['x_coll']
    RT_dTb = physics['dTb']  # * x_tot/(1 + x_tot) * (physics['x_al']+physics['x_coll']+1) / (physics['x_al']+physics['x_coll'])

    beta_a = physics['xal_coda_style'] / x_tot / (1 + x_tot)  # physics['beta_a'] #
    beta_a_HM = GS['x_al'] / (GS['x_al'] + GS['x_cl']) / (1 + GS['x_al'] + GS['x_cl'])

    Tbg = Tcmb0 * (1 + 135) * (1 + RT_zz) ** 2 / (1 + 135) ** 2
    fr = 1  # (physics['Tk']-Tbg)/physics['Tk'] ### Halomodel primordial/heating decomp
    beta_T = Tcmb / (physics['Tk'] - Tcmb)  # physics['beta_T'] * fr  #

    beta_T_HM = (1 + GS['z']) * Tcmb0 / (GS['Tgas'] - (1 + GS['z']) * Tcmb0)
    beta_r = physics['beta_r']  # -physics['x_HII']/(1-physics['x_HII'])
    beta_r_HM = -(1 - GS['x_HI']) / (GS['x_HI'])
    RT_iz = np.argmin(np.abs(RT_zz - z))

    HM_zz, HM_dTb = GS['z'], GS['dTb']
    HM_iz = np.argmin(np.abs(PS['z'] - RT_zz[RT_iz]))

    fig = plt.figure(figsize=(10, 10))

    ################################################ dTb
    axis1 = fig.add_subplot(421)
    axis1.plot(HM_zz, HM_dTb, label='HM')
    axis1.plot(RT_zz, RT_dTb_GS, label='rt dTb_GS f(mean)', ls='--')
    axis1.plot(RT_zz, RT_dTb, label='rt mean(dTb) ', ls='--')
    # axis1.plot(RT_zz , RT_dTb,label='radtrans',ls='-.')
    axis1.set_xlim(6, 21)
    axis1.set_ylabel('dTb')
    axis1.legend()

    ################################################ xHII
    axis2 = fig.add_subplot(422)
    axis2.plot(HM_zz, 1 - GS['x_HI'])
    axis2.plot(RT_zz, physics['x_HII'], ls='--')
    axis2.set_xlim(6, 20)
    axis2.set_ylabel('xHII')
    axis2.set_xlabel('z')

    ################################################ PS(k)
    axis3 = fig.add_subplot(423)
    print('z is', z, '. Radtrans z is :', RT_zz[RT_iz])

    PS_RT = PS_Dict
    kk = PS_RT['k']
    PS_xHI = PS_RT['PS_xHII'] * beta_r[:, None] ** 2  ### sizes are size(kk)
    PS_T = PS_RT['PS_T'] * beta_T[:, None] ** 2
    PS_dTb = PS_RT['PS_dTb']
    PS_rho = PS_RT['PS_rho']
    PS_xal = PS_RT['PS_xal'] * beta_a[:, None] ** 2

    PS_bb = PS_rho  # gas (matter)
    PS_aa = PS_xal  #
    PS_TT = PS_T    #
    PS_rr = PS_xHI  # reio

    PS_ab = beta_a[:, None] * PS_RT['PS_rho_xal']
    PS_Tb = beta_T[:, None] * PS_RT['PS_rho_T']
    PS_aT = beta_a[:, None] * beta_T[:, None] * PS_RT['PS_T_lyal']
    PS_rb = beta_r[:, None] * PS_RT['PS_rho_xHII']
    PS_ra = beta_r[:, None] * beta_a[:, None] * PS_RT['PS_lyal_xHII']
    PS_rT = beta_r[:, None] * beta_T[:, None] * PS_RT['PS_T_xHII']

    PS_dTb_HMstyle = PS_bb + PS_aa + PS_TT + PS_rr + 2 * (PS_ab + PS_Tb + PS_aT + PS_rb + PS_ra + PS_rT)

    dTb_RT_z = RT_dTb_GS[RT_iz]
    dTb_HM_z = GS['dTb'][HM_iz]
    #### dTb
    axis3.loglog(PS['k'], PS['k'] ** 3 * dTb_HM_z ** 0 * PS['P_mu0'][HM_iz] / 2 / np.pi ** 2, color='r', label='PS_dTb')
    axis3.loglog(kk, kk ** 3 * abs(dTb_RT_z) ** 0 * PS_dTb_HMstyle[RT_iz] / 2 / np.pi ** 2,
                 color='r')  # ,alpha=0.5,ls='--',lw=4,label='PS perturb')
    axis3.loglog(kk, kk ** 3 * abs(dTb_RT_z) ** 0 * PS_dTb[RT_iz] / 2 / np.pi ** 2, color='r', alpha=0.5, ls=':',
                 label='PS(dTb)', lw=4)
    print('mean dTb is ', dTb_RT_z)
    # test_HM = PS['k']**3 * dTb_HM_z**2 * PS['P_mu0'][HM_iz]/2/np.pi**2
    # test_RT = abs(dTb_RT_z)**2 * PS_dTb/2/np.pi**2

    #### mm
    axis3.loglog(PS['k'], PS['k'] ** 3 * PS['P_mm'][HM_iz] / 2 / np.pi ** 2, color='k', label='PS_mm')
    axis3.loglog(kk, kk ** 3 * PS_rho[RT_iz] / 2 / np.pi ** 2, color='k', alpha=0.5, ls='--', lw=4)
    #### TT
    axis3.loglog(PS['k'], PS['k'] ** 3 * dTb_HM_z ** 0 * PS['P_TT'][HM_iz] / 2 / np.pi ** 2, color='g', label='PS_TT')
    axis3.loglog(kk, kk ** 3 * abs(dTb_RT_z) ** 0 * PS_T[RT_iz] / 2 / np.pi ** 2, color='g', alpha=0.5, ls='--', lw=4)
    #### aa
    axis3.loglog(PS['k'], PS['k'] ** 3 * dTb_HM_z ** 0 * PS['P_aa'][HM_iz] / 2 / np.pi ** 2, color='b', label='PS_aa')
    axis3.loglog(kk, kk ** 3 * abs(dTb_RT_z) ** 0 * PS_xal[RT_iz] / 2 / np.pi ** 2, color='b', alpha=0.5, ls='--', lw=4)
    ##### bubbles
    axis3.loglog(PS['k'], PS['k'] ** 3 * dTb_HM_z ** 0 * PS['P_rr'][HM_iz] / 2 / np.pi ** 2, color='darkorange',
                 label='PS_bbub')
    axis3.loglog(kk, kk ** 3 * abs(dTb_RT_z) ** 0 * PS_xHI[RT_iz] / 2 / np.pi ** 2, color='darkorange', ls='--')
    print('beta_T is ', beta_T[RT_iz])

    axis3.set_xlim(3e-2, 1e1)
    axis3.set_ylim(3e-4, 2e0)
    axis3.legend(title='z={}'.format(z), loc='best')
    axis3.set_xlabel('k [h/Mpc]', fontsize=14)
    axis3.set_ylabel('$\Delta^{2} = k^{3}P(k)/(2\pi^{2})$   ', fontsize=14)

    ################################################ PS(z)
    axis4 = fig.add_subplot(424)
    ax = plt.gca()
    for k0 in k_array:
        color = next(ax._get_lines.prop_cycler)['color']
        ind_k_RT = np.argmin(np.abs(PS_RT['k'] - k0))
        axis4.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * RT_dTb_GS ** 2 * PS_dTb_HMstyle[:, ind_k_RT] / 2 / np.pi ** 2,
                       color=color, alpha=0.5, ls='--', lw=4)
        axis4.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * RT_dTb ** 2 * PS_dTb[:, ind_k_RT] / 2 / np.pi ** 2,
                       color=color, ls='-.')

        ind_k = np.argmin(np.abs(PS['k'] - PS_RT['k'][ind_k_RT]))
        axis4.semilogy(PS['z'], PS['k'][ind_k] ** 3 * HM_dTb ** 2 * PS['P_mu0'][:, ind_k] / 2 / np.pi ** 2, color=color,
                       label='k={}'.format(k0))
        # axis4.semilogy(PS['z'],PS['k'][ind_k]**3 * HM_dTb**2 * PS['P_angav'][:,ind_k]/2/np.pi**2,ls=':',lw=4,alpha=0.7,color=color,label='k={}'.format(k0))
        print(PS['P_angav'][:, ind_k] / PS['P_mu0'][:, ind_k])
        print('k RT is ', round(PS_RT['k'][ind_k_RT], 4), 'k HM is,', PS['k'][ind_k])

    #  axis4.semilogy(PS['z'],PS['k'][ind_k]**3 * HM_dTb**2* np.abs(PS['P_rT'][:,ind_k])/2/np.pi**2,color=color,label='k={}'.format(k0))
    #  axis4.semilogy(RT_zz, k0**3 * RT_dTb_GS**2 * np.abs(PS_rT[:,ind_k_RT])/2/np.pi**2,color=color,alpha=0.5,ls='--',lw=4)
    axis4.semilogy([], [], color='gray', label='PS Perturbtiv', alpha=0.5, ls='--', lw=4)
    axis4.semilogy([], [], color='gray', label='PS(dTb)', ls='-.')

    axis4.set_ylim(1e-1, 1e3)
    axis4.set_xlim(6, 20)

    axis4.legend(title='$ k^{3}P(k)/(2\pi^{2})$')  # dTb^{2}
    axis4.set_xlabel('z', fontsize=14)

    ################################################ PS_lymanalpha(z)
    axis5 = fig.add_subplot(425)
    ax = plt.gca()
    for k0 in [k_array[0]]:
        color = next(ax._get_lines.prop_cycler)['color']
        ind_k_RT = np.argmin(np.abs(PS_RT['k'] - k0))
        # axis5.semilogy(RT_zz, PS_RT['k'][ind_k_RT]**3 * RT_dTb_GS**0 * PS_xal[:,ind_k_RT]/2/np.pi**2,color=color,alpha=0.5,ls='--',lw=4)

        # axis5.semilogy(RT_zz, PS_RT['k'][ind_k_RT]**3 * np.abs(PS_ab[:,ind_k_RT])/2/np.pi**2,color='C0',alpha=0.5,ls='--',lw=4)
        axis5.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * np.abs(PS_Tb[:, ind_k_RT]) / 2 / np.pi ** 2, color='C1',
                       alpha=0.5, ls='--', lw=4)
        axis5.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * np.abs(PS_aT[:, ind_k_RT]) / 2 / np.pi ** 2, color='C2',
                       alpha=0.5, ls='--', lw=4)
        # axis5.semilogy(RT_zz, PS_RT['k'][ind_k_RT]**3 * np.abs(PS_rb[:,ind_k_RT])/2/np.pi**2,color='C3',alpha=0.5,ls='--',lw=4)
        # axis5.semilogy(RT_zz, PS_RT['k'][ind_k_RT]**3 * np.abs(PS_ra[:,ind_k_RT])/2/np.pi**2,color='C4',alpha=0.5,ls='--',lw=4)
        axis5.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * np.abs(PS_rT[:, ind_k_RT]) / 2 / np.pi ** 2, color='C5',
                       alpha=0.5, ls='--', lw=4)

        ind_k = np.argmin(np.abs(PS['k'] - PS_RT['k'][ind_k_RT]))

        # axis5.semilogy(PS['z'],PS['k'][ind_k]**3 * HM_dTb**0* PS['P_aa'][:,ind_k]/2/np.pi**2,color=color,label='k={}'.format(k0))
        # axis5.semilogy(PS['z'],PS['k'][ind_k]**3 * np.abs(PS['P_am'][:,ind_k])/2/np.pi**2,color='C0',label='k=am'.format(k0))
        axis5.semilogy(PS['z'], PS['k'][ind_k] ** 3 * np.abs(PS['P_Tm'][:, ind_k]) / 2 / np.pi ** 2, color='C1',
                       label='k=Tm'.format(k0))
        axis5.semilogy(PS['z'], PS['k'][ind_k] ** 3 * np.abs(PS['P_aT'][:, ind_k]) / 2 / np.pi ** 2, color='C2',
                       label='k=aT'.format(k0))
        # axis5.semilogy(PS['z'],PS['k'][ind_k]**3 * np.abs(PS['P_rb'][:,ind_k])/2/np.pi**2,color='C3',label='k=rb'.format(k0))
        # axis5.semilogy(PS['z'],PS['k'][ind_k]**3 * np.abs(PS['P_ra'][:,ind_k])/2/np.pi**2,color='C4',label='k=ra'.format(k0))
        axis5.semilogy(PS['z'], PS['k'][ind_k] ** 3 * np.abs(PS['P_rT'][:, ind_k]) / 2 / np.pi ** 2, color='C5',
                       label='k=rT'.format(k0))

        print('k RT is ', round(PS_RT['k'][ind_k_RT], 2), 'k HM is,', PS['k'][ind_k])

    axis5.set_ylim(1e-5, 1e1)
    axis5.set_xlim(6, 18)

    axis5.set_ylabel('$\Delta^{2} = k^{3}P(k)/(2\pi^{2})$   ', fontsize=14)

    axis5.legend(title='cross Temp PS ')
    axis5.set_xlabel('z', fontsize=14)

    ################################################ Beta factors
    axis6 = fig.add_subplot(426)
    ax = plt.gca()

    axis6.semilogy(RT_zz, np.abs(beta_a), 'r', ls='--')
    axis6.semilogy(RT_zz, np.abs(beta_T), 'b', ls='--')
    axis6.semilogy(RT_zz, np.abs(beta_r), 'g', ls='--')

    axis6.semilogy(HM_zz, np.abs(beta_a_HM), 'r', label='|beta_a|')
    axis6.semilogy(HM_zz, np.abs(beta_T_HM), 'b', label='|beta_T|')
    axis6.semilogy(HM_zz, np.abs(beta_r_HM), 'g', label='|beta_r|')
    axis6.legend()
    axis6.set_ylim(1e-5, 1e2)
    axis6.set_xlim(6, 18)

    ################################################ PS_TT(z)
    axis7 = fig.add_subplot(427)
    ax = plt.gca()

    for k0 in k_array:
        color = next(ax._get_lines.prop_cycler)['color']
        ind_k_RT = np.argmin(np.abs(PS_RT['k'] - k0))
        axis7.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * RT_dTb ** 0 * PS_TT[:, ind_k_RT] / 2 / np.pi ** 2,
                       color=color, alpha=0.5, ls='--', lw=4)

        ind_k = np.argmin(np.abs(PS['k'] - PS_RT['k'][ind_k_RT]))
        axis7.semilogy(PS['z'], PS['k'][ind_k] ** 3 * HM_dTb ** 0 * PS['P_TT'][:, ind_k] / 2 / np.pi ** 2, color=color,
                       label='k={}'.format(k0))
        # PS_CC = ((1+PS['Plin']**0.5)**(2/3) - 1)**2
        # axis7.semilogy(PS['z'],PS['k'][ind_k]**3 * HM_dTb**0* PS_CC[:,ind_k]/2/np.pi**2,color=color,label='k={}'.format(k0))
        print('k RT is ', round(PS_RT['k'][ind_k_RT], 2), 'k HM is,', PS['k'][ind_k])

    axis7.set_ylim(1e-4, 1e3)
    axis7.set_xlim(6, 22)

    axis7.set_ylabel('$\Delta^{2} = dTb^{2} k^{3}P(k)/(2\pi^{2})$   ', fontsize=14)

    axis7.legend(title='TT PS ')
    axis7.set_xlabel('z', fontsize=14)

    ################################################ PS_reio_reio(z)
    axis8 = fig.add_subplot(428)
    ax = plt.gca()

    for k0 in k_array:
        color = next(ax._get_lines.prop_cycler)['color']

        ind_k_RT = np.argmin(np.abs(PS_RT['k'] - k0))
        axis8.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3 * RT_dTb ** 0 * PS_rr[:, ind_k_RT] / 2 / np.pi ** 2,
                       color=color, alpha=0.5, ls='--', lw=4)

        ind_k = np.argmin(np.abs(PS['k'] - PS_RT['k'][ind_k_RT]))
        axis8.semilogy(PS['z'], PS['k'][ind_k] ** 3 * HM_dTb ** 0 * PS['P_rr'][:, ind_k] / 2 / np.pi ** 2, color=color,
                       label='k={}'.format(np.round(PS_RT['k'][ind_k_RT], 2)))

        axis8.set_ylim(1e-7, 1e0)
        print('k RT is ', round(PS_RT['k'][ind_k_RT], 2), 'k HM is,', PS['k'][ind_k])

    axis8.set_xlim(6, 18)

    axis8.set_ylabel('$\Delta^{2} = dTb^{2} k^{3}P(k)/(2\pi^{2})$   ', fontsize=14)

    axis8.legend(title='reio-reio PS ')
    axis8.set_xlabel('z', fontsize=14)

    if save_loc is not None:
        plt.savefig(save_loc)


def Plotting_PS_TT(k_array, physics, PS_RT, GS,PS, save_loc=None):
    """""""""""
    k_array,z : values for the power spectra plot
    physics,PS_Dict : GS history and PS dictionnary data from radtrans
    GS, PS : from halo model
    """""""""""
    ################################################ loading
    from matplotlib import pyplot as plt
    RT_zz = physics['z']  #
    fig = plt.figure(figsize=(10, 10))
    Beta_T_HM = GS['Tcmb'] / (GS['Tgas'] - GS['Tcmb'])
    ################################################ P_TT (z)
    axis4 = fig.add_subplot(111)
    ax = plt.gca()
    for k0 in k_array:
        color = next(ax._get_lines.prop_cycler)['color']
        ind_k_RT = np.argmin(np.abs(PS_RT['k'] - k0))
        axis4.semilogy(RT_zz, PS_RT['k'][ind_k_RT] ** 3  * PS_RT['PS_T'][:, ind_k_RT] / 2 / np.pi ** 2,  color=color, alpha=0.5, ls='--', lw=4)

        ind_k = np.argmin(np.abs(PS['k'] - PS_RT['k'][ind_k_RT]))
        axis4.semilogy(PS['z'], PS['k'][ind_k] ** 3 * PS['P_TT'][:, ind_k]/Beta_T_HM**2 / 2 / np.pi ** 2, color=color,label='k={}'.format(k0))
        print('k RT is ', round(PS_RT['k'][ind_k_RT], 4), 'k HM is,', PS['k'][ind_k])


    axis4.semilogy([], [], color='gray', label='Boern', alpha=0.5, ls='--', lw=4)
    axis4.semilogy([], [], color='gray', label='HM', ls='-.')

    #axis4.set_ylim(1e-1, 1e3)
    axis4.set_xlim(6, 20)

    axis4.legend(title='$ k^{3}P(k)/(2\pi^{2})$')  # dTb^{2}
    axis4.set_xlabel('z', fontsize=14)

    if save_loc is not None:
        plt.savefig(save_loc)


def Plot_heat_profiles(profile,HM_PS,zz,label,color,rho='heat'):
    """""""""
    RT_profile : profile dictionnary from radtrans
    HM_PS : output dic of coda.halomodel
    rho : str, either xray or heat
    """""""""
    #profile = load_profile(RT_profile)
    PS = HM_PS
    ind_z = np.argmin(np.abs(profile.z_history - zz))
    z_RT = profile.z_history[ind_z]
    M_halo = profile.Mh_history[ind_z]
    print('solid is HaloModel dashed is RadTrans')
    print('in RT z is', z_RT, 'Mhalo is {:2e}'.format(M_halo))
    if rho == 'heat':
        plt.loglog(profile.r_grid_cell, profile.T_history[str(round(z_RT, 2))],color=color,ls='--')
    elif rho =='xray':
        plt.loglog(profile.r_grid_cell, profile.rhox_history[str(round(z_RT, 2))],color=color,ls='--')

    rmin,rmax,Nr = 0.1,6000,50
    rr = np.logspace(np.log10(rmin), np.log10(rmax), Nr)

    M0, rho_heat,rho_xray  = PS['M0'], PS['rho_heat_prof'], PS['rho_X_prof']
    ind_z_HM = np.argmin(np.abs(PS['z'] - z_RT))

    if rho == 'xray':
        ind_M_HM = np.argmin(np.abs(PS['M_accr'][ind_z_HM] - M_halo))
        plt.loglog(rr, rho_xray[ind_z_HM, :, ind_M_HM]*eV_per_erg,label=label,color=color,ls='-')
        print('in HM z is', PS['z'][ind_z_HM], 'Mhalo is {:2e}'.format(PS['M_accr'][ind_z_HM, ind_M_HM]))
        plt.ylim(1e-30,1e-12)
    if rho == 'heat':
        ind_M_HM = np.argmin(np.abs(PS['M0'] - M_halo))
        plt.loglog(rr, rho_heat[ind_z_HM, :, ind_M_HM], label=label, color=color, ls='-')
        print('in HM z is', PS['z'][ind_z_HM], 'Mhalo is {:2e}'.format(PS['M0'][ind_M_HM]))
        plt.ylim(1e-7, 1e6)













#########HALO MODEL PARAMETER FUNCTION

def default_par_UpperLim(fst, g1, g2, g4, Mp, bias, g3=4, Nion=4000, Nal=9690, Mt=1e9, c=2.7, fX=0.2, fesc0=0.15,
                         Emin=273, q=0.8, Emax_sed=8000, Emax=10000, M_min=1e4):  #
    import cosmicdawn as coda
    par = coda.par()
    par.code.bias = bias
    par.cosmo.h0 = 0.68
    par.cosmo.Om = 0.31
    par.cosmo.Ob = 0.045
    par.code.zmin = 5
    par.code.zmax = 40
    par.code.Nz = 65
    par.code.dz_prime_lyal = 0.01
    par.code.dz_prime_xray = 0.1
    par.code.Mmin = 5e4
    par.code.Mmax = 2e15
    par.code.NM = 100
    par.code.sfrd_from_MA = True

    par.code.kmin = 0.001
    par.code.kmax = 100
    par.code.Nk = 200
    par.code.Emin = Emin
    par.code.Emax = Emax
    par.code.NE = 40

    par.file.ps = '/Users/timotheeschaeffer/Documents/PHD/21cm/P_k_WDM_Class/Pk_CDM.dat'

    par.code.Mdark = M_min
    par.code.MA = 'EXP'
    par.mf.window = 'tophat'  # ''sharpk'   #'sharpk'#'
    par.mf.c = c  # 2.5
    par.mf.q = q  # 1   #1.02 #0.75
    par.mf.p = 0.3  # 0.3
    par.mf.delta_c = 1.675

    par.lyal.N_ph = Nal
    par.lyal.pl_sed = 1.0001
    par.lyal.f0_sfe = fst
    par.lyal.g1_sfe = g1  # -0.5
    par.lyal.g2_sfe = g2  # -0.5
    par.lyal.Mt_sfe = Mt
    par.lyal.Mp_sfe = Mp
    par.lyal.g3_sfe = g3
    par.lyal.g4_sfe = g4

    par.xray.pl_sed = 2.5
    par.xray.cX = 3.4e40
    par.xray.Emin_sed = 500
    par.xray.Emax_sed = Emax_sed
    par.xray.fX = fX
    par.xray.f0_sfe = fst
    par.xray.g1_sfe = g1  # -0.5
    par.xray.g2_sfe = g2  # -0.5
    par.xray.Mt_sfe = Mt
    par.xray.Mp_sfe = Mp
    par.xray.g3_sfe = g3
    par.xray.g4_sfe = g4
    par.reio.N_ph  = Nion
    par.reio.f0_esc = fesc0
    par.reio.pl_esc = 0.0

    par.reio.f0_sfe = fst
    par.reio.g1_sfe = g1 #-0.5
    par.reio.g2_sfe = g2 #-0.5
    par.reio.Mt_sfe = Mt
    par.reio.Mp_sfe = Mp
    par.reio.g3_sfe = g3
    par.reio.g4_sfe = g4
    return par










########### FUNCTIONS TO DO COMPARISON PLOTS WITH ------>21CM_FAST<--------
def plot_FAST(Fast_Model,qty = 'zz',xlim = None,ylim = None,label='',color='C0',ls='-'):
    """""""""
    This functions plots 21cmFAST global quantities.
    
    Parameters
    ----------    
    Fast_Model : dictionnary, output of GlogalSignal_21cmFast (function below)
    qty : str. quantity to plot
    """""""""
    try:
        zz, quantity  = Fast_Model['zz'],Fast_Model[qty]
        plt.plot(zz, quantity, label=label, color=color, ls=ls)
    except Exception :
        print('available qty in Fast: ',Fast_Model.keys())
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()


def plot_Beorn(physics, qty='dTb', xlim=None, ylim=None, label='', color='C0', ls='-', lw=1, alpha=1):
    """""""""
    This functions plots Beorn global quantities.

    Parametersu
    ----------    
    physics : GS pickle file given by Beorn.
    """""""""

    try:
        plt.plot(physics['z'], physics[qty], label=label, color=color, ls=ls,lw=lw,alpha=alpha)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
    except Exception:
        if qty == 'x_HI' or qty == 'xHI':
            plt.plot(physics['z'], 1 - physics['x_HII'], label=label, color=color, ls=ls,lw=lw,alpha=alpha)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.legend()
        print('available qty in Beorn: ', physics.keys())



def GlogalSignal_21cmFast(path):
    """""""""
    Transform the 21cmFAST globalsignal file into a clear dictionnary

    Parameters
    ----------    
    path : global_evolution data file output by 21cmFAST run (int Output_files folder)
    """""""""
    file = np.loadtxt(path)
    zz = file[:,0]
    xHI = file[:,1]
    Tk = file[:,2]
    Tspin = file[:,4]
    Tcmb = file[:,5]
    Om,Ob,h0 = 0.31,0.045,0.68
    factor = 27 * (1 / 10) ** 0.5 * (Ob * h0 ** 2 / 0.023) * (Om * h0 ** 2 / 0.15) ** (-0.5)
    dTb = factor * np.sqrt(1 + zz) * (1 - Tcmb / Tspin) * xHI
    return {'zz':zz,'xHI': xHI, 'Tk':Tk, 'Tspin':Tspin,'Tcmb': Tcmb, 'dTb':dTb}



def plot_FAST_RT_PS_of_z(k, k_values_fast,path_to_FAST_PS, GS_Beorn, PS_Beorn, color='b',RSD = False):

    ind_k_fast = np.argmin(np.abs(k_values_fast - k))

    FAST_PS_of_z = np.loadtxt(path_to_FAST_PS+'PS_k' + str(ind_k_fast))
    plt.semilogy(FAST_PS_of_z[:, 0], FAST_PS_of_z[:, 1], ls='--')
    print('k fast is', k_values_fast[ind_k_fast])

    ind_k = np.argmin(np.abs(PS_Beorn['k'] * 0.68 - k))
    print('k RT is', PS_Beorn['k'][ind_k] * 0.68)

    kk, PS_dTb_RT = PS_Beorn['k'][ind_k], PS_Beorn['PS_dTb'][:, ind_k]
    dTb_RT = GS_Beorn['dTb']
    plt.semilogy(GS_Beorn['z'], kk ** 3 * dTb_RT ** 2 * PS_dTb_RT / 2 / np.pi ** 2, lw=5, alpha=0.4,label='k=' + str(round(k_values_fast[ind_k_fast], 2)), color=color)
    if RSD :
        dTb_RSD,PS_dTb_RT_RSD = GS_Beorn['dTb_RSD'],PS_Beorn['PS_dTb_RSD'][:, ind_k]
        plt.semilogy(GS_Beorn['z'], kk ** 3 * dTb_RSD ** 2 * PS_dTb_RT_RSD / 2 / np.pi ** 2, lw=1, alpha=1,label='k=' + str(round(k_values_fast[ind_k_fast], 2)), color=color)
        print(PS_dTb_RT_RSD/PS_dTb_RT)