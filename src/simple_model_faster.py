"""""""""
Here we develop a fast and simple bubble model + Temp model
"""""""""
from .constants import *
from .astro import *
from .cross_sections import sigma_HI, sigma_HeI
import numpy as np
from scipy.integrate import cumtrapz, trapz, odeint
from scipy.interpolate import splrep, splev, interp1d
from .cosmo import comoving_distance, Hubble, hubble, cosmo_astropy
from scipy.optimize import fsolve
from astropy import units as u
#from astropy.cosmology import WMAP7 as pl
from astropy.cosmology import FlatLambdaCDM
from radtrans.cross_sections import alpha_HII




class simple_solver_faster:
    """
    Source which ionizes the surrounding H and He gas along a radial direction.

    This class initiates a source with a given mass, redshift and a default power law source, which can be changed to
    any spectral energy distribution. Given a starting point and a ending point, the radiative transfer equations are
    solved along the radial direction and the H and He densities are evolved for some given time.

    Rbubble is in comoving Mpc/h.
    r_grid is in cMpc/h
    Parameters
    ----------
    """

    def __init__(self, param):
        self.z_initial = param.solver.z  # starting redshift
        if self.z_initial < 35:
            print('WARNING : z_start (param.solver.z) should be larger than 35 when simple model is chosen.  ')
        self.z_end = param.solver.z_end  # starting redshift
        self.alpha = param.source.alpha_MAR
        self.M_halo = param.source.M_halo #Msol/h
        rmin = 1e-2
        rmax = 600
        Nr = 200
        rr = np.logspace(np.log10(rmin), np.log10(rmax), Nr)
        self.r_grid = rr  ##cMpc/h

        if isinstance(param.solver.Nz, int):
            print('param.solver.Nz is given as an integer. We define z values in linspace from ',self.z_initial,'to ',self.z_end)
            self.z_arr  = np.linspace(self.z_initial,self.z_end,param.solver.Nz)
        elif isinstance(param.solver.Nz, str):
            self.z_arr = np.loadtxt(param.solver.Nz)
            print('param.solver.Nz is given as a string. We read zvalues from ', param.solver.Nz)
        else :
            print('param.solver.Nz should be a string or an int.')

        M_i_min = param.sim.M_i_min
        M_i_max = param.sim.M_i_max
        binn  = param.sim.binn  # let's start with 10 bins
        self.M_Bin = np.logspace(np.log10(M_i_min), np.log10(M_i_max), binn, base=10)


    def solve(self, param):

        Mh_history = self.M_Bin * np.exp(param.source.alpha_MAR * (self.z_initial - self.z_arr[:,None])) #shape is [zz, Mass]
        zz = self.z_arr
        dMh_dt    = dMh_dt_EXP(param, Mh_history, zz[:,None])
        rho_xray_ = rho_xray(self.r_grid, Mh_history, dMh_dt, zz, param)
        rho_heat_ = rho_heat(self.r_grid, rho_xray_, zz, param)
        #R_bubble_ = R_bubble(param,zz,Mh_history).clip(min=0) #cMpc
        R_bubble_ = R_bubble(param, zz,Mh_history).clip(min=0) #c Mpc


        T_history = {}
        rhox_history = {}
        for i in range(len(zz)):
            T_history[str(zz[i])] = rho_heat_[i]
            rhox_history[str(zz[i])] =rho_xray_[i]

        self.rhox_history = rhox_history
        self.Mh_history = Mh_history
        self.z_history = zz
        self.R_bubble = R_bubble_     # cMpc/h (zz,M)
        #self.T_profile = rho_heat_    # Kelvins
        self.T_history = T_history    # Kelvins
        self.T_neutral_hist = T_history    # Kelvins
        self.rho_heat = rho_heat_  #shape (z,r,M)
        self.r_grid_cell = self.r_grid
        self.Ngdot_ion = Ngdot_ion(param, zz[:,None], Mh_history)



def Ngdot_ion(param, zz, Mh):
    """
    zz : redshift. Matters for the mass accretion rate!!
    Mass : mass of the halo in Msol/h
    Number of ionising photons emitted per sec for a given source model and source parameter. [s**-1] for ion and ev/s for xray
    Mass : extra halo mass, when one want to compute Ngdot for a different mass than param.source.Mhalo. Only for param.source.type == SED

    Returns
    ----------
    Number of ionizing photons emitted per sec [s**-1].
    """
    Ob, Om, h0 = param.cosmo.Ob, param.cosmo.Om, param.cosmo.h

    if (param.source.type == 'SED'):
        dMh_dt = param.source.alpha_MAR * Mh * (zz + 1) * Hubble(zz, param)  ## [(Msol/h) / yr]
        Ngam_dot_ion = dMh_dt / h0 * f_star_Halo(param, Mh) * Ob / Om * f_esc(param, Mh) * param.source.Nion / sec_per_year / m_H * M_sun
        Ngam_dot_ion[np.where(Mh < param.source.M_min)] = 0
        return Ngam_dot_ion

    elif param.source.type == 'constant':
        print('constant number of ionising photons chosen. Param.source.Nion becomes Ngam_dot_ion.')
        return np.full(len(zz),param.source.Nion)

    elif (param.source.type == 'Ross'):
        return Mhalo / h0 * Ob / Om / (10 * 1e6 * sec_per_year) / m_p_in_Msun
    else:
        print('Source Type not available. Should be SED or Ross.')
        exit()


def dMh_dt_EXP(param,Mh,z):
    return param.source.alpha_MAR * Mh * (z + 1) * Hubble(z, param)


def R_bubble(param, zz, M_accr):
    """
    Parameters
    ----------
    param : dictionnary containing all the input parameters
    M_accr : halo mass history as a function of redshift zz_arr

    Returns
    ----------
    comoving size [cMpc/h] of the ionized bubble around the source, as a function of time. Array of size len(zz_arr)
    zz : redshift. Matters for the mass accretion rate!!
    """
    Ngam_dot = Ngdot_ion(param, zz[:,None], M_accr)  # s-1
    Ob, Om, h0 = param.cosmo.Ob, param.cosmo.Om, param.cosmo.h
    nb0 = (Ob * rhoc0) / (m_p_in_Msun * h0)  # comoving nbr density of baryons [Mpc/h]**-3
    aa = 1/(zz+1)
    nb0_z = nb0 * (1 + zz) ** 3 # physical baryon density

    nb0_interp  = interp1d(aa, nb0_z, fill_value='extrapolate')
    Ngam_interp = interp1d(aa, Ngam_dot, axis=0,fill_value='extrapolate')
    C = 1.0 #clumping factor

    #source = lambda r, a: km_per_Mpc / (hubble(1 / a - 1, param) * a) * (Ngam_interp(a) / (4 * np.pi * r ** 2 * nb0) - alpha_HII( 1e4) / cm_per_Mpc ** 3 * a ** -3 * h0 ** 3 * nb0 * r / 3) # nb0 * a**-3 is physical baryon density
    #source = lambda V, t: Ngam_interp(t) / nb0 - alpha_HII(1e4) * C / cm_per_Mpc ** 3 * h0 ** 3 * nb0_interp(t) * V  # eq 65 from barkana and loeb
    source = lambda V, a: km_per_Mpc / (hubble(1 / a - 1, param) * a) * (Ngam_interp(a) / nb0 - alpha_HII(1e4) * C / cm_per_Mpc ** 3 * h0 ** 3 * nb0_interp(a) * V)  # eq 65 from barkana and loeb

#bubble_size = odeint(source, 1e-20, time) ## initial condition 1e-20. We get nan if put to zero.
    bubble_vol = odeint(source, np.zeros(len(M_accr[0])), aa)

    return (3*bubble_vol/4/np.pi)**(1/3)




def rho_xray(rr, M_accr, dMdt_accr, zz, param):
    """
    X-ray profile
    of shape rho(zz,rr,MM) (M_accr, dMdt_accr all have same dimension (zz,Masses))
    zz is in decreasing order
    M_accr is function of zz and hence increases
    rr is comoving distance
    """

    Om = param.cosmo.Om
    Ob = param.cosmo.Ob
    h0 = param.cosmo.h
    zstar = 35
    Emin = param.source.E_min_xray
    Emax = param.source.E_max_xray
    NE = 50

    nb0 = rhoc0 * Ob / (m_p_in_Msun * h0)  # [h/Mpc]^3

    # zprime binning
    dz_prime = 0.1

    # define frequency bin
    nu_min = Emin / h_eV_sec
    nu_max = Emax / h_eV_sec
    N_mu = NE
    nu = np.logspace(np.log(nu_min), np.log(nu_max), N_mu, base=np.e)

    f_He_bynumb = 1 - param.cosmo.HI_frac
    # hydrogen
    nH0 = (1-f_He_bynumb) * nb0
    # helium
    nHe0 = f_He_bynumb * nb0

    rho_xray = np.zeros((len(zz), len(rr),len(M_accr[0])))
    for i in range(len(zz)):

        if (zz[i] < zstar):
           # rr_comoving = rr * (1 + zz[i])
            z_max = zstar
            zrange = z_max - zz[i]
            N_prime = int(zrange / dz_prime)



            if (N_prime < 4):
                N_prime = 4
            z_prime = np.logspace(np.log(zz[i]), np.log(z_max), N_prime, base=np.e)
            rcom_prime = comoving_distance(z_prime, param) * h0  # comoving distance

            dMdt_int = interp1d(zz[:i+1], (Ob / Om) * f_star_Halo(param,M_accr[:i+1,:]) * dMdt_accr[:i+1,:],axis=0, fill_value='extrapolate')

            flux = np.zeros((len(nu), len(rr),len(M_accr[0])))

            for j in range(len(nu)):
                tau_prime = cum_optical_depth(z_prime, nu[j] * h_eV_sec, param)
                eps_X = eps_xray(nu[j] * (1 + z_prime) / (1 + zz[i]), param)[:,None] * np.exp(-tau_prime)[:,None] * dMdt_int(z_prime)  # [1/s/Hz]
                eps_int = interp1d(rcom_prime, eps_X, axis=0, fill_value=0.0, bounds_error=False)
                flux[j, :,:] = np.array(eps_int(rr))


            fXh =  0.11       # 1.0 # 0.13 # 0.15 ---> 0.11 matches the f_heat we have in cross_sections.py, for T_neutral

            pref_nu = fXh * ((nH0 / nb0) * sigma_HI(nu * h_eV_sec) * (nu * h_eV_sec - E_HI) + (nHe0 / nb0) * sigma_HeI(nu * h_eV_sec) * (nu * h_eV_sec - E_HeI))   # [cm^2 * eV] 4 * np.pi *

            heat_nu = pref_nu[:, None,None] * flux  # [cm^2*eV/s/Hz]
            heat_of_r = trapz(heat_nu, nu, axis=0)  # [cm^2*eV/s]
            rho_xray[i, :,:] = heat_of_r / (4 * np.pi * (rr/(1+zz[i])) ** 2)[:,None] / (cm_per_Mpc/h0) ** 2  # [eV/s]  1/(rr/(1 + zz[i]))**2

    return rho_xray




def rho_heat(rr, rho_xray, zz, param):
    """
    Going from heating to temperatue.
    Units : Kelvin
    Shape : (zz,rr)

    """
    # decoupling redshift as ic
    z0 = param.cosmo.z_decoupl
    zz = np.concatenate((np.array([z0]),zz))
    aa = np.array(list((1 / (1 + zz)))) #scale factor
    zero = np.zeros((1,len(rho_xray[0,:,0]),len(rho_xray[0,0,:])))  #,len(rho_xray[0,0,:])))
    rho_xray = np.vstack((zero,rho_xray))

    # perturbations
    rho_heat = np.zeros((len(zz)-1,len(rr),len(rho_xray[0,0,:])))
    for j in range(len(rr)):
       # rho_intp = interp1d(aa, rho_xray[:, j,:],axis=0, fill_value="extrapolate")
        rho_intp = interp1d(aa,rho_xray[:, j,:],axis=0,fill_value="extrapolate")
        Gamma_heat = lambda a: 2 * rho_intp(a)/ (3 * kb_eV_per_K * a * hubble(1 / a - 1, param)) * km_per_Mpc  #rho_intp(a)
        source = lambda T_h, a: Gamma_heat(a) - 2 * T_h / a
        T_h_j = odeint(source,np.zeros(len(rho_xray[0,0,:])), aa)

        # print('shape of T_h_j is ',T_h_j.shape,'shape of aa_rev is ',aa_rev.shape,'shape of M0',M0.shape,'shape of zz',zz.shape)
        T_h_j = np.delete(T_h_j,0,axis=0)
        rho_heat[:, j,:] = T_h_j

    return rho_heat




def cum_optical_depth(zz,E,param):
    """
    Cumulative optical optical depth of array zz.
    See e.g. Eq. 6 of 1406.4120
    """
    Ob = param.cosmo.Ob
    h0 = param.cosmo.h

    # Energy of a photon observed at (zz[0], E) and emitted at zz
    if type(E) == np.ndarray:
        Erest = np.outer(E,(1 + zz)/(1 + zz[0]))
    else:
        Erest = E * (1 + zz)/(1 + zz[0])

    #hydrogen and helium cross sections
    sHI   = sigma_HI(Erest)*(h0/cm_per_Mpc)**2   #[Mpc/h]^2
    sHeI  = sigma_HeI(Erest)*(h0/cm_per_Mpc)**2  #[Mpc/h]^2



    nb0   = rhoc0*Ob/(m_p_in_Msun*h0)                    # [h/Mpc]^3

    f_He_bynumb = 1 - param.cosmo.HI_frac

    #H and He abundances
    nHI   = (1-f_He_bynumb)*nb0 *(1+zz)**3       # [h/Mpc]^3
    nHeI  = f_He_bynumb * nb0 *(1+zz)**3

    #proper line element
    dldz = c_km_s*h0/hubble(zz,param)/(1+zz) # [Mpc/h]

    #integrate
    tau_int = dldz * (nHI*sHI + nHeI*sHeI)

    if type(E) == np.ndarray:
        tau = cumtrapz(tau_int,x=zz,axis=1,initial=0.0)
    else:
        tau = cumtrapz(tau_int,x=zz,initial=0.0)

    return tau
