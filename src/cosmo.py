"""

FUNCTIONS RELATED TO COSMOLOGY

"""
import os.path
import numpy as np
from scipy.integrate import cumtrapz, trapz, quad
from scipy.interpolate import splrep,splev
from .constants import rhoc0,c_km_s, Tcmb0, sec_per_year, km_per_Mpc
import scipy.integrate as integrate
from astropy.cosmology import FlatLambdaCDM


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
    Om, Ol = param.cosmo.Om, param.cosmo.Ol
    return param.cosmo.h * 100.0 * sec_per_year / km_per_Mpc * np.sqrt(Om*(1+z)**3 + (1.0-Om-Ol)*(1+z)**2+Ol)


def comoving_distance(z,param):
    """
    Comoving distance between z[0] and z[-1]
    """
    return cumtrapz(c_km_s/hubble(z,param),z,initial=0)  # [Mpc]


def T_cmb(z):
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


def T_adiab(z,param):
    """
    Temperature of the gas assuming it decoupled from CMB at z = param.cosmo.z_decoupl and then cooled adiabatically.
    """
    return Tcmb0 * (1 + z) ** 2 / (1 + param.cosmo.z_decoupl)




#define Hubble factor H=H0*E
def E(x,param):
    return np.sqrt(param.cosmo.Om*(x**(-3))+1-param.cosmo.Om)

#define D(a) non-normalized
def D_non_normalized(a,param):
    w=integrate.quad(lambda u: 1/(u*E(u,param))**3,0,a)[0]
    return  (5*param.cosmo.Om*E(a,param)/(2))*w

#define D normalized
def D(a,param):
    """
    Growth factor. Normalized to 1 at z = 0.
    """
    return D_non_normalized(a,param)/D_non_normalized(1, param)


def rhoc_of_z(param,z):
    """
    Redshift dependence of critical density
    (in comoving units)
    Outputs is in Msol/cMpc**3
    """
    Om = param.cosmo.Om
    rhoc = 2.775e11 * param.cosmo.h**2  ## in Msol/cMpc**3
    return rhoc * (Om * (1.0 + z) ** 3.0 + (1.0 - Om)) / (1.0 + z) ** 3.0



def siny_ov_y(y):
    s = np.sin(y) / y
    s[np.where(y > 100)] = 0
    return s

def cosmo_astropy(param):
    Ob, Om, h0 = param.cosmo.Ob, param.cosmo.Om, param.cosmo.h
    cosmo = FlatLambdaCDM(H0=100 * h0, Om0= Om, Ob0 = Ob, Tcmb0=2.725)
    return cosmo



def correlation_fct(param):
    """
    This function is called when the RT solver is initialized.
    If the path to a new power spectrum (other than in src/files/PCDM_Planck.dat) is given in param.cosmo.ps,
    then the corr_function at z=0 is recomputed and then written at the location param.cosmo.corr_fct.
    Otherwise this function simply prints that it will read in the corr_fct from param.cosmo.corr_fct.
    """
    rmin = 0.005
    rmax = 100            # Mpc/h. Controls the maximum comoving scale that we compute. Can be important for very large scales 2h profile at high redshift. 100 should be safe
    PS_ = param.cosmo.ps  # z=0 linear power spectrum of matter perturb.
    path_to_corr_file = param.cosmo.corr_fct

    if os.path.isfile(path_to_corr_file):
        print('Correlation function already computed : par.cosmo.corr_fct')
    else:
        try:
            names = "k, PS"
            Power_Spec = np.loadtxt(PS_)
        except IOError:
            print('IOERROR: Cannot read power spec. Try: par.cosmo.ps = "/path/to/file"')
            exit()
        print('Computing the z=0 correlation function from the PS given in par.cosmo.ps')
        bin_N = 200
        bin_r = np.logspace(np.log(rmin), np.log(rmax), bin_N, base=np.e)
        krange = Power_Spec[:, 0]
        PS_values = Power_Spec[:, 1]
        bin_corr = np.trapz(krange ** 3 * PS_values * siny_ov_y(krange * bin_r[:, None]) / 2 / np.pi ** 2,np.log(krange))
        try:
            np.savetxt(path_to_corr_file, np.transpose([bin_r, bin_corr]))
            print('Saving the correlation function in ' + path_to_corr_file)
        except IOError:
            print('IOERROR: cannot write Cosmofct file in a non-existing directory!')
            exit()


