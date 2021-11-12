
import scipy.integrate as integrate
import math
import numpy as np



#define Hubble factor H=H0*E
def E(x,param):
    return math.sqrt(param.cosmo.Om*(x**(-3))+1-param.cosmo.Om)

#define D(a) non-normalized
def D_non_normalized(a,param):
    w=integrate.quad(lambda u: 1/(u*E(u,param))**3,0,a)[0]
    return  (5*param.cosmo.Om*E(a,param)/(2))*w

#define D normalized
def D(a,param):
    return D_non_normalized(a,param)/D_non_normalized(1,param)

delta_c = 1.686

def delt_c(z,param):
    return delta_c/D(1/(1+z),param)


def wf_sharpk(y):
    return np.heaviside(1 - y, 0)

def wf_tophat(x):
    return 3 * (np.sin(x) - x * np.cos(x)) / (x) ** 3


def read_powerspectrum(param):
    """
    Linear power spectrum from file
    """
    names= 'k, P'
    PS = np.genfromtxt(param.cosmo.ps,usecols=(0,1),comments='#',dtype=None, names=names) #
    return PS

def Variance_tophat(param,mm):
    """
    Sigma**2 at z=0 computed with a tophat filter. Used to compute the barrier.
    Output : Var and dlnVar_dlnM
    We reintroduce little h units to be consistent with Power spec units.
    """
    ps = read_powerspectrum(param)
    kk_ = ps['k']
    PS_ = ps['P']
    rhoc = 2.775e11 ### with h
    R_ = ((3 * mm / (4 * rhoc * param.cosmo.Om * np.pi)) ** (1. / 3))
    #Var = np.trapz(kk_ ** 2 * PS_ * wf_tophat(kk_ * R_[:, None]) ** 2 / (2 * np.pi ** 2), kk_, axis=-1)
    Var = np.trapz(kk_ ** 2 * PS_ * wf_tophat(kk_ * R_) ** 2 / (2 * np.pi ** 2), kk_)
   # dlnVar_dlnM = np.gradient(np.log(Var), np.log(mm) )
    return Var #, dlnVar_dlnM


def bias(z,param):
    q = 0.73 # sometimes called a
    p = 0.15
    dcz = delt_c(z,param)
    M = param.source.M_halo * param.cosmo.h ### Msol/h
    var = Variance_tophat(param,M)
    nu = dcz ** 2.0 / var
    # cooray and sheth
    e1 = (q * nu - 1.0) / delta_c
    E1 = 2.0 * p / delta_c / (1.0 + (q * nu) ** p)
    bias = 1.0 + e1 + E1
    return bias



def rho_2h(bias_, cosmo_corr_ ,param, z):
    return (bias_ * cosmo_corr_ + 1.0) * param.cosmo.Om * rhoc_of_z(param, z)


def rhoNFW_fct(rbin,param):
    """
    NFW density profile.
    """
    Mvir = param.source.M_halo
    cvir = param.source.C_halo
    rvir = (3.0*Mvir/(4.0*np.pi*200*rhoc_of_z(param)))**(1.0/3.0)
    rho0 = 200*rhoc_of_z(param)*cvir**3.0/(3.0*np.log(1.0+cvir)-3.0*cvir/(1.0+cvir))
    x = cvir*rbin/rvir
    return rho0/(x * (1.0+x)**2.0)

def R_halo(M_halo,z,param):
    """
    M_halo in Msol.
    """
    return (3*M_halo/(4*math.pi*200*rhoc_of_z(param,z)*(1+z)**3))**(1.0/3)

def rhoc_of_z(param,z):
    """
    Redshift dependence of critical density
    (in comoving units where rho_b=const; same as in AHF)
    Outputs is in Msol/Mpc**3
    """
    Om = param.cosmo.Om
    rhoc = 2.775e11 * param.cosmo.h**2 ## in Msol/Mpc**3
    return rhoc * (Om * (1.0 + z) ** 3.0 + (1.0 - Om)) / (1.0 + z) ** 3.0


def profile(bias_,cosmo_corr_,param, z):
    """
    Global profile, in (Msol/h)/(Mpc/h)**3, normalized to the total matter density.
    """
    return rho_2h(bias_, cosmo_corr_, param, z) #+ rhoNFW_fct(rbin,param)