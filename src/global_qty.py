"""

Global quantity computed directly from halo catalog
"""
import os.path
import numpy as np
from .cosmo import Hubble, hubble
import pickle
from .constants import rhoc0,c_km_s, Tcmb0, sec_per_year, km_per_Mpc
from .astro import Read_Rockstar, f_star_Halo
from .couplings import sigma_HI

def global_signal(param,heat=None):
    catalog_dir = param.sim.halo_catalogs
    xHII = []
    G_heat = []
    z = []
    sfrd = []

    for ii, filename in enumerate(os.listdir(catalog_dir)):
        catalog = catalog_dir + filename
        halo_catalog = Read_Rockstar(catalog)

        if heat is not None :
            heat_per_baryon = G_heat_approx(param,halo_catalog)
        else :
            heat_per_baryon = 0

        zz_, SFRD = sfrd_approx(param,halo_catalog)
        zz_, x_HII = xHII_approx(param,halo_catalog)

        xHII.append(min(x_HII, 1))
        z.append(zz_)
        G_heat.append(heat_per_baryon)
        sfrd.append(SFRD)

    sfrd, xHII, z_array, G_heat = np.array(sfrd), np.array(xHII), np.array(z), np.array(G_heat)
    matrice = np.array([z, xHII,sfrd,G_heat])
    z, xHII,sfrd,G_heat = matrice[:, matrice[0].argsort()] ## sort according to zarray

    return {'z':z,'xHII':xHII,'sfrd':sfrd,'Gamma_heat':G_heat}




def xHII_approx(param,halo_catalog):
    """
    Approximation of the mean ionization fraction (maybe more correct to say volume filling factor.)
    We compute for each halo the volume of the surounding ionized bubble. Sum all these volumes and normalize to the total simulation volume.
    """
    LBox = param.sim.Lbox       # Mpc/h
    M_Bin = np.logspace(np.log10(param.sim.M_i_min), np.log10(param.sim.M_i_max), param.sim.binn, base=10)
    z_start = param.solver.z
    model_name = param.sim.model_name

    H_Masses = halo_catalog['M']
    z = halo_catalog['z']

    # quick load to find matching redshift between solver output and simulation snapshot.
    grid_model = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]), 'rb'))
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]
    Indexing = np.argmin(np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))), axis=1) ## values of Mh at z_start, binned via M_Bin.

    Ionized_vol = 0
    for i in range(len(M_Bin)):
        nbr_halos = np.where(Indexing == i)[0].size
        if nbr_halos > 0:
            grid_model = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]),'rb'))
            xHII_profile = grid_model.xHII_history[str(round(zgrid, 2))]
            r_grid_ = grid_model.r_grid_cell
            bubble_volume = np.trapz(4 * np.pi * r_grid_ ** 2 * xHII_profile,r_grid_)
            Ionized_vol += bubble_volume * nbr_halos  ##physical volume !!
    x_HII = Ionized_vol / (LBox / (1 + z)) ** 3  # normalize by total physical volume
    return zgrid, x_HII



def sfrd_approx(param,halo_catalog):
    """
    Approximation of the sfrd of a given snapshot. We sum over all halos from  halo_catalog, acoording to the source model in param. We then normlize to sim volume.
    Output is in  [(Msol/h) / yr /(cMpc/h)**3]
    """
    LBox = param.sim.Lbox       # Mpc/h
    M_Bin = np.logspace(np.log10(param.sim.M_i_min), np.log10(param.sim.M_i_max), param.sim.binn, base=10)
    z_start = param.solver.z
    model_name = param.sim.model_name

    H_Masses = halo_catalog['M']
    z = halo_catalog['z']
    print('There are', H_Masses.size, 'halos at z=', z, )

    # quick load to find matching redshift between solver output and simulation snapshot.
    grid_model = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]), 'rb'))
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]
    Indexing = np.argmin(np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))), axis=1) ## values of Mh at z_start, binned via M_Bin.

    SFRD = 0
    for i in range(len(M_Bin) ):
        nbr_halos = np.where(Indexing == i)[0].size
        if nbr_halos > 0:
            grid_model = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]),'rb'))
            M_halo = grid_model.Mh_history[ind_z]
            dMh_dt = param.source.alpha_MAR * M_halo * (z + 1) * Hubble(z, param)  ## [(Msol/h) / yr]
            SFRD += nbr_halos * dMh_dt * f_star_Halo(param, M_halo) * param.cosmo.Ob / param.cosmo.Om

    SFRD = SFRD / LBox ** 3  ## [(Msol/h) / yr /(cMpc/h)**3]

    return zgrid, SFRD


def G_heat_approx(param, halo_catalog):
    """
    Compute the energy deposited as heat per baryon in [eV.s-1]
    Similar to xHII : We take the heat profile, integrate over volume, sum over halos and normlize to simulation volume. Gives an average Gamma_heat.
    """
    LBox = param.sim.Lbox  # Mpc/h
    M_Bin = np.logspace(np.log10(param.sim.M_i_min), np.log10(param.sim.M_i_max), param.sim.binn, base=10)
    z_start = param.solver.z
    model_name = param.sim.model_name

    H_Masses = halo_catalog['M']
    z = halo_catalog['z']

    # quick load to find matching redshift between solver output and simulation snapshot.
    grid_model = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[0]), 'rb'))
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]
    Indexing = np.argmin(np.abs(np.log10(H_Masses[:, None] / (M_Bin * np.exp(-param.source.alpha_MAR * (z - z_start))))),axis=1)  ## values of Mh at z_start, binned via M_Bin.

    heat_per_baryon = 0
    for i in range(len(M_Bin)):
        nbr_halos = np.where(Indexing == i)[0].size
        if nbr_halos > 0:
            grid_model = pickle.load(file=open('./profiles_output/SolverMAR_' + model_name + '_zi{}_Mh_{:.1e}.pkl'.format(z_start, M_Bin[i]),'rb'))
            heat_profile = grid_model.heat_history[str(round(zgrid, 2))]
            r_grid_ = grid_model.r_grid_cell
            heat_per_baryon += np.trapz(4 * np.pi * r_grid_ ** 2 * heat_profile,r_grid_ ) * nbr_halos

    heat_per_baryon = heat_per_baryon / (LBox / (1 + z)) ** 3

    return heat_per_baryon


def cum_optical_depth(zz,E,param):
    """
    Cumulative optical optical depth of array zz.
    See e.g. Eq. 6 of 1406.4120
    """

    Om = param.cosmo.Om
    Ob = param.cosmo.Ob
    h0 = param.cosmo.h

    # Energy of a photon observed at (zz[0], E) and emitted at zz
    if type(E) == np.ndarray:
        Erest = np.outer(E,(1 + zz)/(1 + zz[0]))
    else:
        Erest = E * (1 + zz)/(1 + zz[0])

    #hydrogen and helium cross sections
    sHI   = sigma_HI(Erest)*(h0/cm_per_Mpc)**2 #[Mpc/h]^2
    #sHeI  = PhotoIonizationCrossSection(Erest, species=1)*(h0/cm_per_Mpc)**2 #[Mpc/h]^2

    sHeII = 0.0

    nb0   = rhoc0*Ob/(m_p*h0)           # [h/Mpc]^3

    #H and He abundances
    nHI   = nb0 *(1+zz)**3   # [h/Mpc]^3
    #nHeI  = f_He_bynumb * nb0 *(1+zz)**3
    #nHeII = 0.0

    #proper line element
    dldz = c_km_s*h0/hubble(zz,param)/(1+zz) # [Mpc/h]

    #integrate
    tau_int = dldz * (nHI*sHI) #+ nHeI*sHeI + nHeII*sHeII)

    if type(E) == np.ndarray:
        tau = cumtrapz(tau_int,x=zz,axis=1,initial=0.0)
    else:
        tau = cumtrapz(tau_int,x=zz,initial=0.0)

    return tau


def J_xray_nu(zz, sfrd, param):
    """
    X-ray flux per frequency nu
    The numerical method is explained in
    Sec. 3.1 of arXiv:1406.4120
    """
    Om = param.cosmo.Om
    Ob = param.cosmo.Ob
    h0 = param.cosmo.h
    zstar = 25
    Emin = param.code.Emin
    Emax = param.code.Emax
    NE = param.code.NE

    # zprime binning
    dz_prime = param.code.dz_prime_xray

    # define frequency bin
    nu_min = Emin / hP
    nu_max = Emax / hP
    N_mu = NE
    nu = np.logspace(np.log(nu_min), np.log(nu_max), N_mu, base=np.e)

    # energy bin
    EE = nu * hP

    # SFRD
    sfrd_tck = splrep(zz, sfrd)  # [(Msun/h)/yr/(Mpc/h)^3]

    J_X_nu = []
    for i in range(len(zz)):
        J_X_nu_z = []
        if (zz[i] < zstar):
            for j in range(len(nu)):

                z_max = zstar
                zrange = z_max - zz[i]
                N_prime = int(zrange / dz_prime)
                if (N_prime < 2):
                    N_prime = 2
                z_prime = np.logspace(np.log(zz[i]), np.log(z_max), N_prime, base=np.e)
                tau_prime = cum_optical_depth(z_prime, nu[j] * hP, param)

                eps = eps_xray(nu[j] * (1 + z_prime) / (1 + zz[i]), param) * splev(z_prime,sfrd_tck)  # [1/s * (h/Mpc)^3]
                itd = c * h0 / hubble(z_prime, param) * eps * np.exp(-tau_prime)

                J_X_nu_j = (1 + zz[i]) ** 2 / (4 * np.pi) * trapz(itd, z_prime)  # [1/s * (h/Mpc)^2]
                J_X_nu_j = J_X_nu_j * (h0 / cm_per_Mpc) ** 2  # [1/s * (1/cm)^2]
                J_X_nu_z += [J_X_nu_j]
            J_X_nu += [J_X_nu_z]  # [1/s * (1/cm)^2]
        else:
            for j in range(len(nu)):
                J_X_nu_z += [0.0]
            J_X_nu += [J_X_nu_z]

    J_X_nu = np.array(J_X_nu)
    J_X_nu = np.ndarray.transpose(J_X_nu)

    return nu, J_X_nu