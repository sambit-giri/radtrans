from scipy.signal import fftconvolve
import radtrans as rad
import pickle
from radtrans.Only_Hydrogen import cm_per_Mpc, BB_Planck, profile_1D
from radtrans.bias import bias,profile
sec_per_year = 3600*24*365.25
M_sun = 1.988 * 10 ** 30
m_H    = 1.6 * 10 ** - 27



# Let's start with a simple model with only galaxies, with T = 50000K, a constant fstar and fesc (via f_c2ray = 4000).
# We bin Halo masses, and redshifts, and compute the profiles for all these.

M_min = 1e7
M_max = 1e12
binn = 12 # let's start with 10 bins
M_Bin = np.logspace(np.log10(M_min),np.log10(M_max),binn,base = 10)
z_start = 25
z_end  = 5
binn = 1
z_Bin   = np.linspace(z_start,z_end,binn)


####We need gamma dot to get an estimate of r_end....
def NGamDot(param):
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


# Fix the parameters, and loop over M_HALO and Z_BIN
parameters = rad.par()
parameters.solver.dn = 20
parameters.solver.dn_table = 1000
parameters.solver.precision = 1e10  # No adaptive mesh
parameters.solver.C = 1

parameters.source.E_0 = 13.6
parameters.source.E_upp = 1e3
parameters.source.type = 'Galaxies'
parameters.source.fc2ray = 4000
parameters.source.lifetime = 10
parameters.solver.evol = 10
parameters.solver.Nt = 50

parameters.cosmo.h = 0.7
parameters.cosmo.corr_fct = './radtrans/files/cosmofct.dat'
parameters.cosmo.ps = './radtrans/files/CDM_PLANCK_tk.dat'

parameters.table.import_table = True

for Mhalo in M_Bin:
    for zz_ in z_Bin:
        parameters.source.M_halo = Mhalo

        ### Let's deal with r_end :
        N_gam_dot = NGamDot(parameters)
        cosmofile = parameters.cosmo.corr_fct
        vc_r, vc_m, vc_bias, vc_corr = np.loadtxt(cosmofile, usecols=(0, 1, 2, 3), unpack=True)
        corr_tck = splrep(vc_r, vc_corr, s=0)
        r_MaxiMal = 5  ## Maximum k-value available in cosmofct.dat
        cosmo_corr = splev(r_MaxiMal * (1 + zz_), corr_tck)
        halo_bias = bias(zz_, parameters)
        # baryonic density profile in [cm**-3]
        nHI0_profile = profile(halo_bias, cosmo_corr, parameters, zz_) * parameters.cosmo.Ob / parameters.cosmo.Om * \
                       M_sun * parameters.cosmo.h ** 2 / (cm_per_Mpc) ** 3 / m_H
        r_End = (3 * N_gam_dot * 10 * sec_per_year * 1e6 / 4 / np.pi / parameters.cosmo.Ob / np.mean(nHI0_profile)) ** (
                    1.0 / 3) / cm_per_Mpc
        # print(r_End/10,np.mean(nHI0_profile))
        parameters.solver.r_end = r_End / 10
        parameters.solver.z = zz_
        parameters.table.filename_table = 'Gamma_Gal_HIOnly_10Myr_Mh_1e{}_z{}'.format(round(np.log10(Mhalo), 2),
                                                                                      round(zz_, 2))
        #
        #
        grid_model = rad.Source_Only_H(parameters)
        grid_model.solve(parameters)
        grid_model.fit()
        pickle.dump(
            file=open('Solver_Gal_HIOnly_10Myr_Mh_1e{}_z{}_Nt50'.format(round(np.log10(Mhalo), 2), round(zz_, 2)), 'wb'),obj=grid_model)


