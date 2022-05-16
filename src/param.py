"""
External Parameters
"""
import pkg_resources
class Bunch(object):
    """
    translates dic['name'] into dic.name 
    """

    def __init__(self, data):
        self.__dict__.update(data)


def source_par():
    par = {
        "M_halo": 1e6,               # Msol/h

        "type": 'SED',                   # source type. Can be 'Galaxies' or 'Miniqsos' or SED

        "E_min_sed_ion": 13.6,                # minimum energy of normalization of ionizing photons in eV
        "E_max_sed_ion": 1000,             # minimum energy of normalization of ionizing photons in eV


        "E_min_sed_xray": 500,             # minimum energy of normalization of xrays in eV
        "E_max_sed_xray": 8000,            # minimum energy of normalization of xrays in eV

        "E_min_xray": 100,
        "E_max_xray": 10000,  # min and max energy that define what we call xrays.

        "alS_ion": 1.5,                 ##PL sed ion part
        "alS_xray": 2.5,                 ##PL sed Xray part N ~ nu**-alS [nbr of photons/s/Hz]
        "cX":  3.4e40,                # Xray normalization [(erg/s) * (yr/Msun)] (astro-ph/0607234 eq22)

        "N_al":9690,    #nbr of lyal photons per baryons in stars
        "alS_lyal":  1.001, ##PL for lyal

        "xray_in_ion" : 1,
        "ion_in_xray" : 1, ##whether or not to include xray in ionising gamma tables calculation and reciprocally

        "alpha_MAR" : 0.79,              # coefficient for exponential MAR
        "M_min" : 1e5,               # Minimum mass of star forming halo. Mdark in HM
        'f_st': 0.05,
        'Mp': 1e11,
        'g1': 0.49,
        'g2': -0.61,
        'Mt': 1e7,
        'g3': 4,
        'g4': -1,

        'Nion': 2665,
        "f0_esc": 0.15,  # photon escape fraction f_esc = f0_esc * (M/Mp)^pl_esc
        "Mp_esc": 1e10,
        "pl_esc": 0.0,

        ## params for old source parametrization, galaxy or Miniqso.
        "E_0" : 10.4,
        "E_upp": 10000,
        "T_gal": 5e4,       # galaxy Temperature. We let it to 50 000K for now.
        "M_miniqso": 1e4,  # miniquasar mass in the case of miniqsos (Msol)
        "alpha": 1,         # SED of the miniqso spectra

    }

    return Bunch(par)

def solver_par():
    par = {
        "z" : 25,                ## Starting redshift
        "z_end" : 6,             ## Only for MAR. Redshift where to stop the solver
        "r_end" : 3,             ## physical Mpc/h
        "dn"  : 100,              ## number of radial sample points to initialize the RT solver (then adaptive refinement goes on)
        "dn_table" : 100,        ## number of radial sample points for the table
        "method": 'bruteforce',  ## "sol" for using the clean solver and "bruteforce" to just solve the equation by discretizing independently nH and T
        "time_step" : 0.1,       ## time step for the solver, in Myr.
        "full_output" : False,  # # if True, then will store dTb, Tspin, and nHI profiles. If False, only Temperature and ionised fractions.
    }
    return Bunch(par)

def sim_par(): ## used when computing and painting profiles on a grid
    par = {
         "M_i_min" : 1e-2,
         "M_i_max" : 1e9,
         "binn" : 12,               # to define the initial halo mass at z_ini = solver.z
         "model_name": 'SED',       # Give a name to your sim, will be used to name all the files created.
         "Ncell" : 128,             # nbr of pixels of the final grid.
         "Lbox" : 100,              # Box lenght, in [Mpc/h]
          "mpi4py": 'no',           # run in parallel or not.
        "halo_catalogs": None,      # path to the directory containing all the halo catalogs.
        "store_grids": True,        # whether or not to store the grids. If not, will just store the power spectra.
        "dens_field": None,         # path and name of the gridded density field. Used in run.py to compute dTb
        "Nh_part_min":50,            # Minimum number of particles in halo to trust
        "cores" : 2, #number of cores used in parallelisation
        "kmin": 3e-2,
        "kmax": 4,
        "kbin": 400,  ## min, max and number of bins in k-space for measuring power spectra.
        "thresh_xHII" :0.1  ### mean(Tk_neutral) and mean(Tspin) are computed over cells that are below this fraction of ionized hydrogen.

    }
    return Bunch(par)



def cosmo_par():
    par = {
    'Om' : 0.3,
    'Ob' : 0.045,
    'Ol' : 0.7,
    'rho_c' : 2.775e11,
    'h' : 0.7,
    's8': None,
    'ps': pkg_resources.resource_filename('radtrans', "files/PCDM_Planck.dat"),      ### This is the path to the input Linear Power Spectrum
    'corr_fct' : pkg_resources.resource_filename('radtrans', "files/corr_fct.dat"),  ### This is the path where the corresponding correlation function will be stored. You can change it to anything.
    'HI_frac' : 0.08,     #fraction of Hydrogen. Only used when running H_He_Final. 1-fraction is Helium then.  0.2453 of total mass is in He according to BBN, so in terms of number density it is  1/(1+4*(1-f_He_bymass)/f_He_bymass)  ~0.075
    "clumping" : 1,         # to rescale the background density. set to 1 to get the normal 2h profile term.
    "profile" : 0,          #0 for constant background density, 1 for 2h term profile
    "z_decoupl" : 135,      # redshift at which the gas decouples from CMB and starts cooling adiabatically according to Tcmb0*(1+z)**2/(1+zdecoupl)
    }
    return Bunch(par)



def table_par():
    par = {
        "import_table": False,              # Whether or not to import an external Gamma table
        "filename_table": None,             # Filename of the table to import OR filename to write in
        }
    return Bunch(par)


def par():
    par = Bunch({
        "source": source_par(),
        "table": table_par(),
        "solver": solver_par(),
        "cosmo" : cosmo_par(),
        "sim" : sim_par(),
        })
    return par
