"""
External Parameters
"""

class Bunch(object):
    """
    translates dic['name'] into dic.name 
    """

    def __init__(self, data):
        self.__dict__.update(data)


def source_par():
    par = {
        "fc2ray": 4000,              # coefficient for the galaxy model, Nion_dot
        "M_halo": 1e6,               # galaxy halo mass, miniqso halo mass (required for the profiles, i.e. for the bias )
        "C_halo": 1,                 # halo concentration. Needed for the profiles
        "T_gal": 5e4,                # galaxy Temperature. We let it to 50 000K for now.
        "M_miniqso": 1e4,            # miniquasar mass in the case of miniqsos (Msol)
        "alpha": 1,                  # SED of the miniqso spectra
        "type": 'Miniqsos',          # source type. Can be 'Galaxies' or 'Miniqsos'
        "E_0" : 10.4,                # minimum energy of ionizing photons in eV
        "E_upp" : 10000,             # minimum energy of ionizing photons in e
        "lifetime" : 10,             # time [Myr] until which we switch off the photon production from the source
        "alpha_MAR" : 0.79,              # coefficient for exponential MAR
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
    }

    return Bunch(par)

def solver_par():
    par = {
        "z" : 6,
        "z_end" : 6,       ## Only for MAR. Redshift where to stop the solver
        "r_end" : 3,
        "dn"  : 10,       ## number of radial sample points to initialize the RT solver (then adaptive refinement goes on)
        "Nt"  : 150,      ## number of time slices
        "dn_table" : 100, ## number of radial sample points for the table
        "refinement": False, ## Bool, wheter or not to refine.
        "precision": 0.05, ## degree of precision for the ionization front (to decide when to stop the refinement)
        "evol" : 10,      ## evolution time, typically 3-10 Myr
        "C" : 1,          ## Clumping factor
        "method": 'sol',  ## sol for using the clean solver and bruteforce to just solve the equation by discretizing independently nH and T

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
    'ps': None,
    'corr_fct' : None,
    'HI_frac' : 0.7,     #fraction of Hydrogen. Only used when running H_He_Final. 1-fraction is Helium then.
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
        })
    return par
