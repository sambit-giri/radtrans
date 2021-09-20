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
        "M_halo": 1e6,               # galaxy halo mass
        "T_gal": 5e4 ,               # galaxy Temperature. We let it to 50 000K for now.
        "M_miniqso": 1e4,            # miniquasar mass in the case of miniqsos
        "alpha" : 1,                 # SED of the miniqso spectra
        "type": 'Miniqsos',          # source type. Can be 'Galaxies' or 'Miniqsos'
        "E_0" : 10.4        ,        # minimum energy of ionizing photons in eV
        "E_upp" : 10000        ,     # minimum energy of ionizing photons in eV
        }
    return Bunch(par)

def solver_par():
    par = {
        "z" : 6,
        "r_start" : 0.001,
        "r_end" : 3,
        "dn"  : 10,       ## number of radial sample points to initialize the RT solver (then adaptive refinement goes on)
        "dn_table" : 100, ## number of radial sample points for the table
        "evol" : 10,      ## evolution time, typically 3-10 Myr
        "C" : 1,          ## Clumping factor

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
        })
    return par
