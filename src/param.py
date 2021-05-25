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
        "fc2ray": 4000,              # coefficient 
        "M_halo": 1e6,               # galaxy halo mass
        "M_miniqso": 1e4,            # miniquasar mass in the case of miniqsos
        "type": 'Miniqsos',          # source type. Can be 'Galaxies' or 'Miniqsos' 
        }
    return Bunch(par)


def table_par():
    par = {
        "import_table": False,              # Whether or not to import an external Gamma table 
        "recalculate_table": True,          # Not really needed
        "filename_table": None,             # Filename of the table to import OR filename to write in
        }
    return Bunch(par)


def par():
    par = Bunch({
        "source": source_par(),
        "table": table_par(),
        })
    return par
