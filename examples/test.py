import numpy as np

import radtrans as rad
from radtrans.H_He_final import *

param = rad.par() ###parameter dictionnary
param.source.type = 'Galaxies' ### you can choose the source type 'Miniqsos', 'Galaxies'
param.source.M_halo = 1e9 ### you can set the halo mass and f_c2ray
grid_model = Source(param,1e5,10,10,alpha=0,r_end=1,dn=10,import_table=False) 

grid_model = Source(
    param,
    1e9, #M,
    10, #z,
    10, #evol,
    r_start=0.01,
    r_end=100,
    dn=50,
    LE=None,
    alpha=None,
    sed=None,
    lifetime=None,
    filename_table='table_alpha1_0_uv.p',
    C=None,
    recalculate_table=False,
    import_table=False,
    import_profiles=False,
)

grid_model.solve(param)

