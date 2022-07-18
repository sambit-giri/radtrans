"""
A test run of the RT solver. Here we include Helium
"""

import pickle
import numpy as np
import radtrans as rad
from radtrans.solver_Helium import Source_MAR_Helium

param = rad.par()


for Emin in [500, ]:
    for Nion in [4000, ]:
        # solver param
        param.solver.time_step = 0.05
        param.solver.dn = 50  # rad bins
        param.solver.z_end = 15
        param.solver.z = 25
        param.solver.r_end = 3

        # cosmo
        param.cosmo.Om = 0.31
        param.cosmo.Ob = 0.045
        param.cosmo.Ol = 0.69
        param.cosmo.h = 0.68
        param.cosmo.profile = 0

        ## Source sed
        param.source.N_al = 9690
        param.source.Nion = Nion
        param.source.xray_in_ion = 0
        param.source.ion_in_xray = 0
        param.source.E_min_xray = Emin
        param.source.E_max_xray = 10000
        param.source.E_min_sed_xray = 500
        param.source.E_max_sed_xray = 8000
        param.source.E_min_sed_ion = 13.6
        param.source.E_max_sed_ion = 273  # should be equal to E_min_xray to be sure that we get Nion photons in total !
        param.source.alS_ion = 0
        param.source.cX = 0.2 * 3.4e40  # * 1e-90

        # fesc
        param.source.f0_esc = 0.15

        # fstar
        param.source.f_st = 0.2
        param.source.g1 = 0.49
        param.source.g2 = -0.61
        param.source.g3 = 4
        param.source.g4 = -1
        param.source.Mp = 1e11
        param.source.Mt = 4e9

        param.source.M_halo = 1e10

        grid_model = Source_MAR_Helium(param)
        grid_model.solve(param)
        # pickle.dump(file=open('./TProfiles_test_AsHM_Nion'+str(Nion)+'_Emin'+str(Emin)+'.pkl','wb'),obj=grid_model )

