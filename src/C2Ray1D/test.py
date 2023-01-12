import numpy as np
from glob import glob 
import os
from . import pyccray

class C2RAY_wrapper:
    '''
    The object that uses wrapped C2Ray functions.

    Args:
        param (Bunch): The parameters. 
        verbose (Bool): Verbose (Default: True).

    Attributes:
        param (Bunch): The parameters are stored here.
    '''
    def __init__(self, param, verbose=True):
        self.param = param 
        self.verbose = verbose
        
    def run_c2ray1D_inputfile(self,inputfile=None):
        if inputfile is None: inputfile = self.param.RTsolver.inputfile
        pyccray.run_inputfile(inputfile)
        Ifront1D_files = np.array(glob('Ifront1_*'))
        Ifront1D_xfs   = np.array([ff.split('Ifront1_')[-1].split('.dat')[0] for ff in Ifront1D_files]).astype(float)
        Ifront1D_files = Ifront1D_files[np.argsort(Ifront1D_xfs)]
        Ifront1D_xfs   = Ifront1D_xfs[np.argsort(Ifront1D_xfs)]
        Ifront1D_dict = {Ifront1D_xfs[i]:np.loadtxt(ff) for i,ff in enumerate(Ifront1D_files)}
        for ff in Ifront1D_files: os.remove(ff)
        return Ifront1D_dict
 