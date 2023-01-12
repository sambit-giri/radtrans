import numpy as np
from glob import glob 
import os, scipy
from scipy.interpolate import splev, splrep
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


class C2RAY_solver:
    '''
    The C2Ray solver object.

    Args:
        param (Bunch): The parameter file. 

    Attributes:
        param (Bunch): The parameters are stored here.
    '''
    def __init__(self, param, spectrum=None, verbose=True):
        self.param = param 
        self.spectrum = spectrum
        self.verbose  = verbose
        
        self.Mpc_to_cm = 3.086e24 #cm
        self.Myr_to_s  = 3.156e13 #s 

    def alphaB(self,T):
        return 2.59e-13 * (T/1e4)**-0.7 #cm^3 s^-1

    def sigma_HI_fit(self,nu):
        '''Cross section for ionising photons. 
        A fit from Osterbrock & Ferland (2006).

        Args:
            nu (float): The frequency in Hz or with astropy unit.

        Returns:
            sigma_HI: Cross section for ionising photons in cm.
        '''
        try: nu = nu.to('Hz').value
        except: nu = nu  # Hz
        sigma0 = 6.30e-18  # cm^2
        nu_HI  = 13.6/const.h_eV_sec
        x = lambda nu: np.sqrt(nu/nu_HI-1)
        sigmaHI = lambda nu: sigma0 * x(nu)**-4 * \
                np.exp(4-4*np.arctan(x(nu))/x(nu))/(1-np.exp(-2*np.pi/x(nu)))
        def _out(nu):
            if nu>nu_HI: return sigmaHI(nu)
            else: 
                nus = 10**np.linspace(15.6,17)
                return 10**splev(np.log10(nu), splrep(np.log10(nus),np.log10(sigmaHI(nus))))
        return np.vectorize(_out)(nu)

    def create_lookuptable_VshellnHIGamma1(self, nbins=20, tau_min=1e-3, tau_max=1e2, spectrum=None, **kargs):
        if spectrum is None: spectrum = self.spectrum
        L_nu = spectrum.L_nu
        if self.verbose: print('Creating the look-up table for Gamma intergration...')
        log_taus = np.linspace(np.log10(tau_min),np.log10(tau_max),nbins)
        VshellnHIGamma1 = np.zeros_like(log_taus)
        nus = 10**np.linspace(np.log10(spectrum.Emin/spectrum.h_P),
                              np.log10(spectrum.Emax/spectrum.h_P),200)  # Hz
        for i,ltau in enumerate(log_taus):
            Itg = lambda x: L_nu(x)*np.exp(-10**ltau)/spectrum.h_P/x 
            VshellnHIGamma1[i] = simpson(Itg(nus), nus)
            # VshellnHIGamma1[i] = quad(Itg, nus[0],nus[-1])
        tck = splrep(log_taus,np.log10(VshellnHIGamma1))
        VshellnHIGamma1_fit = lambda tau0: 10**splev(np.log10(tau0), tck)
        self.VshellnHIGamma1_fit = VshellnHIGamma1_fit
        if self.verbose: print('...done')
        return VshellnHIGamma1_fit

    def get_Gamma_from_fit(self, tau, dtau, nHI, Vshell, **kwargs):
        try:
            VshellnHIGamma1_fit = self.VshellnHIGamma1_fit
        except:
            VshellnHIGamma1_fit = self.create_lookuptable_VshellnHIGamma1(**kwargs)
        return (VshellnHIGamma1_fit(tau)-VshellnHIGamma1_fit(tau+dtau))/(nHI*Vshell) #1/s

    def time_average_ionization_fraction(self, Gamma, ne, x0, Tgas=1e4, dt=None):
        alphaH = self.alphaB(Tgas)
        C_H = 0
        ti  = 1/(Gamma+ne*C_H+ne*alphaH)
        xeq = (Gamma+ne*C_H)/ti
        if dt is None: dt = self.param.RTsolver.dt
        return xeq + (x0-xeq)*(1-np.exp(-dt/ti))*ti/dt

    def solve_1d_t(self, nH_grid, xHII_initial, spectrum=None, count_max=100, x_tol=1e-4, dt=None):
        assert nH_grid.ndim == 1
        if spectrum is None: spectrum = self.spectrum
        L_nu = spectrum.L_nu
        dr_cm = self.param.RTsolver.LB*self.Mpc_to_cm/nH_grid.size
        # NHI_grid = nH_grid*dr_cm # np.cumsum(nH_grid*dr_cm)
        xHII_final = (xHII_initial).copy(); xHII_final[xHII_final<1e-4] = 1e-4
        # tau_nu = lambda nu,N_HI: self.sigma_HI_fit(nu) * N_HI  # dimensionless
        tau_grid = 11.5*np.ones_like(xHII_final) 
        r_grid  = np.arange(n_cells)*dr_cm
        Vs_grid = 4*np.pi/3*((r_grid+dr_cm/2)**3-(r_grid-dr_cm/2)**3)
        get_Gamma_from_fit = self.get_Gamma_from_fit
        time_average_ionization_fraction = self.time_average_ionization_fraction
        for i in range(xHII_final.shape[0]):
            # dtau = (1-xi0)*nHI_grid[i]*self.sigma_HI_fit(nu)*dr_cm
            Vshell = Vs_grid[i]
            count  = 0
            x0 = xHII_final[i]
            x1 = x0*1.1
            while count<count_max and np.abs(x1/x0-1)>x_tol:
                count += 1
                x0    = xHII_final[i]
                nhi0  = (1-x0)*nH_grid[i]
                ne    = x0*nH_grid[i]
                tau0  = tau_grid[:i].sum()
                if tau0<1e-3: tau0 = 1e-3
                dtau0 = tau_grid[i]
                G1 = get_Gamma_from_fit(tau0, dtau0, nhi0, Vshell)
                x1 = time_average_ionization_fraction(G1, ne, x0, Tgas=1e4, dt=dt)
                dtau1 = tau_grid[i]*(1-x1)/(1-x0)
                xHII_final[i] = x1
                tau_grid[i] = dtau1
                # print(x0,x1)
                if count==count_max: 
                    print('Cell number {}'.format(i))
                    print('Calculation is not converged')
        return np.nan_to_num(xHII_final)

    def solve_1d(self, nH_grid, xHII_initial, spectrum=None):
        t_stamps = np.arange(0,param.RTsolver.t_evol,param.RTsolver.dt)+param.RTsolver.dt
        xHIIs = {}
        for ti in t_stamps:
            for dt in np.linspace(0,param.RTsolver.dt,101)[1:]:
                xHII_t = self.solve_1d_t(nH_grid, xHII_initial, spectrum=spectrum, dt=dt)
                xHII_initial = xHII_t
            xHIIs[ti] = xHII_t




