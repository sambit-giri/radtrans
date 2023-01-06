import numpy as np 
from scipy.integrate import quad, simpson
from scipy.interpolate import splev, splrep

from . import constants as const

class SourceSpectrum:
    '''
    The SourceSpectrum object contains several models to describe the photon sources.

    Args:
        spect_type (str): Define the name of source. 
        verbose (bool): Set to True if you want to print verbose.

    Attributes:
        spect_name (str): The source name is stored here.
        k_B (float): Boltzmann constant.
        h_P (float): Planck constant.
        Emax (float): Maximum energy to normalise the spectrum.
    '''
    def __init__(self, spect_name='BB', verbose=False, Emax=200):
        self.spect_name = spect_name
        self.verbose    = verbose
        self.k_B = const.kb_eV_per_K  # [eV/K]
        self.h_P = const.h_eV_sec     # [eV.s]
        self.Emax = Emax              # eV

    def test_source_model(self, L_nu=None):
        '''
        The SourceSpectrum object contains several models to describe the photon sources.

        Example:
            S = SourceSpectrum()
            TBB = 1e5      # K
            L0  = 1.9e31   # 1/eV^2
            Lnu = S.black_body(TBB, L0=L0)
            S.test_source_model(Lnu)

        Args:
            L_nu (str): Function modelling the luminosity distribution (in eV/s/Hz) of a black-body source. 

        Attributes:
            
        '''
        if L_nu is None: L_nu = self.L_nu
        h_P = self.h_P
        nuHI   = const.E_HI/h_P   # Hz
        nu_max = self.Emax/h_P
        Ndot_est = self.spectrum_to_Ndot(L_nu=L_nu)
        if self.verbose:
            print('Estimated photon emission rate = {:.1e} s^-1'.format(Ndot_est))

        # Plot
        nus = 10**np.linspace(10,20,200)  # Hz
        import matplotlib.pyplot as plt 
        fig, ax = plt.subplots(1,1,figsize=(6,5))
        ax.loglog(nus, L_nu(nus), ls='-', c='k', label='Spectrum')
        ax.axvline(nuHI, label='H ionisation', ls='--')
        ax.axvline(nu_max, label='$\\nu_\mathrm{max}$', ls='-.')
        ax.set_ylim(bottom=1)
        # ax.axis([1e10,1e15,1e5,1e27])
        plt.legend()
        plt.show()

    def spectrum_to_Ndot(self, L_nu=None):
        h_P = self.h_P
        nuHI = const.E_HI/h_P   # Hz
        nu_max = 200/h_P
        if L_nu is None: L_nu = self.L_nu 
        Itg = lambda x: L_nu(x)/h_P/x
        # nus = 10**np.linspace(np.log10(nuHI),np.log(nu_max),100)  # Hz
        # Ndot_est = simpson(Itg(nus), nus)
        Ndot_est = quad(Itg, nuHI, nu_max)[0]
        if self.verbose: 
            print('Ionisation threshold of hydrogen: E={:.1f}eV and nu={:.1e}Hz'.format(const.E_HI,nuHI))
            print('Photon emission rate: {:.2e} s^-1'.format(Ndot_est))
        return Ndot_est

    def black_body(self, T_BB, L0=1, Ndot=None):
        '''Black Body spectrum. 
        The 10^5 K black body roughly represents a high mass Pop II or Pop III star (Bond et al. 1984), 
        whilst the 10^6 K black body (hotter than expected for stellar atmospheres), is presented as a 
        contrasting source with a greater proportion of ionizing photons able to fully ionize helium. 

        Args:
            T_BB (float): The temperature of the black-body spectrum in Kelvin or with astropy unit.
            L0 (float)  : Normalisation coefficient in 1/eV^2 or with astropy unit.
            Ndot (float): Photon emission rate above the hydrogen ionization threshold in 1/s or in astropy unit.

        Returns:
            L_nu: Function modelling the source luminosity distribution (in eV/s/Hz). 
        '''
        assert L0 is not None or Ndot is not None
        try: self.T_BB = T_BB.to('K').value # K
        except: self.T_BB = T_BB            # K
        k_B, h_P, T_BB = self.k_B, self.h_P, self.T_BB
        if self.verbose:
            from astropy import units as u
            b = 2.898e-3*u.m*u.K # m K
            l_peak  = (b/T_BB/u.K).to('nm')
            nu_peak = (3e8*u.m/u.s/l_peak).to('Hz')
            print('Peak wavelength = {:.2f}'.format(l_peak))
            print('Peak frequency = {:.2e}'.format(nu_peak))
        L_nu = lambda nu: L0*(h_P*nu)**3/(np.exp(h_P*nu/k_B/T_BB)-1) # eV/s/Hz
        if Ndot is not None: 
            Ndot1 = self.spectrum_to_Ndot(L_nu=L_nu)
            L0 = L0 * Ndot/Ndot1
            self.L0, self.Ndot = L0, Ndot
            if self.verbose:
                print('Model spectrum normalised to emit {} photons per second between 13.6 eV and {:.1f} eV.'.format(self.Ndot,self.Emax))
        else:
            self.Ndot = self.spectrum_to_Ndot(L_nu=L_nu)
        self.L_nu = L_nu
        L_E = lambda E: L0*E**3/(np.exp(E/k_B/T_BB)-1) # eV/s/Hz
        self.L_E = L_E
        return L_nu

    def power_law(self, a_S=0.5, L_L=0.56e31, nu_L=3.29e15):
        '''Power law. 
        The power-law spectra represent QSOs, although they also approximate the photon emission rates 
        for galaxies dominated by Pop II or Pop III stars (Meiksin 2005). 

        Args:
            a_S (float) : The power index of the spectrum (Default: 0.5).
            L_L (float) : Normalisation coefficient in erg/s/Hz or with astropy unit.
            nu_L (float): the ionization threshold in Hz or in astropy unit.

        Returns:
            L_nu: Function modelling the source luminosity distribution (in eV/s/Hz). 
        '''
        self.a_S = a_S
        try: self.L_L = L_L.to('erg/s/Hz').value # erg/s/Hz
        except: self.L_L = L_L                   # erg/s/Hz
        try: self.nu_L = nu_L.to('Hz').value # Hz
        except: self.nu_L = nu_L             # Hz
        k_B, h_P = self.k_B, self.h_P
        L_nu = lambda nu: self.L_L*(nu/self.nu_L)**(-self.a_S) # erg/s/Hz
        # self.Ndot = self.spectrum_to_Ndot(L_nu=L_nu)
        self.L_nu = L_nu
        L_E = lambda E: self.L_L*(E/self.nu_L/h_P)**(-self.a_S) # erg/s/Hz
        self.L_E = L_E
        return L_nu

