import numpy as np 
import matplotlib.pyplot as plt 
import radtrans

h_P  = radtrans.constants.h_eV_sec # eV s
E_HI = radtrans.constants.E_HI     # eV
nuHI = E_HI/h_P                    # Hz
nu_max = 200/h_P
nus = 10**np.linspace(10,20,200)   # Hz

T1 = 1e5      # K
L1 = 1.9e31   # 1/eV^2
N1 = 5e48     # 1/s
S1 = radtrans.SourceSpectrum()
S1.black_body(T1, L0=L1)
s1 = radtrans.SourceSpectrum()
s1.black_body(T1, Ndot=N1)

T2 = 1e6      # K
L2 = 1.9e27   # 1/eV^2
N2 = 7e47     # 1/s
S2 = radtrans.SourceSpectrum()
S2.black_body(T2, L0=L2)
s2 = radtrans.SourceSpectrum(verbose=True)
s2.black_body(T2, Ndot=N2)

fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.loglog(nus, S1.L_nu(nus), ls='-', label='T={:.0e}'.format(T1))
ax.loglog(nus, s1.L_nu(nus), ls='--')
ax.loglog(nus, S2.L_nu(nus), ls='-', label='T={:.0e}'.format(T2))
ax.loglog(nus, s2.L_nu(nus), ls='--')
ax.axvline(nuHI, label='H ionisation', c='k', ls='--')
ax.axvline(nu_max, label='$\\nu_\mathrm{max}$', c='k', ls='-.')
ax.set_ylim(bottom=1)
# ax.axis([1e10,1e15,1e5,1e27])
plt.legend()
plt.show()