import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from astropy import constants as const

## Simulation setting
LB = 100  # in Mpc
nGrid = 256
z = 9


# ## Number density of H
# def f_nH_1D(x, z):
# 	out = (3.2/u.cm**3*(91.5*u.pc/x)**2).to('cm^-3')
# 	mean_rho_b = cosmo.Ob(z)*cosmo.critical_density(z)
# 	mean_nH = (mean_rho_b/(const.m_p+const.m_e)).to('cm^-3')       # All the gas is hydrogen
# 	if type(out.value)==np.ndarray:
# 		out[out<mean_nH] = mean_nH
# 	else:
# 		if out<mean_nH: out = mean_nH
# 	return out

# rs = 10**np.linspace(-2,np.log10(LB.value*np.sqrt(3)),200)*LB.unit

## Assuming a Gaussian profile

def profile_1D(r, A=1, c1=0.5, c2=5,  sigma=5): #
    '''
    r is the distance from the source in Mpc.
    c1 controls the shaprness of the profile (sharp = high c1)
    c2 is  the ionization front
    '''
    #out = np.exp(-r ** 2 / sigma ** 2)
    out = 1-1/(1+np.exp(-c1*(np.abs(r)-c2)))
    return out * A


rs = 10 ** np.linspace(-2, np.log10(LB), 200)

plt.figure()
plt.plot(rs, profile_1D(rs, sigma=10))
plt.plot(rs, profile_1D(rs, sigma=5), '--')
plt.xlabel('r (in Mpc)')
plt.ylabel('profiles')
plt.show()


## Creating a profile kernel

def profile_to_3Dkernel(profile, nGrid, LB):
    '''''''''
    Profile, nGrid and LB are profile function, number of grids and boxsize (in Mpc) respectively.
    '''''''''
    # out = np.zeros((nGrid,nGrid,nGrid))
    x = np.linspace(-LB / 2, LB / 2, nGrid)  # y, z will be the same.
    rx, ry, rz = np.meshgrid(x, x, x, sparse=True)
    rgrid = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    kern = profile(rgrid)
    return kern


## Putting profile near sources

def put_profiles(source_pos, profile_kern, nGrid=None):
    '''
    source_pos and profile_kern are source positions and profile_kern respectively.
    '''
    if nGrid is None: nGrid = profile_kern.shape[0]
    source_grid = np.zeros((nGrid, nGrid, nGrid))
    for i, j, k in source_pos:
        source_grid[i, j, k] = 1
    out = fftconvolve(source_grid, profile_kern, mode='same')
    return out


## Toy case

### 1 source

source_pos = np.array([[128, 128, 128]])
profile1 = lambda x: profile_1D(x, sigma=10)
profile_kern = profile_to_3Dkernel(profile1, nGrid, LB)
arr = put_profiles(source_pos, profile_kern)

plt.figure(figsize=(9, 5))
plt.suptitle('1 source at the centre')
plt.subplot(121)
plt.imshow(arr[:, :, 128])
plt.subplot(122)
rx = np.linspace(0, LB, nGrid)
plt.plot(rx, arr[:, 128, 128], label='from our grid')
plt.plot(rx, profile1(rx - LB / nGrid * source_pos[0][0]), label='from profile function', ls='--')
plt.ylim(-0.1, 1.5)
plt.legend()
plt.show()

source_pos = np.array([
    [70, 128, 128]
])
profile1 = lambda x: profile_1D(x, sigma=10)
profile_kern = profile_to_3Dkernel(profile1, nGrid, LB)
arr = put_profiles(source_pos, profile_kern)

plt.figure(figsize=(9, 5))
plt.suptitle('1 source closer to an edge')
plt.subplot(121)
plt.imshow(arr[:, :, 128])
plt.subplot(122)
rx = np.linspace(0, LB, nGrid)
plt.plot(rx, arr[:, 128, 128], label='from our grid')
plt.plot(rx, profile1(rx - LB / nGrid * source_pos[0][0]), label='from profile function', ls='--')
plt.ylim(-0.1, 1.5)
plt.legend()
plt.show()

### 2 source

source_pos = np.array([
    [70, 128, 128],
    [130, 128, 128]
])
profile1 = lambda x: profile_1D(x, sigma=10)
profile_kern = profile_to_3Dkernel(profile1, nGrid, LB)
arr = put_profiles(source_pos, profile_kern)

plt.figure(figsize=(9, 5))
plt.suptitle('2 sources with same profile')
plt.subplot(121)
plt.imshow(arr[:, :, 128])
plt.subplot(122)
rx = np.linspace(0, LB, nGrid)
prof1 = profile1(rx - LB / nGrid * source_pos[0][0])
prof2 = profile1(rx - LB / nGrid * source_pos[1][0])
plt.plot(rx, arr[:, 128, 128], label='from our grid')
plt.plot(rx, prof1, label='from profile function, source 1', ls='--')
plt.plot(rx, prof2, label='from profile function, source 2', ls='--')
plt.plot(rx, prof1 + prof2, label='from profile function', ls=':')
plt.ylim(-0.1, 1.5)
plt.legend()
plt.show()

### 3 source

source_pos1 = np.array([
    [70, 128, 128],
    [130, 128, 128]
])
profile1 = lambda x: profile_1D(x, sigma=10)
profile_kern = profile_to_3Dkernel(profile1, nGrid, LB)
arr1 = put_profiles(source_pos1, profile_kern)

source_pos2 = np.array([
    [180, 128, 128]
])
profile2 = lambda x: profile_1D(x, sigma=5)
profile_kern = profile_to_3Dkernel(profile2, nGrid, LB)
arr2 = put_profiles(source_pos2, profile_kern)

arr = arr1 + arr2

plt.figure(figsize=(6, 5))
plt.suptitle('sources 1,2 - profile 1; source 3 - profile 2')
plt.imshow(arr[:, :, 128])
plt.show()
