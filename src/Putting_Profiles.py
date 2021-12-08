
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import pickle
from astropy.cosmology import Planck15 as cosmo
from astropy import constants as const
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from astropy.convolution import convolve_fft

def profile_1D(r, c1=2, c2=5):  #
    """
    1D profile, sigmoid function.

    Parameters
    ----------
    r  : the distance from the source [Mpc].
    c1 : shaprness of the profile (sharp = high c1)
    c2 : the ionization front [Mpc]
    """
    out = 1 - 1 / (1 + np.exp(-c1 * (np.abs(r) - c2)))
    return out


## Creating a profile kernel
def profile_to_3Dkernel(profile, nGrid, LB):
    """
    Put profile_1D on a grid

    Parameters
    ----------
    profile  : profile_1D(r, c1=2, c2=5).
    nGrid, LB  : number of grids and boxsize (in Mpc) respectively

    Returns
    -------
    meshgrid of size (nGrid,nGrid,nGrid), with the profile at the center.
    """
    x = np.linspace(-LB / 2, LB / 2, nGrid)  # y, z will be the same.
    rx, ry, rz = np.meshgrid(x, x, x, sparse=True)
    rgrid = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    kern = profile(rgrid)
    return kern

def put_profiles_Middle(profile_kern, nGrid=None):
    """
    Similar to profile_to_3Dkernel.
    """
    if nGrid is None: nGrid = profile_kern.shape[0]
    source_grid = np.zeros((nGrid, nGrid, nGrid))
    source_grid[int(nGrid / 2), int(nGrid / 2), int(nGrid / 2)] = 1

    out = convolve_fft(source_grid, profile_kern, boundary='wrap', normalize_kernel=False)
    #out = fftconvolve(source_grid, profile_kern, mode='same')
    return out


def put_profiles_Sources(out, source_pos, nGrid=None):
    """
    Takes a kernel out (or similarly the output from put_profiles_Middle), and shifts it to center it at source_pos.
    Parameters
    ----------
    Source_pos : Position of the source in grid units [0,nGrid]**3
    Out : Meshgrid. The result of fft convolv.
    """

    if nGrid is None: nGrid = out.shape[0]
    out_ = np.roll(out, shift=int(nGrid / 2 + source_pos[0]), axis=0)
    out_ = np.roll(out_, shift=int(nGrid / 2 + source_pos[1]), axis=1)
    out_ = np.roll(out_, shift=int(nGrid / 2 + source_pos[2]), axis=2)

    return out_





def Spreading_Excess(Grid_Storage):
    """
    Spread the excess photons using scipy.measure.label and distance_transform_edt. For each connected regions, spread the photons to the first closest set of pixels, The last boundary will be filled with an equal fraction of the remaining excess x_ion.
    """
    Grid = np.copy(Grid_Storage)
    nGrid = Grid.shape[0]
    Binary_Grid = np.copy(Grid)
    Binary_Grid[np.where(Grid < 0.999)] = 0
    Binary_Grid[np.where(Grid >= 0.999)] = 1
    connected_regions = label(Binary_Grid)

    Nbr_regions = np.max(connected_regions) + 1

    Grid_of_1 = np.full(((nGrid, nGrid, nGrid)), 1)
    Grid_of_0 = np.zeros((nGrid, nGrid, nGrid))

    # When i = 0, the region if the full region outside the bubbles
    X_Ion_Tot_i = np.sum(Grid)
    print('initial sum of ionized fraction :', np.sum(Grid))

    if X_Ion_Tot_i > Grid.size :
        print('Universe is fully ionized.')
        return 1

    else:
        for i in range(1, Nbr_regions):
            connected_indices = np.where(connected_regions == i)
            Grid_connected = np.copy(Grid_of_0)  ## Grid with the fiducial value only for the region i.
            Grid_connected[connected_indices] = Grid[connected_indices]
            ## take sub grid with only the connected region, find pixels where xion>1, sum the excess, and set these pixels to 1.
            overlap = np.where(Grid_connected > 1)

            excess_ion = np.sum(Grid_connected[overlap] - 1)
            initial_excess = excess_ion
            Grid[overlap] = 1

            Inverted_grid = np.copy(Grid_of_1)
            Inverted_grid[connected_indices] = 0

            sum_distributed_xion = 0
           # print(excess_ion)
            if excess_ion > 1e-7:  ### small value but non zero to avoid doing that step when excess ion is very small
                dist_from_boundary = distance_transform_edt(Inverted_grid)
                dist_from_boundary[np.where(dist_from_boundary == 0)] = 2 * nGrid  ### eliminate pixels inside boundary
                dist_from_boundary[np.where(Grid > 1)] = 2 * nGrid  ### eliminate pixels that already have excess x_ion (belonging to another connected regions..)
                minimum = np.min(dist_from_boundary)
                boundary = np.where(
                    dist_from_boundary == minimum)  # np.where((dist_from_boundary == minimum )& ( Grid<1))

                if np.sum(1 - Grid[boundary]) > excess_ion:  # if their is room for the excess ion,
                    #  you add in each cell a fraction of the neutral fraction available.
                    Grid[boundary] += (1 - Grid[boundary]) * excess_ion / np.sum(1 - Grid[boundary])
                    if np.any(Grid[boundary] > 1):
                        print('x_ion > 1')
                    sum_distributed_xion += excess_ion
                else:

                    while np.sum(1 - Grid[boundary]) < excess_ion:
                        #print('have to go for more than 1 layer')
                        sum_distributed_xion += np.sum(1 - Grid[boundary])
                        excess_ion = excess_ion - np.sum(1 - Grid[boundary])
                        Grid[boundary] = 1
                        dist_from_boundary[boundary] = nGrid * 2  ### exclude this layer for next step
                        minimum = np.min(dist_from_boundary)
                        boundary = np.where(
                            dist_from_boundary == minimum)  ### new closest region to fill with excess ion
                    # you go out of the *while* when np.sum(1 - Grid[boundary]) > excess_ion
                    residual_excess = (1 - Grid[boundary]) * excess_ion / np.sum(1 - Grid[boundary])
                    Grid[boundary] += residual_excess
                    sum_distributed_xion += excess_ion

                    if np.any(Grid[boundary] > 1):
                        print('x_ion > 1 at the end of the process', aaaa)
                        break

        if np.any(Grid > 1):
            print('3. x_ion > 1 ')

        print('final xion sum: ', np.sum(Grid))
        X_Ion_Tot_f = np.sum(Grid)
        if int(X_Ion_Tot_f) != int(X_Ion_Tot_i):
            print('smtg is wrong when spreading xion_excess.')

    return Grid




def Spreading_Excess_HR(Grid_Storage):
    """
    Same function than *Spreading_Excess*, but we add a step to speed up the procedure (relevant when choosing HR Grids (>=256**3)) : for each connected region we run distance_transform only for a subbox centered on the connected region.
    The size of the sub-boxes is N_subgrid.

    """
    Grid = np.copy(Grid_Storage)
    nGrid = Grid.shape[0]
    Binary_Grid = np.copy(Grid)
    Binary_Grid[np.where(Grid < 0.999)] = 0
    Binary_Grid[np.where(Grid >= 0.999)] = 1
    connected_regions = label(Binary_Grid)

    Nbr_regions = np.max(connected_regions) + 1

    Grid_of_1 = np.full(((nGrid, nGrid, nGrid)), 1)
    # When i = 0, the region if the full region outside the bubbles
    X_Ion_Tot_i = np.sum(Grid)
    print('initial sum of ionized fraction :', np.sum(Grid))
    print(Nbr_regions, 'connected regions.')

    if X_Ion_Tot_i > Grid.size:
        print('Universe is fully ionized.')
        Grid = 1

    else:
        for i in range(1, Nbr_regions):

            connected_indices = np.where(connected_regions == i)
            overlap = np.where(Grid[connected_indices] > 1)[0]
            initial_excess = np.sum(Grid[connected_indices][overlap] - 1)
            Grid[connected_indices] = np.where(Grid[connected_indices] > 1, 1, Grid[connected_indices])
            excess_ion = initial_excess

            if excess_ion > 1e-8:

                Inverted_grid = np.copy(Grid_of_1)
                Inverted_grid[connected_indices] = 0
                sum_distributed_xion = 0

                Delta_pixel = int(excess_ion ** (1. / 3) / 2) + 1

                Min_X, Max_X = np.min(connected_indices[0]), np.max(connected_indices[0])
                Min_Y, Max_Y = np.min(connected_indices[1]), np.max(connected_indices[1])
                Min_Z, Max_Z = np.min(connected_indices[2]), np.max(connected_indices[2])
                Delta_max = np.max((Max_X - Min_X + 0, Max_Y - Min_Y + 0, Max_Z - Min_Z + 0))
                Center_X, Center_Y, Center_Z = int((Min_X + Max_X) / 2), int((Min_Y + Max_Y) / 2), int(
                    (Min_Z + Max_Z) / 2)

                N_subgrid = 2 * (Delta_max + 2 * Delta_pixel)  ## length of subgrid embedding the connected region
                if N_subgrid % 2 == 1:
                    N_subgrid += 1  ###Nsubgrid needs to be even to make things easier

                if N_subgrid > nGrid:
                    dist_from_boundary = distance_transform_edt(Inverted_grid)
                    dist_from_boundary[
                        np.where(dist_from_boundary == 0)] = 2 * nGrid  ### eliminate pixels inside boundary
                    dist_from_boundary[np.where(
                        Grid > 1)] = 2 * nGrid  ### eliminate pixels that already have excess x_ion (belonging to another connected regions..)
                    minimum = np.min(dist_from_boundary)
                    boundary = np.where(
                        dist_from_boundary == minimum)  # np.where((dist_from_boundary == minimum )& ( Grid<1))

                    while np.sum(1 - Grid[boundary]) < excess_ion:
                        sum_distributed_xion += np.sum(1 - Grid[boundary])
                        excess_ion = excess_ion - np.sum(1 - Grid[boundary])
                        Grid[boundary] = 1
                        dist_from_boundary[boundary] = nGrid * 2  ### exclude this layer for next step
                        minimum = np.min(dist_from_boundary)
                        boundary = np.where(dist_from_boundary == minimum)  ### new closest region to fill with excess ion
                        # you go out of the *while* when np.sum(1 - Grid[boundary]) > excess_ion
                    residual_excess = (1 - Grid[boundary]) * excess_ion / np.sum(1 - Grid[boundary])
                    Grid[boundary] += residual_excess
                    sum_distributed_xion += excess_ion



                else:

                    Sub_Grid = np.full(((N_subgrid, N_subgrid, N_subgrid)), 0)

                    Sub_Grid = Sub_Grid.astype('float64')

                    Sub_Grid[:] = Grid[np.max((Center_X - int(N_subgrid / 2), 0)) - np.max(
                        (0, Center_X + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                        (nGrid, Center_X + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_X)),
                                  np.max((Center_Y - int(N_subgrid / 2), 0)) - np.max(
                                      (0, Center_Y + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                                      (nGrid, Center_Y + int(N_subgrid / 2) + 0)) + np.max(
                                      (0, int(N_subgrid / 2) - Center_Y)),
                                  np.max((Center_Z - int(N_subgrid / 2), 0)) - np.max(
                                      (0, Center_Z + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                                      (nGrid, Center_Z + int(N_subgrid / 2) + 0)) + np.max(
                                      (0, int(N_subgrid / 2) - Center_Z))]

                    while np.sum(1 - Sub_Grid) < excess_ion:  ### for very small regions there might be no room for excess ion.
                        N_subgrid = N_subgrid + 2
                        Sub_Grid = np.full(((N_subgrid, N_subgrid, N_subgrid)), 0)
                        Sub_Grid = Sub_Grid.astype('float64')
                        Sub_Grid[:] = Grid[np.max((Center_X - int(N_subgrid / 2), 0)) - np.max(
                            (0, Center_X + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                            (nGrid, Center_X + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_X)),
                                      np.max((Center_Y - int(N_subgrid / 2), 0)) - np.max(
                                          (0, Center_Y + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                                          (nGrid, Center_Y + int(N_subgrid / 2) + 0)) + np.max(
                                          (0, int(N_subgrid / 2) - Center_Y)),
                                      np.max((Center_Z - int(N_subgrid / 2), 0)) - np.max(
                                          (0, Center_Z + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                                          (nGrid, Center_Z + int(N_subgrid / 2) + 0)) + np.max(
                                          (0, int(N_subgrid / 2) - Center_Z))]

                    Sub_Inverted_Grid = np.full(((N_subgrid, N_subgrid, N_subgrid)), 1)
                    Sub_Inverted_Grid = Sub_Inverted_Grid.astype('float64')
                    Sub_Inverted_Grid[:] = Inverted_grid[np.max((Center_X - int(N_subgrid / 2), 0)) - np.max(
                        (0, Center_X + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                        (nGrid, Center_X + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_X)),
                                           np.max((Center_Y - int(N_subgrid / 2), 0)) - np.max(
                                               (0, Center_Y + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                                               (nGrid, Center_Y + int(N_subgrid / 2) + 0)) + np.max(
                                               (0, int(N_subgrid / 2) - Center_Y)),
                                           np.max((Center_Z - int(N_subgrid / 2), 0)) - np.max(
                                               (0, Center_Z + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                                               (nGrid, Center_Z + int(N_subgrid / 2) + 0)) + np.max(
                                               (0, int(N_subgrid / 2) - Center_Z))]

                    Sub_Grid_Initiale = np.copy(Sub_Grid)

                    dist_from_boundary = distance_transform_edt(Sub_Inverted_Grid)
                    dist_from_boundary[
                        np.where(dist_from_boundary == 0)] = 2 * N_subgrid  ### eliminate pixels inside boundary
                    dist_from_boundary[np.where(
                        Sub_Grid > 1)] = 2 * N_subgrid  ### eliminate pixels that already have excess x_ion (belonging to another connected regions..)
                    minimum = np.min(dist_from_boundary)
                    boundary = np.where(dist_from_boundary == minimum)
                    #

                    excess_ion_i = excess_ion
                    while np.sum(1 - Sub_Grid[boundary]) < excess_ion:
                        sum_distributed_xion += np.sum(1 - Sub_Grid[boundary])
                        excess_ion = excess_ion - np.sum(1 - Sub_Grid[boundary])
                        Sub_Grid[boundary] = 1
                        dist_from_boundary[boundary] = N_subgrid * 2  ### exclude this layer for nex
                        minimum = np.min(dist_from_boundary)
                        boundary = np.where(dist_from_boundary == minimum)  ### new closest region to fill
                    # you go out of the *while* when np.sum(1 - Grid[boundary]) > excess_ion

                    residual_excess = (1 - Sub_Grid[boundary]) * excess_ion / np.sum(1 - Sub_Grid[boundary])

                    Sub_Grid[boundary] = np.add(Sub_Grid[boundary], residual_excess)
                    sum_distributed_xion += excess_ion

                    Grid[np.max((Center_X - int(N_subgrid / 2), 0)) - np.max(
                        (0, Center_X + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                        (nGrid, Center_X + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_X)),
                    np.max((Center_Y - int(N_subgrid / 2), 0)) - np.max(
                        (0, Center_Y + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                        (nGrid, Center_Y + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_Y)),
                    np.max((Center_Z - int(N_subgrid / 2), 0)) - np.max(
                        (0, Center_Z + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                        (nGrid, Center_Z + int(N_subgrid / 2) + 0)) + np.max(
                        (0, int(N_subgrid / 2) - Center_Z))] = Sub_Grid[:]

                    if np.any(Sub_Grid[boundary] > 1) or np.any(np.isnan(Sub_Grid[boundary])):
                        print('2. xion>1')

                    #if round(np.sum(Sub_Grid)) != int(np.sum(Sub_Grid_Initiale) + excess_ion_i):
                    if round(np.sum(Sub_Grid) / int(np.sum(Sub_Grid_Initiale) + excess_ion_i)) !=1 :
                    ### just a trick to avoid exiting when half a photon is lost...
                        print('loosing photons')
                        exit()


        if np.any(Grid > 1):
            print('3. xion>1.')

        print('final xion sum: ', np.sum(Grid))
        X_Ion_Tot_f = np.sum(Grid)
        if int(X_Ion_Tot_f) != int(X_Ion_Tot_i):
            print('smtg is wrong when spreading xion_excess.')


    return Grid


'''''''''''

### Toy model to run :
Halo_File = pickle.load(open('./Snapshots/FastPM_512_B100_z5/halos/fof0.1685','rb'))
H_Masses = Halo_File['Masses']
H_positions =  Halo_File['Position']
H_Radii =  Halo_File['Radius']
#H_positions[:,0] = H_positions[:,0] *3 ### shift the positions for weird Fast PM halo catalogs
#H_positions[:,0][np.where(H_positions[:,0]>100)] = H_positions[:,0][np.where(H_positions[:,0]>100)]/3

R_Bubbles = H_Radii * 100
Pos_Bubles = H_positions


#Bin halo masses, assuming same profile for halo in each bin.
M_min = np.min(H_Masses)
M_max = np.max(H_Masses)
M_binning = np.logspace(np.log10(M_min), np.log10(M_max), 10, base=10)
Indexing = np.digitize(H_Masses, M_binning)
nGrid, LB = 64, 100

Pos_Bubbles_Grid = np.array([Pos_Bubles / LB * nGrid]).astype(int)[0]
Grid = np.zeros((nGrid, nGrid, nGrid))


# Put profiles on the grid, with overlapp
for i in range(len(M_binning)):
    indices = np.where(Indexing == i)[0]
    R_mean = np.mean(R_Bubbles[indices])
    print(R_mean)
    profile = lambda x: profile_1D(x, c2=R_mean)
    kernel = profile_to_3Dkernel(profile, nGrid, LB)
    Mid_profile = put_profiles_Middle(kernel, nGrid=nGrid)

    for p_ in range(len(indices)):
        if p_ < 700:
            Grid += put_profiles_Sources(Mid_profile, Pos_Bubbles_Grid[indices[p_]], nGrid=None)
        else:
            break

Grid_Storage = np.copy(Grid)
Grid = np.copy(Grid_Storage)

Binary_Grid = np.copy(Grid)
Binary_Grid[np.where(Grid<0.999)] = 0
Binary_Grid[np.where(Grid>=0.999)] = 1
connected_regions = label(Binary_Grid)

Nbr_regions = np.max(connected_regions) + 1

Grid_of_1 = np.full(((nGrid, nGrid, nGrid)) ,1)
Grid_of_0 = np.zeros((nGrid, nGrid, nGrid))



print('initial sum of ionized fraction :', np.sum(Grid))

for i in range(1, Nbr_regions):
    connected_indices = np.where(connected_regions == i)
    Grid_connected = np.copy(Grid_of_0)  ## Grid with the fiducial value only for the region i.
    Grid_connected[connected_indices] = Grid[connected_indices]
    ## take sub grid with only the connected region, find pixels where xion>1, sum the excess, and set these pixels to 1.
    overlap = np.where(Grid_connected > 1)

    excess_ion = np.sum(Grid_connected[overlap] - 1)
    initial_excess = excess_ion
    Grid[overlap] = 1

    Inverted_grid = np.copy(Grid_of_1)
    Inverted_grid[connected_indices] = 0
    Inverted_grid[np.where(Grid > 1)] = 0  # We do not want to touch the regions that already have excess xion.
    # So we exclud them by putting there distance to zero like this..

    sum_distributed_xion = 0

    # print('seum:',np.sum(Grid))

    if excess_ion > 0:
        dist_from_boundary = distance_transform_edt(Inverted_grid)
        dist_from_boundary[np.where(dist_from_boundary == 0)] = 2 * nGrid  ### eliminate pixels inside boundary
        minimum = np.min(dist_from_boundary)
        boundary = np.where(dist_from_boundary == minimum)  # np.where((dist_from_boundary == minimum )& ( Grid<1))

        if np.sum(1 - Grid[boundary]) > excess_ion:  # if their is room for the excess ion,
            ### then you add in each cell a fraction the neutral fraction available.
            Grid[boundary] += (1 - Grid[boundary]) * excess_ion / np.sum(1 - Grid[boundary])
            if np.any(Grid[boundary] > 1):
                print('1. Thats where we trigger')
            sum_distributed_xion += excess_ion
        else:
            # print('has to go for more than 1 layer')
            while np.sum(1 - Grid[boundary]) < excess_ion:
                # print('yes')
                sum_distributed_xion += np.sum(1 - Grid[boundary])  # np.minimum(1,Grid[boundary]) )
                excess_ion = excess_ion - np.sum(1 - Grid[boundary])  # np.minimum(1,Grid[boundary]) )
                Grid[boundary] = 1
                dist_from_boundary[boundary] = nGrid * 2  ##to be sure to be out of reach
                minimum = np.min(dist_from_boundary)
                boundary = np.where(
                    dist_from_boundary == minimum)  # np.where((dist_from_boundary == minimum) & (Grid<1) )

            # you go out of the while when     excess_ion < bound_len
            aaaa = (1 - Grid[boundary]) * excess_ion / np.sum(1 - Grid[boundary])  # np.minimum(1,Grid[boundary]))
            Grid[boundary] += aaaa
            sum_distributed_xion += excess_ion

            if np.any(Grid[boundary] > 1):
                print('2. Thats where we trigger', aaaa)
                break

    if np.any(Grid[boundary] > 1):
        print('okay thats it')
        # print('we distributed : ',sum_distributed_xion==initial_excess)

print('i: ', i, 'sum: ', np.sum(Grid))

'''''''''''




