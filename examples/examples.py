
import radtrans as rad


'''''''''''
-Spatial gridding :
According to C2ray paper, the Gamma values are converged for a spatial tau resolution such that delta_Tau < 0.1 (1 being okayish).
This gives a prescription for the number of radial bins, given a certain  r_start, r_end . (r_grid is lin spaced)

delta_R = delta_Tau / sigma_HI(13.6) / cm_per_Mpc / (n_H_0*(1+z)**3)       (assuming that sigma_HI(13.6)*n_H_0 is larger than sigma_HeI(13.6) * n_He_0.)
delta_Tau --> precision you choose (0.1 or 1). 
Then :
dn = (r_end-r_start)/delta_R --> yields your dn.

-Time gridding :
parameters.solver.precision controls the degree of convergence to decide when to stop increasing the number of time steps.
If you don't want adaptive time refinement, set this parameter to a very high value (1e10)

-Nt : 
For the initial number of time steps, you can start with the time that a photon takes to go through one cell (times a factor)
c = 9.71561e-12 * sec_per_year * 1e6  # speed of light in kpc/s
frac = 100
Delta_t = (r_end - r_start) * 1e3 / dn /c * frac
Nt = parameters.solver.evol / Delta_t
print(Nt) and see

-r start is by default the halo size at the starting redshift

-r_end : 
The stromgren sphere in a non expanding background can be used.
r_S = (N_gam_dot * 3 / 4/ np.pi/ alpha_B / C**2 / n_H_0**2 )**(1/3) with n_H_0 = n_H(z,C)
Or, knowing Ngamma_dot and t_lifetime,  
your can take (some factor) * (3*Ngdot*tlife*sec_per_year *1e6/4/np.pi/parameters.cosmo.Ob/rho_c/(1+z)**3)**(1.0/3)
with rho_c = 2.775e11 * parameters.cosmo.h**2 * 2e30 / 1.67e-27 ### in Mpc**-3

'''''''''''


parameters = rad.par()
parameters.solver.r_end   = 0.1
parameters.solver.dn_table = 50
parameters.solver.dn =100
parameters.solver.evol =3
parameters.source.lifetime = 3
parameters.solver.z =10
parameters.table.import_table = False


#### Mini qsos model with Mass 1e4, minimum ionizing energy 200
parameters.table.filename_table = 'Miniqsos_1e4' # read or write in it
parameters.source.type = 'Miniqsos'
parameters.source.sed  = 1
parameters.source.M_miniqso  = 1e4
parameters.source.M_halo = 1e6   ## Halo mass is required for the miniqsos (for the bias)

grid_model = rad.Source(parameters) 
grid_model.solve(parameters)


####Galaxy model with Halo Mass 1e8, minimum ionizing energy 200
parameters.table.filename_table = 'Galaxies_1e6'
parameters.source.type = 'Galaxies'
parameters.source.T_gal = 50000 # Kelvins
parameters.source.M_halo = 1e6

grid_model = rad.Source(parameters) 
grid_model.solve(parameters)



