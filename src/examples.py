
import radtrans as rad

parameters = rad.par()
parameters.solver.r_start = 0.0001
parameters.solver.r_end   = 2
parameters.solver.dn =10
parameters.solver.evol =3
parameters.solver.z =10
parameters.table.import_table = False


#### Mini qsos model with Mass 1e4, minimum ionizing energy 200
parameters.table.filename_table = 'Miniqsos_1e4_E0_200' # read or write in it
parameters.source.E_0 = 200
parameters.source.type = 'Miniqsos'
parameters.source.sed  = 1 

grid_model = rad.Source(parameters) 
grid_model.solve(parameters)


####Galaxy model with Halo Mass 1e8, minimum ionizing energy 200
parameters.table.filename_table = 'Galaxies_1e8_E0_200' 
parameters.source.E_0 = 200
parameters.source.type = 'Galaxies'
parameters.source.T_gal = 50000 # Kelvins
parameters.source.M_halo = 1e8

grid_model = rad.Source(parameters) 
grid_model.solve(parameters)



