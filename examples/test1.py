import numpy as np 
import radtrans

## Source model
Ndot  = 1e54    #s^1
nHI   = 1.87e-4 #cm^-3
C     = 5
t_evol= 500     #Myr
x_box = 5e24    #cm
Mpc_to_cm = 3.086e24 #cm

## Code setup
n_cells = 256

## one-dimension