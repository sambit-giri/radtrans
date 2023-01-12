#!/bin/sh
rm -f *.o *~ *.mod fort* *.nfs0* *.so ccrayfortlib.a pyccray.so

gfortran -c -O -fPIC -shared  precision.f90
gfortran -c -O -fPIC -shared  mathconstants.f90
gfortran -c -O -fPIC -shared  cgsconstants.f90
gfortran -c -O -fPIC -shared  cgsphotoconstants.f90
gfortran -c -O -fPIC -shared  cgsastroconstants.f90
gfortran -c -O -fPIC -shared  c2ray_parameters.f90
gfortran -c -O -fPIC -shared  abundances.f90
gfortran -c -O -fPIC -shared  atomic.f90
gfortran -c -O -fPIC -shared  file_admin.f90
gfortran -c -O -fPIC -shared  no_mpi.F90
gfortran -c -O -fPIC -shared  clocks.f90
gfortran -c -O -fPIC -shared  sizes.F90
gfortran -c -O -fPIC -shared  string.f90 
gfortran -c -O -fPIC -shared  grid.F90
gfortran -c -O -fPIC -shared  tped.f90
gfortran -c -O -fPIC -shared  cosmology.F90
gfortran -c -O -fPIC -shared  material.F90
gfortran -c -O -fPIC -shared  cooling.f90
gfortran -c -O -fPIC -shared  romberg.f90
gfortran -c -O -fPIC -shared  radiation.F90
gfortran -c -O -fPIC -shared  thermal.f90
gfortran -c -O -fPIC -shared  time.F90
gfortran -c -O -fPIC -shared  doric.f90
gfortran -c -O -fPIC -shared  photonstatistics.f90
gfortran -c -O -fPIC -shared  cosmological_evolution.f90
gfortran -c -O -fPIC -shared  evolve.F90
gfortran -c -O -fPIC -shared  output.f90
gfortran -c -O -fPIC -shared  C2Ray.F90
gfortran -c -O -fPIC -shared  subr_main.f90
gfortran -c -O -fPIC -shared  controler.f90
ar -rcs libccrayfortlib.a precision.o romberg.o string.o file_admin.o sizes.o no_mpi.o clocks.o grid.o tped.o cosmology.o material.o cooling.o radiation.o thermal.o time.o doric.o photonstatistics.o cosmological_evolution.o  evolve.o output.o C2Ray.o subr_main.o
f2py -c controler.f90 -L. -lccrayfortlib -m pyccray

rm -f *.o *~ *.mod fort* *.nfs0*