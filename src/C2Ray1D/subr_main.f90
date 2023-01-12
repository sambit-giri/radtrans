!>
!! \brief Main program for C2Ray-1D
!!
!! C2Ray-1D does a one-dimensional photo-ionization calculation 
!! one of a series of test problems.\n
!! The main programme calls a number of initialization routines
!! and then enters the main integration loop, which ends when one
!! of the stopping conditions is met. At specified times it calls
!! the output module routines to produce output.
!! After the integration loop ends, a number of closing down routines
!! are called and the programme stops.
!!
!! \b Author: Garrelt Mellema \n
!!
!! \b Date: 10-Jan-2023
!<

module subr_main
contains

subroutine grid_ini_subr(r_in,r_out)

use precision, only: dp
use sizes, only: Ndim, mesh
use astroconstants, only: Mpc

use mathconstants, only: pi
use string_manipulation, only: convert_case
use astroconstants, only: pc,kpc,Mpc,AU

implicit none

real(kind=dp) :: dr !< cell size
real(kind=dp),dimension(mesh) :: r !< spatial cooridnate
real(kind=dp),dimension(mesh) :: vol !< volume of grid cell

integer :: i
real(kind=dp), intent(in) :: r_in,r_out
character(len=10) :: str_length_unit="cm"
real(kind=dp) :: conversion_factor=1.0

! print *, r_in,r_out,str_length_unit

dr=(r_out-r_in)/real(mesh)

! Radial coordinate of a cell
do i=1,mesh
    r(i)=(real(i)-0.5)*dr+r_in
enddo

! Volume of a cell.
do i=1,mesh
    vol(i)=4.0*pi/3.0*((r(i)+0.5*dr)**3-(r(i)-0.5*dr)**3)
    !vol(i)=4.0*pi*r(i)*r(i)*dr
enddo

! print *, vol

end subroutine grid_ini_subr

subroutine mat_ini_subr (restart,testnum,clumping,temper_val,answer,gamma_uvb_h,dens_val_input,r_core)

! Initializes material properties on grid

! Author: Garrelt Mellema

! Date: 20-Aug-2006 (f77 21-May-2005 (derives from mat_ini_cosmo2.f))

! Version: 
! - 1D
! - Four different test problems
! - Initially completely neutral

use precision, only: dp,si
use cgsconstants, only: bh00, albpow
use astroconstants, only: YEAR
use sizes, only: mesh
use file_admin, only: stdinput, file_input
use my_mpi
use grid, only: r,vol
use material, only: find_ionfractions_from_uvb
use cgsconstants, only: m_p
use abundances, only: mu
use cosmology, only: cosmology_init,H0,t0,zred_t0

implicit none

! ndens - number density (cm^-3) of a cell
! temper - temperature (K) of a cell
! xh - ionization fractions for one cell
real(kind=dp) :: ndens(mesh) !< number density (cm^-3) of a cell
real(kind=dp) :: temper(mesh) !< temperature (K) of a cell
real(kind=dp) :: xh(mesh,0:1) !< ionization fractions for one cell
real(kind=dp),intent(in) :: clumping !< global clumping factor
real(kind=dp),intent(in) :: r_core !< core radius (for problems 2 and 3) 
real(kind=dp) :: dens_core !< core density (for problems 2 and 3)
integer, intent(in) :: testnum !< number of test problem (1 to 4)
logical :: isothermal !< is the run isothermal?
real(kind=dp),intent(in) :: gamma_uvb_h
! needed for analytical solution of cosmological Ifront
real(kind=dp) :: t1 !< parameter for analytical solution of test 4 
real(kind=dp) :: eta !< parameter for analytical solution of test 4 

integer,intent(out) :: restart !< will be /= 0 if a restart is intended

integer :: i,n ! loop counters
real(kind=dp) :: dens_val
real(kind=dp),intent(in) :: temper_val
real(kind=dp) :: alpha
real(kind=dp) :: zfactor
real(kind=dp) :: xions
character(len=1),intent(in) :: answer

! Input variables
real(kind=dp) :: dens_val_input

dens_val = dens_val_input

! restart
restart=0 ! no restart by default

! Set alpha according to test problem
select case (testnum)
case(1,4) 
    alpha=0.0
case(2) 
    alpha=-1.0
case(3) 
    alpha=-2.0
end select

    
! For test problem 4: cosmological parameters
if (testnum == 4) then
    call cosmology_init (.true.)
else
    call cosmology_init (.false.)
endif

! Assign density and temperature to grid

select case (testnum)
case(1)
    do i=1,mesh
        ndens(i)=dens_val
        temper(i)=temper_val
    enddo
    
case(2,3)
    dens_core=dens_val
    do i=1,mesh
        !     This is an attempt to make the initial conditions more
        !     correct: the density of a cell is the integrated density
        !     (mass) divided by the volume. Seems to give a worse fit
        !     to the analytical solution (slightly)
        !              rl=r(i)-0.5*dr
        !              rr=r(i)+0.5*dr
        !              ndens(i)=4.0*pi*dens_val*r_core**(-alpha)/vol(i)*
        !     $            (rr**(3.0+alpha)-rl**(3.0+alpha))/(3.0+alpha)
        
        !     This is just a straight sampling of the density distribution
        !     using the value at the cell centre.
        if (testnum == 3 .and. r(i) <= r_core) then
            ! Flat core for test 3
            ndens(i)=dens_val
        else
            ndens(i)=dens_val*(r(i)/r_core)**alpha
        endif
        temper(i)=temper_val
    enddo
    
case(4)
    ! For cosmological simulations, mean IGM
    dens_core=dens_val    ! save initial density value
    ! Parameters needed for the analytical solution
    ! Recombination time at z0
    t1 = 1./(bh00*clumping*dens_core) 
    eta = t0/t1*(1.+zred_t0)**3
    
    ! Set the density to the comoving value 
    ! (as is the spatial coordinate)
    ! evol_cosmo will set it to proper values.
    do i=1,mesh
        ndens(i)=dens_val
        temper(i)=temper_val
    enddo
    dens_val=dens_val*(1.+zred_t0)**3 !otherwise recombination time 
    !scale below would be wrong
end select

! Initialize zfactor for cosmological problems
if (testnum == 4) then
    zfactor=(1.+zred_t0)**3
else
    zfactor=1.0
endif

! Assign ionization fractions
! Use Gamma_UVB_H for this if it is not zero
if (gamma_uvb_h > 0.0) then
    do i=1,mesh
        call find_ionfractions_from_uvb(i, ndens(i), temper(i), xions)
        xh(i,0)=xions
        xh(i,1)=1.0-xh(i,0)
    enddo
else
    do i=1,mesh
        xh(i,0)=1.0-1e-8
        xh(i,1)=1e-8
    enddo
endif

! Report recombination time scale (in case of screen input)
write(*,'(A,1pe10.3,A)') 'Recombination time scale: ', &
        1.0/(dens_val*clumping*bh00*YEAR),' years'

end subroutine mat_ini_subr

end module subr_main