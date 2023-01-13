!>
!! \brief This module contains routines that interfaces with python.
!!
!! Module for Capreole / C2-Ray (f90)
!!
!! \b Author: Sambit K. Giri
!!
!! \b Date: 10 January 2023
!!
!! \b Version: 

subroutine grid_ini_fn(r_in_cm,r_out_cm, dr)
! =====================================================
! Creates the initial 1D grid for the simulation
! =====================================================
    use subr_main
    use precision, only: dp
    use sizes, only: Ndim, mesh

    implicit none

    real(kind=8), intent(in) :: r_in_cm,r_out_cm

    real(kind=dp), intent(out) :: dr !< cell size
    ! real(kind=dp),dimension(mesh) :: r !< spatial cooridnate
    ! real(kind=dp),dimension(mesh) :: vol !< volume of grid cell

    call grid_ini_subr(r_in_cm,r_out_cm, dr)
    
end subroutine

subroutine mat_ini_fn(restart,testnum,clumping,temper_val,isotherm_answer,gamma_uvb_h,dens_val_input,r_core)
    use subr_main
    use precision, only: dp

    implicit none

    integer, intent(in) :: testnum !< number of test problem (1 to 4)
    real(kind=8), intent(in) :: clumping !< global clumping factor
    real(kind=8), intent(in) :: temper_val, dens_val_input, r_core
    real(kind=8), intent(in) :: gamma_uvb_h
    character(len=1), intent(in) :: isotherm_answer

    integer, intent(out) :: restart !< restart if not zero (not used in 1D code)

    call mat_ini_subr (restart,testnum,clumping,temper_val,isotherm_answer,gamma_uvb_h,dens_val_input,r_core)
    
end subroutine

