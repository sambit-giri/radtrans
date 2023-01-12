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

module C2Ray_routines
contains

SUBROUTINE wrapper()
    

    call setup_output ()
end SUBROUTINE wrapper

end module C2Ray_routines