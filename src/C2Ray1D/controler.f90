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


subroutine sieve(is_prime, n_max)
! =====================================================
! Uses the sieve of Eratosthenes to compute a logical
! array of size n_max, where .true. in element i
! indicates that i is a prime.
! =====================================================
    integer, intent(in)   :: n_max
    logical, intent(out)  :: is_prime(n_max)
    integer :: i
    is_prime = .true.
    is_prime(1) = .false.
    do i = 2, int(sqrt(real(n_max)))
        if (is_prime (i)) is_prime (i * i : n_max : i) = .false.
    end do
    return
end subroutine

subroutine logical_to_integer(prime_numbers, is_prime, num_primes, n)
! =====================================================
! Translates the logical array from sieve to an array
! of size num_primes of prime numbers.
! =====================================================
    integer                 :: i, j=0
    integer, intent(in)     :: n
    logical, intent(in)     :: is_prime(n)
    integer, intent(in)     :: num_primes
    integer, intent(out)    :: prime_numbers(num_primes)
    do i = 1, size(is_prime)
        if (is_prime(i)) then
            j = j + 1
            prime_numbers(j) = i
        end if
    end do
end subroutine


subroutine grid_ini_fn(r_in_cm, r_out_cm)
    use subr_main
    use precision, only: dp
    use sizes, only: Ndim, mesh

    implicit none

    real(kind=8), intent(in) :: r_in_cm,r_out_cm

    call grid_ini_subr(r_in_cm,r_out_cm)
    
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

subroutine run_inputfile(inputfile,outfile)
    ! Needs following modules
    use precision, only: dp
    use clocks, only: setup_clocks, update_clocks, report_clocks
    use file_admin, only: stdinput, logf, file_input, flag_for_file_input
    use astroconstants, only: YEAR
    use my_mpi, only: mpi_setup, mpi_end, rank
    use output_module, only: setup_output,output,close_down
    use grid, only: grid_ini
    use subr_main
    use radiation, only: rad_ini
    use cosmology, only: cosmology_init, redshift_evol, &
        time2zred, zred2time, zred, cosmological
    use cosmological_evolution, only: cosmo_evol
    use material, only: mat_ini, testnum
    use times, only: time_ini, end_time,dt,output_time
    use evolve, only: evolve1D

    implicit none

    ! Integer variables
    integer :: nstep !< time step counter
    integer :: restart !< restart if not zero (not used in 1D code)

    ! Time variables
    real(kind=dp) :: sim_time !< actual time (s)
    real(kind=dp) :: next_output_time !< time of next output (s)
    real(kind=dp) :: actual_dt !< actual time step (s)

    !> Input file
    character(len=512), intent(in)  :: inputfile
    character(len=512), intent(out) :: outfile
    outfile = inputfile


    ! Initialize clocks (cpu and wall)
    call setup_clocks

    ! Set up MPI structure (compatibility mode) & open log file
    call mpi_setup()

    ! Set up input stream (either standard input or from file given
    ! by first argument)

    write(logf,*) "reading input from ",trim(adjustl(inputfile))
    open(unit=stdinput,file=inputfile,status="old")
    call flag_for_file_input(.true.)

    ! Initialize output
    call setup_output ()

    ! Initialize grid
    call grid_ini ()

    ! Initialize the material properties
    call mat_ini (restart)

    ! Initialize photo-ionization calculation
    call rad_ini( )

    ! Initialize time step parameters
    call time_ini ()

    ! Set time to zero
    sim_time=0.0
    next_output_time=0.0

    ! Update cosmology (transform from comoving to proper values)
    if (cosmological) then
        call redshift_evol(sim_time)
        call cosmo_evol( )
        !write(*,*) zred
    endif

    ! Loop until end time is reached
    nstep=0
    do

        ! Write output
        if (abs(sim_time-next_output_time) <= 1e-6*sim_time) then
        call output(nstep,sim_time,dt,end_time)
        next_output_time=next_output_time+output_time
        endif
        
        ! Make sure you produce output at the correct time
        ! dt=YEAR*10.0**(min(5.0,(-2.0+real(nstep)/1e5*10.0)))
        actual_dt=min(next_output_time-sim_time,dt)
        nstep=nstep+1

        ! Report time and time step
        write(logf,'(A,2(1pe10.3,1x),A)') 'Time, dt:', &
            sim_time/YEAR,actual_dt/YEAR,' (years)'
        
        ! For cosmological simulations evolve proper quantities
        if (cosmological) then
        call redshift_evol(sim_time+0.5*actual_dt)
        call cosmo_evol()
        endif

        ! Take one time step
        call evolve1D(actual_dt)

        ! Update time
        sim_time=sim_time+actual_dt
            
        if (abs(sim_time-end_time) < 1e-6*end_time) exit

        ! Update clock counters (cpu + wall, to avoid overflowing the counter)
        call update_clocks ()

    enddo

    ! Scale to the current redshift
    if (cosmological) then
        call redshift_evol(sim_time)
        call cosmo_evol()
    endif

    ! Write final output
    call output(nstep,sim_time,dt,end_time)

    ! Clean up some stuff
    call close_down ()

    ! Report clocks (cpu and wall)
    call report_clocks ()

    ! End the run
    call mpi_end ()

end subroutine

subroutine run_param(r_in_cm,r_out_cm, testnum,clumping,temper_val,isotherm_answer,gamma_uvb_h,dens_val_input,r_core)
    ! Needs following modules
    use precision, only: dp
    use clocks, only: setup_clocks, update_clocks, report_clocks
    use file_admin, only: stdinput, logf, file_input, flag_for_file_input
    use astroconstants, only: YEAR
    use my_mpi, only: mpi_setup, mpi_end, rank
    use output_module, only: setup_output,output,close_down
    ! use grid, only: grid_ini
    use subr_main
    use radiation, only: rad_ini
    use cosmology, only: cosmology_init, redshift_evol, &
        time2zred, zred2time, zred, cosmological
    use cosmological_evolution, only: cosmo_evol
    ! use material, only: mat_ini, testnum
    use times, only: time_ini, end_time,dt,output_time
    use evolve, only: evolve1D

    implicit none

    ! Integer variables
    integer :: nstep !< time step counter
    integer :: restart !< restart if not zero (not used in 1D code)

    ! Time variables
    real(kind=dp) :: sim_time !< actual time (s)
    real(kind=dp) :: next_output_time !< time of next output (s)
    real(kind=dp) :: actual_dt !< actual time step (s)

    !> Input file
    ! character(len=512), intent(in)  :: inputfile
    ! character(len=512), intent(out) :: outfile
    ! outfile = inputfile

    !> Input parameters 
    real(kind=8), intent(in) :: r_in_cm,r_out_cm
    integer, intent(in) :: testnum !< number of test problem (1 to 4)
    real(kind=8), intent(in) :: clumping !< global clumping factor
    real(kind=8), intent(in) :: temper_val, dens_val_input, r_core
    real(kind=8), intent(in) :: gamma_uvb_h
    character(len=1), intent(in) :: isotherm_answer

    ! Initialize clocks (cpu and wall)
    call setup_clocks

    ! Set up MPI structure (compatibility mode) & open log file
    call mpi_setup()

    ! Set up input stream (either standard input or from file given
    ! by first argument)
    !if (rank == 0) then
    !    write(logf,*) "screen input or file input?"
    !    flush(logf)
    !    if (COMMAND_ARGUMENT_COUNT () > 0) then
    !        call GET_COMMAND_ARGUMENT(1,inputfile)
    !        write(logf,*) "reading input from ",trim(adjustl(inputfile))
    !        open(unit=stdinput,file=inputfile,status="old")
    !        call flag_for_file_input(.true.)
    !    else
    !        write(logf,*) "reading input from command line"
    !    endif
    !    flush(logf)
    !endif

    ! Initialize output
    call setup_output ()

    ! Initialize grid
    ! call grid_ini ()
    call grid_ini_subr(r_in_cm,r_out_cm)

    ! Initialize the material properties
    ! call mat_ini (restart)
    call mat_ini_subr (restart,testnum,clumping,temper_val,isotherm_answer,gamma_uvb_h,dens_val_input,r_core)

    ! Initialize photo-ionization calculation
    call rad_ini( )

    ! Initialize time step parameters
    call time_ini ()

    ! Set time to zero
    sim_time=0.0
    next_output_time=0.0

    ! Update cosmology (transform from comoving to proper values)
    if (cosmological) then
        call redshift_evol(sim_time)
        call cosmo_evol( )
        !write(*,*) zred
    endif

    ! Loop until end time is reached
    nstep=0
    do

        ! Write output
        if (abs(sim_time-next_output_time) <= 1e-6*sim_time) then
        call output(nstep,sim_time,dt,end_time)
        next_output_time=next_output_time+output_time
        endif
        
        ! Make sure you produce output at the correct time
        ! dt=YEAR*10.0**(min(5.0,(-2.0+real(nstep)/1e5*10.0)))
        actual_dt=min(next_output_time-sim_time,dt)
        nstep=nstep+1

        ! Report time and time step
        write(logf,'(A,2(1pe10.3,1x),A)') 'Time, dt:', &
            sim_time/YEAR,actual_dt/YEAR,' (years)'
        
        ! For cosmological simulations evolve proper quantities
        if (cosmological) then
        call redshift_evol(sim_time+0.5*actual_dt)
        call cosmo_evol()
        endif

        ! Take one time step
        call evolve1D(actual_dt)

        ! Update time
        sim_time=sim_time+actual_dt
            
        if (abs(sim_time-end_time) < 1e-6*end_time) exit

        ! Update clock counters (cpu + wall, to avoid overflowing the counter)
        call update_clocks ()

    enddo

    ! Scale to the current redshift
    if (cosmological) then
        call redshift_evol(sim_time)
        call cosmo_evol()
    endif

    ! Write final output
    call output(nstep,sim_time,dt,end_time)

    ! Clean up some stuff
    call close_down ()

    ! Report clocks (cpu and wall)
    call report_clocks ()

    ! End the run
    call mpi_end ()


end subroutine