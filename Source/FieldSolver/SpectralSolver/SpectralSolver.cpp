/* Copyright 2019 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/SpectralSolver/SpectralAlgorithms/SpectralBaseAlgorithm.H"
#include "FieldSolver/SpectralSolver/SpectralFieldData.H"
#include "SpectralAlgorithms/ComovingPsatdAlgorithm.H"
#include "SpectralAlgorithms/PMLPsatdAlgorithm.H"
#include "SpectralAlgorithms/PsatdAlgorithm.H"
#include "SpectralKSpace.H"
#include "SpectralSolver.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <memory>

#if WARPX_USE_PSATD

SpectralSolver::SpectralSolver(
                const int lev,
                const amrex::BoxArray& realspace_ba,
                const amrex::DistributionMapping& dm,
                const int norder_x, const int norder_y,
                const int norder_z, const bool nodal,
                const amrex::IntVect& fill_guards,
                const amrex::Array<amrex::Real,3>& v_galilean,
                const amrex::Array<amrex::Real,3>& v_comoving,
                const amrex::RealVect dx, const amrex::Real dt,
                const bool pml, const bool periodic_single_box,
                const bool update_with_rho,
                const bool fft_do_time_averaging,
                const bool J_linear_in_time,
                const bool dive_cleaning,
                const bool divb_cleaning,
                const bool plot_Ex_lowfreq,
                const bool plot_Ey_lowfreq,
                const bool plot_Ez_lowfreq,
                const bool plot_Bx_lowfreq,
                const bool plot_By_lowfreq,
                const bool plot_Bz_lowfreq,
                const amrex::Real Ex_lowfreq_fundamental_wavelength,
                const amrex::Real Ey_lowfreq_fundamental_wavelength,
                const amrex::Real Ez_lowfreq_fundamental_wavelength,
                const amrex::Real Bx_lowfreq_fundamental_wavelength,
                const amrex::Real By_lowfreq_fundamental_wavelength,
                const amrex::Real Bz_lowfreq_fundamental_wavelength,
                const amrex::Real Ex_lowfreq_cutoff_harmonic,
                const amrex::Real Ey_lowfreq_cutoff_harmonic,
                const amrex::Real Ez_lowfreq_cutoff_harmonic,
                const amrex::Real Bx_lowfreq_cutoff_harmonic,
                const amrex::Real By_lowfreq_cutoff_harmonic,
                const amrex::Real Bz_lowfreq_cutoff_harmonic)
{
    // Initialize all structures using the same distribution mapping dm

    // - Initialize k space object (Contains info about the size of
    // the spectral space corresponding to each box in `realspace_ba`,
    // as well as the value of the corresponding k coordinates)
    const SpectralKSpace k_space= SpectralKSpace(realspace_ba, dm, dx);

    m_spectral_index = SpectralFieldIndex(update_with_rho, fft_do_time_averaging,
                                          J_linear_in_time, dive_cleaning, divb_cleaning, pml,
                                          plot_Ex_lowfreq, plot_Ey_lowfreq, plot_Ez_lowfreq,
                                          plot_Bx_lowfreq, plot_By_lowfreq, plot_Bz_lowfreq);

    // - Select the algorithm depending on the input parameters
    //   Initialize the corresponding coefficients over k space

    if (pml) {
        algorithm = std::make_unique<PMLPsatdAlgorithm>(
            k_space, dm, m_spectral_index, norder_x, norder_y, norder_z, nodal,
            fill_guards, dt, dive_cleaning, divb_cleaning);
    }
    else {
        // Comoving PSATD algorithm
        if (v_comoving[0] != 0. || v_comoving[1] != 0. || v_comoving[2] != 0.) {
            algorithm = std::make_unique<ComovingPsatdAlgorithm>(
                k_space, dm, m_spectral_index, norder_x, norder_y, norder_z, nodal,
                fill_guards, v_comoving, dt, update_with_rho);
        }
        // PSATD algorithms: standard, Galilean, or averaged Galilean
        else {
            algorithm = std::make_unique<PsatdAlgorithm>(
                k_space, dm, m_spectral_index, norder_x, norder_y, norder_z, nodal, fill_guards,
                v_galilean, dt, update_with_rho, fft_do_time_averaging, J_linear_in_time,
                dive_cleaning, divb_cleaning, plot_Ex_lowfreq, plot_Ey_lowfreq, plot_Ez_lowfreq,
                plot_Bx_lowfreq, plot_By_lowfreq, plot_Bz_lowfreq,
                Ex_lowfreq_fundamental_wavelength, Ey_lowfreq_fundamental_wavelength, Ez_lowfreq_fundamental_wavelength,
                Bx_lowfreq_fundamental_wavelength, By_lowfreq_fundamental_wavelength, Bz_lowfreq_fundamental_wavelength,
                Ex_lowfreq_cutoff_harmonic, Ey_lowfreq_cutoff_harmonic, Ez_lowfreq_cutoff_harmonic,
                Bx_lowfreq_cutoff_harmonic, By_lowfreq_cutoff_harmonic, Bz_lowfreq_cutoff_harmonic);
        }
    }

    // - Initialize arrays for fields in spectral space + FFT plans
    field_data = SpectralFieldData(lev, realspace_ba, k_space, dm,
                                   m_spectral_index.n_fields, periodic_single_box);

    m_fill_guards = fill_guards;
}

void
SpectralSolver::ForwardTransform( const int lev,
                                  const amrex::MultiFab& mf,
                                  const int field_index,
                                  const int i_comp )
{
    WARPX_PROFILE("SpectralSolver::ForwardTransform");
    field_data.ForwardTransform( lev, mf, field_index, i_comp );
}

void
SpectralSolver::BackwardTransform( const int lev,
                                   amrex::MultiFab& mf,
                                   const int field_index,
                                   const int i_comp )
{
    WARPX_PROFILE("SpectralSolver::BackwardTransform");
    field_data.BackwardTransform(lev, mf, field_index, i_comp, m_fill_guards);
}

void
SpectralSolver::pushSpectralFields(){
    WARPX_PROFILE("SpectralSolver::pushSpectralFields");
    // Virtual function: the actual function used here depends
    // on the sub-class of `SpectralBaseAlgorithm` that was
    // initialized in the constructor of `SpectralSolver`
    algorithm->pushSpectralFields( field_data );
}

#endif // WARPX_USE_PSATD
