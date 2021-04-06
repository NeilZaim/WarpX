/* Copyright 2021 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef OMEGAP2DEPOSITION_H_
#define OMEGAP2DEPOSITION_H_

#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/ShapeFactors.H"

#include <AMReX.H>

/* \brief Omegap2 Deposition for thread thread_num
 * \param GetPosition : A functor for returning the particle position.
 * \param wp           : Pointer to array of particle weights.
 * \param ion_lev      : Pointer to array of particle ionization level. This is
                         required to have the charge of each macroparticle
                         since q is a scalar. For non-ionizable species,
                         ion_lev is a null pointer.
 * \param omegap2_fab  : FArrayBox of plasma frequency squared, either full array or tile.
 * \param np_to_depose : Number of particles for which current is deposited.
 * \param dx           : 3D cell size
 * \param xyzmin       : Physical lower bounds of domain.
 * \param lo           : Index lower bounds of domain.
 * \param q            : species charge.
 * \param m            : species mass.
 * \param n_rz_azimuthal_modes: Number of azimuthal modes when using RZ geometry.
 * \param cost: Pointer to (load balancing) cost corresponding to box where present particles deposit current.
 * \param load_balance_costs_update_algo: Selected method for updating load balance costs.
 */
template <int depos_order>
void doOmegap2DepositionShapeN (const GetParticlePosition& GetPosition,
                               const amrex::ParticleReal * const wp,
                               const int * const ion_lev,
                               amrex::FArrayBox& omegap2_fab,
                               const long np_to_depose,
                               const std::array<amrex::Real,3>& dx,
                               const std::array<amrex::Real, 3> xyzmin,
                               const amrex::Dim3 lo,
                               const amrex::Real q,
                               const amrex::Real m,
                               const int n_rz_azimuthal_modes,
                               amrex::Real* cost,
                               const long load_balance_costs_update_algo)
{
    using namespace amrex;

#if !defined(AMREX_USE_GPU)
    amrex::ignore_unused(cost, load_balance_costs_update_algo);
#endif

    // Whether ion_lev is a null pointer (do_ionization=0) or a real pointer
    // (do_ionization=1)
    const bool do_ionization = ion_lev;
    const amrex::Real dxi = 1.0_rt/dx[0];
    const amrex::Real dzi = 1.0_rt/dx[2];
#if (AMREX_SPACEDIM == 2)
    const amrex::Real invvol = dxi*dzi;
#elif (defined WARPX_DIM_3D)
    const amrex::Real dyi = 1.0_rt/dx[1];
    const amrex::Real invvol = dxi*dyi*dzi;
#endif

    const amrex::Real xmin = xyzmin[0];
#if (defined WARPX_DIM_3D)
    const amrex::Real ymin = xyzmin[1];
#endif
    const amrex::Real zmin = xyzmin[2];

    amrex::Array4<amrex::Real> const& omegap2_arr = omegap2_fab.array();
    amrex::IntVect const omegap2_type = omegap2_fab.box().type();

    constexpr int zdir = (AMREX_SPACEDIM - 1);
    constexpr int NODE = amrex::IndexType::NODE;
    constexpr int CELL = amrex::IndexType::CELL;

    const amrex::Real omegap2_factor = q*q/m/PhysConst::ep0;

    // Loop over particles and deposit into omegap2_fab
    amrex::Real* cost_real = (amrex::Real*) amrex::The_Managed_Arena()->alloc(sizeof(amrex::Real));
    *cost_real = 0.;
    amrex::ParallelFor(
        np_to_depose,
        [=] AMREX_GPU_DEVICE (long ip) {
#if (defined AMREX_USE_GPU)
            KernelTimer KnlTimer(cost && load_balance_costs_update_algo
                                 == LoadBalanceCostsUpdateAlgo::GpuClock, cost_real);
#endif
            // --- Get particle quantities
            amrex::Real womegap2 = wp[ip]*invvol*omegap2_factor;
            if (do_ionization){
                womegap2 *= std::pow(ion_lev[ip],2);
            }

            amrex::ParticleReal xp, yp, zp;
            GetPosition(ip, xp, yp, zp);

            // --- Compute shape factors
            // x direction
            // Get particle position in grid coordinates
#if (defined WARPX_DIM_RZ)
            const amrex::Real rp = std::sqrt(xp*xp + yp*yp);
            amrex::Real costheta;
            amrex::Real sintheta;
            if (rp > 0.) {
                costheta = xp/rp;
                sintheta = yp/rp;
            } else {
                costheta = 1._rt;
                sintheta = 0._rt;
            }
            const Complex xy0 = Complex{costheta, sintheta};
            const amrex::Real x = (rp - xmin)*dxi;
#else
            const amrex::Real x = (xp - xmin)*dxi;
#endif

            // Compute shape factor along x
            // i: leftmost grid point that the particle touches
            amrex::Real sx[depos_order + 1] = {0._rt};
            int i = 0;
            Compute_shape_factor< depos_order > const compute_shape_factor;
            if (omegap2_type[0] == NODE) {
                i = compute_shape_factor(sx, x);
            } else if (omegap2_type[0] == CELL) {
                i = compute_shape_factor(sx, x - 0.5_rt);
            }

#if (defined WARPX_DIM_3D)
            // y direction
            const amrex::Real y = (yp - ymin)*dyi;
            amrex::Real sy[depos_order + 1] = {0._rt};
            int j = 0;
            if (omegap2_type[1] == NODE) {
                j = compute_shape_factor(sy, y);
            } else if (omegap2_type[1] == CELL) {
                j = compute_shape_factor(sy, y - 0.5_rt);
            }
#endif
            // z direction
            const amrex::Real z = (zp - zmin)*dzi;
            amrex::Real sz[depos_order + 1] = {0._rt};
            int k = 0;
            if (omegap2_type[zdir] == NODE) {
                k = compute_shape_factor(sz, z);
            } else if (omegap2_type[zdir] == CELL) {
                k = compute_shape_factor(sz, z - 0.5_rt);
            }

            // Deposit omegap2 into omegap2_arr
#if (defined WARPX_DIM_XZ) || (defined WARPX_DIM_RZ)
            for (int iz=0; iz<=depos_order; iz++){
                for (int ix=0; ix<=depos_order; ix++){
                    amrex::Gpu::Atomic::AddNoRet(
                        &omegap2_arr(lo.x+i+ix, lo.y+k+iz, 0, 0),
                        sx[ix]*sz[iz]*womegap2);
#if (defined WARPX_DIM_RZ)
                    Complex xy = xy0; // Throughout the following loop, xy takes the value e^{i m theta}
                    for (int imode=1 ; imode < n_rz_azimuthal_modes ; imode++) {
                        // The factor 2 on the weighting comes from the normalization of the modes
                        amrex::Gpu::Atomic::AddNoRet( &omegap2_arr(lo.x+i+ix, lo.y+k+iz, 0, 2*imode-1), 2._rt*sx[ix]*sz[iz]*womegap2*xy.real());
                        amrex::Gpu::Atomic::AddNoRet( &omegap2_arr(lo.x+i+ix, lo.y+k+iz, 0, 2*imode  ), 2._rt*sx[ix]*sz[iz]*womegap2*xy.imag());
                        xy = xy*xy0;
                    }
#endif
                }
            }
#elif (defined WARPX_DIM_3D)
            for (int iz=0; iz<=depos_order; iz++){
                for (int iy=0; iy<=depos_order; iy++){
                    for (int ix=0; ix<=depos_order; ix++){
                        amrex::Gpu::Atomic::AddNoRet(
                            &omegap2_arr(lo.x+i+ix, lo.y+j+iy, lo.z+k+iz),
                            sx[ix]*sy[iy]*sz[iz]*womegap2);
                    }
                }
            }
#endif
        }
        );
        amrex::The_Managed_Arena()->free(cost_real);

#ifndef WARPX_DIM_RZ
        amrex::ignore_unused(n_rz_azimuthal_modes);
#endif
}

#endif // OMEGAP2DEPOSITION_H_
