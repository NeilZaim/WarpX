#ifndef WARPX_FLUSHFORMATPLOTFILE_H_
#define WARPX_FLUSHFORMATPLOTFILE_H_

#include "FlushFormat.H"

#include "Diagnostics/ParticleDiag/ParticleDiag_fwd.H"

#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <iosfwd>
#include <string>

/**
 * \brief This class aims at dumping diags data to disk using the AMReX Plotfile format.
 * In particular, function WriteToFile takes fields and particles as input arguments,
 * and writes data to file.
 */
class FlushFormatPlotfile : public FlushFormat
{
public:
    /** Flush fields and particles to plotfile */
    virtual void WriteToFile (
        const amrex::Vector<std::string> varnames,
        const amrex::Vector<amrex::MultiFab>& mf,
        amrex::Vector<amrex::Geometry>& geom,
        const amrex::Vector<int> iteration, const double time,
        const amrex::Vector<ParticleDiag>& particle_diags, int nlev, const std::string prefix,
        bool plot_raw_fields,
        bool plot_raw_fields_guards,
        bool plot_raw_rho, bool plot_raw_F,
        bool isBTD = false, int snapshotID = -1,
        const amrex::Geometry& full_BTD_snapshot = amrex::Geometry(),
        bool isLastBTDFlush = false) const override;

    /** Write general info of the run into the plotfile */
    void WriteJobInfo(const std::string& dir) const;
    /** Write WarpX-specific plotfile header */
    void WriteWarpXHeader(const std::string& name, const amrex::Vector<ParticleDiag>& particle_diags, amrex::Vector<amrex::Geometry>& geom) const;
    void WriteAllRawFields (const bool plot_raw_fields, const int nlevels,
                            const std::string& plotfilename,
                            const bool plot_raw_fields_guards,
                            const bool plot_raw_rho, bool plot_raw_F) const;
    void WriteHeaderParticle(std::ostream& os,
                             const amrex::Vector<ParticleDiag>& particle_diags) const;
    /** \brief Write particles data to file.
     * \param[in] filename name of output directory
     * \param[in] particle_diags Each element of this vector handles output of 1 species.
     */
    void WriteParticles(const std::string& filename,
                        const amrex::Vector<ParticleDiag>& particle_diags) const;

    ~FlushFormatPlotfile() {}
};

#endif // WARPX_FLUSHFORMATPLOTFILE_H_
