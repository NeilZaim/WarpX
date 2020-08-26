/* Copyright 2019-2020 Neil Zaim
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Resampling.H"
#include "LevelingThinning.H"

Resampling::Resampling ()
{
    amrex::ParmParse pp("resampling_algorithm");
    std::string resampling_algorithm_string = "leveling_thinning";
    pp.query("type", resampling_algorithm_string);

    if (resampling_algorithm_string.compare("leveling_thinning") == 0)
    {
        m_resampling_algorithm.reset (new LevelingThinning );
    }
    else
    { amrex::Abort("Unknown resampling algorithm."); }
}

bool Resampling::triggered (const int n, const amrex::Real global_numparts)
{
    return m_resampling_trigger.triggered(n, global_numparts);
}

void Resampling::run (WarpXParIter& pti) const
{
    m_resampling_algorithm->run(pti);
}
