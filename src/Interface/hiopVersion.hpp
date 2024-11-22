#pragma once
#include <string>
#include <sstream>
#include <cstdlib>
#include "hiop_defs.hpp"

namespace hiop
{
/**
 * @brief Contains statically determinable information about the current HiOp
 * build.
 */
struct hiopVersion
{
  static constexpr bool useGPU =
#ifdef HIOP_USE_GPU
      true;
#else
      false;
#endif

  static constexpr bool useMPI =
#ifdef HIOP_USE_MPI
      true;
#else
      false;
#endif

  static constexpr bool useMagma =
#ifdef HIOP_USE_MAGMA
      true;
#else
      false;
#endif

  static constexpr bool useRAJA =
#ifdef HIOP_USE_RAJA
      true;
#else
      false;
#endif

  static constexpr bool useSparse =
#ifdef HIOP_SPARSE
      true;
#else
      false;
#endif

  static constexpr bool useCOINHSL =
#ifdef HIOP_USE_COINHSL
      true;
#else
      false;
#endif

  static constexpr bool useSTRUMPACK =
#ifdef HIOP_USE_STRUMPACK
      true;
#else
      false;
#endif

  static constexpr bool usePARDISO =
#ifdef HIOP_USE_PARDISO
      true;
#else
      false;
#endif

  static constexpr bool useReSolve =
#ifdef HIOP_USE_RESOLVE
      true;
#else
      false;
#endif

  static constexpr bool useGINKGO =
#ifdef HIOP_USE_GINKGO
      true;
#else
      false;
#endif

  static inline void version(int& major, int& minor, int& patch)
  {
    major = std::atoi(HIOP_VERSION_MAJOR);
    minor = std::atoi(HIOP_VERSION_MINOR);
    patch = std::atoi(HIOP_VERSION_PATCH);
  }

  static inline std::string version() { return HIOP_VERSION; }
  static inline std::string releaseDate() { return HIOP_RELEASE_DATE; }

  static inline std::string fullVersionInfo()
  {
    auto fmt = [](bool use) { return use ? "YES" : "NO"; };
    std::stringstream ss;
    ss << "HiOp " << version() << " compiled on " << releaseDate() << "\n"
       << "Built with:"
       << "\nGPU: " << fmt(useGPU) << "\nMPI: " << fmt(useMPI) << "\nMAGMA: " << fmt(useMagma) << "\nRAJA: " << fmt(useRAJA)
       << "\nSparse: " << fmt(useSparse) << "\nCOINHSL: " << fmt(useCOINHSL) << "\nSTRUMPACK: " << fmt(useSTRUMPACK)
       << "\nPARDISO: " << fmt(usePARDISO) << "\nReSolve: " << fmt(useReSolve) << "\n";
    return ss.str();
  }
};

}  // namespace hiop
