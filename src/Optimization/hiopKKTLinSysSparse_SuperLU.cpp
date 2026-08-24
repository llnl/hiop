/**
 * @file hiopKKTLinSysSparse_SuperLU.cpp
 *
 * @brief Separate compilation unit for SuperLU instantiation factory
 *
 * This file is compiled separately to provide a factory function
 * without exposing SuperLU headers to hiopKKTLinSysSparse.cpp.
 */

#include "hiopLinSolverSparseSuperLU.hpp"
#include "hiopNlpFormulation.hpp"

namespace hiop
{

/**
 * Factory function to create SuperLU solver instance.
 * This is defined in a separate compilation unit so that
 * hiopKKTLinSysSparse.cpp doesn't need to include SuperLU headers.
 */
hiopLinSolverSymSparse* createSuperLUSolver(int n, int nnz, hiopNlpFormulation* nlp)
{
  return new hiopLinSolverSymSparseSuperLU(n, nnz, nlp);
}

} // namespace hiop
