set(hiopOptimization_SRC
  hiopNlpFormulation.cpp
  hiopFactAcceptor.cpp
  hiopIterate.cpp 
  hiopResidual.cpp 
  hiopFilter.cpp 
  hiopAlgFilterIPM.cpp 
  hiopKKTLinSys.cpp
  KktLinSysLowRank.cpp
  hiopKKTLinSysMDS.cpp 
  HessianDiagPlusRowRank.cpp
  VectorSpace.cpp
  hiopDualsUpdater.cpp 
  hiopNlpTransforms.cpp
  hiopPDPerturbation.cpp
  hiopAlgPrimalDecomp.cpp
  hiopFRProb.cpp
)

set(hiopOptimization_SPARSE_SRC
  hiopKKTLinSysSparse.cpp
  hiopKKTLinSysSparseCondensed.cpp
  hiopKKTLinSysSparseNormalEqn.cpp
  )

set(hiopOptimization_INTERFACE_HEADERS
  hiopAlgFilterIPM.hpp
  hiopAlgPrimalDecomp.hpp
  hiopDualsUpdater.hpp
  hiopFactAcceptor.hpp
  hiopFilter.hpp
  HessianDiagPlusRowRank.hpp
  VectorSpace.hpp
  hiopIterate.hpp
  hiopKKTLinSys.hpp
  hiopKKTLinSysDense.hpp
  hiopKKTLinSysMDS.hpp
  hiopKKTLinSysSparse.hpp
  hiopKKTLinSysSparseCondensed.hpp
  hiopKKTLinSysSparseNormalEqn.hpp
  hiopLogBarProblem.hpp
  hiopNlpFormulation.hpp
  hiopNlpTransforms.hpp
  hiopPDPerturbation.hpp
  hiopResidual.hpp
  )

if(HIOP_SPARSE)
  list(APPEND hiopOptimization_SRC ${hiopOptimization_SPARSE_SRC})
endif()

install(
  FILES ${hiopOptimization_INTERFACE_HEADERS}
  DESTINATION include
  )

add_library(hiopOptimization OBJECT ${hiopOptimization_SRC})
target_link_libraries(hiopOptimization PRIVATE hiop_tpl)
