add_library(${LIB_NAME}_solver SHARED 
    OptSolverBase.h 
    GaussNewtonSolver.cpp GaussNewtonSolver.h
    GaussNewtonSchurSolver.cpp GaussNewtonSchurSolver.h
    GaussNewtonSparseSchurSolver.cpp GaussNewtonSparseSchurSolver.h
    LevenbergMarquartSparseSchurSolver.cpp LevenbergMarquartSparseSchurSolver.h
)

target_link_libraries(${LIB_NAME}_solver 
${LIB_NAME}_util
)