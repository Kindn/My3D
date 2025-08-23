add_library(${PROJECT_NAME}_estimators 
    RANSAC.h RANSAC.cpp
    HomographyEstimator.cpp HomographyEstimator.h
    EssentialMatrixEstimator.cpp EssentialMatrixEstimator.h
    FundamentalMatrixEstimator.cpp FundamentalMatrixEstimator.h
    AP3PPoseEstimator.cpp AP3PPoseEstimator.h 
    EPnPPoseEstimator.cpp EPnPPoseEstimator.h 
    TriangulationEstimator.cpp TriangulationEstimator.h 
    estimator_utils.cpp estimator_utils.h
)
target_link_libraries(${PROJECT_NAME}_estimators ${PROJECT_NAME}_base ${PROJECT_NAME}_utils)

add_executable(ransac_test ransac_test.cpp)
target_link_libraries(ransac_test ${PROJECT_NAME}_estimators ${Boost_LIBRARIES}) 

add_executable(loransac_test loransac_test.cpp)
target_link_libraries(loransac_test ${PROJECT_NAME}_estimators ${Boost_LIBRARIES})

add_executable(homography_estimator_test homography_estimator_test.cpp)
target_link_libraries(homography_estimator_test ${PROJECT_NAME}_estimators ${Boost_LIBRARIES})

add_executable(essential_matrix_estimator_test essential_matrix_estimator_test.cpp)
target_link_libraries(essential_matrix_estimator_test ${PROJECT_NAME}_estimators 
                                                      ${PROJECT_NAME}_base
                                                      ${PROJECT_NAME}_utils
                                                      ${Boost_LIBRARIES})

add_executable(fundamental_matrix_estimator_test fundamental_matrix_estimator_test.cpp)
target_link_libraries(fundamental_matrix_estimator_test ${PROJECT_NAME}_estimators 
                                                      ${PROJECT_NAME}_base
                                                      ${PROJECT_NAME}_utils
                                                      ${Boost_LIBRARIES})

add_executable(ap3p_pose_estimator_test ap3p_pose_estimator_test.cpp)
target_link_libraries(ap3p_pose_estimator_test ${PROJECT_NAME}_estimators 
                                               ${PROJECT_NAME}_base
                                               ${PROJECT_NAME}_utils
                                               ${Boost_LIBRARIES})

add_executable(epnp_pose_estimator_test epnp_pose_estimator_test.cpp)
target_link_libraries(epnp_pose_estimator_test ${PROJECT_NAME}_estimators 
                                               ${PROJECT_NAME}_base
                                               ${PROJECT_NAME}_utils
                                               ${Boost_LIBRARIES})

add_executable(triangulation_estimator_test triangulation_estimator_test.cpp)
target_link_libraries(triangulation_estimator_test ${PROJECT_NAME}_estimators 
                                                   ${PROJECT_NAME}_base
                                                   ${PROJECT_NAME}_utils
                                                   ${Boost_LIBRARIES})