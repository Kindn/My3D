add_library(${PROJECT_NAME}_sfm 
    SceneBuilder.cpp SceneBuilder.h
    IncrementalSFM.cpp IncrementalSFM.h
)   
target_link_libraries(${PROJECT_NAME}_sfm ${PROJECT_NAME}_estimators
                                          ${PROJECT_NAME}_base 
                                          ${PROJECT_NAME}_feature 
                                          ${PROJECT_NAME}_utils)

add_executable(scene_builder_test scene_builder_test.cpp)
target_link_libraries(scene_builder_test ${PROJECT_NAME}_base
                                         ${PROJECT_NAME}_estimators
                                         ${PROJECT_NAME}_feature 
                                         ${PROJECT_NAME}_utils
                                         ${PROJECT_NAME}_sfm
                                         ${Boost_LIBRARIES} )    
                                         
                                         add_executable(scene_builder_test_main scene_builder_test_main.cpp)
                                         target_link_libraries(scene_builder_test_main ${PROJECT_NAME}_base
                                                                                  ${PROJECT_NAME}_estimators
                                                                                  ${PROJECT_NAME}_feature 
                                                                                  ${PROJECT_NAME}_utils
                                                                                  ${PROJECT_NAME}_sfm
                                                                                  ${Boost_LIBRARIES} )
