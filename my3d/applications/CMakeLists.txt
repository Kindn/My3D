set(MY3D_LIBRARIES 
${PROJECT_NAME}_base
${PROJECT_NAME}_estimators
${PROJECT_NAME}_feature
${PROJECT_NAME}_sfm
${PROJECT_NAME}_utils )

add_executable(sparse_reconstruction sparse_reconstruction.cpp)
target_link_libraries(sparse_reconstruction 
${MY3D_LIBRARIES} 
${GOPT_LIBRARIES}
${OpenCV_LIBRARIES}
${Boost_LIBRARIES}
${SimpleJSON_LIBRARIES})

add_executable(two_view_geometry_demo two_view_geometry_demo.cpp)
target_link_libraries(two_view_geometry_demo 
${MY3D_LIBRARIES} 
${GOPT_LIBRARIES}
${OpenCV_LIBRARIES}
${Boost_LIBRARIES})

add_executable(gaussian_blur_demo gaussian_blur_demo.cpp)
target_link_libraries(gaussian_blur_demo 
${MY3D_LIBRARIES} 
${GOPT_LIBRARIES}
${OpenCV_LIBRARIES}
${Boost_LIBRARIES})

add_executable(sift_detection_demo sift_detection_demo.cpp)
target_link_libraries(sift_detection_demo 
${MY3D_LIBRARIES} 
${GOPT_LIBRARIES}
${OpenCV_LIBRARIES}
${Boost_LIBRARIES})

add_executable(sift_matching_demo sift_matching_demo.cpp)
target_link_libraries(sift_matching_demo 
${MY3D_LIBRARIES} 
${GOPT_LIBRARIES}
${OpenCV_LIBRARIES}
${Boost_LIBRARIES})

add_executable(sfm_visualizer sfm_visualizer.cpp)
target_link_libraries(sfm_visualizer
${MY3D_LIBRARIES} 
${GOPT_LIBRARIES}
${OpenCV_LIBRARIES}
${Boost_LIBRARIES})