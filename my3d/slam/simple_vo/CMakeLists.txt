add_library(${PROJECT_NAME}_simple_vo 
  FeatureTracker.cpp FeatureTracker.h
  Estimator.cpp Estimator.h
  visualization.cpp visualization.h
)
target_link_libraries(${PROJECT_NAME}_simple_vo 
                      ${PROJECT_NAME}_utils 
                      ${PROJECT_NAME}_estimators
                      ${Boost_LIBRARIES} 
                      ${OpenCV_LIBRARIES})

add_executable(simple_vo_tum_main simple_vo_tum_main.cpp)
target_link_libraries(simple_vo_tum_main ${PROJECT_NAME}_simple_vo )

add_executable(simple_vo_video_main simple_vo_video_main.cpp)
target_link_libraries(simple_vo_video_main 
                      ${PROJECT_NAME}_simple_vo)

add_executable(simple_vo_camera_main simple_vo_camera_main.cpp)
target_link_libraries(simple_vo_camera_main 
                      ${PROJECT_NAME}_simple_vo)