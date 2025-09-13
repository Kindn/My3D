add_library(${PROJECT_NAME}_base 
    camera/CameraBase.h 
    camera/RadialPinHoleCamera.cpp camera/RadialPinHoleCamera.h 
    camera/CameraDatabase.cpp camera/CameraDatabase.h 
    camera/projection.cpp camera/projection.h 
    image/exif.cpp image/exif.h
    image/Image.cpp image/Image.h 
    image/ImageReaderBase.h
    image/OpenCVImageReader.cpp image/OpenCVImageReader.h
    image/image_io.h
    image/image_processing.cpp image/image_processing.h
    reconstruction/Track.cpp reconstruction/Track.h
    reconstruction/Scene.cpp reconstruction/Scene.h
    reconstruction/BundleAdjustment.cpp reconstruction/BundleAdjustment.h 
    reconstruction/Reconstruction.cpp reconstruction/Reconstruction.h 
    reconstruction/Visualizer.cpp reconstruction/Visualizer.h 
    triangulation.cpp triangulation.h
    pose.cpp pose.h
    points.cpp points.h
    polynomial.cpp polynomial.h
)

target_link_libraries(${PROJECT_NAME}_base ${PROJECT_NAME}_utils 
                                            tinyexif
                                            ${OpenCV_LIBRARIES}
                                            ${GOPT_LIBRARIES}
                                            ${BOOST_LIBRARIES}
                                            OpenMP::OpenMP_CXX)

add_executable(image_test image/image_test.cpp)
target_link_libraries(image_test ${PROJECT_NAME}_base
                                 ${Boost_LIBRARIES} 
                                 ${OpenCV_LIBRARIES})

add_executable(polynomial_test polynomial_test.cpp)
target_link_libraries(polynomial_test ${PROJECT_NAME}_base
                                      ${Boost_LIBRARIES})