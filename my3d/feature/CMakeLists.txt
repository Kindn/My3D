add_library(${PROJECT_NAME}_feature 
    feature_utils.cpp feature_utils.h
    OpenCVORBExtractor.cpp OpenCVORBExtractor.h
    SIFTExtractor.cpp SIFTExtractor.h
    OpenCVMatcher.cpp OpenCVMatcher.h 
    # NNMatcher.cpp NNMatcher.h 
    BrutalForceMatcher.h
    OpenCVBatchFeatureExtractor.cpp OpenCVBatchFeatureExtractor.h
    OpenCVBatchFeatureMatcher.cpp OpenCVBatchFeatureMatcher.h
    BatchFeatureExtractorBase.h
    BatchFeatureMatcherBase.h  
    FeatureExtractorBase.h
    FeatureMatcherBase.h
    sift.cpp sift.h
)
target_link_libraries(${PROJECT_NAME}_feature ${PROJECT_NAME}_base 
                                              ${OpenCV_LIBRARIES})       