add_library(${PROJECT_NAME}_utils 
RandomNumberGenerator.cpp RandomNumberGenerator.h
unit_test.h
eigen_types.h 
math.cpp math.h
io.cpp io.h
TicToc.cpp TicToc.h
)

target_link_libraries(${PROJECT_NAME}_utils ${Boost_LIBRARIES})
