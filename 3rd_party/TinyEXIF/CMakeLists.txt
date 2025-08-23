cmake_minimum_required(VERSION 2.8.3)
project(tinyexif)

add_library(tinyexif SHARED TinyEXIF.cpp tinyxml2.cpp)

add_executable(test test.cpp)
target_link_libraries(test tinyexif)