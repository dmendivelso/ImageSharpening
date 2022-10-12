cmake_minimum_required(VERSION 2.8)
PROJECT( test )
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( test test.cpp )
TARGET_LINK_LIBRARIES( test ${OpenCV_LIBS} )

find_package(OpenMP)

# add_executable(...

if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME}
    $<$<CONFIG:Release>:OpenMP::OpenMP_CXX> )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()