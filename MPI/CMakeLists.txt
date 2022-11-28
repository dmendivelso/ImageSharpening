
cmake_minimum_required(VERSION 2.8)
PROJECT( Sharp )
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( Sharp test.cpp )
TARGET_LINK_LIBRARIES( Sharp ${OpenCV_LIBS} )

find_package(OpenMP)


if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME}
    # OpenMP::OpenMP_CXX is an imported target with INTERFACE_**
    $<$<CONFIG:Release>:OpenMP::OpenMP_CXX> )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# MPI
find_package(MPI REQUIRED)
target_link_libraries(Sharp PUBLIC MPI::MPI_CXX)