cmake_minimum_required(VERSION 3.16)
project(opencl_minimal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCL REQUIRED)

add_executable(opencl_minimal src/main.cpp)
target_link_libraries(opencl_minimal PRIVATE OpenCL::OpenCL)
