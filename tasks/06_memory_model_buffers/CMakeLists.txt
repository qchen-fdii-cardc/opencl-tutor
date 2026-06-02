cmake_minimum_required(VERSION 3.16)
project(task_06 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCL REQUIRED)

add_executable(task_06 src/main.cpp)
target_link_libraries(task_06 PRIVATE OpenCL::OpenCL)
