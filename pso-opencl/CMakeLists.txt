cmake_minimum_required(VERSION 3.16)

add_executable(pso_opencl src/main.cpp)
target_compile_features(pso_opencl PRIVATE cxx_std_17)
target_link_libraries(pso_opencl PRIVATE OpenCL::OpenCL)
