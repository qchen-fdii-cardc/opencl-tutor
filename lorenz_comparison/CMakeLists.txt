cmake_minimum_required(VERSION 3.16)

add_executable(lorenz_cpu_vs_opencl src/main.cpp)
target_compile_features(lorenz_cpu_vs_opencl PRIVATE cxx_std_17)
target_link_libraries(lorenz_cpu_vs_opencl PRIVATE OpenCL::OpenCL)
