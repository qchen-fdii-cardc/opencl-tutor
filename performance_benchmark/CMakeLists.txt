cmake_minimum_required(VERSION 3.16)

add_executable(opencl_multi_device_stress src/main.cpp)
target_compile_features(opencl_multi_device_stress PRIVATE cxx_std_11)
target_link_libraries(opencl_multi_device_stress PRIVATE OpenCL::OpenCL)
