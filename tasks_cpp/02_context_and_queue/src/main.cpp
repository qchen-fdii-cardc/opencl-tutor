#if __has_include(<CL/cl.hpp>)
#include <CL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include <cstdlib>
#include <iostream>
#include <vector>

static bool pick_first_device(cl_platform_id* out_platform, cl_device_id* out_device) {
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0) {
        return false;
    }

    std::vector<cl_platform_id> platforms(platform_count);
    if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS) {
        return false;
    }

    for (cl_uint i = 0; i < platform_count; ++i) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count) != CL_SUCCESS ||
            device_count == 0) {
            continue;
        }

        std::vector<cl_device_id> devices(device_count);
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr) != CL_SUCCESS) {
            continue;
        }

        *out_platform = platforms[i];
        *out_device = devices[0];
        return true;
    }

    return false;
}

int main() {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    if (!pick_first_device(&platform, &device)) {
        std::cerr << "No usable OpenCL device found.\n";
        return EXIT_FAILURE;
    }

    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || context == nullptr) {
        std::cerr << "clCreateContext failed, error=" << err << "\n";
        return EXIT_FAILURE;
    }

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS || queue == nullptr) {
        std::cerr << "clCreateCommandQueue failed, error=" << err << "\n";
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    std::cout << "Context and command queue created successfully.\n";

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return EXIT_SUCCESS;
}
