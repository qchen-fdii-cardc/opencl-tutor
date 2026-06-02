#include <CL/cl.h>

#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    cl_int err = CL_SUCCESS;

    cl_uint platform_count = 0;
    err = clGetPlatformIDs(0, nullptr, &platform_count);
    if (err != CL_SUCCESS || platform_count == 0) {
        std::cerr << "No OpenCL platform found (error " << err << ")\n";
        return EXIT_FAILURE;
    }

    std::vector<cl_platform_id> platforms(platform_count);
    err = clGetPlatformIDs(platform_count, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to query OpenCL platforms (error " << err << ")\n";
        return EXIT_FAILURE;
    }

    cl_platform_id platform = platforms[0];

    cl_uint device_count = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 0, nullptr, &device_count);
    if (err != CL_SUCCESS || device_count == 0) {
        std::cerr << "No OpenCL device found on first platform (error " << err << ")\n";
        return EXIT_FAILURE;
    }

    std::vector<cl_device_id> devices(device_count);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, device_count, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to query OpenCL devices (error " << err << ")\n";
        return EXIT_FAILURE;
    }

    cl_device_id device = devices[0];

    char device_name[256] = {};
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to query device name (error " << err << ")\n";
        return EXIT_FAILURE;
    }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || context == nullptr) {
        std::cerr << "Failed to create OpenCL context (error " << err << ")\n";
        return EXIT_FAILURE;
    }

    std::cout << "OpenCL initialized successfully\n";
    std::cout << "Using device: " << device_name << "\n";

    clReleaseContext(context);
    return EXIT_SUCCESS;
}
