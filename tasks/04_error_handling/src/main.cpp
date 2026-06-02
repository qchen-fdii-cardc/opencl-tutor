#include <CL/cl.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static const char* cl_error_to_string(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        default: return "CL_UNKNOWN_ERROR";
    }
}

static void check_cl(cl_int err, const char* what) {
    if (err == CL_SUCCESS) {
        return;
    }
    throw std::runtime_error(std::string(what) + " failed: " + cl_error_to_string(err) +
                             " (" + std::to_string(err) + ")");
}

#define CL_CHECK(call) check_cl((call), #call)

static cl_device_id pick_first_device() {
    cl_uint platform_count = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &platform_count));
    if (platform_count == 0) {
        throw std::runtime_error("No OpenCL platform found");
    }

    std::vector<cl_platform_id> platforms(platform_count);
    CL_CHECK(clGetPlatformIDs(platform_count, platforms.data(), nullptr));

    for (cl_uint i = 0; i < platform_count; ++i) {
        cl_uint device_count = 0;
        cl_int err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
        if (err != CL_SUCCESS || device_count == 0) {
            continue;
        }

        std::vector<cl_device_id> devices(device_count);
        CL_CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr));
        return devices[0];
    }

    throw std::runtime_error("No OpenCL device found");
}

int main() {
    try {
        cl_device_id device = pick_first_device();

        cl_int err = CL_SUCCESS;
        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        check_cl(err, "clCreateContext");

        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
        check_cl(err, "clCreateCommandQueue");

        const int values[4] = {1, 2, 3, 4};
        cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(values), const_cast<int*>(values), &err);
        check_cl(err, "clCreateBuffer");

        std::cout << "Unified error handling flow completed successfully.\n";

        clReleaseMemObject(buffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
