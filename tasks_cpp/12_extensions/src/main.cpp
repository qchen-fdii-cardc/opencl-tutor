#if __has_include(<CL/cl.hpp>)
#include <CL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string query_device_string(cl_device_id device, cl_device_info param) {
    size_t size = 0;
    if (clGetDeviceInfo(device, param, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return "";
    }
    std::vector<char> buf(size, '\0');
    if (clGetDeviceInfo(device, param, size, buf.data(), nullptr) != CL_SUCCESS) {
        return "";
    }
    return std::string(buf.data());
}

int main() {
    cl_uint platform_count = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platform_count);
    if (err != CL_SUCCESS || platform_count == 0) {
        std::cerr << "No OpenCL platform found.\n";
        return EXIT_FAILURE;
    }

    std::vector<cl_platform_id> platforms(platform_count);
    if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS) {
        return EXIT_FAILURE;
    }

    for (cl_uint p = 0; p < platform_count; ++p) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count) != CL_SUCCESS ||
            device_count == 0) {
            continue;
        }

        std::vector<cl_device_id> devices(device_count);
        if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr) != CL_SUCCESS) {
            continue;
        }

        for (cl_uint d = 0; d < device_count; ++d) {
            std::string name = query_device_string(devices[d], CL_DEVICE_NAME);
            std::string c_version = query_device_string(devices[d], CL_DEVICE_OPENCL_C_VERSION);
            std::string extensions = query_device_string(devices[d], CL_DEVICE_EXTENSIONS);

            std::cout << "Device: " << name << "\n";
            std::cout << "  OpenCL C version: " << c_version << "\n";
            std::cout << "  Extensions:\n";

            std::istringstream iss(extensions);
            std::string ext;
            while (iss >> ext) {
                std::cout << "    - " << ext << "\n";
            }
            std::cout << "\n";
        }
    }

    return EXIT_SUCCESS;
}
