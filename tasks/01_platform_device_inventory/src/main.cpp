#include <CL/cl.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static const char* device_type_to_string(cl_device_type type) {
    if (type & CL_DEVICE_TYPE_GPU) return "GPU";
    if (type & CL_DEVICE_TYPE_CPU) return "CPU";
    if (type & CL_DEVICE_TYPE_ACCELERATOR) return "ACCELERATOR";
    if (type & CL_DEVICE_TYPE_CUSTOM) return "CUSTOM";
    return "DEFAULT/UNKNOWN";
}

static std::string get_platform_name(cl_platform_id platform) {
    size_t size = 0;
    if (clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return "<unknown platform>";
    }
    std::vector<char> buf(size, '\0');
    if (clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, buf.data(), nullptr) != CL_SUCCESS) {
        return "<unknown platform>";
    }
    return std::string(buf.data());
}

static std::string get_device_name(cl_device_id device) {
    size_t size = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return "<unknown device>";
    }
    std::vector<char> buf(size, '\0');
    if (clGetDeviceInfo(device, CL_DEVICE_NAME, size, buf.data(), nullptr) != CL_SUCCESS) {
        return "<unknown device>";
    }
    return std::string(buf.data());
}

int main() {
    cl_uint platform_count = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platform_count);
    if (err != CL_SUCCESS) {
        std::cerr << "clGetPlatformIDs failed, error=" << err << "\n";
        return EXIT_FAILURE;
    }

    if (platform_count == 0) {
        std::cout << "No OpenCL platform found.\n";
        return EXIT_SUCCESS;
    }

    std::vector<cl_platform_id> platforms(platform_count);
    err = clGetPlatformIDs(platform_count, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to query platform list, error=" << err << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "OpenCL platform count: " << platform_count << "\n\n";

    for (cl_uint i = 0; i < platform_count; ++i) {
        const std::string platform_name = get_platform_name(platforms[i]);
        std::cout << "[Platform " << i << "] " << platform_name << "\n";

        cl_uint device_count = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
        if (err != CL_SUCCESS) {
            std::cout << "  Failed to query device count, error=" << err << "\n\n";
            continue;
        }

        if (device_count == 0) {
            std::cout << "  No devices found on this platform.\n\n";
            continue;
        }

        std::vector<cl_device_id> devices(device_count);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "  Failed to query device list, error=" << err << "\n\n";
            continue;
        }

        for (cl_uint d = 0; d < device_count; ++d) {
            cl_device_type dtype = 0;
            clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr);
            std::cout << "  - Device " << d << ": " << get_device_name(devices[d])
                      << " [" << device_type_to_string(dtype) << "]\n";
        }

        std::cout << "\n";
    }

    return EXIT_SUCCESS;
}
