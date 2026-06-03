#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#if __has_include(<CL/cl.hpp>)
#include <CL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static std::string device_type_to_string(cl_device_type type) {
    if (type & CL_DEVICE_TYPE_GPU) return "GPU";
    if (type & CL_DEVICE_TYPE_CPU) return "CPU";
    if (type & CL_DEVICE_TYPE_ACCELERATOR) return "ACCELERATOR";
    if (type & CL_DEVICE_TYPE_CUSTOM) return "CUSTOM";
    return "DEFAULT/UNKNOWN";
}

int main() {
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        std::cout << "OpenCL platform count: " << platforms.size() << "\n\n";

        for (size_t i = 0; i < platforms.size(); ++i) {
            const auto platform_name = platforms[i].getInfo<CL_PLATFORM_NAME>();
            std::cout << "[Platform " << i << "] " << platform_name << "\n";

            std::vector<cl::Device> devices;
            platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
            if (devices.empty()) {
                std::cout << "  No devices found on this platform.\n\n";
                continue;
            }

            for (size_t d = 0; d < devices.size(); ++d) {
                const auto dtype = devices[d].getInfo<CL_DEVICE_TYPE>();
                const auto name = devices[d].getInfo<CL_DEVICE_NAME>();
                std::cout << "  - Device " << d << ": " << name
                          << " [" << device_type_to_string(dtype) << "]\n";
            }
            std::cout << "\n";
        }
        return EXIT_SUCCESS;
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    }
}
