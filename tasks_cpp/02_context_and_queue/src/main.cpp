#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

static cl::Device pick_first_device() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (size_t i = 0; i < platforms.size(); ++i) {
        std::vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (!devices.empty()) return devices[0];
    }
    throw std::runtime_error("No usable OpenCL device found.");
}

int main() {
    try {
        cl::Device device = pick_first_device();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        std::cout << "Context and command queue created successfully: "
                  << device.getInfo<CL_DEVICE_NAME>() << "\n";
        return EXIT_SUCCESS;
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
