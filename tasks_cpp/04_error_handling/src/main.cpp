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
    cl::Device device;
    cl::Program program;
    try {
        device = pick_first_device();
        cl::Context context(device);

        const char* broken =
            "__kernel void broken(__global float* out) { out[get_global_id(0)] = ; }";
        program = cl::Program(context, broken);
        program.build({device});

        std::cout << "Unexpected: broken kernel compiled.\n";
        return EXIT_FAILURE;
    } catch (const cl::Error& e) {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE && program() != nullptr) {
            std::cout << "Build error captured successfully.\n";
            std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
            return EXIT_SUCCESS;
        }
        std::cerr << "OpenCL runtime error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
