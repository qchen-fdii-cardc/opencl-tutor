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
        const cl::Device device = pick_first_device();
        cl::Context context(device);

        const char* broken =
            "__kernel void broken(__global float* out) { out[get_global_id(0)] = ; }";
        cl::Program program(context, broken);
        program.build({device});

        std::cout << "Unexpected: broken kernel compiled.\n";
        return EXIT_FAILURE;
    } catch (const cl::BuildError& e) {
        std::cout << "Build error captured successfully.\n";
        const std::vector<std::pair<cl::Device, std::string> > logs = e.getBuildLog();
        for (size_t i = 0; i < logs.size(); ++i) {
            std::cerr << logs[i].second << "\n";
        }
        return EXIT_SUCCESS;
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL runtime error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
