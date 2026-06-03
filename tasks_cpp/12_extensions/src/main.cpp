#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
            std::cout << "No OpenCL platforms found.\n";
            return EXIT_SUCCESS;
        }

        for (size_t p = 0; p < platforms.size(); ++p) {
            std::cout << "[Platform " << p << "] "
                      << platforms[p].getInfo<CL_PLATFORM_NAME>() << "\n";

            std::vector<cl::Device> devices;
            platforms[p].getDevices(CL_DEVICE_TYPE_ALL, &devices);
            for (size_t d = 0; d < devices.size(); ++d) {
                const std::string name = devices[d].getInfo<CL_DEVICE_NAME>();
                const std::string exts = devices[d].getInfo<CL_DEVICE_EXTENSIONS>();
                std::istringstream iss(exts);
                size_t count = 0;
                std::string token;
                while (iss >> token) ++count;
                std::cout << "  - Device " << d << ": " << name
                          << ", extension count=" << count << "\n";
            }
        }

        return EXIT_SUCCESS;
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    }
}
