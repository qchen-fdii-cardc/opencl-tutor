#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

static const char* kKernelSource =
    "__kernel void vec_add(__global const float* a, __global const float* b, __global float* c) {"
    "  const int gid = get_global_id(0);"
    "  c[gid] = a[gid] + b[gid];"
    "}";

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
    bool program_ready = false;
    try {
        const size_t n = 1024;
        std::vector<float> a(n), b(n), c(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(2 * i);
        }

        device = pick_first_device();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        program = cl::Program(context, kKernelSource);
        program_ready = true;
        program.build({device});
        cl::Kernel kernel(program, "vec_add");

        cl::Buffer a_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, a.data());
        cl::Buffer b_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, b.data());
        cl::Buffer c_buf(context, CL_MEM_WRITE_ONLY, sizeof(float) * n);

        kernel.setArg(0, a_buf);
        kernel.setArg(1, b_buf);
        kernel.setArg(2, c_buf);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);
        queue.enqueueReadBuffer(c_buf, CL_TRUE, 0, sizeof(float) * n, c.data());

        bool ok = true;
        for (size_t i = 0; i < n; ++i) {
            if (std::fabs(c[i] - (a[i] + b[i])) > 1e-5f) {
                ok = false;
                break;
            }
        }

        std::cout << (ok ? "Vector add verified.\n" : "Vector add failed.\n");
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const cl::Error& e) {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE && program_ready) {
            std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        }
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
