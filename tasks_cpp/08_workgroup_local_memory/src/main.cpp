#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#if __has_include(<CL/cl.hpp>)
#include <CL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

static const char* kKernelSource =
    "__kernel void local_add(__global const float* a, __global const float* b, __global float* c, __local float* lmem) {"
    "  const int gid = get_global_id(0);"
    "  const int lid = get_local_id(0);"
    "  lmem[lid] = a[gid] + b[gid];"
    "  barrier(CLK_LOCAL_MEM_FENCE);"
    "  c[gid] = lmem[lid];"
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
    try {
        const size_t n = 1024;
        const size_t local_size = 64;
        std::vector<float> a(n), b(n), c(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(2 * i);
        }

        const cl::Device device = pick_first_device();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cl::Program program(context, kKernelSource);
        program.build({device});
        cl::Kernel kernel(program, "local_add");

        cl::Buffer a_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, a.data());
        cl::Buffer b_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, b.data());
        cl::Buffer c_buf(context, CL_MEM_WRITE_ONLY, sizeof(float) * n);

        kernel.setArg(0, a_buf);
        kernel.setArg(1, b_buf);
        kernel.setArg(2, c_buf);
        kernel.setArg(3, cl::Local(sizeof(float) * local_size));

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NDRange(local_size));
        queue.enqueueReadBuffer(c_buf, CL_TRUE, 0, sizeof(float) * n, c.data());

        bool ok = true;
        for (size_t i = 0; i < n; ++i) {
            if (std::fabs(c[i] - (a[i] + b[i])) > 1e-5f) {
                ok = false;
                break;
            }
        }

        std::cout << (ok ? "Workgroup/local memory task verified.\n" : "Workgroup/local memory task failed.\n");
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
