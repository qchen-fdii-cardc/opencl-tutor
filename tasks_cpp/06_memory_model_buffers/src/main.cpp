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
    "__kernel void saxpy(__global const float* x, __global float* y, float alpha) {"
    "  const int gid = get_global_id(0);"
    "  y[gid] = alpha * x[gid] + y[gid];"
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
        const size_t n = 2048;
        std::vector<float> x(n), y(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = static_cast<float>(i) * 0.5f;
            y[i] = 1.0f;
        }

        const cl::Device device = pick_first_device();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        cl::Program program(context, kKernelSource);
        program.build({device});
        cl::Kernel kernel(program, "saxpy");

        cl::Buffer x_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, x.data());
        cl::Buffer y_buf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, y.data());

        kernel.setArg(0, x_buf);
        kernel.setArg(1, y_buf);
        kernel.setArg(2, 2.0f);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);
        queue.finish();

        float* mapped = static_cast<float*>(queue.enqueueMapBuffer(y_buf, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * n));
        bool ok = true;
        for (size_t i = 0; i < n; ++i) {
            const float expected = 2.0f * x[i] + 1.0f;
            if (std::fabs(mapped[i] - expected) > 1e-5f) {
                ok = false;
                break;
            }
        }
        queue.enqueueUnmapMemObject(y_buf, mapped);
        queue.finish();

        std::cout << (ok ? "Buffer memory model task verified.\n" : "Buffer memory model task failed.\n");
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
