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
    "__kernel void vec_add(__global const float* a, __global const float* b, __global float* c) {"
    "  const int gid = get_global_id(0);"
    "  c[gid] = a[gid] + b[gid];"
    "}";

static std::vector<cl::Device> pick_devices() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (size_t i = 0; i < platforms.size(); ++i) {
        std::vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (!devices.empty()) return devices;
    }
    throw std::runtime_error("No usable OpenCL devices found.");
}

int main() {
    try {
        const size_t n = 1 << 20;
        std::vector<float> a(n), b(n), c(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i % 127);
            b[i] = static_cast<float>(i % 89);
        }

        const std::vector<cl::Device> devices = pick_devices();
        const size_t dev_count = devices.size();
        cl::Context context(devices);
        cl::Program program(context, kKernelSource);
        program.build(devices);

        for (size_t di = 0; di < dev_count; ++di) {
            cl::CommandQueue queue(context, devices[di]);
            cl::Kernel kernel(program, "vec_add");

            const size_t begin = (n * di) / dev_count;
            const size_t end = (n * (di + 1)) / dev_count;
            const size_t count = end - begin;

            cl::Buffer a_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * count, a.data() + begin);
            cl::Buffer b_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * count, b.data() + begin);
            cl::Buffer c_buf(context, CL_MEM_WRITE_ONLY, sizeof(float) * count);

            kernel.setArg(0, a_buf);
            kernel.setArg(1, b_buf);
            kernel.setArg(2, c_buf);

            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(count), cl::NullRange);
            queue.enqueueReadBuffer(c_buf, CL_TRUE, 0, sizeof(float) * count, c.data() + begin);
        }

        bool ok = true;
        for (size_t i = 0; i < n; ++i) {
            if (std::fabs(c[i] - (a[i] + b[i])) > 1e-5f) {
                ok = false;
                break;
            }
        }

        std::cout << "Devices used: " << dev_count << "\n";
        std::cout << (ok ? "Multi-device task verified.\n" : "Multi-device task failed.\n");
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
