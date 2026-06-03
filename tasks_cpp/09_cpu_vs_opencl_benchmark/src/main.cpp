#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#if __has_include(<CL/cl.hpp>)
#include <CL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include <chrono>
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
    try {
        const size_t n = 1 << 20;
        std::vector<float> a(n), b(n), c_cpu(n), c_gpu(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i % 131);
            b[i] = static_cast<float>(i % 47);
        }

        const auto cpu_begin = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; ++i) c_cpu[i] = a[i] + b[i];
        const auto cpu_end = std::chrono::high_resolution_clock::now();
        const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();

        const cl::Device device = pick_first_device();
        cl::Context context(device);
        cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
        cl::Program program(context, kKernelSource);
        program.build({device});
        cl::Kernel kernel(program, "vec_add");

        cl::Buffer a_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, a.data());
        cl::Buffer b_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, b.data());
        cl::Buffer c_buf(context, CL_MEM_WRITE_ONLY, sizeof(float) * n);
        kernel.setArg(0, a_buf);
        kernel.setArg(1, b_buf);
        kernel.setArg(2, c_buf);

        cl::Event evt;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange, nullptr, &evt);
        evt.wait();
        queue.enqueueReadBuffer(c_buf, CL_TRUE, 0, sizeof(float) * n, c_gpu.data());

        const cl_ulong t0 = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        const cl_ulong t1 = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        const double gpu_ms = static_cast<double>(t1 - t0) / 1e6;

        bool ok = true;
        for (size_t i = 0; i < n; ++i) {
            if (std::fabs(c_gpu[i] - c_cpu[i]) > 1e-5f) {
                ok = false;
                break;
            }
        }

        std::cout << "CPU time: " << cpu_ms << " ms\n";
        std::cout << "OpenCL kernel time: " << gpu_ms << " ms\n";
        std::cout << (ok ? "Benchmark result verified.\n" : "Benchmark result mismatch.\n");
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
