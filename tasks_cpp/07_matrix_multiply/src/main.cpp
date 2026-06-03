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
    "__kernel void matmul(__global const float* A, __global const float* B, __global float* C, int N) {"
    "  const int row = get_global_id(0);"
    "  const int col = get_global_id(1);"
    "  float sum = 0.0f;"
    "  for (int k = 0; k < N; ++k) {"
    "    sum += A[row * N + k] * B[k * N + col];"
    "  }"
    "  C[row * N + col] = sum;"
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
        const int N = 32;
        const size_t elems = static_cast<size_t>(N * N);
        std::vector<float> A(elems), B(elems), C(elems, 0.0f), ref(elems, 0.0f);
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                A[r * N + c] = static_cast<float>((r + c) % 7);
                B[r * N + c] = static_cast<float>((r * 3 + c) % 11);
            }
        }

        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                float s = 0.0f;
                for (int k = 0; k < N; ++k) s += A[r * N + k] * B[k * N + c];
                ref[r * N + c] = s;
            }
        }

        const cl::Device device = pick_first_device();
        cl::Context context(device);
        cl::CommandQueue queue(context, device);
        cl::Program program(context, kKernelSource);
        program.build({device});
        cl::Kernel kernel(program, "matmul");

        cl::Buffer A_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * elems, A.data());
        cl::Buffer B_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * elems, B.data());
        cl::Buffer C_buf(context, CL_MEM_WRITE_ONLY, sizeof(float) * elems);

        kernel.setArg(0, A_buf);
        kernel.setArg(1, B_buf);
        kernel.setArg(2, C_buf);
        kernel.setArg(3, N);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N, N), cl::NullRange);
        queue.enqueueReadBuffer(C_buf, CL_TRUE, 0, sizeof(float) * elems, C.data());

        bool ok = true;
        for (size_t i = 0; i < elems; ++i) {
            if (std::fabs(C[i] - ref[i]) > 1e-3f) {
                ok = false;
                break;
            }
        }

        std::cout << (ok ? "Matrix multiply verified.\n" : "Matrix multiply failed.\n");
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
