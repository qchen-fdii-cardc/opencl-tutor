#include <CL/cl.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static const char* kKernelSource =
    "__kernel void mat_mul(__global const float* a, __global const float* b, __global float* c, int n) {"
    "  int row = get_global_id(1);"
    "  int col = get_global_id(0);"
    "  if (row >= n || col >= n) return;"
    "  float sum = 0.0f;"
    "  for (int k = 0; k < n; ++k) sum += a[row * n + k] * b[k * n + col];"
    "  c[row * n + col] = sum;"
    "}";

static bool pick_first_device(cl_device_id* out_device) {
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0) return false;
    std::vector<cl_platform_id> platforms(platform_count);
    if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS) return false;
    for (cl_uint i = 0; i < platform_count; ++i) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count) != CL_SUCCESS ||
            device_count == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(device_count);
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr) == CL_SUCCESS) {
            *out_device = devices[0];
            return true;
        }
    }
    return false;
}

int main() {
    const int n = 32;
    const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = total * sizeof(float);

    std::vector<float> a(total), b(total), c_gpu(total, 0.0f), c_cpu(total, 0.0f);
    for (size_t i = 0; i < total; ++i) {
        a[i] = static_cast<float>((i % 13) + 1);
        b[i] = static_cast<float>((i % 7) + 1);
    }

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += a[row * n + k] * b[k * n + col];
            }
            c_cpu[row * n + col] = sum;
        }
    }

    cl_device_id device = nullptr;
    if (!pick_first_device(&device)) {
        std::cerr << "No usable OpenCL device found.\n";
        return EXIT_FAILURE;
    }

    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    cl_program program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    cl_kernel kernel = clCreateKernel(program, "mat_mul", &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, a.data(), &err);
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, b.data(), &err);
    cl_mem c_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buf);
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    size_t global_size[2] = {static_cast<size_t>(n), static_cast<size_t>(n)};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr);
    if (err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(queue, c_buf, CL_TRUE, 0, bytes, c_gpu.data(), 0, nullptr, nullptr);
    }

    bool ok = (err == CL_SUCCESS);
    for (size_t i = 0; ok && i < total; ++i) {
        if (std::fabs(c_gpu[i] - c_cpu[i]) > 1e-3f) {
            ok = false;
        }
    }

    std::cout << (ok ? "Matrix multiply verified.\n" : "Matrix multiply failed.\n");

    clReleaseMemObject(c_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(a_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
