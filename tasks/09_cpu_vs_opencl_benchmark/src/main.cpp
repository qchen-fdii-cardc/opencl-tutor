#include <CL/cl.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static const char* kKernelSource =
    "__kernel void vec_add(__global const float* a, __global const float* b, __global float* c) {"
    "  int gid = get_global_id(0);"
    "  c[gid] = a[gid] + b[gid];"
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
    const size_t n = 1 << 22;
    const size_t bytes = n * sizeof(float);
    std::vector<float> a(n), b(n), c_cpu(n), c_gpu(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i % 131);
        b[i] = static_cast<float>(i % 47);
    }

    const auto cpu_begin = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; ++i) {
        c_cpu[i] = a[i] + b[i];
    }
    const auto cpu_end = std::chrono::high_resolution_clock::now();
    const double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();

    cl_device_id device = nullptr;
    if (!pick_first_device(&device)) {
        std::cerr << "No usable OpenCL device found.\n";
        return EXIT_FAILURE;
    }

    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    cl_program program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    cl_kernel kernel = clCreateKernel(program, "vec_add", &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, a.data(), &err);
    if (err != CL_SUCCESS || a_buf == nullptr) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, b.data(), &err);
    if (err != CL_SUCCESS || b_buf == nullptr) {
        clReleaseMemObject(a_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }
    cl_mem c_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    if (err != CL_SUCCESS || c_buf == nullptr) {
        clReleaseMemObject(b_buf);
        clReleaseMemObject(a_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buf);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(c_buf);
        clReleaseMemObject(b_buf);
        clReleaseMemObject(a_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    size_t global_size = n;
    cl_event event = nullptr;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, &event);
    if (err == CL_SUCCESS) {
        clWaitForEvents(1, &event);
        err = clEnqueueReadBuffer(queue, c_buf, CL_TRUE, 0, bytes, c_gpu.data(), 0, nullptr, nullptr);
    }

    cl_ulong start_ns = 0;
    cl_ulong end_ns = 0;
    if (event != nullptr) {
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_ns), &start_ns, nullptr);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end_ns), &end_ns, nullptr);
    }

    bool ok = (err == CL_SUCCESS);
    for (size_t i = 0; ok && i < n; ++i) {
        if (std::fabs(c_cpu[i] - c_gpu[i]) > 1e-5f) {
            ok = false;
        }
    }

    const double ocl_ms = (end_ns > start_ns) ? (end_ns - start_ns) / 1e6 : 0.0;
    std::cout << "CPU time: " << cpu_ms << " ms\n";
    std::cout << "OpenCL kernel time: " << ocl_ms << " ms\n";
    if (ocl_ms > 0.0) {
        std::cout << "Speedup (CPU/OpenCL): " << (cpu_ms / ocl_ms) << "x\n";
    }
    std::cout << (ok ? "Benchmark result validated.\n" : "Benchmark validation failed.\n");

    if (event != nullptr) clReleaseEvent(event);
    clReleaseMemObject(c_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(a_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
