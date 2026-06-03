#if __has_include(<CL/cl.hpp>)
#include <CL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static const char* kKernelSource =
    "__kernel void scale(__global const float* in, __global float* out, float alpha) {"
    "  int gid = get_global_id(0);"
    "  out[gid] = in[gid] * alpha;"
    "}";

static bool pick_first_device(cl_device_id* out_device) {
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0) {
        return false;
    }
    std::vector<cl_platform_id> platforms(platform_count);
    if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS) {
        return false;
    }

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
    const size_t n = 1 << 20;
    const float alpha = 2.5f;
    std::vector<float> input(n), output(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i % 251);
    }

    cl_device_id device = nullptr;
    if (!pick_first_device(&device)) {
        std::cerr << "No usable OpenCL device found.\n";
        return EXIT_FAILURE;
    }

    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    cl_program program = clCreateProgramWithSource(context, 1, &kKernelSource, nullptr, &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    cl_kernel kernel = clCreateKernel(program, "scale", &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    const size_t bytes = n * sizeof(float);
    cl_mem in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, input.data(), &err);
    if (err != CL_SUCCESS || in_buf == nullptr) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }
    cl_mem out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    if (err != CL_SUCCESS || out_buf == nullptr) {
        clReleaseMemObject(in_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(float), &alpha);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(out_buf);
        clReleaseMemObject(in_buf);
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
        err = clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);
    }

    cl_ulong start_ns = 0;
    cl_ulong end_ns = 0;
    if (event != nullptr) {
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_ns), &start_ns, nullptr);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end_ns), &end_ns, nullptr);
    }

    bool ok = (err == CL_SUCCESS);
    for (size_t i = 0; ok && i < n; ++i) {
        if (std::fabs(output[i] - input[i] * alpha) > 1e-4f) {
            ok = false;
        }
    }

    const double kernel_ms = (end_ns > start_ns) ? (end_ns - start_ns) / 1e6 : 0.0;
    std::cout << "Kernel time (profiling): " << kernel_ms << " ms\n";
    std::cout << (ok ? "Timing task verified.\n" : "Timing task failed.\n");

    if (event != nullptr) clReleaseEvent(event);
    clReleaseMemObject(out_buf);
    clReleaseMemObject(in_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
