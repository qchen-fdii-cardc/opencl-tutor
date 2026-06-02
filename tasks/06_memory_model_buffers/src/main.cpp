#include <CL/cl.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static const char* kKernelSource =
    "__kernel void saxpy(__global const float* x, __global const float* y, __global float* z, float a) {"
    "  int gid = get_global_id(0);"
    "  z[gid] = a * x[gid] + y[gid];"
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
    const size_t n = 2048;
    const float alpha = 3.0f;
    std::vector<float> x(n), y(n), z_host(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i % 97);
        y[i] = static_cast<float>(i % 71);
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

    cl_kernel kernel = clCreateKernel(program, "saxpy", &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    const size_t bytes = n * sizeof(float);
    cl_mem x_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bytes, x.data(), &err);
    if (err != CL_SUCCESS || x_buf == nullptr) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }
    cl_mem y_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, y.data(), &err);
    if (err != CL_SUCCESS || y_buf == nullptr) {
        clReleaseMemObject(x_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }
    cl_mem z_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, &err);
    if (err != CL_SUCCESS || z_buf == nullptr) {
        clReleaseMemObject(y_buf);
        clReleaseMemObject(x_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &x_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &y_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &z_buf);
    clSetKernelArg(kernel, 3, sizeof(float), &alpha);

    size_t global_size = n;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return EXIT_FAILURE;
    clFinish(queue);

    void* mapped = clEnqueueMapBuffer(queue, z_buf, CL_TRUE, CL_MAP_READ, 0, bytes, 0, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || mapped == nullptr) return EXIT_FAILURE;

    const float* z_mapped = static_cast<const float*>(mapped);
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        z_host[i] = z_mapped[i];
        if (std::fabs(z_host[i] - (alpha * x[i] + y[i])) > 1e-4f) {
            ok = false;
        }
    }

    clEnqueueUnmapMemObject(queue, z_buf, mapped, 0, nullptr, nullptr);
    clFinish(queue);

    std::cout << "Buffer flags used: USE_HOST_PTR, COPY_HOST_PTR, ALLOC_HOST_PTR\n";
    std::cout << (ok ? "Memory task verified.\n" : "Memory task failed.\n");

    clReleaseMemObject(z_buf);
    clReleaseMemObject(y_buf);
    clReleaseMemObject(x_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
