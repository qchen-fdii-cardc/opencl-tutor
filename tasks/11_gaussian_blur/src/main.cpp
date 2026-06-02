#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static const char* kKernelSource =
    "__kernel void gaussian3x3(__global const float* src, __global float* dst, int w, int h) {"
    "  int x = get_global_id(0);"
    "  int y = get_global_id(1);"
    "  if (x >= w || y >= h) return;"
    "  const float k[9] = {1,2,1,2,4,2,1,2,1};"
    "  float sum = 0.0f;"
    "  int idx = 0;"
    "  for (int dy = -1; dy <= 1; ++dy) {"
    "    for (int dx = -1; dx <= 1; ++dx) {"
    "      int xx = clamp(x + dx, 0, w - 1);"
    "      int yy = clamp(y + dy, 0, h - 1);"
    "      sum += src[yy * w + xx] * k[idx++];"
    "    }"
    "  }"
    "  dst[y * w + x] = sum / 16.0f;"
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
    const int w = 128;
    const int h = 128;
    const size_t total = static_cast<size_t>(w) * static_cast<size_t>(h);
    const size_t bytes = total * sizeof(float);

    std::vector<float> src(total), cpu(total), gpu(total);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            src[y * w + x] = static_cast<float>((x * 3 + y * 5) % 256) / 255.0f;
        }
    }

    const auto cpu_begin = std::chrono::high_resolution_clock::now();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            const float k[9] = {1,2,1,2,4,2,1,2,1};
            int idx = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = std::max(0, std::min(w - 1, x + dx));
                    int yy = std::max(0, std::min(h - 1, y + dy));
                    sum += src[yy * w + xx] * k[idx++];
                }
            }
            cpu[y * w + x] = sum / 16.0f;
        }
    }
    const auto cpu_end = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();

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

    cl_kernel kernel = clCreateKernel(program, "gaussian3x3", &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    cl_mem src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, src.data(), &err);
    if (err != CL_SUCCESS || src_buf == nullptr) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }
    cl_mem dst_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    if (err != CL_SUCCESS || dst_buf == nullptr) {
        clReleaseMemObject(src_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_buf);
    clSetKernelArg(kernel, 2, sizeof(int), &w);
    clSetKernelArg(kernel, 3, sizeof(int), &h);

    size_t global_size[2] = {static_cast<size_t>(w), static_cast<size_t>(h)};
    cl_event event = nullptr;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, nullptr, 0, nullptr, &event);
    if (err == CL_SUCCESS) {
        clWaitForEvents(1, &event);
        err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, bytes, gpu.data(), 0, nullptr, nullptr);
    }

    cl_ulong start_ns = 0;
    cl_ulong end_ns = 0;
    if (event != nullptr) {
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_ns), &start_ns, nullptr);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end_ns), &end_ns, nullptr);
    }

    bool ok = (err == CL_SUCCESS);
    for (size_t i = 0; ok && i < total; ++i) {
        if (std::fabs(cpu[i] - gpu[i]) > 1e-4f) {
            ok = false;
        }
    }

    const double gpu_ms = (end_ns > start_ns) ? (end_ns - start_ns) / 1e6 : 0.0;
    std::cout << "CPU blur time: " << cpu_ms << " ms\n";
    std::cout << "OpenCL blur time: " << gpu_ms << " ms\n";
    std::cout << (ok ? "Gaussian blur comparison verified.\n" : "Gaussian blur comparison failed.\n");

    if (event != nullptr) clReleaseEvent(event);
    clReleaseMemObject(dst_buf);
    clReleaseMemObject(src_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
