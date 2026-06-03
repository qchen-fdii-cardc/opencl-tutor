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
    "__kernel void monte_carlo_pi(__global const uint* seeds, __global uint* hits, int samples_per_item) {"
    "  int gid = get_global_id(0);"
    "  uint s = seeds[gid];"
    "  uint count = 0;"
    "  for (int i = 0; i < samples_per_item; ++i) {"
    "    s = 1664525u * s + 1013904223u;"
    "    float x = (float)(s & 0x00FFFFFF) / 16777216.0f;"
    "    s = 1664525u * s + 1013904223u;"
    "    float y = (float)(s & 0x00FFFFFF) / 16777216.0f;"
    "    if (x * x + y * y <= 1.0f) count++;"
    "  }"
    "  hits[gid] = count;"
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
    const size_t work_items = 4096;
    const int samples_per_item = 2048;
    std::vector<cl_uint> seeds(work_items), hits(work_items, 0);
    for (size_t i = 0; i < work_items; ++i) {
        seeds[i] = static_cast<cl_uint>(1234567u + i * 2654435761u);
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

    cl_kernel kernel = clCreateKernel(program, "monte_carlo_pi", &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    cl_mem seed_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(cl_uint) * seeds.size(), seeds.data(), &err);
    if (err != CL_SUCCESS || seed_buf == nullptr) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }
    cl_mem hit_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(cl_uint) * hits.size(), nullptr, &err);
    if (err != CL_SUCCESS || hit_buf == nullptr) {
        clReleaseMemObject(seed_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &seed_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &hit_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &samples_per_item);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(hit_buf);
        clReleaseMemObject(seed_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return EXIT_FAILURE;
    }

    size_t global_size = work_items;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(queue, hit_buf, CL_TRUE, 0,
                                  sizeof(cl_uint) * hits.size(), hits.data(), 0, nullptr, nullptr);
    }

    unsigned long long total_hits = 0;
    for (size_t i = 0; i < hits.size(); ++i) {
        total_hits += hits[i];
    }

    const double total_samples = static_cast<double>(work_items) * static_cast<double>(samples_per_item);
    const double pi_estimate = 4.0 * static_cast<double>(total_hits) / total_samples;
    const double error = std::fabs(pi_estimate - 3.14159265358979323846);
    const bool ok = (err == CL_SUCCESS) && (error < 0.05);

    std::cout << "Estimated pi: " << pi_estimate << "\n";
    std::cout << "Absolute error: " << error << "\n";
    std::cout << (ok ? "Non-graphics OpenCL task verified.\n" : "Non-graphics OpenCL task failed.\n");

    clReleaseMemObject(hit_buf);
    clReleaseMemObject(seed_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
