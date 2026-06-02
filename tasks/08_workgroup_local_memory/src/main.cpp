#include <CL/cl.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

static const char* kKernelSource =
    "__kernel void reduce_local(__global const float* input, __global float* partial, __local float* cache, int n) {"
    "  int gid = get_global_id(0);"
    "  int lid = get_local_id(0);"
    "  int group = get_group_id(0);"
    "  int local_size = get_local_size(0);"
    "  cache[lid] = (gid < n) ? input[gid] : 0.0f;"
    "  barrier(CLK_LOCAL_MEM_FENCE);"
    "  for (int stride = local_size / 2; stride > 0; stride >>= 1) {"
    "    if (lid < stride) cache[lid] += cache[lid + stride];"
    "    barrier(CLK_LOCAL_MEM_FENCE);"
    "  }"
    "  if (lid == 0) partial[group] = cache[0];"
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
    const int n = 1 << 16;
    const size_t local_size = 128;
    const size_t group_count = (static_cast<size_t>(n) + local_size - 1) / local_size;
    const size_t global_size = group_count * local_size;

    std::vector<float> input(n), partial(group_count, 0.0f);
    for (int i = 0; i < n; ++i) {
        input[i] = 1.0f + static_cast<float>(i % 5);
    }

    const float cpu_sum = std::accumulate(input.begin(), input.end(), 0.0f);

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

    cl_kernel kernel = clCreateKernel(program, "reduce_local", &err);
    if (err != CL_SUCCESS) return EXIT_FAILURE;

    cl_mem in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * input.size(), input.data(), &err);
    cl_mem partial_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        sizeof(float) * partial.size(), nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &partial_buf);
    clSetKernelArg(kernel, 2, sizeof(float) * local_size, nullptr);
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
    if (err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(queue, partial_buf, CL_TRUE, 0,
                                  sizeof(float) * partial.size(), partial.data(), 0, nullptr, nullptr);
    }

    float gpu_sum = std::accumulate(partial.begin(), partial.end(), 0.0f);
    bool ok = (err == CL_SUCCESS) && (std::fabs(gpu_sum - cpu_sum) < 1e-2f);

    std::cout << "CPU sum: " << cpu_sum << "\n";
    std::cout << "GPU(local memory) sum: " << gpu_sum << "\n";
    std::cout << (ok ? "Local memory reduction verified.\n" : "Local memory reduction failed.\n");

    clReleaseMemObject(partial_buf);
    clReleaseMemObject(in_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
